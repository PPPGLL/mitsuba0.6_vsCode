/*
    This file is part of Mitsuba, a physically based rendering system.

    Copyright (c) 2007-2014 by Wenzel Jakob and others.

    Mitsuba is free software; you can redistribute it and/or modify
    it under the terms of the GNU General Public License Version 3
    as published by the Free Software Foundation.

    Mitsuba is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program. If not, see <http://www.gnu.org/licenses/>.
*/

#include <mitsuba/render/util.h>
#include <mitsuba/core/timer.h>
#include <mitsuba/core/fresolver.h>
#include <mitsuba/core/fstream.h>
#include <mitsuba/core/bitmap.h>
#include <mitsuba/core/plugin.h>
#include <boost/algorithm/string.hpp>
#include <chrono>
#include <fstream>
#if defined(WIN32)
# include <mitsuba/core/getopt.h>
#endif
#ifdef MTS_OPENMP
# include <omp.h>
#endif

MTS_NAMESPACE_BEGIN

class show_pdf_BSDF : public Utility {
public:
	void help() {
		cout << endl;
		cout << "Synopsis: Loads one or more EXR/RGBE images and writes tonemapped 8-bit PNG/JPGs";
		cout << endl;
		cout << "Usage: mtsutil tonemap [options] <EXR/RGBE file (s)>" << endl;
		cout << "Options/Arguments:" << endl;
		cout << "   -h             Display this help text" << endl << endl;
		cout << "   -g gamma       Specify the gamma value (The default is -1 => sRGB)" << endl << endl;
		cout << "   -m multiplier  Multiply the pixel values by 'multiplier' (Default = 1)" << endl << endl;
		cout << "   -b r,g,b       Color balance: apply the specified per-channel multipliers" << endl << endl;
		cout << "   -c x,y,w,h     Crop: tonemap a given rectangle instead of the entire image" << endl << endl;
		cout << "   -s w,h         Resize the output image to the specified resolution" << endl << endl;
		cout << "   -F name        Name of the resampling filter used by -s (Default = lanczos)" << endl << endl;
		cout << "   -r x,y,w,h,i   Add a rectangle at the specified position and intensity, e.g." << endl;
		cout << "                  to make paper figures. The intensity should be in [0, 255]." << endl << endl;
		cout << "   -f fmt         Request a certain output format (png/jpg, default:png)" << endl << endl;
		cout << "   -a             Require the output image to have an alpha channel" << endl << endl;
		cout << "   -p key,burn    Run Reinhard et al.'s photographic tonemapping operator. 'key'" << endl;
		cout << "                  between [0, 1] chooses between low and high-key images and" << endl;
		cout << "                  'burn' (also [0, 1]) controls how much highlights may burn out" << endl << endl;
		cout << "   -B fov         Apply a bloom filter that simulates scattering in the human" << endl;
		cout << "                  eye. Requires the approx. field of view of the images to be" << endl;
		cout << "                  processed in order to compute a point spread function." << endl << endl;
		cout << "   -x             Temporal coherence mode: activate this flag when tonemapping " << endl;
		cout << "                  frames of an animation using the '-p' option to avoid flicker" << endl << endl;
		cout << "   -o file        Save the output with a given filename" << endl << endl;
		cout << "   -t             Multithreaded: process several files in parallel" << endl << endl;
		cout << " The operations are ordered as follows: 1. crop, 2. bloom, 3. resize, 4. color" << endl;
		cout << " balance, 5. tonemap, 6. annotate. To simply process a directory full of EXRs" << endl;
		cout << " in parallel, run the following: 'mtsutil tonemap -t path-to-directory/*.exr'" << endl;
	}

	//void WriteValue(Vector3 wi, Vector3 wo, Float pdf, Spectrum throughput){
	//	std::cout << "wi: (" << wi.x << "," << wi.y << "," << wi.z << ")" << std::endl;
	//	std::cout << "wo: (" << wo.x << "," << wo.y << "," << wo.z << ")" << std::endl;
	//	std::cout << "pdf: " << pdf << std::endl;
	//	std::cout << "throughput: (" << throughput[0] << "," << throughput[1] << "," << throughput[2] << ")" << std::endl;
	//}

	void WriteValue(Vector3 wi, Vector3 wo, Float pdf, Spectrum throughput, const std::string& filename) {
		std::ofstream file(filename, std::ios::app);
		if (!file.is_open()) {
			std::cerr << "Unable to open file: " << filename << std::endl;
			return;
		}

		file << wi.x << "," << wi.y << "," << wi.z << std::endl;
		file << wo.x << "," << wo.y << "," << wo.z << std::endl;
		file << pdf << std::endl;
		file << throughput[0] << "," << throughput[1] << "," << throughput[2] << std::endl;
		file << std::endl;

		file.close();
	}

	void WriteValueToBuffer(std::ostringstream& buffer, Vector3 wi, Vector3 wo, Float pdf, Spectrum throughput) {
		buffer << wi.x << "," << wi.y << "," << wi.z << std::endl;
		buffer << wo.x << "," << wo.y << "," << wo.z << std::endl;
		buffer << pdf << std::endl;
		buffer << throughput[0] << "," << throughput[1] << "," << throughput[2] << std::endl;
		buffer << std::endl;
	}


	int run(int argc, char **argv) {
		ref<FileResolver> fileResolver = Thread::getThread()->getFileResolver();

		optind = 1;
		if (argv[optind][1] == 'p'){
			optind++;
			ref<Scene> scene;
			ref<ShapeKDTree> kdtree;

			std::string lowercase = boost::to_lower_copy(std::string(argv[optind]));
			if (boost::ends_with(lowercase, ".xml")) {
				fs::path
					filename = fileResolver->resolve(argv[optind]),
					filePath = fs::absolute(filename).parent_path(),
					baseName = filename.stem();
				ref<FileResolver> frClone = fileResolver->clone();
				frClone->prependPath(filePath);
				Thread::getThread()->setFileResolver(frClone);
				scene = loadScene(argv[optind]);
				kdtree = scene->getKDTree();
			}
			else {
				Log(EError, "The supplied scene filename must end in XML!");
			}

			optind++;
			int spp = std::stoi(argv[optind]);

			/* Create an bitmap */
			int resolution = 512;
			ref<Bitmap> bitmap_pdf = new Bitmap(Bitmap::ELuminance, Bitmap::EFloat32, Vector2i(resolution));

			Ray ray = Ray(Point(0.f, 0.f, 0.f), Vector3(0.f, 0.f, 1.f), 0.0f);
			Intersection its;
			scene->rayIntersect(ray, its);
			Sampler *sampler = scene->getSampler();
			BSDFSamplingRecord bRec(its, sampler);
			BSDF* bsdf = scene->getShapes()[0]->getBSDF();

			const Vector3 wi_input = { 0.0f, 0.0f, 1.0f };
			//const Vector3 wi_input = { 1.0f, 0.0f, 0.05f };
			//its.uv = Point2(0.9, 0.9);
			Vector3 wi = normalize(wi_input);
			bRec.wi = wi;
			Float wox = 0, woy = 0, wox_spp = 0, woy_spp = 0;
			Float pdf = 0;

			int samples_pix = 0;
			for (int i = 0; i < resolution; i++) {
				wox = -1.0f + 2.0f * i / (resolution - 1);
				for (int j = 0; j < resolution; j++) {
					woy = -1.0f + 2.0f * j / (resolution - 1);
					pdf = 0.0f;

					for (int s = 0; s < spp; s++){
						Point2 r = sampler->next2D();
						wox_spp = wox + 2.0f * 1 / (resolution - 1) * r.x;
						woy_spp = woy + 2.0f * 1 / (resolution - 1) * r.y;
						float z_squared = 1.0f - wox_spp * wox_spp - woy_spp * woy_spp;
						if (z_squared < 0){
							continue;
						}
						else {
							Float woz = std::sqrt(z_squared);
							bRec.wo = Vector3(wox_spp, woy_spp, woz);
							pdf += bsdf->pdf(bRec) / spp;
						}
					}

					bitmap_pdf->setPixel(Point2i(j, i), Spectrum(pdf));   // Exchange i,j here, don't know why but it works :)
				}
			}
			ref<FileStream> fs = new FileStream("PDF_GT.exr", FileStream::ETruncReadWrite);
			bitmap_pdf->write(Bitmap::EOpenEXR, fs);
			cout << "write map down" << endl;
		}
		else if (argv[optind][1] == 'e'){
			optind++;
			ref<Scene> scene;
			ref<ShapeKDTree> kdtree;

			std::string lowercase = boost::to_lower_copy(std::string(argv[optind]));
			if (boost::ends_with(lowercase, ".xml")) {
				fs::path
					filename = fileResolver->resolve(argv[optind]),
					filePath = fs::absolute(filename).parent_path(),
					baseName = filename.stem();
				ref<FileResolver> frClone = fileResolver->clone();
				frClone->prependPath(filePath);
				Thread::getThread()->setFileResolver(frClone);
				scene = loadScene(argv[optind]);
				kdtree = scene->getKDTree();
			}
			else {
				Log(EError, "The supplied scene filename must end in XML!");
			}

			optind++;
			int spp = std::stoi(argv[optind]);

			/* Create an bitmap */
			int resolution = 256;
			ref<Bitmap> bitmap_pdf = new Bitmap(Bitmap::ERGB, Bitmap::EFloat32, Vector2i(resolution));

			Ray ray = Ray(Point(0.f, 0.f, 0.f), Vector3(0.f, 0.f, 1.f), 0.0f);
			Intersection its;
			scene->rayIntersect(ray, its);
			Sampler *sampler = scene->getSampler();
			BSDFSamplingRecord bRec(its, sampler);
			BSDF* bsdf = scene->getShapes()[0]->getBSDF();

			const Vector3 wi_input = { 0.0f, 0.0f, 1.0f };
			//const Vector3 wi_input = { 1.0f, 0.0f, 0.05f };

			Vector3 wi = normalize(wi_input);
			bRec.wi = wi;
			Float wox = 0, woy = 0, wox_spp = 0, woy_spp = 0;
			Float pdf = 0;
			Spectrum eval(0.0f);
			int samples_pix = 0;
			for (int i = 0; i < resolution; i++) {
				wox = -1.0f + 2.0f * i / (resolution - 1);
				for (int j = 0; j < resolution; j++) {
					woy = -1.0f + 2.0f * j / (resolution - 1);
					pdf = 0.0f;
					eval.fromSRGB(0, 0, 0);
					for (int s = 0; s < spp; s++){
						Point2 r = sampler->next2D();
						wox_spp = wox + 2.0f * 1 / (resolution - 1) * r.x;
						woy_spp = woy + 2.0f * 1 / (resolution - 1) * r.y;
						float z_squared = 1.0f - wox_spp * wox_spp - woy_spp * woy_spp;
						if (z_squared < 0){
							continue;
						}
						else {
							Float woz = std::sqrt(z_squared);
							bRec.wo = Vector3(wox_spp, woy_spp, woz);
							eval += bsdf->eval(bRec) / spp;
						}
					}

					bitmap_pdf->setPixel(Point2i(j, i), Spectrum(eval));   // Exchange i,j here, don't know why but it works :)
				}
			}
			ref<FileStream> fs = new FileStream("EVAL_GT.exr", FileStream::ETruncReadWrite);
			bitmap_pdf->write(Bitmap::EOpenEXR, fs);
			cout << "write map down" << endl;
		}
		else if (argv[optind][1] == 't'){
			optind++;
			ref<Scene> scene;
			ref<ShapeKDTree> kdtree;

			std::string lowercase = boost::to_lower_copy(std::string(argv[optind]));
			if (boost::ends_with(lowercase, ".xml")) {
				fs::path
					filename = fileResolver->resolve(argv[optind]),
					filePath = fs::absolute(filename).parent_path(),
					baseName = filename.stem();
				ref<FileResolver> frClone = fileResolver->clone();
				frClone->prependPath(filePath);
				Thread::getThread()->setFileResolver(frClone);
				scene = loadScene(argv[optind]);
				kdtree = scene->getKDTree();
			}
			else {
				Log(EError, "The supplied scene filename must end in XML!");
			}

			optind++;
			int spp = std::stoi(argv[optind]);

			/* Create an bitmap */
			int resolution = 256;
			ref<Bitmap> bitmap_pdf = new Bitmap(Bitmap::ERGB, Bitmap::EFloat32, Vector2i(resolution));

			Ray ray = Ray(Point(0.f, 0.f, 0.f), Vector3(0.f, 0.f, 1.f), 0.0f);
			Intersection its;
			scene->rayIntersect(ray, its);
			Sampler *sampler = scene->getSampler();
			BSDFSamplingRecord bRec(its, sampler);
			BSDF* bsdf = scene->getShapes()[0]->getBSDF();

			const Vector3 wi_input = { 0.0f, 0.0f, 1.0f };

			Vector3 wi = normalize(wi_input);
			bRec.wi = wi;
			Float wox = 0, woy = 0, wox_spp = 0, woy_spp = 0;
			Float pdf = 0;
			Spectrum throughput(0.0f);
			Spectrum eval(0.0f);
			int samples_pix = 0;
			for (int i = 0; i < resolution; i++) {
				wox = -1.0f + 2.0f * i / (resolution - 1);
				for (int j = 0; j < resolution; j++) {
					woy = -1.0f + 2.0f * j / (resolution - 1);
					pdf = 0.0f;
					eval.fromSRGB(0, 0, 0);
					throughput.fromSRGB(0, 0, 0);
					for (int s = 0; s < spp; s++){
						Point2 r = sampler->next2D();
						wox_spp = wox + 2.0f * 1 / (resolution - 1) * r.x;
						woy_spp = woy + 2.0f * 1 / (resolution - 1) * r.y;
						float z_squared = 1.0f - wox_spp * wox_spp - woy_spp * woy_spp;
						if (z_squared < 0){
							continue;
						}
						else {
							Float woz = std::sqrt(z_squared);
							bRec.wo = Vector3(wox_spp, woy_spp, woz);
							eval = bsdf->eval(bRec);
							pdf = bsdf->pdf(bRec);
							if (pdf <= 1e-5){
								throughput += Spectrum(0.0);
							}
							else {
								throughput += eval / pdf / spp;
							}
							

						}
					}

					bitmap_pdf->setPixel(Point2i(j, i), Spectrum(throughput));   // Exchange i,j here, don't know why but it works :)
				}
			}
			ref<FileStream> fs = new FileStream("THROUGHPUT_GT.exr", FileStream::ETruncReadWrite);
			bitmap_pdf->write(Bitmap::EOpenEXR, fs);
			cout << "write map down" << endl;
		}
		else if (argv[optind][1] == 'u'){
			optind++;
			ref<Scene> scene;
			ref<ShapeKDTree> kdtree;

			std::string lowercase = boost::to_lower_copy(std::string(argv[optind]));
			if (boost::ends_with(lowercase, ".xml")) {
				fs::path
					filename = fileResolver->resolve(argv[optind]),
					filePath = fs::absolute(filename).parent_path(),
					baseName = filename.stem();
				ref<FileResolver> frClone = fileResolver->clone();
				frClone->prependPath(filePath);
				Thread::getThread()->setFileResolver(frClone);
				scene = loadScene(argv[optind]);
				kdtree = scene->getKDTree();
			}
			else {
				Log(EError, "The supplied scene filename must end in XML!");
			}

			optind++;
			int spp = std::stoi(argv[optind]);

			/* Create an bitmap */
			int resolution = 256;
			ref<Bitmap> bitmap_pdf = new Bitmap(Bitmap::ERGB, Bitmap::EFloat32, Vector2i(resolution));

			Ray ray = Ray(Point(0.f, 0.f, 0.f), Vector3(0.f, 0.f, 1.f), 0.0f);
			Intersection its;
			scene->rayIntersect(ray, its);
			Sampler *sampler = scene->getSampler();
			BSDFSamplingRecord bRec(its, sampler);
			BSDF* bsdf = scene->getShapes()[0]->getBSDF();

			const Vector3 wi_input = { 0.0f, 0.0f, 1.0f };
			//const Vector3 wi_input = { 1.0f, 0.0f, 0.05f };

			Vector3 wi = normalize(wi_input);
			bRec.wi = wi;
			Float wox = 0, woy = 0, wox_spp = 0, woy_spp = 0;
			Float pdf = 0;
			Spectrum eval(0.0f);
			int samples_pix = 0;
			for (int i = 0; i < resolution; i++) {
				wox = -1.0f + 2.0f * i / (resolution - 1);
				for (int j = 0; j < resolution; j++) {
					woy = -1.0f + 2.0f * j / (resolution - 1);
					pdf = 0.0f;
					eval.fromSRGB(0, 0, 0);
					
					for (int s = 0; s < spp; s++){			
						int N = 512; // ÿ�������ϻ��ֵ�����
						int x = 0;  // ѡ�е��ӿ��ں������������ 0 ��ʼ��
						int y = 0; // ѡ�е��ӿ������������

						Point2 uv_raw = sampler->next2D(); // ԭʼ�� [0,1) ����
						Point2 uv_tile = Point2(
							(x + uv_raw.x) / float(N),
							(y + uv_raw.y) / float(N)
							);
						/*Point2 uv_tile = Point2(
							Float(147/N),
							Float(296 / N)
							);*/
						//bRec.its.uv.x = uv.x;
						its.uv = uv_tile;
						BSDFSamplingRecord bRec_temp(its, sampler);
						bRec_temp.wi = wi;

						Point2 r = sampler->next2D();
						wox_spp = wox + 2.0f * 1 / (resolution - 1) * r.x;
						woy_spp = woy + 2.0f * 1 / (resolution - 1) * r.y;
						float z_squared = 1.0f - wox_spp * wox_spp - woy_spp * woy_spp;
						if (z_squared < 0){
							continue;
						}
						else {
							Float woz = std::sqrt(z_squared);
							bRec_temp.wo = Vector3(wox_spp, woy_spp, woz);
							eval += bsdf->eval(bRec_temp) / spp;
						}
					}

					bitmap_pdf->setPixel(Point2i(j, i), Spectrum(eval));   // Exchange i,j here, don't know why but it works :)
				}
			}
			ref<FileStream> fs = new FileStream("EVAL_GT.exr", FileStream::ETruncReadWrite);
			bitmap_pdf->write(Bitmap::EOpenEXR, fs);
			cout << "write map down" << endl;
		}

		else if (argv[optind][1] == 'U'){
			optind++;
			ref<Scene> scene;
			ref<ShapeKDTree> kdtree;

			std::string lowercase = boost::to_lower_copy(std::string(argv[optind]));
			if (boost::ends_with(lowercase, ".xml")) {
				fs::path
					filename = fileResolver->resolve(argv[optind]),
					filePath = fs::absolute(filename).parent_path(),
					baseName = filename.stem();
				ref<FileResolver> frClone = fileResolver->clone();
				frClone->prependPath(filePath);
				Thread::getThread()->setFileResolver(frClone);
				scene = loadScene(argv[optind]);
				kdtree = scene->getKDTree();
			}
			else {
				Log(EError, "The supplied scene filename must end in XML!");
			}

			optind++;
			int spp = std::stoi(argv[optind]);

			/* Create an bitmap */
			int resolution = 256;
			ref<Bitmap> bitmap_pdf = new Bitmap(Bitmap::ERGB, Bitmap::EFloat32, Vector2i(resolution));

			Ray ray = Ray(Point(0.f, 0.f, 0.f), Vector3(0.f, 0.f, 1.f), 0.0f);
			Intersection its;
			scene->rayIntersect(ray, its);
			Sampler *sampler = scene->getSampler();
			BSDFSamplingRecord bRec(its, sampler);
			BSDF* bsdf = scene->getShapes()[0]->getBSDF();

			const Vector3 wi_input = { 0.0f, 0.0f, 1.0f };
			//const Vector3 wi_input = { 1.0f, 0.0f, 0.05f };

			Vector3 wi = normalize(wi_input);
			bRec.wi = wi;
			Float wox = 0, woy = 0, wox_spp = 0, woy_spp = 0;
			Float pdf = 0;
			Spectrum eval(0.0f);
			int samples_pix = 0;
			int N = 16; // tile ������

			for (int tile_x = 0; tile_x < N; ++tile_x) {
				for (int tile_y = 0; tile_y < N; ++tile_y) {

					ref<Bitmap> bitmap_pdf = new Bitmap(Bitmap::ERGB, Bitmap::EFloat32, Vector2i(resolution));

					for (int i = 0; i < resolution; i++) {
						wox = -1.0f + 2.0f * i / (resolution - 1);
						for (int j = 0; j < resolution; j++) {
							woy = -1.0f + 2.0f * j / (resolution - 1);
							pdf = 0.0f;
							eval.fromSRGB(0, 0, 0);

							for (int s = 0; s < spp; s++){
								Point2 uv_raw = sampler->next2D(); // ԭʼ�� [0,1) ����
								Point2 uv_tile = Point2(
									(tile_x + uv_raw.x) / float(N),
									(tile_y + uv_raw.y) / float(N)
									);
								its.uv = uv_tile;

								BSDFSamplingRecord bRec_temp(its, sampler);
								bRec_temp.wi = wi;

								Point2 r = sampler->next2D();
								wox_spp = wox + 2.0f * 1 / (resolution - 1) * r.x;
								woy_spp = woy + 2.0f * 1 / (resolution - 1) * r.y;
								float z_squared = 1.0f - wox_spp * wox_spp - woy_spp * woy_spp;
								if (z_squared < 0){
									continue;
								}
								else {
									Float woz = std::sqrt(z_squared);
									bRec_temp.wo = Vector3(wox_spp, woy_spp, woz);
									eval += bsdf->eval(bRec_temp) / spp;
								}
							}

							bitmap_pdf->setPixel(Point2i(j, i), Spectrum(eval));
						}
					}

					std::ostringstream filename;
					filename << "vis/EVAL_GT_tile_" << tile_x << "_" << tile_y << ".exr";
					ref<FileStream> fs = new FileStream(filename.str(), FileStream::ETruncReadWrite);
					bitmap_pdf->write(Bitmap::EOpenEXR, fs);
					cout << "Wrote " << filename.str() << endl;
				}
			}
		}
		
		return 0;
	}

	MTS_DECLARE_UTILITY()
};

MTS_EXPORT_UTILITY(show_pdf_BSDF, "Command line samlpe bsdf")
MTS_NAMESPACE_END
