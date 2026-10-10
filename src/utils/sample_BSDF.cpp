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
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/xml_parser.hpp>
#include <chrono>
#include <fstream>
#if defined(WIN32)
#pragma message("Compiling on Windows")
# include <mitsuba/core/getopt.h>
#endif
#ifdef MTS_OPENMP
# include <omp.h>
#endif

#include "hdf5.h"
#include "hdf5_util.h"
#define FILE_NAME "test.h5"
#define DATASET_NAME "data"

MTS_NAMESPACE_BEGIN

/*
 * The dataset generator is parallelized over incident directions.  Sharing the
 * scene sampler between OpenMP workers races its mutable random-number state
 * and makes both the samples and the output non-reproducible.  Keep a small
 * sampler local to every worker instead.  Distinct, deterministic SFMT seeds
 * give each worker an independent stream while preserving reproducibility for
 * a fixed worker count.
 */
class SampleBSDFThreadSampler : public Sampler {
public:
	SampleBSDFThreadSampler(uint64_t seed)
		: Sampler(Properties()), m_random(new Random(seed)) {
		m_sampleCount = 1;
	}

	SampleBSDFThreadSampler(Random *random)
		: Sampler(Properties()), m_random(new Random(random)) {
		m_sampleCount = 1;
	}

	ref<Sampler> clone() {
		return new SampleBSDFThreadSampler(m_random.get());
	}

	Float next1D() {
		return m_random->nextFloat();
	}

	Point2 next2D() {
		return Point2(m_random->nextFloat(), m_random->nextFloat());
	}

private:
	ref<Random> m_random;
};

class samlple_BSDF : public Utility {
public:
	void help() {
		cout << endl;
		cout << "Synopsis: Generate wi & wo from a bsdf";
		cout << endl;
		cout << "Usage: mtsutil tonemap [options] <EXR/RGBE file (s)>" << endl;
		cout << "Options/Arguments:" << endl;
		cout << "   -h                         Display this help text" << endl << endl;
		cout << "   -m scene wi_num wo_num     Generate 'wi_num' wi and for each wi generate 'wo_num' wo" << endl << endl;
		cout << "   -p scene num               Generate 'num' wi and wo pairs" << endl << endl;
		cout << "   -s scene num               Generate 'num' wo with the wi=(0,0,1)" << endl << endl;
		cout << " Note: This program will read the BSDF component of the first shape in the scene file." << endl;

	}

	//void WriteValue(Vector3 wi, Vector3 wo, Float pdf, Spectrum throughput){
	//	std::cout << "wi: (" << wi.x << "," << wi.y << "," << wi.z << ")" << std::endl;
	//	std::cout << "wo: (" << wo.x << "," << wo.y << "," << wo.z << ")" << std::endl;
	//	std::cout << "pdf: " << pdf << std::endl;
	//	std::cout << "throughput: (" << throughput[0] << "," << throughput[1] << "," << throughput[2] << ")" << std::endl;
	//}

	static std::vector<std::array<float, 3>> wi_list, wo_list, throughput_list, normal_list;
	static std::vector<std::array<float, 2>> uv_list;

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

	void find_and_modify_spectrum(boost::property_tree::ptree& tree, const std::string& targetName, const std::string& newValue) {
		for (auto& node : tree) {
			if (node.first == "spectrum" && node.second.get<std::string>("name") == targetName) {
				node.second.put("value", newValue);
				break;
			}
			find_and_modify_spectrum(node.second, targetName, newValue);
		}
	}

	Vector3 w_disk2sphere(Point2 w){
		auto z = 1 - w.x*w.x - w.y*w.y;
		return Vector3(w.x, w.y, z);
	}

	struct RayData_conditional {
		int condition_len = 3;
		Float* condition;
		Vector3f wo;
		Spectrum throughput;
		RayData_conditional() = default;
		// ??????
		RayData_conditional(const int condition_len_, Float* condition_, const Vector3f& wo_, const Spectrum& throughput_)
			: condition_len(condition_len_), condition(condition_), wo(wo_), throughput(throughput_) {}
	};

	struct RayData {
		Vector3f wi;
		Vector3f wo;
		Spectrum throughput;
		RayData() = default;
		// ??????
		RayData(const Vector3f& wi_, const Vector3f& wo_, const Spectrum& throughput_)
			: wi(wi_), wo(wo_), throughput(throughput_) {}
	};

	void WriteValueToBuffer(std::ostringstream& buffer, Vector3f wi, Vector3f wo, Float pdf, Spectrum throughput) {
		buffer << wi.x << "," << wi.y << "," << wi.z << std::endl;
		buffer << wo.x << "," << wo.y << "," << wo.z << std::endl;
		buffer << pdf << std::endl;
		buffer << throughput[0] << "," << throughput[1] << "," << throughput[2] << std::endl;
		buffer << std::endl;
	}

	void WriteValueToBuffer(std::ostringstream& buffer, RayData raydata) {
		auto wi = raydata.wi;
		auto wo = raydata.wo;
		auto throughput = raydata.throughput;
		buffer << wi.x << "," << wi.y << "," << wi.z << std::endl;
		buffer << wo.x << "," << wo.y << "," << wo.z << std::endl;
		buffer << 1.0f << std::endl;
		buffer << throughput[0] << "," << throughput[1] << "," << throughput[2] << std::endl;
		buffer << std::endl;
	}

	void WriteValueToBufferBinary(std::ostringstream& buffer, const RayData& raydata) {
		auto wi = raydata.wi;
		auto wo = raydata.wo;
		auto throughput = raydata.throughput;

		buffer.write(reinterpret_cast<const char*>(&wi[0]), sizeof(Float)); // ��?? wi (vec3)
		buffer.write(reinterpret_cast<const char*>(&wi[1]), sizeof(Float));
		buffer.write(reinterpret_cast<const char*>(&wi[2]), sizeof(Float));
		buffer.write(reinterpret_cast<const char*>(&wo[0]), sizeof(Float)); // ��?? wo (vec3)
		buffer.write(reinterpret_cast<const char*>(&wo[1]), sizeof(Float));
		buffer.write(reinterpret_cast<const char*>(&wo[2]), sizeof(Float));
		buffer.write(reinterpret_cast<const char*>(&throughput[0]), sizeof(Float)); // ��?? throughput (vec3)
		buffer.write(reinterpret_cast<const char*>(&throughput[1]), sizeof(Float));
		buffer.write(reinterpret_cast<const char*>(&throughput[2]), sizeof(Float));
	}

	void WriteValueToBufferVaringSigmaT(std::ostringstream& buffer, Vector3 wi, Vector3 wo, Float pdf, Spectrum throughput, Float sigmaT) {
		buffer << wi.x << "," << wi.y << "," << wi.z << std::endl;
		buffer << wo.x << "," << wo.y << "," << wo.z << std::endl;
		buffer << pdf << std::endl;
		buffer << throughput[0] << "," << throughput[1] << "," << throughput[2] << std::endl;
		buffer << sigmaT << std::endl;
		buffer << std::endl;
	}

	void WriteValueToBufferConditional(std::ostringstream& buffer, RayData_conditional raydata) {
		auto wo = raydata.wo;
		auto throughput = raydata.throughput;
		auto condition_len = raydata.condition_len;
		for (int i = 0; i < condition_len - 1; i++){
			buffer << raydata.condition[i] << ",";
		}
		buffer << raydata.condition[condition_len - 1] << endl;

		buffer << wo.x << "," << wo.y << "," << wo.z << std::endl;
		buffer << 1.0f << std::endl;
		buffer << throughput[0] << "," << throughput[1] << "," << throughput[2] << std::endl;
		buffer << std::endl;
	}

	void GenDataMultiple(char **argv){
		ref<FileResolver> fileResolver = Thread::getThread()->getFileResolver();
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
		int wi_num = std::stoi(argv[optind]);
		optind++;
		int wo_num_per_wi = std::stoi(argv[optind]);

		Ray ray = Ray(Point(0.f, 0.f, 0.f), Vector3(0.f, 0.f, 1.f), 0.0f);
		Intersection its;
		scene->rayIntersect(ray, its);
			BSDF* bsdf = scene->getShapes()[0]->getBSDF();

			int total_num = wi_num * wo_num_per_wi;
			wi_list.resize(wi_num);
			wo_list.resize(total_num);
			throughput_list.resize(total_num);

	#pragma omp parallel
			{
				int thread_id = 0;
	#ifdef MTS_OPENMP
				thread_id = omp_get_thread_num();
	#endif
				ref<Sampler> sampler = new SampleBSDFThreadSampler(
					UINT64_C(0x53414d504c455242) + (uint64_t) thread_id);

	#pragma omp for schedule(static)
				for (int i = 0; i < wi_num; i++) {
				Vector3 wi = warp::squareToUniformHemisphere(sampler->next2D());
				std::array<float, 3> wi_array = { wi.x, wi.y, wi.z };
				wi_list[i] = wi_array;

			for (int j = 0; j < wo_num_per_wi; j++) {
				Point2 rand = sampler->next2D();
				Float pdf = 0;
					BSDFSamplingRecord bRec(its, sampler.get());
				bRec.wi = wi;
				Spectrum throughput = bsdf->sample(bRec, rand);
				std::array<float, 3> wo_array = { bRec.wo.x, bRec.wo.y, bRec.wo.z };
				std::array<float, 3> throughput_array = { throughput[0], throughput[1], throughput[2] };
				int index = i * wo_num_per_wi + j;
				wo_list[index] = wo_array;
				throughput_list[index] = throughput_array;
			}

		}
		}
	}


		void GenDataSingleWi(char **argv){
		ref<FileResolver> fileResolver = Thread::getThread()->getFileResolver();
		ref<Scene> scene;
		ref<ShapeKDTree> kdtree;
		char* argv_file = argv[optind];
		std::string lowercase = boost::to_lower_copy(std::string(argv[optind]));

		Vector3 normal = normalize(Vector3(0.0, 0.0, 1));

		if (boost::ends_with(lowercase, ".xml")) {
			using boost::property_tree::ptree;
			ptree tree;
			read_xml(std::string(argv_file), tree);

			std::string normal_x_str = std::to_string(normal.x);
			std::string normal_y_str = std::to_string(normal.y);
			std::string normal_z_str = std::to_string(normal.z);

			//print_tree(tree);
			for (auto& bsdf : tree.get_child("scene.bsdf")) {
				if (bsdf.first == "vector") {
					auto name_node = bsdf.second.get_optional<std::string>("<xmlattr>.name");
					if (name_node && name_node.get() == "normal_0") {
						bsdf.second.put("<xmlattr>.x", normal_x_str);
						bsdf.second.put("<xmlattr>.y", normal_y_str);
						bsdf.second.put("<xmlattr>.z", normal_z_str);
						break;
					}
				}
			}

			write_xml(std::string(argv_file), tree);

			fs::path
				filename = fileResolver->resolve(argv[optind]),
				filePath = fs::absolute(filename).parent_path(),
				baseName = filename.stem();
			ref<FileResolver> frClone = fileResolver->clone();
			frClone->prependPath(filePath);
			Thread::getThread()->setFileResolver(frClone);
			scene = loadScene(argv_file);
			kdtree = scene->getKDTree();

		}
		else {
			Log(EError, "The supplied scene filename must end in XML!");
		}
		
		optind++;
		int wi_num = 1;
		int wo_num_per_wi = std::stoi(argv[optind]);

		Ray ray = Ray(Point(0.f, 0.f, 0.f), Vector3(0.f, 0.f, 1.f), 0.0f);
		Intersection its;
		scene->rayIntersect(ray, its);
		Sampler *sampler = scene->getSampler();

		BSDF* bsdf = scene->getShapes()[0]->getBSDF();

		int total_num = wi_num * wo_num_per_wi;
		wi_list.resize(wi_num);
		wo_list.resize(total_num);
		throughput_list.resize(total_num);
		normal_list.resize(wi_num);
		uv_list.resize(wi_num);

		
		Vector3 wi = normalize(Vector3(0.0, 0.0, 1));
		
		std::array<float, 3> wi_array = { wi.x, wi.y, wi.z };
		std::array<float, 3> normal_array = { normal.x, normal.y, normal.z };
		wi_list[0] = wi_array;
		normal_list[0] = normal_array;

#pragma omp parallel for
		for (int j = 0; j < wo_num_per_wi; j++) {
			Point2 rand = sampler->next2D();
			Float pdf = 0;
			BSDFSamplingRecord bRec(its, sampler);
			bRec.wi = wi;
			Spectrum throughput = bsdf->sample(bRec, rand);
			std::array<float, 3> wo_array = { bRec.wo.x, bRec.wo.y, bRec.wo.z };
			std::array<float, 3> throughput_array = { throughput[0], throughput[1], throughput[2] };
			int index = j;
			wo_list[index] = wo_array;
			throughput_list[index] = throughput_array;
		}

		
	}

	void print_tree(const boost::property_tree::ptree& tree, const std::string& prefix = "") {
		for (const auto& node : tree) {
			std::cout << prefix << node.first << " (";
			for (const auto& attr : node.second) {
				std::cout << attr.first << ": " << attr.second.data() << ", ";
			}
			std::cout << ")" << std::endl;


			print_tree(node.second, prefix + "  ");
		}
	}

	void GenDataMultipleVaryingNormal(char **argv){
		ref<FileResolver> fileResolver = Thread::getThread()->getFileResolver();
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
		char* argv_file = argv[optind];
		optind++;
		int condition_num = std::stoi(argv[optind]);
		optind++;
		int wo_num_per_wi = std::stoi(argv[optind]);
		int total_num = condition_num * wo_num_per_wi;
		wi_list.resize(condition_num);
		wo_list.resize(total_num);
		throughput_list.resize(total_num);
		normal_list.resize(condition_num);
		if (boost::ends_with(lowercase, ".xml")) {
			using boost::property_tree::ptree;
			ptree tree;
			fs::path
				filename = fileResolver->resolve(argv_file),
				filePath = fs::absolute(filename).parent_path(),
				baseName = filename.stem();
			read_xml(std::string(argv_file), tree);
			scene = loadScene(argv_file);
			Ray ray = Ray(Point(0.f, 0.f, 0.f), Vector3(0.f, 0.f, 1.f), 0.0f);
			Intersection its;
			scene->rayIntersect(ray, its);
			Sampler *sampler = scene->getSampler();
			for (int i = 0; i < condition_num; i++) {
				Vector3 normal = warp::squareToUniformHemisphere(sampler->next2D());
				std::string normal_x_str = std::to_string(normal.x);
				std::string normal_y_str = std::to_string(normal.y);
				std::string normal_z_str = std::to_string(normal.z);

				//print_tree(tree);
				for (auto& bsdf : tree.get_child("scene.bsdf")) {
					if (bsdf.first == "vector") {
						auto name_node = bsdf.second.get_optional<std::string>("<xmlattr>.name");
						if (name_node && name_node.get() == "normal_0") {
							bsdf.second.put("<xmlattr>.x", normal_x_str);
							bsdf.second.put("<xmlattr>.y", normal_y_str);
							bsdf.second.put("<xmlattr>.z", normal_z_str);
							break;
						}
					}
				}

				write_xml(std::string(argv_file), tree);
				//cout << "change normal" << endl;
				auto scene_new = loadScene(argv_file);

				scene_new->rayIntersect(ray, its);
				BSDF* bsdf = scene_new->getShapes()[0]->getBSDF();
				//Vector3 wi = normalize(Vector3(0.0, 0.0, 1));
				Vector3 wi = warp::squareToUniformHemisphere(sampler->next2D());
				std::array<float, 3> wi_array = { wi.x, wi.y, wi.z };
				std::array<float, 3> normal_array = { normal.x, normal.y, normal.z };
				wi_list[i] = wi_array;
				normal_list[i] = normal_array;
				BSDFSamplingRecord bRec(its, sampler);
				bRec.wi = wi;
#pragma omp parallel for
					for (int j = 0; j < wo_num_per_wi; j++) {
						Point2 rand = sampler->next2D();
						BSDFSamplingRecord bRec(its, sampler);
					bRec.wi = wi;
					Spectrum throughput = bsdf->sample(bRec, rand);
					
					std::array<float, 3> wo_array = { bRec.wo.x, bRec.wo.y, bRec.wo.z };
					std::array<float, 3> throughput_array = { throughput[0], throughput[1], throughput[2] };
					

					int index = i * wo_num_per_wi + j;

					
					wo_list[index] = wo_array;
					throughput_list[index] = throughput_array;
					
				}

			}
		}

	}


	void GenDataMultipleVaryingUV(char **argv){
		ref<FileResolver> fileResolver = Thread::getThread()->getFileResolver();
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
		char* argv_file = argv[optind];
		optind++;
		int condition_num = std::stoi(argv[optind]);
		optind++;
		int wo_num_per_wi = std::stoi(argv[optind]);
		int total_num = condition_num * wo_num_per_wi;
		wi_list.resize(condition_num);
		wo_list.resize(total_num);
		throughput_list.resize(total_num);
		uv_list.resize(condition_num);

		if (boost::ends_with(lowercase, ".xml")) {
			using boost::property_tree::ptree;
			ptree tree;
			fs::path
				filename = fileResolver->resolve(argv_file),
				filePath = fs::absolute(filename).parent_path(),
				baseName = filename.stem();
			read_xml(std::string(argv_file), tree);
			scene = loadScene(argv_file);
			Ray ray = Ray(Point(0.f, 0.f, 0.f), Vector3(0.f, 0.f, 1.f), 0.0f);
			Intersection its;
			scene->rayIntersect(ray, its);
			Sampler *sampler = scene->getSampler();
#pragma omp parallel for
			for (int i = 0; i < condition_num; i++) {
				Point2 uv = sampler->next2D();
				Intersection its_sample = Intersection(its);
				//print_tree(tree);
				its_sample.uv = uv;
				BSDF* bsdf = scene->getShapes()[0]->getBSDF();
				//Vector3 wi = normalize(Vector3(0.0, 0.0, 1));
				Vector3 wi = warp::squareToUniformHemisphere(sampler->next2D());
				std::array<float, 3> wi_array = { wi.x, wi.y, wi.z };
				std::array<float, 2> uv_array = { uv.x, uv.y };

				wi_list[i] = wi_array;
				uv_list[i] = uv_array;
				BSDFSamplingRecord bRec(its_sample, sampler);
				bRec.wi = wi;
				for (int j = 0; j < wo_num_per_wi; j++) {
					Point2 rand = sampler->next2D();
					BSDFSamplingRecord bRec(its_sample, sampler);
					bRec.wi = wi;
					Spectrum throughput = bsdf->sample(bRec, rand);

					std::array<float, 3> wo_array = { bRec.wo.x, bRec.wo.y, bRec.wo.z };
					std::array<float, 3> throughput_array = { throughput[0], throughput[1], throughput[2] };


					int index = i * wo_num_per_wi + j;


					wo_list[index] = wo_array;
					throughput_list[index] = throughput_array;

				}

			}
		}

	}

	std::ostringstream GenDataMultipleVaryingSigmaT(char **argv){
		ref<FileResolver> fileResolver = Thread::getThread()->getFileResolver();
		ref<Scene> scene;
		ref<ShapeKDTree> kdtree;
		char* argv_file = argv[optind];
		optind++;
		int wi_num = std::stoi(argv[optind]);
		optind++;
		int wo_num_per_wi = std::stoi(argv[optind]);
		optind++;
		int sigmaT_num = std::stoi(argv[optind]);
		std::string lowercase = boost::to_lower_copy(std::string(argv_file));

		std::ostringstream buffer;
		buffer << wi_num << " " << wo_num_per_wi << " " << sigmaT_num << std::endl;

		if (boost::ends_with(lowercase, ".xml")) {
			using boost::property_tree::ptree;
			ptree tree;
			fs::path
				filename = fileResolver->resolve(argv_file),
				filePath = fs::absolute(filename).parent_path(),
				baseName = filename.stem();
			read_xml(std::string(argv_file), tree);

			for (int s = 0; s<sigmaT_num; s++){
				Float sigmaT = s / Float(sigmaT_num - 1);
				std::string sigmaT_str = std::to_string(sigmaT);

				//print_tree(tree);
				for (auto& bsdf : tree.get_child("scene.bsdf")) {
					if (bsdf.first == "spectrum") {
						auto name_node = bsdf.second.get_optional<std::string>("<xmlattr>.name");
						if (name_node && name_node.get() == "sigmaT_0") {
							bsdf.second.put("<xmlattr>.value", sigmaT_str);
							break;
						}
					}
				}

				write_xml(std::string(argv_file), tree);

				ref<FileResolver> frClone = fileResolver->clone();
				frClone->prependPath(filePath);
				Thread::getThread()->setFileResolver(frClone);
				scene = loadScene(argv_file);
				kdtree = scene->getKDTree();

				Ray ray = Ray(Point(0.f, 0.f, 0.f), Vector3(0.f, 0.f, 1.f), 0.0f);
				Intersection its;
				scene->rayIntersect(ray, its);
				Sampler *sampler = scene->getSampler();
				BSDFSamplingRecord bRec(its, sampler);
				BSDF* bsdf = scene->getShapes()[0]->getBSDF();
				for (int i = 0; i < wi_num; i++) {
					Vector3 wi = warp::squareToUniformHemisphere(sampler->next2D());
					//Vector3 wi = Vector3(0.f, 0.f, 1.f);
					bRec.wi = wi;
					for (int j = 0; j < wo_num_per_wi; j++) {
						Float pdf = 0;
						Spectrum throughput = bsdf->sample(bRec, pdf, sampler->next2D());
						Vector3 wo = bRec.wo;
						WriteValueToBufferVaringSigmaT(buffer, wi, wo, pdf, throughput, sigmaT);
					}
				}
			}
		}
		else {
			Log(EError, "The supplied scene filename must end in XML!");
		}


		return buffer;
	}

	int run(int argc, char **argv) {

		omp_set_num_threads(24);


		optind = 1;

		auto start = std::chrono::high_resolution_clock::now();

		//std::vector

		int wi_num = 0;
		Float wox = 0, woy = 0;

		switch (argv[optind][1])
		{
		case 'h': {
			help();
			return 0;
			break;
		}
			
		case 'm':
			optind++;
			GenDataMultiple(argv);
			break;

		case 's':
			optind++;
			GenDataSingleWi(argv);
			break;

		case 'N':
			optind++;
			GenDataMultipleVaryingNormal(argv);
			break;

		case 'U':
			optind++;
			GenDataMultipleVaryingUV(argv);
			break;
			/*case 'S':
			optind++;
			buffer = GenDataMultipleVaryingSigmaT(argv);
			break;
			case 'b':
			optind++;
			buffer = GenDataMultipleBinning(argv);
			break;*/

		default:
			cout << "Error: Unknown command! run '-h' to show the useage." << endl;
			return 0;
			break;
		}
		auto end = std::chrono::high_resolution_clock::now();

		HDF5Writer writer("bsdf_dataset/positionfree_data.h5");
		writer.writeVector3Array("wi", wi_list);
		writer.writeVector3Array("wo", wo_list);
		writer.writeVector3Array("throughput", throughput_list);
		writer.writeVector3Array("normal", normal_list);
		writer.writeVector2Array("uv", uv_list);

		auto duration = std::chrono::duration_cast<std::chrono::seconds>(end - start);
		std::cout << "Time cost: " << duration.count() << " s" << std::endl;

		return 0;
	}

	MTS_DECLARE_UTILITY()
};

std::vector<std::array<float, 3>> samlple_BSDF::wi_list;
std::vector<std::array<float, 3>> samlple_BSDF::wo_list;
std::vector<std::array<float, 3>> samlple_BSDF::throughput_list;
std::vector<std::array<float, 3>> samlple_BSDF::normal_list;
std::vector<std::array<float, 2>> samlple_BSDF::uv_list;

MTS_EXPORT_UTILITY(samlple_BSDF, "Command line samlpe bsdf")
MTS_NAMESPACE_END
