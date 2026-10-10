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
#include <mitsuba/core/getopt.h>
#endif
#ifdef MTS_OPENMP
# include <omp.h>
#endif

#include "hdf5.h"
#include "hdf5_util.h"
#define FILE_NAME "test.h5"
#define DATASET_NAME "data"
#define LOGVec(v) (cout<<#v<<": "<<v.x<<","<<v.y<<","<<v.z<<endl)

MTS_NAMESPACE_BEGIN

class sample_microstructure : public Utility {
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


	Vector3 w_disk2sphere(Point2 w){
		auto z = 1 - w.x*w.x - w.y*w.y;
		return Vector3(w.x, w.y, z);
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
			scene->initialize();
			kdtree = scene->getKDTree();
		}
		else {
			Log(EError, "The supplied scene filename must end in XML!");
		}

		optind++;
		int wi_num = std::stoi(argv[optind]);
		optind++;
		int wo_num_per_wi = std::stoi(argv[optind]);

		int total_num = wi_num * wo_num_per_wi;
		wi_list.resize(wi_num);
		wo_list.resize(total_num);
		throughput_list.resize(total_num);
		normal_list.resize(wi_num);
		uv_list.resize(wi_num);

		Sampler *sampler = scene->getSampler();

		Float scale = 0.5f;
		int mat_size_x = 95;
		int mat_size_y = 95;
		int mat_size_z = 4;
		int maxDepth = 10;

		mat_size_x *= scale;
		mat_size_y *= scale;
		mat_size_z *= scale;
#pragma omp parallel for
		for (int i = 0; i < wi_num; i++) {
			Point2 start_uv = sampler->next2D();
			Point start_pos = Point((start_uv.x - 0.5)*mat_size_x, (start_uv.y - 0.5)*mat_size_y, 0);
			Vector3 wi = warp::squareToUniformHemisphere(sampler->next2D());
			Point3 ori_pos = start_pos + wi*100;
			//LOGVec(ori_pos);
			Ray ray = Ray(ori_pos, -wi, 0.0f);
			//LOGVec(-wi);
			Intersection its;
			bool hit = kdtree->rayIntersect(ray, its);
			//cout <<"ori hit? "<< hit << endl;
			//cout << "time is:" << its.t << endl;
			
			Vector3 normal(0.0f, 0.0f, 1.0f);
			std::array<float, 3> wi_array = { wi.x, wi.y, wi.z };
			std::array<float, 2> uv_array = { start_uv.x, start_uv.y };
			wi_list[i] = wi_array;
			uv_list[i] = uv_array;
			for (int j = 0; j < wo_num_per_wi; j++){
				Spectrum throughput(1.0f);
				Float eta = 1.0f;
				Vector wo(0.0f, 0.0f, 1.0f);
				int depth = 0;
				Intersection its_traced = Intersection(its);
				while (depth <= maxDepth || maxDepth < 0) {
					if (!its_traced.isValid() && depth == 0){
						//cout << "first bounce miss! depth is: "<< depth << endl;
						throughput = Spectrum(0.0f);
						break;
					}
					const BSDF *bsdf = its_traced.getBSDF();
					BSDFSamplingRecord bRec(its_traced, sampler, ERadiance);
					Spectrum bsdfWeight = bsdf->sample(bRec, sampler->next2D());
					wo = its_traced.toWorld(bRec.wo);
					throughput *= bsdfWeight;
					eta *= bRec.eta;
					//depth++;
					ray = Ray(its_traced.p, wo, ray.time);
					if (!scene->rayIntersect(ray, its_traced)) {
						//cout << "miss! depth is: " << depth << endl;
						if (dot(wo, normal) < 0){
							throughput = Spectrum(0.0f);
							//cout << "transmission! depth is: " << depth << endl;

						}
						break;
					}

					if (depth++ >= 5) {
						/* Russian roulette: try to keep path weights equal to one,
						while accounting for the solid angle compression at refractive
						index boundaries. Stop with at least some probability to avoid
						getting stuck (e.g. due to total internal reflection) */

						Float q = std::min(throughput.max() * eta * eta, (Float) 0.95f);
						if (sampler->next1D() >= q)
							break;
						throughput /= q;
					}

				}
				//cout << "depth is: " << depth << endl;
				std::array<float, 3> wo_array = { wo.x, wo.y, wo.z };
				std::array<float, 3> throughput_array = { throughput[0], throughput[1], throughput[2] };
				int index = i * wo_num_per_wi + j;
				wo_list[index] = wo_array;
				throughput_list[index] = throughput_array;

			}

		}

	}

	void GenDataMultiplePGL(char **argv){
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
			scene->initialize();
			kdtree = scene->getKDTree();
		}
		else {
			Log(EError, "The supplied scene filename must end in XML!");
		}

		optind++;
		int wi_num = std::stoi(argv[optind]);
		optind++;
		int wo_num_per_wi = std::stoi(argv[optind]);

		int total_num = wi_num * wo_num_per_wi;
		wi_list.resize(wi_num);
		wo_list.resize(total_num);
		throughput_list.resize(total_num);
		normal_list.resize(wi_num);
		uv_list.resize(wi_num);

		Sampler *sampler = scene->getSampler();

		Float scale = 0.5f;
		int mat_size_x = 95;
		int mat_size_y = 95;
		int mat_size_z = 4;
		int maxDepth = 10;

		mat_size_x *= scale;
		mat_size_y *= scale;
		mat_size_z *= scale;
#pragma omp parallel for
		for (int i = 0; i < wi_num; i++) {
			Point2 start_uv = sampler->next2D();
			Point start_pos_up = Point((start_uv.x - 0.5)*mat_size_x, (start_uv.y - 0.5)*mat_size_y, 10);
			Vector3 up = Vector3(0.0f, 0.0f, 1.0f);
			Ray upRay = Ray(start_pos_up, -up, 0.0f);
			Intersection surfaceIts;
			kdtree->rayIntersect(upRay, surfaceIts);
			Point start_pos = surfaceIts.p;

			Vector3 wi = warp::squareToUniformHemisphere(sampler->next2D());
			Point3 ori_pos = start_pos + wi*100;
			//LOGVec(ori_pos);
			Ray ray = Ray(ori_pos, -wi, 0.0f);
			//LOGVec(-wi);
			Intersection its;
			bool hit = kdtree->rayIntersect(ray, its);
			//cout <<"ori hit? "<< hit << endl;
			//cout << "time is:" << its.t << endl;

			// bool occlusion = false;
			// Vector3 dif = its.p - surfaceIts.p;
			// if(dif.length() > Epsilon)
			// 	occlusion = true;
			
			Vector3 normal(0.0f, 0.0f, 1.0f);
			std::array<float, 3> wi_array = { wi.x, wi.y, wi.z };
			std::array<float, 2> uv_array = { start_uv.x, start_uv.y };
			wi_list[i] = wi_array;
			uv_list[i] = uv_array;
			for (int j = 0; j < wo_num_per_wi; j++){
				Spectrum throughput(1.0f);
				Float eta = 1.0f;
				Vector wo(0.0f, 0.0f, 1.0f);
				int depth = 0;
				Intersection its_traced = Intersection(its);
				while (depth <= maxDepth || maxDepth < 0) {
					// if (occlusion){
					// 	throughput = Spectrum(0.0f);
					// 	break;
					// }
					if (!its_traced.isValid() && depth == 0){
						//cout << "first bounce miss! depth is: "<< depth << endl;
						throughput = Spectrum(0.0f);
						break;
					}
					const BSDF *bsdf = its_traced.getBSDF();
					BSDFSamplingRecord bRec(its_traced, sampler, ERadiance);
					Spectrum bsdfWeight = bsdf->sample(bRec, sampler->next2D());
					wo = its_traced.toWorld(bRec.wo);
					throughput *= bsdfWeight;
					eta *= bRec.eta;
					//depth++;
					ray = Ray(its_traced.p, wo, ray.time);
					if (!scene->rayIntersect(ray, its_traced)) {
						//cout << "miss! depth is: " << depth << endl;
						if (dot(wo, normal) < 0){
							throughput = Spectrum(0.0f);
							//cout << "transmission! depth is: " << depth << endl;

						}
						break;
					}

					if (depth++ >= 5) {
						/* Russian roulette: try to keep path weights equal to one,
						while accounting for the solid angle compression at refractive
						index boundaries. Stop with at least some probability to avoid
						getting stuck (e.g. due to total internal reflection) */

						Float q = std::min(throughput.max() * eta * eta, (Float) 0.95f);
						if (sampler->next1D() >= q){
							throughput = Spectrum(0.0f);
							break;
						}
						throughput /= q;
					}

				}
				if (depth > maxDepth){
					throughput = Spectrum(0.0f);
				}
				//cout << "depth is: " << depth << endl;
				std::array<float, 3> wo_array = { wo.x, wo.y, wo.z };
				std::array<float, 3> throughput_array = { throughput[0], throughput[1], throughput[2] };
				int index = i * wo_num_per_wi + j;
				wo_list[index] = wo_array;
				throughput_list[index] = throughput_array;
			}

		}

	}

	void GenDataMultipleVis(char **argv){
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
			scene->initialize();
			kdtree = scene->getKDTree();
		}
		else {
			Log(EError, "The supplied scene filename must end in XML!");
		}

		optind++;
		int wi_num = std::stoi(argv[optind]);
		optind++;
		int wo_num_per_wi = std::stoi(argv[optind]);

		int total_num = wi_num * wo_num_per_wi;
		wi_list.resize(wi_num);
		wo_list.resize(total_num);
		throughput_list.resize(total_num);
		normal_list.resize(wi_num);
		uv_list.resize(wi_num);

		Sampler *sampler = scene->getSampler();

		Float scale = 0.5f;
		int mat_size_x = 95;
		int mat_size_y = 95;
		int mat_size_z = 4;
		int maxDepth = 10;

		mat_size_x *= scale;
		mat_size_y *= scale;
		mat_size_z *= scale;
#pragma omp parallel for
		for (int i = 0; i < wi_num; i++) {
			Point2 start_uv(0.5, 0.5005263);
			Point start_pos_up = Point((start_uv.x - 0.5) * mat_size_x, (start_uv.y - 0.5) * mat_size_y, 10);
			Vector3 up = Vector3(0.0f, 0.0f, 1.0f);
			Ray upRay = Ray(start_pos_up, -up, 0.0f);
			Intersection surfaceIts;
			kdtree->rayIntersect(upRay, surfaceIts);
			Point start_pos = surfaceIts.p;

			Vector3 wi = Vector3(0.0f, 0.0f, 1.0f);//warp::squareToUniformHemisphere(sampler->next2D());
			Point3 ori_pos = start_pos + wi*100;
			//LOGVec(ori_pos);
			Ray ray = Ray(ori_pos, -wi, 0.0f);
			//LOGVec(-wi);
			Intersection its;
			bool hit = kdtree->rayIntersect(ray, its);
			//cout <<"ori hit? "<< hit << endl;
			//cout << "time is:" << its.t << endl;

			// bool occlusion = false;
			// Vector3 dif = its.p - surfaceIts.p;
			// if(dif.length() > Epsilon)
			// 	occlusion = true;
			
			Vector3 normal(0.0f, 0.0f, 1.0f);
			std::array<float, 3> wi_array = { wi.x, wi.y, wi.z };
			std::array<float, 2> uv_array = { start_uv.x, start_uv.y };
			wi_list[i] = wi_array;
			uv_list[i] = uv_array;
			for (int j = 0; j < wo_num_per_wi; j++){
				Spectrum throughput(1.0f);
				Float eta = 1.0f;
				Vector wo(0.0f, 0.0f, 1.0f);
				int depth = 0;
				Intersection its_traced = Intersection(its);
				while (depth <= maxDepth || maxDepth < 0) {
					// if (occlusion){
					// 	throughput = Spectrum(0.0f);
					// 	break;
					// }
					if (!its_traced.isValid() && depth == 0){
						//cout << "first bounce miss! depth is: "<< depth << endl;
						throughput = Spectrum(0.0f);
						break;
					}
					const BSDF *bsdf = its_traced.getBSDF();
					BSDFSamplingRecord bRec(its_traced, sampler, ERadiance);
					Spectrum bsdfWeight = bsdf->sample(bRec, sampler->next2D());
					wo = its_traced.toWorld(bRec.wo);
					throughput *= bsdfWeight;
					eta *= bRec.eta;
					//depth++;
					ray = Ray(its_traced.p, wo, ray.time);
					if (!scene->rayIntersect(ray, its_traced)) {
						//cout << "miss! depth is: " << depth << endl;
						if (dot(wo, normal) < 0){
							throughput = Spectrum(0.0f);
							//cout << "transmission! depth is: " << depth << endl;

						}
						break;
					}

					if (depth++ >= 5) {
						/* Russian roulette: try to keep path weights equal to one,
						while accounting for the solid angle compression at refractive
						index boundaries. Stop with at least some probability to avoid
						getting stuck (e.g. due to total internal reflection) */

						Float q = std::min(throughput.max() * eta * eta, (Float) 0.95f);
						if (sampler->next1D() >= q)
							break;
						throughput /= q;
					}

				}
				//cout << "depth is: " << depth << endl;
				std::array<float, 3> wo_array = { wo.x, wo.y, wo.z };
				std::array<float, 3> throughput_array = { throughput[0], throughput[1], throughput[2] };
				int index = i * wo_num_per_wi + j;
				wo_list[index] = wo_array;
				throughput_list[index] = throughput_array;
			}

		}

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
		}
			break;
		case 'm':
			optind++;
			GenDataMultiple(argv);
			break;

		case 'p':
			optind++;
			GenDataMultiplePGL(argv);
			break;

		case 'v':
			optind++;
			GenDataMultipleVis(argv);
			break;

		default:
			cout << "Error: Unknown command! run '-h' to show the useage." << endl;
			return 0;
			break;
		}
		auto end = std::chrono::high_resolution_clock::now();

		HDF5Writer writer("bsdf_dataset/microstrcture_data.h5");
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

std::vector<std::array<float, 3>> sample_microstructure::wi_list;
std::vector<std::array<float, 3>> sample_microstructure::wo_list;
std::vector<std::array<float, 3>> sample_microstructure::throughput_list;
std::vector<std::array<float, 3>> sample_microstructure::normal_list;
std::vector<std::array<float, 2>> sample_microstructure::uv_list;

MTS_EXPORT_UTILITY(sample_microstructure, "Command line samlpe microstructure")
MTS_NAMESPACE_END