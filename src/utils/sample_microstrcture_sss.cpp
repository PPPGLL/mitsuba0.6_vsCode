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
#include <cmath>
#include <cstdint>
#include <fstream>
#include <limits>
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
		cout << "   -u scene wi_num wo_num     Generate fixed wi=(0,0,1) with stratified-uniform incident UV" << endl << endl;
		cout << " Note: This program will read the BSDF component of the first shape in the scene file." << endl;

	}

	//void WriteValue(Vector3 wi, Vector3 wo, Float pdf, Spectrum throughput){
	//	std::cout << "wi: (" << wi.x << "," << wi.y << "," << wi.z << ")" << std::endl;
	//	std::cout << "wo: (" << wo.x << "," << wo.y << "," << wo.z << ")" << std::endl;
	//	std::cout << "pdf: " << pdf << std::endl;
	//	std::cout << "throughput: (" << throughput[0] << "," << throughput[1] << "," << throughput[2] << ")" << std::endl;
	//}

	static std::vector<std::array<float, 3>> wi_list, wo_list, throughput_list, normal_list;
	static std::vector<std::array<float, 2>> uv_list, uv_wo_list;
	static std::vector<uint32_t> bounce_count_list, flat_bounce_count_list,
		micro_bounce_count_list, surface_mask_list, first_surface_type_list,
		first_bsdf_type_list, bsdf_type_mask_list, path_class_list,
		termination_code_list;
	static std::vector<int32_t> first_bsdf_component_list;
	static std::vector<float> first_hit_z_list, max_abs_hit_z_list;
	static bool path_labels_available;

	enum ESurfaceType : uint32_t {
		ESurfaceFlatBase = 1,
		ESurfaceMicrogeometry = 2
	};

	enum EPathClass : uint32_t {
		EPathInvalid = 0,
		EPathFlatSingle = 1,
		EPathMicroSingle = 2,
		EPathFlatMulti = 3,
		EPathMicroMulti = 4,
		EPathMixedMulti = 5
	};

	enum ETerminationCode : uint32_t {
		ETerminationUnknown = 0,
		ETerminationEscapedUpper = 1,
		ETerminationEscapedLower = 2,
		ETerminationRussianRoulette = 3,
		ETerminationMaxDepth = 4,
		ETerminationInitialMiss = 5
	};

	uint32_t classifySurface(const Intersection &its) const {
		// Heightfield scenes use z=0 for the unstructured base and signed
		// non-zero height for designed microgeometry (including negative pits).
		return std::abs(its.p.z) > 1e-5f
			? ESurfaceMicrogeometry
			: ESurfaceFlatBase;
	}

	uint32_t classifyPath(uint32_t bounceCount, uint32_t surfaceMask,
		uint32_t terminationCode, const Spectrum &throughput) const {
		if (terminationCode != ETerminationEscapedUpper || throughput.max() <= Epsilon)
			return EPathInvalid;
		if (bounceCount == 1)
			return surfaceMask == ESurfaceFlatBase ? EPathFlatSingle : EPathMicroSingle;
		if (bounceCount >= 2) {
			if (surfaceMask == ESurfaceFlatBase)
				return EPathFlatMulti;
			if (surfaceMask == ESurfaceMicrogeometry)
				return EPathMicroMulti;
			return EPathMixedMulti;
		}
		return EPathInvalid;
	}


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

	void GenDataMultiplePGL(char **argv, bool fixed_wi_uniform_uv = false){
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
		uv_wo_list.resize(total_num);
		bounce_count_list.assign(total_num, 0);
		flat_bounce_count_list.assign(total_num, 0);
		micro_bounce_count_list.assign(total_num, 0);
		surface_mask_list.assign(total_num, 0);
		first_surface_type_list.assign(total_num, 0);
		first_bsdf_type_list.assign(total_num, 0);
		bsdf_type_mask_list.assign(total_num, 0);
		path_class_list.assign(total_num, EPathInvalid);
		termination_code_list.assign(total_num, ETerminationUnknown);
		first_bsdf_component_list.assign(total_num, -1);
		first_hit_z_list.assign(total_num, 0.f);
		max_abs_hit_z_list.assign(total_num, 0.f);
		path_labels_available = true;

		// Sampler *sampler = scene->getSampler();

		Float mat_size_x = 95;
		Float mat_size_y = 95;
		Float mat_size_z = 4;
		int maxDepth = 10;

		Float scale = 0.13810526f;//phone81
		//Float scale = 0.1398105f;//phone82
		//Float scale = 0.7f/mat_size_x;//qua

		//mat_size_x *= scale;
		//mat_size_y *= scale;
		//mat_size_x = 122.64;
		//mat_size_y = 122.64;
		// mat_size_x = 12.96;
		// mat_size_y = 12.96;
		//mat_size_x = 16; //example
		//mat_size_y = 32;
		mat_size_x = 2; //stage3
		mat_size_y = 2;
		mat_size_z *= scale;

		int strat_N = (int)std::ceil(std::sqrt((Float)wi_num));
		Float invN = 1.0f / strat_N;

#pragma omp parallel for
		for (int i = 0; i < wi_num; i++) {
			ref<Sampler> sampler = scene->getSampler()->clone();
			sampler->generate(Point2i(i,0));
    		// sampler->setSampleIndex(i);
				Point2 start_uv;
				if (fixed_wi_uniform_uv) {
					int ix = i % strat_N;
					int iy = i / strat_N;
					Point2 jitter = sampler->next2D();
					Float u = (ix + jitter.x) * invN;
					Float v = (iy + jitter.y) * invN;
					// HDF5 stores UV as float32 even when Mitsuba's Float is double.
					// Compute the exact rational boundaries in double precision first.
					// Boundaries derived from float invN can already lie on the wrong
					// side of k/N by more than one float32 ULP (e.g. 0.879000008 >
					// 879/1000), so simply applying nextafterf to them is not enough.
					double x0Exact = (double) ix / (double) strat_N;
					double x1Exact = (double) (ix + 1) / (double) strat_N;
					double y0Exact = (double) iy / (double) strat_N;
					double y1Exact = (double) (iy + 1) / (double) strat_N;
					float xLowerFloat = (float) x0Exact;
					float xUpperFloat = (float) x1Exact;
					float yLowerFloat = (float) y0Exact;
					float yUpperFloat = (float) y1Exact;
					if ((double) xLowerFloat < x0Exact)
						xLowerFloat = std::nextafterf(
							xLowerFloat, std::numeric_limits<float>::infinity());
					if ((double) xUpperFloat >= x1Exact)
						xUpperFloat = std::nextafterf(
							xUpperFloat, -std::numeric_limits<float>::infinity());
					if ((double) yLowerFloat < y0Exact)
						yLowerFloat = std::nextafterf(
							yLowerFloat, std::numeric_limits<float>::infinity());
					if ((double) yUpperFloat >= y1Exact)
						yUpperFloat = std::nextafterf(
							yUpperFloat, -std::numeric_limits<float>::infinity());
					Float xLowerStored = (Float) xLowerFloat;
					Float xUpperStored = (Float) xUpperFloat;
					Float yLowerStored = (Float) yLowerFloat;
					Float yUpperStored = (Float) yUpperFloat;
					u = std::max(xLowerStored, std::min(u, xUpperStored));
					v = std::max(yLowerStored, std::min(v, yUpperStored));
					start_uv = Point2(u, v);
				} else {
					start_uv = sampler->next2D();
				}

				Vector3 wi = fixed_wi_uniform_uv
					? Vector3(0.0f, 0.0f, 1.0f)
					: warp::squareToUniformHemisphere(sampler->next2D());
			std::array<float, 3> wi_array = { wi.x, wi.y, wi.z };
			std::array<float, 2> uv_array = { start_uv.x, start_uv.y };
			wi_list[i] = wi_array;
			uv_list[i] = uv_array;

			Point start_pos_up = Point((start_uv.x - 0.5) * mat_size_x, (start_uv.y - 0.5) * mat_size_y, 10);
			Vector3 up = Vector3(0.0f, 0.0f, 1.0f);
			Ray upRay = Ray(start_pos_up, -up, 0.0f);
			Intersection surfaceIts;
			if (!kdtree->rayIntersect(upRay, surfaceIts) || !surfaceIts.isValid()) {
				for (int j = 0; j < wo_num_per_wi; j++) {
					int index = i * wo_num_per_wi + j;
					wo_list[index] = { 0.f, 0.f, 1.f };
					throughput_list[index] = { 0.f, 0.f, 0.f };
					uv_wo_list[index] = { -1.f, -1.f };
					termination_code_list[index] = ETerminationInitialMiss;
					path_class_list[index] = EPathInvalid;
				}
				continue;
			}
			Point start_pos = surfaceIts.p;

			//Point start_pos = Point((start_uv.x - 0.5) * mat_size_x, (start_uv.y - 0.5) * mat_size_y, 0);

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
			for (int j = 0; j < wo_num_per_wi; j++){
				Point2 uv_wo(0.f,0.f);
				Spectrum throughput(1.0f);
				Float eta = 1.0f;
				Vector wo(0.0f, 0.0f, 1.0f);
				int depth = 0;
				uint32_t bounceCount = 0;
				uint32_t flatBounceCount = 0;
				uint32_t microBounceCount = 0;
				uint32_t surfaceMask = 0;
				uint32_t firstSurfaceType = 0;
				uint32_t firstBsdfType = 0;
				uint32_t bsdfTypeMask = 0;
				uint32_t terminationCode = ETerminationUnknown;
				int32_t firstBsdfComponent = -1;
				float firstHitZ = 0.f;
				float maxAbsHitZ = 0.f;
				Intersection its_traced = Intersection(its);
				while (depth <= maxDepth || maxDepth < 0) {
					if (!its_traced.isValid() && depth == 0){
						throughput = Spectrum(0.0f);
						terminationCode = ETerminationInitialMiss;
						break;
					}

					uint32_t surfaceType = classifySurface(its_traced);
					surfaceMask |= surfaceType;
					float absHitZ = std::abs((float) its_traced.p.z);
					maxAbsHitZ = std::max(maxAbsHitZ, absHitZ);
					if (surfaceType == ESurfaceFlatBase)
						flatBounceCount++;
					else
						microBounceCount++;
					if (bounceCount == 0) {
						firstSurfaceType = surfaceType;
						firstHitZ = (float) its_traced.p.z;
					}

					const BSDF *bsdf = its_traced.getBSDF();
					BSDFSamplingRecord bRec(its_traced, sampler, ERadiance);
					Spectrum bsdfWeight = bsdf->sample(bRec, sampler->next2D());
					if (bounceCount == 0) {
						firstBsdfType = (uint32_t) bRec.sampledType;
						firstBsdfComponent = (int32_t) bRec.sampledComponent;
					}
					bsdfTypeMask |= (uint32_t) bRec.sampledType;
					bounceCount++;
					wo = its_traced.toWorld(bRec.wo);
					throughput *= bsdfWeight;
					eta *= bRec.eta;

					Point newOrigin = its_traced.p;
					ray = Ray(newOrigin, wo, ray.time);
					if (!scene->rayIntersect(ray, its_traced)) {
						if (dot(wo, normal) < 0){
							throughput = Spectrum(0.0f);
							terminationCode = ETerminationEscapedLower;
						} else {
							terminationCode = ETerminationEscapedUpper;
						}
						break;
					}

					if (depth++ >= 5) {
						Float q = std::min(throughput.max() * eta * eta, (Float) 0.95f);
						if (sampler->next1D() >= q){
							throughput = Spectrum(0.0f);
							terminationCode = ETerminationRussianRoulette;
							break;
						}
						throughput /= q;
					}
				}

				if (depth > maxDepth) {
					throughput = Spectrum(0.0f);
					terminationCode = ETerminationMaxDepth;
				}
				std::array<float, 3> wo_array = { wo.x, wo.y, wo.z };
				std::array<float, 3> throughput_array = { throughput[0], throughput[1], throughput[2] };
				int index = i * wo_num_per_wi + j;
				wo_list[index] = wo_array;
				throughput_list[index] = throughput_array;
				bounce_count_list[index] = bounceCount;
				flat_bounce_count_list[index] = flatBounceCount;
				micro_bounce_count_list[index] = microBounceCount;
				surface_mask_list[index] = surfaceMask;
				first_surface_type_list[index] = firstSurfaceType;
				first_bsdf_type_list[index] = firstBsdfType;
				bsdf_type_mask_list[index] = bsdfTypeMask;
				termination_code_list[index] = terminationCode;
				first_bsdf_component_list[index] = firstBsdfComponent;
				first_hit_z_list[index] = firstHitZ;
				max_abs_hit_z_list[index] = maxAbsHitZ;
				path_class_list[index] = classifyPath(
					bounceCount, surfaceMask, terminationCode, throughput);

				std::array<float, 2> uv_wo_array = { uv_wo.x, uv_wo.y };
				if (throughput.max() > Epsilon) {
					uv_wo_list[index] = uv_wo_array;
				} else {
					uv_wo_list[index] = {-1.f, -1.f};
				}
			}

		}

	}

	inline Vector2 sampleStandardNormal(Sampler *sampler) {
		Float u1 = sampler->next1D();
		Float u2 = sampler->next1D();

		u1 = std::max(u1, 1e-12f);   // 防止 LN(0)

		Float r = std::sqrt(-2.0f * std::log(u1));
		Float theta = 2.0f * M_PI * u2;

		return Vector2(r * std::cos(theta), r * std::sin(theta));
	}

	void GenDataMultipleFixedWiUVi(char **argv){
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
		uv_wo_list.resize(total_num);

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
			Point2 start_uv = Point2(0.5, 0.5);
			Point start_pos_up = Point((start_uv.x - 0.5) * mat_size_x, (start_uv.y - 0.5) * mat_size_y, 10);
			Vector3 up = Vector3(0.0f, 0.0f, 1.0f);
			Ray upRay = Ray(start_pos_up, -up, 0.0f);
			Intersection surfaceIts;
			kdtree->rayIntersect(upRay, surfaceIts);
			Point start_pos = surfaceIts.p;

			Vector3 wi = Vector3(-0.0f, 0.0f, 1.0f);//warp::squareToUniformHemisphere(sampler->next2D());
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
				Point2 uv_wo(0.f,0.f);
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
					Point3 wo_pos = its_traced.p;
					if (!scene->rayIntersect(ray, its_traced)) {
						//cout << "miss! depth is: " << depth << endl;
						if (dot(wo, normal) < 0){
							throughput = Spectrum(0.0f);
							//cout << "transmission! depth is: " << depth << endl;

						}
						uv_wo = Point2(wo_pos.x/mat_size_x + 0.5, wo_pos.y/mat_size_y + 0.5);
						if(uv_wo.x < 0.f || uv_wo.x > 1.f || uv_wo.y < 0.f || uv_wo.y > 1.f){
							throughput = Spectrum(0.0f);
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
				if (depth > maxDepth||depth==0){
					throughput = Spectrum(0.0f);
				}

				//auto temp_uv = sampleStandardNormal(sampler);
				//temp_uv /= 6.f;
				//if(abs(temp_uv.x) > 3.f||abs(temp_uv.y) > 3.f) throughput = Spectrum(0.0f);

				// cout << "depth is: " << depth << endl;
				std::array<float, 3> wo_array = { wo.x, wo.y, wo.z };
				std::array<float, 3> throughput_array = { throughput[0], throughput[1], throughput[2] };
				int index = i * wo_num_per_wi + j;
				wo_list[index] = wo_array;
				throughput_list[index] = throughput_array;
				
				//std::array<float, 2> uv_wo_array = { uv_wo.x, uv_wo.y };
				// auto temp_uv = sampleStandardNormal(sampler);//sampler->next2D();//::squareToUniformHemisphere(sampler->next2D());
				// std::array<float, 2> uv_wo_array = { (temp_uv.x+1)/2, (temp_uv.y+1)/2 };
				//temp_uv += Vector2(0.5,0.5);
				std::array<float, 2> uv_wo_array = { uv_wo.x, uv_wo.y};
				//std::array<float, 2> uv_wo_array = { 0.5, 0.5 };
				//uv_wo_list[index] = uv_wo_array;
				if (throughput.max() > Epsilon) {
					uv_wo_list[index] = uv_wo_array;
				} else {
					uv_wo_list[index] = {-1.f, -1.f};  // 无效样本标记
				}
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
		case 'p':
			optind++;
			GenDataMultiplePGL(argv);
			break;

		case 'u':
			optind++;
			GenDataMultiplePGL(argv, true);
			break;

		case 'v':
			optind++;
			GenDataMultipleVis(argv);
			break;

		case 'f':
			optind++;
			GenDataMultipleFixedWiUVi(argv);
			break;

		default:
			cout << "Error: Unknown command! run '-h' to show the useage." << endl;
			return 0;
			break;
		}
		auto end = std::chrono::high_resolution_clock::now();
		
		optind++;
		std::string output_file = argv[optind];
		if (output_file == "")
		{
			output_file = "default_output.h5";
		}
		
		HDF5Writer writer(output_file);
		writer.writeVector3Array("wi", wi_list);
		writer.writeVector3Array("wo", wo_list);
		writer.writeVector3Array("throughput", throughput_list);
		//writer.writeVector3Array("normal", normal_list);
		writer.writeVector2Array("uv", uv_list);
		//writer.writeVector2Array("uv_wo", uv_wo_list);
		if (path_labels_available && bounce_count_list.size() == wo_list.size()) {
			writer.writeUInt32Dataset("bounce_count", bounce_count_list);
			writer.writeUInt32Dataset("flat_bounce_count", flat_bounce_count_list);
			writer.writeUInt32Dataset("micro_bounce_count", micro_bounce_count_list);
			writer.writeUInt32Dataset("surface_mask", surface_mask_list);
			writer.writeUInt32Dataset("first_surface_type", first_surface_type_list);
			writer.writeUInt32Dataset("first_bsdf_type", first_bsdf_type_list);
			writer.writeUInt32Dataset("bsdf_type_mask", bsdf_type_mask_list);
			writer.writeUInt32Dataset("path_class", path_class_list);
			writer.writeUInt32Dataset("termination_code", termination_code_list);
			writer.writeInt32Dataset("first_bsdf_component", first_bsdf_component_list);
			writer.writeFloatDataset("first_hit_z", first_hit_z_list);
			writer.writeFloatDataset("max_abs_hit_z", max_abs_hit_z_list);
		}

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
std::vector<std::array<float, 2>> sample_microstructure::uv_wo_list;
std::vector<uint32_t> sample_microstructure::bounce_count_list;
std::vector<uint32_t> sample_microstructure::flat_bounce_count_list;
std::vector<uint32_t> sample_microstructure::micro_bounce_count_list;
std::vector<uint32_t> sample_microstructure::surface_mask_list;
std::vector<uint32_t> sample_microstructure::first_surface_type_list;
std::vector<uint32_t> sample_microstructure::first_bsdf_type_list;
std::vector<uint32_t> sample_microstructure::bsdf_type_mask_list;
std::vector<uint32_t> sample_microstructure::path_class_list;
std::vector<uint32_t> sample_microstructure::termination_code_list;
std::vector<int32_t> sample_microstructure::first_bsdf_component_list;
std::vector<float> sample_microstructure::first_hit_z_list;
std::vector<float> sample_microstructure::max_abs_hit_z_list;
bool sample_microstructure::path_labels_available = false;

MTS_EXPORT_UTILITY(sample_microstructure, "Command line samlpe microstructure")
MTS_NAMESPACE_END
