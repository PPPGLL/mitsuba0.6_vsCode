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

	void BSDFProfile(char **argv){
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
		int spp = std::stoi(argv[optind]);

		// 假设场景中至少有一个 shape
        ref_vector<mitsuba::Shape> &shapes = scene->getShapes();
        if (shapes.empty()) {
            std::cerr << "No shapes found in scene." << std::endl;
        }

        // 获取第一个物体的 BSDF
        ref<BSDF> bsdf = shapes[0]->getBSDF();
        if (!bsdf) {
            std::cerr << "Shape has no BSDF." << std::endl;
        }
		
		// Sampler* sampler = static_cast<Sampler*>(
		// 	PluginManager::getInstance()->createObject(MTS_CLASS(Sampler), Properties("independent"))
		// 	);
        ref<Sampler> sampler = scene->getSampler();
        Intersection its;
        its.wi = Vector(0, 0, 1);
        its.uv = Point2(0.5f, 0.5f);

        BSDFSamplingRecord bRec(its, Vector(0, 0, 1), ERadiance);
        bRec.typeMask = BSDF::EAll;
        bRec.sampler = sampler;

        auto start = std::chrono::high_resolution_clock::now();

        for (int i = 0; i < spp; ++i) {
            bRec.wi = warp::squareToCosineHemisphere(sampler->next2D());
            bRec.wo = warp::squareToCosineHemisphere(sampler->next2D());
            Spectrum result = bsdf->eval(bRec);
        }
        auto end = std::chrono::high_resolution_clock::now();
        double elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        double avg_time = elapsed_ms / double(spp);

        std::cout << "Total time: " << elapsed_ms << " ms for " << spp << " samples\n";
        std::cout << "Average time per eval: " << avg_time << " ms" << std::endl;

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
			BSDFProfile(argv);
			break;


		default:
			cout << "Error: Unknown command! run '-h' to show the useage." << endl;
			return 0;
			break;
		}
		auto end = std::chrono::high_resolution_clock::now();

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