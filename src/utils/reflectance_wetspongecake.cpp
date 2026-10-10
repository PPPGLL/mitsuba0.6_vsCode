#include <mitsuba/render/util.h>

#include <mitsuba/core/frame.h>
#include <mitsuba/render/phase.h>
#include <mitsuba/render/medium.h>
#include <mitsuba/render/sampler.h>
#include <mitsuba/core/fresolver.h>
#include <mitsuba/render/bsdf.h>
#include <mitsuba/core/warp.h>
#include "../bsdfs/microfacet.h"
#include <mitsuba/core/plugin.h>
#include <mitsuba/render/scene.h>
#include <chrono>

#include <fstream>
#include <iomanip>

#define BSDFtest "ours"

MTS_NAMESPACE_BEGIN
class ROT13Encoder : public Utility {
public:
	int run(int argc, char **argv) {
		Float time = 0;
		int spp = 100000;

		// 1. 打开文件（模式：覆盖写入）
		std::ofstream outFile("C:\\PGL\\Pan\\porousMedium\\scenes\\exp_measuredData\\reflectance_ours.txt");

		Sampler* sampler = static_cast<Sampler*>(
			PluginManager::getInstance()->createObject(MTS_CLASS(Sampler), Properties("independent"))
			);

		Intersection its;
		Vector wo(0.0f, 0.0f, 1.0f); 

		float theta = degToRad(30.0f);
		Vector wi(std::sin(theta), 0.0f, std::cos(theta));

		its.wi = wi;
		its.uv = Point2(0, 0);
		BSDFSamplingRecord bRec(its, wi, wo, ERadiance);
		bRec.typeMask = BSDF::EAll;
		bRec.sampler = sampler;
		bRec.sampledComponent = 0;
		bRec.sampledType = BSDF::EBSDFType::EGlossyReflection;
		bRec.bounces = 0;
		bRec.sampler = sampler;

		auto start = std::chrono::high_resolution_clock::now();

		Properties phaseProps("sggx_volume");
		phaseProps.setBoolean("precomputation", true);
		phaseProps.setBoolean("fiber", false);
		phaseProps.setBoolean("fiberP", false);
		phaseProps.setFloat("intIOR", 1.71f);
		phaseProps.setFloat("extIOR", 1.0f);
		phaseProps.setFloat("extIOR_wet", 1.33f);
		phaseProps.setFloat("roughness", 1.0f);
		phaseProps.setFloat("roughnessP", 0.425f);
		phaseProps.setVector("orientation", Vector(0, 0, 1));
		PhaseFunction* phase = static_cast<PhaseFunction*>(
			PluginManager::getInstance()->createObject(MTS_CLASS(PhaseFunction), phaseProps)
			);

		for (int i = 0; i <= 100; i += 2) {
			// Properties props("semi-infinit_wetspongecake");
			// Properties props("spongecake_eval");
			Properties props("wetspongecake");
			//Properties props("roughconductor");
			props.setBoolean("Analytic", true);
			props.setBoolean("wet_0", true);
			props.setBoolean("aniso_0", true);
			props.setBoolean("mapping", false);
			props.setBoolean("reflectionOnly", true);
			props.setInteger("MaxDepth", 10);
			props.setInteger("LayerCount", 1);
			props.setFloat("thickness_0", 100);
			props.setFloat("saturation_0", i*0.01f);
			props.setFloat("porosity_0", 0.4f);
			props.setSpectrum("albedo_0", Spectrum(0.8f));
			props.setSpectrum("sigmaWet_0", Spectrum(0.f));
			props.setVector("orientation_0", Vector(0, 0, 1));
			props.setBoolean("multipleNet", false);
			props.setBoolean("multipleTex", false);
			BSDF* bsdf = static_cast<BSDF*>(
				PluginManager::getInstance()->createObject(MTS_CLASS(BSDF), props)
				);

			bsdf->addChild("phase_0", phase);
			bsdf->configure();
			Spectrum tmp(0.f);

			for (int j = 0; j < spp; ++j) 
				tmp += bsdf->eval(bRec);
			tmp *= 1.f / spp;

			outFile << std::fixed << std::setprecision(8) 
            << tmp.getLuminance() << "\n";
		}

		outFile.close();

		auto end = std::chrono::high_resolution_clock::now();
		double elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
		double avg_time = elapsed_ms / double(spp);

		std::cout << "Total time: " << elapsed_ms << " ms for " << spp << " samples" << std::endl;
		std::cout << "Average time per eval: " << avg_time << " ms" << std::endl;

		return 0;
	}
	MTS_DECLARE_UTILITY()
};
MTS_EXPORT_UTILITY(ROT13Encoder, "Perform a ROT13 encryption of a string")
MTS_NAMESPACE_END