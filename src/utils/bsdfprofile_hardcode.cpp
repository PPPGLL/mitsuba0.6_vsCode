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

#define BSDFtest "ours"

MTS_NAMESPACE_BEGIN
class ROT13Encoder : public Utility {
public:
	int run(int argc, char **argv) {
		Float time = 0;
		int spp = 1000000;

		Sampler* sampler = static_cast<Sampler*>(
			PluginManager::getInstance()->createObject(MTS_CLASS(Sampler), Properties("independent"))
			);
		// Properties props("semi-infinit_wetspongecake");
		//Properties props("wetspongecake_hapkeClamped_ours_ani");
		Properties props("roughconductor");
		props.setBoolean("Analytic", true);
		props.setBoolean("wet_0", true);
		props.setBoolean("aniso_0", true);
		props.setBoolean("mapping", false);
		props.setBoolean("reflectionOnly", true);
		props.setInteger("MaxDepth", 1);
		props.setInteger("LayerCount", 1);
		props.setFloat("thickness_0", 100);
		props.setFloat("saturation_0", 0);
		props.setFloat("porosity_0", 0.8);
		props.setSpectrum("albedo_0", Spectrum(1.f));
		props.setSpectrum("sigmaWet_0", Spectrum(0.f));
		props.setVector("orientation_0", Vector(0, 0, 1));
		props.setBoolean("multipleNet", true);
		props.setBoolean("multipleTex", false);
		BSDF* bsdf = static_cast<BSDF*>(
			PluginManager::getInstance()->createObject(MTS_CLASS(BSDF), props)
			);

		Properties phaseProps("sggx_volume");
		phaseProps.setBoolean("precomputation", true);
		phaseProps.setVector("orientation", Vector(0, 0, 1));
		PhaseFunction* phase = static_cast<PhaseFunction*>(
			PluginManager::getInstance()->createObject(MTS_CLASS(PhaseFunction), phaseProps)
			);

		//bsdf->addChild("phase_0", phase);
		bsdf->configure();

		Intersection its;
		Vector wi = warp::squareToCosineHemisphere(sampler->next2D());
		Vector wo = warp::squareToCosineHemisphere(sampler->next2D());
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

		for (int i = 0; i < spp; ++i) {
			bRec.wi = warp::squareToCosineHemisphere(sampler->next2D());
			bRec.wo = warp::squareToCosineHemisphere(sampler->next2D());

			Spectrum tmp = bsdf->eval(bRec);
		}

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