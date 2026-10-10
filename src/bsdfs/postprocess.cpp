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

#include <mitsuba/core/fresolver.h>
#include <mitsuba/render/bsdf.h>
#include <mitsuba/hw/basicshader.h>
#include "microfacet.h"
#include "ior.h"
//#include "roughBeckmann.h"
#define MTS_OPENMP 1
#if defined(MTS_OPENMP)
#include <omp.h>
#endif
#include <mitsuba/core/warp.h>

#define M_PI			3.14159265358979323846f	/* pi */
#define INV_M_PI		0.31830988618379067153f /* 1/pi */
#define M_PI_2			1.57079632679489661923f	/* pi/2 */
#define SQRT_M_PI		1.77245385090551602729f /* sqrt(pi) */
#define SQRT_2			1.41421356237309504880f /* sqrt(2) */
#define INV_SQRT_M_PI	0.56418958354775628694f /* 1/sqrt(pi) */
#define INV_2_SQRT_M_PI	0.28209479177387814347f /* 0.5/sqrt(pi) */
#define INV_SQRT_2_M_PI 0.3989422804014326779f /* 1/sqrt(2*pi) */
#define INV_SQRT_2		0.7071067811865475244f /* 1/sqrt(2) */

#define PI_DIVIDE_180 0.0174532922
MTS_NAMESPACE_BEGIN

/*!\plugin{roughconductor}{Rough conductor material}
 * \order{7}
 * \icon{bsdf_roughconductor}
 * \parameters{
 *     \parameter{distribution}{\String}{
 *          Specifies the type of microfacet normal distribution
 *          used to model the surface roughness.
 *          \vspace{-1mm}
 *       \begin{enumerate}[(i)]
 *           \item \code{beckmann}: Physically-based distribution derived from
 *               Gaussian random surfaces. This is the default.\vspace{-1.5mm}
 *           \item \code{ggx}: The GGX \cite{Walter07Microfacet} distribution (also known as
 *               Trowbridge-Reitz \cite{Trowbridge19975Average} distribution)
 *               was designed to better approximate the long tails observed in measurements
 *               of ground surfaces, which are not modeled by the Beckmann distribution.
 *           \vspace{-1.5mm}
 *           \item \code{phong}: Anisotropic Phong distribution by
 *              Ashikhmin and Shirley \cite{Ashikhmin2005Anisotropic}.
 *              In most cases, the \code{ggx} and \code{beckmann} distributions
 *              should be preferred, since they provide better importance sampling
 *              and accurate shadowing/masking computations.
 *              \vspace{-4mm}
 *       \end{enumerate}
 *     }
 *     \parameter{alpha, alphaU, alphaV}{\Float\Or\Texture}{
 *         Specifies the roughness of the unresolved surface micro-geometry
 *         along the tangent and bitangent directions. When the Beckmann
 *         distribution is used, this parameter is equal to the
 *         \emph{root mean square} (RMS) slope of the microfacets.
 *         \code{alpha} is a convenience parameter to initialize both
 *         \code{alphaU} and \code{alphaV} to the same value. \default{0.1}.
 *     }
 *     \parameter{material}{\String}{Name of a material preset, see
 *           \tblref{conductor-iors}.\!\default{\texttt{Cu} / copper}}
 *     \parameter{eta, k}{\Spectrum}{Real and imaginary components of the material's index of
 *             refraction \default{based on the value of \texttt{material}}}
 *     \parameter{extEta}{\Float\Or\String}{
 *           Real-valued index of refraction of the surrounding dielectric,
 *           or a material name of a dielectric \default{\code{air}}
 *     }
 *     \parameter{sampleVisible}{\Boolean}{
 *         Enables a sampling technique proposed by Heitz and D'Eon~\cite{Heitz1014Importance},
 *         which focuses computation on the visible parts of the microfacet normal
 *         distribution, considerably reducing variance in some cases.
 *         \default{\code{true}, i.e. use visible normal sampling}
 *     }
 *     \parameter{specular\showbreak Reflectance}{\Spectrum\Or\Texture}{Optional
 *         factor that can be used to modulate the specular reflection component. Note
 *         that for physical realism, this parameter should never be touched. \default{1.0}}
 * }
 * \vspace{3mm}
 * This plugin implements a realistic microfacet scattering model for rendering
 * rough conducting materials, such as metals. It can be interpreted as a fancy
 * version of the Cook-Torrance model and should be preferred over
 * heuristic models like \pluginref{phong} and \pluginref{ward} if possible.
 * \renderings{
 *     \rendering{Rough copper (Beckmann, $\alpha=0.1$)}
 *     	   {bsdf_roughconductor_copper.jpg}
 *     \rendering{Vertically brushed aluminium (Anisotropic Phong,
 *         $\alpha_u=0.05,\ \alpha_v=0.3$), see
 *         \lstref{roughconductor-aluminium}}
 *         {bsdf_roughconductor_anisotropic_aluminium.jpg}
 * }
 *
 * Microfacet theory describes rough surfaces as an arrangement of unresolved
 * and ideally specular facets, whose normal directions are given by a
 * specially chosen \emph{microfacet distribution}. By accounting for shadowing
 * and masking effects between these facets, it is possible to reproduce the
 * important off-specular reflections peaks observed in real-world measurements
 * of such materials.
 *
 * This plugin is essentially the ``roughened'' equivalent of the (smooth) plugin
 * \pluginref{conductor}. For very low values of $\alpha$, the two will
 * be identical, though scenes using this plugin will take longer to render
 * due to the additional computational burden of tracking surface roughness.
 *
 * The implementation is based on the paper ``Microfacet Models
 * for Refraction through Rough Surfaces'' by Walter et al.
 * \cite{Walter07Microfacet}. It supports three different types of microfacet
 * distributions and has a texturable roughness parameter.
 * To facilitate the tedious task of specifying spectrally-varying index of
 * refraction information, this plugin can access a set of measured materials
 * for which visible-spectrum information was publicly available
 * (see \tblref{conductor-iors} for the full list).
 * There is also a special material profile named \code{none}, which disables
 * the computation of Fresnel reflectances and produces an idealized
 * 100% reflecting mirror.
 *
 * When no parameters are given, the plugin activates the default settings,
 * which describe copper with a medium amount of roughness modeled using a
 * Beckmann distribution.
 *
 * To get an intuition about the effect of the surface roughness parameter
 * $\alpha$, consider the following approximate classification: a value of
 * $\alpha=0.001-0.01$ corresponds to a material with slight imperfections
 * on an otherwise smooth surface finish, $\alpha=0.1$ is relatively rough,
 * and $\alpha=0.3-0.7$ is \emph{extremely} rough (e.g. an etched or ground
 * finish). Values significantly above that are probably not too realistic.
 * \vspace{4mm}
 * \begin{xml}[caption={A material definition for brushed aluminium}, label=lst:roughconductor-aluminium]
 * <bsdf type="roughconductor">
 *     <string name="material" value="Al"/>
 *     <string name="distribution" value="phong"/>
 *     <float name="alphaU" value="0.05"/>
 *     <float name="alphaV" value="0.3"/>
 * </bsdf>
 * \end{xml}
 *
 * \subsubsection*{Technical details}
 * All microfacet distributions allow the specification of two distinct
 * roughness values along the tangent and bitangent directions. This can be
 * used to provide a material with a ``brushed'' appearance. The alignment
 * of the anisotropy will follow the UV parameterization of the underlying
 * mesh. This means that such an anisotropic material cannot be applied to
 * triangle meshes that are missing texture coordinates.
 *
 * \label{sec:visiblenormal-sampling}
 * Since Mitsuba 0.5.1, this plugin uses a new importance sampling technique
 * contributed by Eric Heitz and Eugene D'Eon, which restricts the sampling
 * domain to the set of visible (unmasked) microfacet normals. The previous
 * approach of sampling all normals is still available and can be enabled
 * by setting \code{sampleVisible} to \code{false}.
 * Note that this new method is only available for the \code{beckmann} and
 * \code{ggx} microfacet distributions. When the \code{phong} distribution
 * is selected, the parameter has no effect.
 *
 * When rendering with the Phong microfacet distribution, a conversion is
 * used to turn the specified Beckmann-equivalent $\alpha$ roughness value
 * into the exponent parameter of this distribution. This is done in a way,
 * such that the same value $\alpha$ will produce a similar appearance across
 * different microfacet distributions.
 *
 * When using this plugin, you should ideally compile Mitsuba with support for
 * spectral rendering to get the most accurate results. While it also works
 * in RGB mode, the computations will be more approximate in nature.
 * Also note that this material is one-sided---that is, observed from the
 * back side, it will be completely black. If this is undesirable,
 * consider using the \pluginref{twosided} BRDF adapter.
 */
class RoughConductor : public BSDF {
public:
	RoughConductor(const Properties &props) : BSDF(props) {
		ref<FileResolver> fResolver = Thread::getThread()->getFileResolver();

		m_specularReflectance = new ConstantSpectrumTexture(
			props.getSpectrum("specularReflectance", Spectrum(1.0f)));

		m_reflectance = new ConstantSpectrumTexture(props.getSpectrum(
			props.hasProperty("reflectance") ? "reflectance"
			: "diffuseReflectance", Spectrum(.5f)));

		m_Sat = new ConstantSpectrumTexture(
			props.getSpectrum("Sat", Spectrum(1.0f)));

		std::string materialName = props.getString("material", "Cu");

		Spectrum intEta, intK;
		if (boost::to_lower_copy(materialName) == "none") {
			intEta = Spectrum(0.0f);
			intK = Spectrum(1.0f);
		} else {
			intEta.fromContinuousSpectrum(InterpolatedSpectrum(
				fResolver->resolve("data/ior/" + materialName + ".eta.spd")));
			intK.fromContinuousSpectrum(InterpolatedSpectrum(
				fResolver->resolve("data/ior/" + materialName + ".k.spd")));
		}

		Float extEta = lookupIOR(props, "extEta", "air");

		m_eta = props.getSpectrum("eta", intEta) / extEta;
		m_k   = props.getSpectrum("k", intK) / extEta;

		MicrofacetDistribution distr(props);
		m_type = distr.getType();
		m_sampleVisible = distr.getSampleVisible();

		m_alphaU = new ConstantFloatTexture(distr.getAlphaU());
		if (distr.getAlphaU() == distr.getAlphaV())
			m_alphaV = m_alphaU;
		else
			m_alphaV = new ConstantFloatTexture(distr.getAlphaV());

		m_uvscale = props.getFloat("uvscale", 1.0);
		Kd = props.getFloat("Kd", 0.5);
		Ks = props.getFloat("Ks", 0.5);
		// Ka = props.getFloat("Ka", 0.5);
		Ka = 1.f-Kd-Ks;
		alphaP = props.getFloat("alphaP", 0.5);
		Sp = props.getFloat("Sp", 0.5);
		Kswater = props.getFloat("Kswater", 1.f);
		// Sat = props.getFloat("Sat", 0.f);
		m_F0 = props.getSpectrum("F0", Spectrum(1.0f));
	}

	RoughConductor(Stream *stream, InstanceManager *manager)
	 : BSDF(stream, manager) {
		m_type = (MicrofacetDistribution::EType) stream->readUInt();
		m_sampleVisible = stream->readBool();
		m_alphaU = static_cast<Texture *>(manager->getInstance(stream));
		m_alphaV = static_cast<Texture *>(manager->getInstance(stream));
		m_specularReflectance = static_cast<Texture *>(manager->getInstance(stream));
		m_eta = Spectrum(stream);
		m_k = Spectrum(stream);

		configure();
	}

	void serialize(Stream *stream, InstanceManager *manager) const {
		BSDF::serialize(stream, manager);

		stream->writeUInt((uint32_t) m_type);
		stream->writeBool(m_sampleVisible);
		manager->serialize(stream, m_alphaU.get());
		manager->serialize(stream, m_alphaV.get());
		manager->serialize(stream, m_specularReflectance.get());
		m_eta.serialize(stream);
		m_k.serialize(stream);
	}

	void configure() {
		unsigned int extraFlags = 0;
		if (m_alphaU != m_alphaV)
			extraFlags |= EAnisotropic;

		if (!m_alphaU->isConstant() || !m_alphaV->isConstant() ||
			!m_specularReflectance->isConstant())
			extraFlags |= ESpatiallyVarying;

		m_components.clear();
		m_components.push_back(EGlossyReflection | EFrontSide | extraFlags);

		/* Verify the input parameters and fix them if necessary */
		m_specularReflectance = ensureEnergyConservation(
			m_specularReflectance, "specularReflectance", 1.0f);

		m_usesRayDifferentials =
			m_alphaU->usesRayDifferentials() ||
			m_alphaV->usesRayDifferentials() ||
			m_specularReflectance->usesRayDifferentials();

		m_usesRayDifferentials = true;

		BSDF::configure();
	}

	/// Helper function: reflect \c wi with respect to a given surface normal
	inline Vector reflect(const Vector &wi, const Normal &m) const {
		return 2 * dot(wi, m) * Vector(m) - wi;
	}

	Spectrum fresnelSchlick(Float cosTheta) const {
        return m_F0 + (Spectrum(1.0f) - m_F0) * std::pow(1 - cosTheta, 5);
    }

	Spectrum eval(const BSDFSamplingRecord &bRec, EMeasure measure) const {
		/* Stop if this component was not requested */
		if (measure != ESolidAngle ||
			Frame::cosTheta(bRec.wi) <= 0 ||
			Frame::cosTheta(bRec.wo) <= 0 ||
			((bRec.component != -1 && bRec.component != 0) ||
			!(bRec.typeMask & EGlossyReflection)))
			return Spectrum(0.0f);

		/* Calculate the reflection half-vector */
		Vector H = normalize(bRec.wo+bRec.wi);

		/* Construct the microfacet distribution matching the
		   roughness values at the current surface position. */
		MicrofacetDistribution distr(
			m_type,
			m_alphaU->eval(bRec.its).average(),
			m_alphaV->eval(bRec.its).average(),
			m_sampleVisible
		);
		const Float D = distr.eval(H);

		//if (D == 0)
		//	return Spectrum(0.0f);

		/* Fresnel factor */
		const Spectrum F = fresnelSchlick(Frame::cosTheta(bRec.wi)) *
			m_specularReflectance->eval(bRec.its);

		/* Smith's shadow-masking function */
		const Float G = distr.G(bRec.wi, bRec.wo, H); //


		/* Calculate the specular BRDF */
		Spectrum Fs = F * D * G / abs(4.0f * Frame::cosTheta(bRec.wi));
	    // for half vector vis:  D * G 
		// for outgoing direction vis, without cos_theta (binning  == eval): D * G / (4.0f * Frame::cosTheta(bRec.wi)* Frame::cosTheta(bRec.wo))

		/* Calculate the diffuse BRDF */
		Spectrum Fd = m_reflectance->eval(bRec.its)
			* (INV_PI * Frame::cosTheta(bRec.wo));

		/* Calculate the mirror BRDF */

		Float Sat = m_Sat->eval(bRec.its)[0];
		Float theta_i = acos(Frame::cosTheta(bRec.wi));
		Float theta_o = acos(Frame::cosTheta(bRec.wo));
		Float Nb = Sp * (3.7 - 2 * (theta_i + theta_o) / 2 - 2 * M_PI / (Sp+6), 2);
		Float Ap = Ka*(1 - powf(1 - Ka, Nb))/(1-(1-Ka));
		Float Kdpw = Kd*(1 - alphaP * G)+(Kd*alphaP*G*(1-Ap)+Ks*(1-Ap)*alphaP*G)*(1-Sat);
		Float Kspw = Ks*(1-alphaP*G);
		Float Kspecular = Kswater*alphaP*G*Sat;
		return Kspw * Fs + Kdpw * Fd;

		// Spectrum Fr = Ks * Fs + Kd * Fd;
		// Float theta_i = acos(Frame::cosTheta(bRec.wi));
		// Float theta_o = acos(Frame::cosTheta(bRec.wo));
		// Float Nb;
		// Nb = Sp * (3.7 - 2 * pow((theta_i + theta_o) / 2 - 2 * M_PI / (Sp+6), 2));
		// Float Ap = 1 - powf(1 - Ka, Nb);
		// //postprocess
		// Spectrum result = (1 - alphaP * G) * Fr //fr1
		// 				+ alphaP * G * (1 - Ap) * (Ks + Kd) * Fd;//fr2
		// return result;
	}

	/// Return whether this BSDF makes use of ray differentials
	float getUVScale() const
	{
		return m_uvscale;
	}


#define INV_2_SQRT_M_PI	0.28209479177387814347f /* 0.5/sqrt(pi) */


	void print(const Vector3 &dir) const
	{
		cout << dir.x << "," << dir.y << "," << dir.z << endl;

		Frame f(dir);

	}

	Float pdf(const BSDFSamplingRecord &bRec, EMeasure measure) const {
		if (measure != ESolidAngle ||
			Frame::cosTheta(bRec.wi) <= 0 ||
			Frame::cosTheta(bRec.wo) <= 0 ||
			((bRec.component != -1 && bRec.component != 0) ||
			!(bRec.typeMask & EGlossyReflection)))
			return 0.0f;

		/* Calculate the reflection half-vector */
		Vector H = normalize(bRec.wo+bRec.wi);

		/* Construct the microfacet distribution matching the
		   roughness values at the current surface position. */
		MicrofacetDistribution distr(
			m_type,
			m_alphaU->eval(bRec.its).average(),
			m_alphaV->eval(bRec.its).average(),
			m_sampleVisible
		);

		if (m_sampleVisible)
			return distr.eval(H) * distr.smithG1(bRec.wi, H)
				/ (4.0f * Frame::cosTheta(bRec.wi));
		else
			return distr.pdf(bRec.wi, H) / (4 * absDot(bRec.wo, H));
	}

	Spectrum sample(BSDFSamplingRecord &bRec, const Point2 &sample) const {
		bRec.wo = warp::squareToCosineHemisphere(sample);
        bRec.eta = 1.0f;
        bRec.sampledComponent = 0;
        bRec.sampledType = EDiffuseReflection;
        Float pdf = warp::squareToCosineHemispherePdf(bRec.wo);
        return eval(bRec, ESolidAngle) / pdf;

		/* Calculate the reflection half-vector */
		Vector H = normalize(bRec.wo+bRec.wi);

		/* Construct the microfacet distribution matching the
		   roughness values at the current surface position. */
		MicrofacetDistribution distr(
			m_type,
			m_alphaU->eval(bRec.its).average(),
			m_alphaV->eval(bRec.its).average(),
			m_sampleVisible
		);
		const Float D = distr.eval(H);

		//if (D == 0)
		//	return Spectrum(0.0f);

		/* Fresnel factor */
		const Spectrum F = fresnelSchlick(Frame::cosTheta(bRec.wi)) *
			m_specularReflectance->eval(bRec.its);

		/* Smith's shadow-masking function */
		const Float G = distr.G(bRec.wi, bRec.wo, H); //


		/* Calculate the specular BRDF */
		Spectrum Fs = F * D * G / abs(4.0f * Frame::cosTheta(bRec.wi));
	    // for half vector vis:  D * G 
		// for outgoing direction vis, without cos_theta (binning  == eval): D * G / (4.0f * Frame::cosTheta(bRec.wi)* Frame::cosTheta(bRec.wo))

		/* Calculate the diffuse BRDF */
		Spectrum Fd = m_reflectance->eval(bRec.its)
			* (INV_PI * Frame::cosTheta(bRec.wo));

		/* Calculate the mirror BRDF */

		Float theta_i = acos(Frame::cosTheta(bRec.wi));
		Float theta_o = acos(Frame::cosTheta(bRec.wo));
		Float Nb = Sp * (3.7 - 2 * (theta_i + theta_o) / 2 - 2 * M_PI / (Sp+6), 2);
		Float Ap = Ka*(1 - powf(1 - Ka, Nb))/(1-(1-Ka));
		Float Kdpw = Kd*(1 - alphaP * G)+(Kd*alphaP*G*(1-Ap)+Ks*(1-Ap)*alphaP*G)*(1-Sat);
		Float Kspw = Ks*(1-alphaP*G);
		Float Kspecular = Kswater*alphaP*G*Sat;
		Float pdfSum = Kdpw+Kspw+Kspecular;
		return Kspw * Fs + Kdpw * Fd;

	// 	float pdf1;
	// 	return this->sample(bRec, pdf1, sample);

	// 	if (Frame::cosTheta(bRec.wi) < 0 ||
	// 		((bRec.component != -1 && bRec.component != 0) ||
	// 		!(bRec.typeMask & EGlossyReflection)))
	// 		return Spectrum(0.0f);

	// 	/* Construct the microfacet distribution matching the
	// 	   roughness values at the current surface position. */
	// 	MicrofacetDistribution distr(
	// 		m_type,
	// 		m_alphaU->eval(bRec.its).average(),
	// 		m_alphaV->eval(bRec.its).average(),
	// 		m_sampleVisible
	// 	);

	// 	/* Sample M, the microfacet normal */
	// 	Float pdf;
	// 	Normal m = distr.sample(bRec.wi, sample, pdf);

	

	// 	/* Perfect specular reflection based on the microfacet normal */
	// 	bRec.wo = reflect(bRec.wi, m);
	// 	bRec.eta = 1.0f;
	// 	bRec.sampledComponent = 0;
	// 	bRec.sampledType = EGlossyReflection;
	// //	return Spectrum(1.0f);
	// 	/* Side check */
	// 	if (Frame::cosTheta(bRec.wo) <= 0)
	// 		return Spectrum(0.0f);

	// 	Spectrum F = fresnelConductorExact(dot(bRec.wi, m),
	// 		m_eta, m_k) * m_specularReflectance->eval(bRec.its);

	// 	Float weight;
	// 	if (m_sampleVisible) {
	// 		weight = distr.smithG1(bRec.wo, m);
	// 	} else {
	// 		weight = distr.eval(m) * distr.G(bRec.wi, bRec.wo, m)
	// 			* dot(bRec.wi, m) / (pdf * Frame::cosTheta(bRec.wi));
	// 	}

	// 	return F * weight;
	}

	// Spectrum sample(BSDFSamplingRecord &bRec, const Point2 &sample) const {
	// 	/* 1. 计算各个 Lobe 的权重/概率 */
	// 	// 注意：这里的概率通常基于各项系数的亮度（Luminance）或平均值
	// 	Float theta_i = acos(Frame::cosTheta(bRec.wi));
	// 	// 假设 theta_o 在采样前未知，通常用入射角估算，或简化 Nb 的计算
		
	// 	// ... 你的系数计算逻辑 ...
	// 	Float prob_spec = Kspw;      // 镜面反射概率
	// 	Float prob_diff = Kdpw;      // 漫反射概率
	// 	Float prob_water = Kspecular; // 水面/特殊镜面概率
		
	// 	Float total = prob_spec + prob_diff + prob_water;
	// 	if (total <= 0.0f) return Spectrum(0.0f);

	// 	// 归一化概率
	// 	Float invTotal = 1.0f / total;
	// 	prob_spec *= invTotal;
	// 	prob_diff *= invTotal;
	// 	prob_water *= invTotal;

	// 	/* 2. 根据随机数选择一个 Lobe 进行采样 */
	// 	Point2 reuseSample = sample; // 复制一份用于后续采样
		
	// 	if (sample.x < prob_spec) {
	// 		// --- 采样 Specular Lobe (Microfacet) ---
	// 		bRec.sampledType = ESpecularReflection;
	// 		// 重新映射随机数到 [0, 1] 范围
	// 		reuseSample.x /= prob_spec;
			
	// 		// 这里应当调用微表面分布采样，例如：
	// 		// MicrofacetDistribution distr(...);
	// 		// Normal m = distr.sample(reuseSample);
	// 		// bRec.wo = reflect(bRec.wi, m);
	// 		// ...
	// 	} 
	// 	else if (sample.x < prob_spec + prob_diff) {
	// 		// --- 采样 Diffuse Lobe ---
	// 		bRec.sampledType = EDiffuseReflection;
	// 		reuseSample.x = (reuseSample.x - prob_spec) / prob_diff;
			
	// 		bRec.wo = warp::squareToCosineHemisphere(reuseSample);
	// 	} 
	// 	else {
	// 		// --- 采样 Water Specular Lobe ---
	// 		bRec.sampledType = EDiffuseReflection; // 或自定义类型
	// 		reuseSample.x = (reuseSample.x - prob_spec - prob_diff) / prob_water;
			
	// 		// 执行对应的采样逻辑...
	// 	}

	// 	/* 3. 计算 PDF 并返回 Eval / PDF */
	// 	// 必须计算混合 PDF：pdf = p1*pdf1 + p2*pdf2 + p3*pdf3
	// 	Float pdf = pdf(bRec); 
		
	// 	if (pdf <= 0.0f || Frame::cosTheta(bRec.wo) <= 0.0f)
	// 		return Spectrum(0.0f);

	// 	return eval(bRec, ESolidAngle) / pdf;
	// }

	Spectrum sample(BSDFSamplingRecord &bRec, Float &pdf, const Point2 &sample) const {
		if (Frame::cosTheta(bRec.wi) < 0 ||
			((bRec.component != -1 && bRec.component != 0) ||
			!(bRec.typeMask & EGlossyReflection)))
			return Spectrum(0.0f);

		bRec.wo = warp::squareToCosineHemisphere(sample);
        bRec.eta = 1.0f;
        bRec.sampledComponent = 0;
        bRec.sampledType = EDiffuseReflection;
        pdf = warp::squareToCosineHemispherePdf(bRec.wo);
        return eval(bRec, ESolidAngle) / pdf;

		// /* Construct the microfacet distribution matching the
		//    roughness values at the current surface position. */
		// MicrofacetDistribution distr(
		// 	m_type,
		// 	m_alphaU->eval(bRec.its).average(),
		// 	m_alphaV->eval(bRec.its).average(),
		// 	m_sampleVisible
		// );

		// /* Sample M, the microfacet normal */
		// Normal m = distr.sample(bRec.wi, sample, pdf);

		// if (pdf == 0)
		// 	return Spectrum(0.0f);

		// /* Perfect specular reflection based on the microfacet normal */
		// bRec.wo = reflect(bRec.wi, m);
		// bRec.eta = 1.0f;
		// bRec.sampledComponent = 0;
		// bRec.sampledType = EGlossyReflection;

		// /* Side check */
		// if (Frame::cosTheta(bRec.wo) <= 0)
		// 	return Spectrum(0.0f);

		// Spectrum F = fresnelConductorExact(dot(bRec.wi, m), m_eta, m_k) *
		// 	m_specularReflectance->eval(bRec.its);

		// Float weight;
		// if (m_sampleVisible) {
		// 	weight = distr.smithG1(bRec.wo, m);
		// } else {
		// 	weight = distr.eval(m) * distr.G(bRec.wi, bRec.wo, m)
		// 		* dot(bRec.wi, m) / (pdf * Frame::cosTheta(bRec.wi));
		// }
		

		// /* Jacobian of the half-direction mapping */
		// pdf /= 4.0f * dot(bRec.wo, m);

		// /*Spectrum weight = this->eval(bRec, ESolidAngle) / pdf;
		// return weight;*/

		// //weight;// 
		// return F * weight;
	}

	void addChild(const std::string &name, ConfigurableObject *child) {
		if (child->getClass()->derivesFrom(MTS_CLASS(Texture))) {
			if (name == "alpha")
				m_alphaU = m_alphaV = static_cast<Texture *>(child);
			else if (name == "alphaU")
				m_alphaU = static_cast<Texture *>(child);
			else if (name == "alphaV")
				m_alphaV = static_cast<Texture *>(child);
			else if (name == "specularReflectance")
				m_specularReflectance = static_cast<Texture *>(child);
			else if (name == "Sat")
				m_Sat = static_cast<Texture *>(child);
			else
				BSDF::addChild(name, child);
		} else {
			BSDF::addChild(name, child);
		}
	}

	Float getRoughness(const Intersection &its, int component) const {
		return 0.5f * (m_alphaU->eval(its).average()
			+ m_alphaV->eval(its).average());
	}

	std::string toString() const {
		std::ostringstream oss;
		oss << "RoughConductor[" << endl
			<< "  id = \"" << getID() << "\"," << endl
			<< "  distribution = " << MicrofacetDistribution::distributionName(m_type) << "," << endl
			<< "  sampleVisible = " << m_sampleVisible << "," << endl
			<< "  alphaU = " << indent(m_alphaU->toString()) << "," << endl
			<< "  alphaV = " << indent(m_alphaV->toString()) << "," << endl
			<< "  specularReflectance = " << indent(m_specularReflectance->toString()) << "," << endl
			<< "  eta = " << m_eta.toString() << "," << endl
			<< "  k = " << m_k.toString() << endl
			<< "]";
		return oss.str();
	}
	Float getPhongExponent(const Intersection &its) const
	{
		MicrofacetDistribution distr(
			MicrofacetDistribution::EPhong,
			m_alphaU->eval(its).average(),
			m_alphaV->eval(its).average(),
			m_sampleVisible
			);
		return distr.getExponent();
	}
	Shader *createShader(Renderer *renderer) const;

	MTS_DECLARE_CLASS()
private:
	MicrofacetDistribution::EType m_type;
	ref<Texture> m_specularReflectance;
	ref<Texture> m_alphaU, m_alphaV;
	ref<Texture> m_Sat;
	bool m_sampleVisible;
	Spectrum m_eta, m_k;
	float m_uvscale;
	ref<Texture> m_reflectance;
	Float Ka;
	Float Kd;
	Float Ks;
	Float alphaP;
	Float Sp;
	Float Kswater;
	Float Sat;
	Spectrum m_F0;
};

/**
 * GLSL port of the rough conductor shader. This version is much more
 * approximate -- it only supports the Ashikhmin-Shirley distribution,
 * does everything in RGB, and it uses the Schlick approximation to the
 * Fresnel reflectance of conductors. When the roughness is lower than
 * \alpha < 0.2, the shader clamps it to 0.2 so that it will still perform
 * reasonably well in a VPL-based preview.
 */
class RoughConductorShader : public Shader {
public:
	RoughConductorShader(Renderer *renderer, const Texture *specularReflectance,
			const Texture *alphaU, const Texture *alphaV, const Spectrum &eta,
			const Spectrum &k) : Shader(renderer, EBSDFShader),
			m_specularReflectance(specularReflectance), m_alphaU(alphaU), m_alphaV(alphaV){
		m_specularReflectanceShader = renderer->registerShaderForResource(m_specularReflectance.get());
		m_alphaUShader = renderer->registerShaderForResource(m_alphaU.get());
		m_alphaVShader = renderer->registerShaderForResource(m_alphaV.get());

		/* Compute the reflectance at perpendicular incidence */
		m_R0 = fresnelConductorExact(1.0f, eta, k);
	}

	bool isComplete() const {
		return m_specularReflectanceShader.get() != NULL &&
			   m_alphaUShader.get() != NULL &&
			   m_alphaVShader.get() != NULL;
	}

	void putDependencies(std::vector<Shader *> &deps) {
		deps.push_back(m_specularReflectanceShader.get());
		deps.push_back(m_alphaUShader.get());
		deps.push_back(m_alphaVShader.get());
	}

	void cleanup(Renderer *renderer) {
		renderer->unregisterShaderForResource(m_specularReflectance.get());
		renderer->unregisterShaderForResource(m_alphaU.get());
		renderer->unregisterShaderForResource(m_alphaV.get());
	}

	void resolve(const GPUProgram *program, const std::string &evalName, std::vector<int> &parameterIDs) const {
		parameterIDs.push_back(program->getParameterID(evalName + "_R0", false));
	}

	void bind(GPUProgram *program, const std::vector<int> &parameterIDs, int &textureUnitOffset) const {
		program->setParameter(parameterIDs[0], m_R0);
	}

	void generateCode(std::ostringstream &oss,
			const std::string &evalName,
			const std::vector<std::string> &depNames) const {
		oss << "uniform vec3 " << evalName << "_R0;" << endl
			<< endl
			<< "float " << evalName << "_D(vec3 m, float alphaU, float alphaV) {" << endl
			<< "    float ct = cosTheta(m), ds = 1-ct*ct;" << endl
			<< "    if (ds <= 0.0)" << endl
			<< "        return 0.0f;" << endl
			<< "    alphaU = 2 / (alphaU * alphaU) - 2;" << endl
			<< "    alphaV = 2 / (alphaV * alphaV) - 2;" << endl
			<< "    float exponent = (alphaU*m.x*m.x + alphaV*m.y*m.y)/ds;" << endl
			<< "    return sqrt((alphaU+2) * (alphaV+2)) * 0.15915 * pow(ct, exponent);" << endl
			<< "}" << endl
			<< endl
			<< "float " << evalName << "_G(vec3 m, vec3 wi, vec3 wo) {" << endl
			<< "    if ((dot(wi, m) * cosTheta(wi)) <= 0 || " << endl
			<< "        (dot(wo, m) * cosTheta(wo)) <= 0)" << endl
			<< "        return 0.0;" << endl
			<< "    float nDotM = cosTheta(m);" << endl
			<< "    return min(1.0, min(" << endl
			<< "        abs(2 * nDotM * cosTheta(wo) / dot(wo, m))," << endl
			<< "        abs(2 * nDotM * cosTheta(wi) / dot(wi, m))));" << endl
			<< "}" << endl
			<< endl
			<< "vec3 " << evalName << "_schlick(float ct) {" << endl
			<< "    float ctSqr = ct*ct, ct5 = ctSqr*ctSqr*ct;" << endl
			<< "    return " << evalName << "_R0 + (vec3(1.0) - " << evalName << "_R0) * ct5;" << endl
			<< "}" << endl
			<< endl
			<< "vec3 " << evalName << "(vec2 uv, vec3 wi, vec3 wo) {" << endl
			<< "   if (cosTheta(wi) <= 0 || cosTheta(wo) <= 0)" << endl
			<< "    	return vec3(0.0);" << endl
			<< "   vec3 H = normalize(wi + wo);" << endl
			<< "   vec3 reflectance = " << depNames[0] << "(uv);" << endl
			<< "   float alphaU = max(0.2, " << depNames[1] << "(uv).r);" << endl
			<< "   float alphaV = max(0.2, " << depNames[2] << "(uv).r);" << endl
			<< "   float D = " << evalName << "_D(H, alphaU, alphaV)" << ";" << endl
			<< "   float G = " << evalName << "_G(H, wi, wo);" << endl
			<< "   vec3 F = " << evalName << "_schlick(1-dot(wi, H));" << endl
			<< "   return reflectance * F * (D * G / (4*cosTheta(wi)));" << endl
			<< "}" << endl
			<< endl
			<< "vec3 " << evalName << "_diffuse(vec2 uv, vec3 wi, vec3 wo) {" << endl
			<< "    if (cosTheta(wi) < 0.0 || cosTheta(wo) < 0.0)" << endl
			<< "    	return vec3(0.0);" << endl
			<< "    return " << evalName << "_R0 * inv_pi * inv_pi * cosTheta(wo);"<< endl
			<< "}" << endl;
	}
	MTS_DECLARE_CLASS()
private:
	ref<const Texture> m_specularReflectance;
	ref<const Texture> m_alphaU;
	ref<const Texture> m_alphaV;
	ref<Shader> m_specularReflectanceShader;
	ref<Shader> m_alphaUShader;
	ref<Shader> m_alphaVShader;
	Spectrum m_R0;
};

Shader *RoughConductor::createShader(Renderer *renderer) const {
	return new RoughConductorShader(renderer,
		m_specularReflectance.get(), m_alphaU.get(), m_alphaV.get(), m_eta, m_k);
}

MTS_IMPLEMENT_CLASS(RoughConductorShader, false, Shader)
MTS_IMPLEMENT_CLASS_S(RoughConductor, false, BSDF)
MTS_EXPORT_PLUGIN(RoughConductor, "Rough conductor BRDF");
MTS_NAMESPACE_END
