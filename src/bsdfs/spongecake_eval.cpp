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

#include <mitsuba/render/bsdf.h>
#include <mitsuba/render/texture.h>
#include <mitsuba/hw/basicshader.h>
#include <mitsuba/core/warp.h>
// #include "wavebrdf/spectrum.h"
MTS_NAMESPACE_BEGIN

/*!\plugin{diffuse}{Smooth diffuse material}
 * \order{1}
 * \icon{bsdf_diffuse}
 * \parameters{
 *     \parameter{reflectance}{\Spectrum\Or\Texture}{
 *       Specifies the diffuse albedo of the
 *       material \default{0.5}
 *     }
 * }
 *
 * \renderings{
 *     \rendering{Homogeneous reflectance, see \lstref{diffuse-uniform}}
 *         {bsdf_diffuse_plain}
 *     \rendering{Textured reflectance, see \lstref{diffuse-textured}}
 *         {bsdf_diffuse_textured}
 * }
 *
 * The smooth diffuse material (also referred to as ``Lambertian'')
 * represents an ideally diffuse material with a user-specified amount of
 * reflectance. Any received illumination is scattered so that the surface
 * looks the same independently of the direction of observation.
 *
 * Apart from a  homogeneous reflectance value, the plugin can also accept
 * a nested or referenced texture map to be used as the source of reflectance
 * information, which is then mapped onto the shape based on its UV
 * parameterization. When no parameters are specified, the model uses the default
 * of 50% reflectance.
 *
 * Note that this material is one-sided---that is, observed from the
 * back side, it will be completely black. If this is undesirable,
 * consider using the \pluginref{twosided} BRDF adapter plugin.
 * \vspace{4mm}
 *
 * \begin{xml}[caption={A diffuse material, whose reflectance is specified
 *     as an sRGB color}, label=lst:diffuse-uniform]
 * <bsdf type="diffuse">
 *     <srgb name="reflectance" value="#6d7185"/>
 * </bsdf>
 * \end{xml}
 *
 * \begin{xml}[caption=A diffuse material with a texture map,
 *     label=lst:diffuse-textured]
 * <bsdf type="diffuse">
 *     <texture type="bitmap" name="reflectance">
 *         <string name="filename" value="wood.jpg"/>
 *     </texture>
 * </bsdf>
 * \end{xml}
 */
class SmoothDiffuse : public BSDF {
public:
	SmoothDiffuse(const Properties &props)
		: BSDF(props) {
		/* For better compatibility with other models, support both
		   'reflectance' and 'diffuseReflectance' as parameter names */
		m_reflectance = new ConstantSpectrumTexture(props.getSpectrum(
			props.hasProperty("reflectance") ? "reflectance"
				: "diffuseReflectance", Spectrum(.5f)));

		/*SpectrumInit();
		vector<float> white;
		white.resize(8);
		float temp[] = { 0.672988117, 1.08663714, 1.10315061, 0.996719301, 0.871498048, 0.773481607, 0.652201116, 0.608076334 };
		for (int i = 0; i < 8; i++)
			white[i] = temp[i] * temp[i]*2;
		float r, g, b;
		SpectrumToRGB(white, r, g, b);
		SLog(EDebug, "The data is %f, %f, %f", r, g, b);*/
	}

	SmoothDiffuse(Stream *stream, InstanceManager *manager)
		: BSDF(stream, manager) {
		m_reflectance = static_cast<Texture *>(manager->getInstance(stream));

		configure();
	}

	void configure() {
		/* Verify the input parameter and fix them if necessary */
		m_reflectance = ensureEnergyConservation(m_reflectance, "reflectance", 1.0f);

		m_components.clear();
		if (m_reflectance->getMaximum().max() > 0)
			m_components.push_back(EDiffuseReflection | EFrontSide
				| (m_reflectance->isConstant() ? 0 : ESpatiallyVarying));
			m_usesRayDifferentials = m_reflectance->usesRayDifferentials();

		BSDF::configure();
	}

	Spectrum getDiffuseReflectance(const Intersection &its) const {
		return m_reflectance->eval(its);
	}

	double   power(double base, double exponent) const { return std::exp(exponent * std::log(base)); }
	Spectrum   power(Float base, Spectrum exponent) const { return exp(exponent * std::log(base)); }
	Spectrum log(const Spectrum& exponent) const {
		Spectrum result;
		for (int i = 0; i < 3; i++)
			result[i] = std::log(exponent[i]);
		return result;
	}
	Spectrum exp(const Spectrum& base, const Float exponent) const {
		Spectrum result;
		for (int i = 0; i < 3; i++)
			result[i] = power(base[i], exponent);
		return result;
	}

	Spectrum exp(const Spectrum& exponent) const {
		Spectrum result;
		for (int i = 0; i < 3; i++)
			result[i] = std::exp(exponent[i]);
		return result;
	}

	Spectrum expm1(const Spectrum &input) const
	{
		Spectrum result;
		for (int i = 0; i < 3; i++)
			result[i] = std::exp(input[i])-1.f;
		return result;
	}

	inline float sqr(float x) const
	{
		return x*x;
	}

	float D(const Vector3 &x, float roughness) const
	{
		if (1)
		{
			// For an SGGX microflake, D(x) = 1 / (pi * sqrt |det S| * (x^T S^-1 x)^2),
			// where S = diag(1, 1, alpha^2) for a fiber-like distribution.
			float tmp = sqr(x[0]) + sqr(x[1]) + sqr(x[2] / roughness);      // tmp = x^T S^-1 x
			return 1 / (M_PI * roughness * sqr(tmp));                       // det S = alpha^2
		}
		else
		{
			// For an SGGX microflake, D(x) = 1 / (pi * sqrt |det S| * (x^T S^-1 x)^2),
			// where S = diag(alpha^2, alpha^2, 1) for a surface-like distribution.
			float tmp = sqr(x[0] / roughness) + sqr(x[1] / roughness) + sqr(x[2]);      // tmp = x^T S^-1 x
			return 1 / (M_PI * roughness * roughness* sqr(tmp));                       // det S = alpha^2
		}

	}

	float sigma(const Vector3 &x, float roughness) const
	{
		if (1)
		{
			// For an SGGX microflake, sigma(x)^2 = x^T * S * x,
			// where S = diag(1, 1, alpha^2) for a fiber-like distribution.
			float s2 = sqr(x[0]) + sqr(x[1]) + sqr(roughness * x[2]);
			return std::sqrt(s2);
		}
		else
		{	// For an SGGX microflake, sigma(x)^2 = x^T * S * x,
			// where S = diag(alpha^2, alpha^2, 1) for a surface-like distribution.
			float s2 = sqr(roughness * x[0]) + sqr(roughness * x[1]) + sqr(x[2]);
			return std::sqrt(s2);

		}
	}

	Spectrum eval(const BSDFSamplingRecord &bRec, EMeasure measure) const {
		if (!(bRec.typeMask & EDiffuseReflection) || measure != ESolidAngle
			|| Frame::cosTheta(bRec.wi) <= 0
			|| Frame::cosTheta(bRec.wo) <= 0)
			return Spectrum(0.0f);

		Vector3 wi_ = bRec.wi;
		Vector3 wo_ = bRec.wo;
		float idotn = wi_[2], rec_idotn = abs(1.f / idotn);
		float odotn = wo_[2], rec_odotn = abs(1.f / odotn);
		float currentprod = 1;
		Spectrum result(0.0f), result1(0.0f);
		float roughness = 0.1f;
		// float sigma_i = sigma(wi_, roughness), sigma_o = sigma(wo_, roughness);
		float sigma_i = 1, sigma_o = 1;


		// Spectrum tmp = Spectrum(-currentprod * (sigma_i* rec_idotn + sigma_o  * rec_odotn));
		Spectrum tmp = -currentprod * Spectrum( sigma_i* rec_idotn+ sigma_o  * rec_odotn);
		Spectrum G = 1 / tmp;
		G *= Spectrum(1.f)-exp(tmp);
		Spectrum brdf = 1 * D(wi_, 0.1) * G * abs(rec_idotn) / 4.f;
		result = 1 * brdf;

		// Spectrum tmp1 = Spectrum(-currentprod * (sigma_i* rec_idotn + sigma_o  * rec_odotn));
		Spectrum tmp1 = -currentprod * Spectrum( sigma_i* rec_idotn+ sigma_o  * rec_odotn);
		Spectrum G1 = 1 / tmp1;
		G1 *= Spectrum(1.f)-exp(tmp1);
		Spectrum brdf1 = 1 * D(wi_, 0.1) * G1 * abs(rec_idotn) / 4.f;
		result1 = 1 * brdf1;

		return result1+result;
	}

	Float pdf(const BSDFSamplingRecord &bRec, EMeasure measure) const {
		if (!(bRec.typeMask & EDiffuseReflection) || measure != ESolidAngle
			|| Frame::cosTheta(bRec.wi) <= 0
			|| Frame::cosTheta(bRec.wo) <= 0)
			return 0.0f;

		return warp::squareToCosineHemispherePdf(bRec.wo);
	}

	Spectrum sample(BSDFSamplingRecord &bRec, const Point2 &sample) const {
		if (!(bRec.typeMask & EDiffuseReflection) || Frame::cosTheta(bRec.wi) <= 0)
			return Spectrum(0.0f);

		bRec.wo = warp::squareToCosineHemisphere(sample);
		bRec.eta = 1.0f;
		bRec.sampledComponent = 0;
		bRec.sampledType = EDiffuseReflection;
		return m_reflectance->eval(bRec.its);
	}

	Spectrum sample(BSDFSamplingRecord &bRec, Float &pdf, const Point2 &sample) const {
		if (!(bRec.typeMask & EDiffuseReflection) || Frame::cosTheta(bRec.wi) <= 0)
			return Spectrum(0.0f);

		bRec.wo = warp::squareToCosineHemisphere(sample);
		bRec.eta = 1.0f;
		bRec.sampledComponent = 0;
		bRec.sampledType = EDiffuseReflection;
		pdf = warp::squareToCosineHemispherePdf(bRec.wo);
		return m_reflectance->eval(bRec.its);
	}

	void addChild(const std::string &name, ConfigurableObject *child) {
		if (child->getClass()->derivesFrom(MTS_CLASS(Texture))
				&& (name == "reflectance" || name == "diffuseReflectance")) {
			m_reflectance = static_cast<Texture *>(child);
		} else {
			BSDF::addChild(name, child);
		}
	}

	void serialize(Stream *stream, InstanceManager *manager) const {
		BSDF::serialize(stream, manager);

		manager->serialize(stream, m_reflectance.get());
	}

	Float getRoughness(const Intersection &its, int component) const {
		return std::numeric_limits<Float>::infinity();
	}

	std::string toString() const {
		std::ostringstream oss;
		oss << "SmoothDiffuse[" << endl
			<< "  id = \"" << getID() << "\"," << endl
			<< "  reflectance = " << indent(m_reflectance->toString()) << endl
			<< "]";
		return oss.str();
	}

	Shader *createShader(Renderer *renderer) const;

	MTS_DECLARE_CLASS()
private:
	ref<Texture> m_reflectance;
};

// ================ Hardware shader implementation ================

class SmoothDiffuseShader : public Shader {
public:
	SmoothDiffuseShader(Renderer *renderer, const Texture *reflectance)
		: Shader(renderer, EBSDFShader), m_reflectance(reflectance) {
		m_reflectanceShader = renderer->registerShaderForResource(m_reflectance.get());
	}

	bool isComplete() const {
		return m_reflectanceShader.get() != NULL;
	}

	void cleanup(Renderer *renderer) {
		renderer->unregisterShaderForResource(m_reflectance.get());
	}

	void putDependencies(std::vector<Shader *> &deps) {
		deps.push_back(m_reflectanceShader.get());
	}

	void generateCode(std::ostringstream &oss,
			const std::string &evalName,
			const std::vector<std::string> &depNames) const {
		oss << "vec3 " << evalName << "(vec2 uv, vec3 wi, vec3 wo) {" << endl
			<< "    if (cosTheta(wi) < 0.0 || cosTheta(wo) < 0.0)" << endl
			<< "    	return vec3(0.0);" << endl
			<< "    return " << depNames[0] << "(uv) * inv_pi * cosTheta(wo);" << endl
			<< "}" << endl
			<< endl
			<< "vec3 " << evalName << "_diffuse(vec2 uv, vec3 wi, vec3 wo) {" << endl
			<< "    return " << evalName << "(uv, wi, wo);" << endl
			<< "}" << endl;
	}

	MTS_DECLARE_CLASS()
private:
	ref<const Texture> m_reflectance;
	ref<Shader> m_reflectanceShader;
};

Shader *SmoothDiffuse::createShader(Renderer *renderer) const {
	return new SmoothDiffuseShader(renderer, m_reflectance.get());
}

MTS_IMPLEMENT_CLASS(SmoothDiffuseShader, false, Shader)
MTS_IMPLEMENT_CLASS_S(SmoothDiffuse, false, BSDF)
MTS_EXPORT_PLUGIN(SmoothDiffuse, "Smooth diffuse BRDF")
MTS_NAMESPACE_END
