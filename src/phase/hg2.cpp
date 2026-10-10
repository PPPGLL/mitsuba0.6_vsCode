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

#include <mitsuba/render/phase.h>
#include <mitsuba/render/sampler.h>
#include <mitsuba/core/properties.h>
#include <mitsuba/core/frame.h>

MTS_NAMESPACE_BEGIN

/*!\plugin{hg}{Henyey-Greenstein phase function}
 * \order{2}
 * \parameters{
 *     \parameter{g}{\Float}{
 *       This parameter must be somewhere in the range $-1$ to $1$
 *       (but not equal to $-1$ or $1$). It denotes the \emph{mean cosine}
 *       of scattering interactions. A value greater than zero indicates that
 *       medium interactions predominantly scatter incident light into a similar
 *       direction (i.e. the medium is \emph{forward-scattering}), whereas
 *       values smaller than zero cause the medium to be
 *       scatter more light in the opposite direction.
 *     }
 * }
 * This plugin implements the phase function model proposed by
 * Henyey and Greenstein \cite{Henyey1941Diffuse}. It is
 * parameterizable from backward- ($g<0$) through
 * isotropic- ($g=0$) to forward ($g>0$) scattering.
 */
class HGPhaseFunction : public PhaseFunction {
public:
	HGPhaseFunction(const Properties &props)
		: PhaseFunction(props) {
		/* Asymmetry parameter: must lie in [-1, 1] where >0 is
		   forward scattering and <0 is backward scattering. */
		m_g1_dry = props.getFloat("g1_dry", 0.8f);
		m_g2_dry = props.getFloat("g2_dry", 0.8f);
		m_w1_dry = props.getFloat("w1_dry", 0.8f);
		m_g1_wet = props.getFloat("g1_wet", 0.8f);
		m_g2_wet = props.getFloat("g2_wet", 0.8f);
		m_w1_wet = props.getFloat("w1_wet", 0.8f);
		m_g = 0;
		if (m_g >= 1 || m_g <= -1)
			Log(EError, "The asymmetry parameter must lie in the interval (-1, 1)!");
	}

	HGPhaseFunction(Stream *stream, InstanceManager *manager)
		: PhaseFunction(stream, manager) {
		m_g = stream->readFloat();
		configure();
	}

	virtual ~HGPhaseFunction() { }

	void serialize(Stream *stream, InstanceManager *manager) const {
		PhaseFunction::serialize(stream, manager);

		stream->writeFloat(m_g);
	}

	void configure() {
		PhaseFunction::configure();
		m_type = EAngleDependence;
	}

	inline Float sample(PhaseFunctionSamplingRecord &pRec,
			Sampler *sampler) const {
		Float sample_g;
		if (sampler->next1D() < m_w1_dry)
			sample_g = m_g1_dry;
		else
			sample_g = m_g2_dry;

		Point2 sample(sampler->next2D());

		Float cosTheta;
		if (std::abs(sample_g) < Epsilon) {
			cosTheta = 1 - 2*sample.x;
		} else {
			Float sqrTerm = (1 - sample_g * sample_g) / (1 - sample_g + 2 * sample_g * sample.x);
			cosTheta = (1 + sample_g * sample_g - sqrTerm * sqrTerm) / (2 * sample_g);
		}

		Float sinTheta = math::safe_sqrt(1.0f-cosTheta*cosTheta),
			  sinPhi, cosPhi;

		math::sincos(2*M_PI*sample.y, &sinPhi, &cosPhi);

		pRec.wo = Frame(-pRec.wi).toWorld(Vector(
			sinTheta * cosPhi,
			sinTheta * sinPhi,
			cosTheta
		));

		return 1.0f;
	}

	Float sample(PhaseFunctionSamplingRecord &pRec,
			Float &pdf, Sampler *sampler) const {
		HGPhaseFunction::sample(pRec, sampler);
		pdf = HGPhaseFunction::eval(pRec);
		return 1.0f;
	}

	Float eval(const PhaseFunctionSamplingRecord &pRec) const {
		Float temp1 = 1.0f + m_g1_dry*m_g1_dry + 2.0f * m_g1_dry * dot(pRec.wi, pRec.wo);
		Float fp1 = INV_FOURPI * (1 - m_g1_dry*m_g1_dry) / (temp1 * std::sqrt(temp1));

		Float temp2 = 1.0f + m_g2_dry*m_g2_dry + 2.0f * m_g2_dry * dot(pRec.wi, pRec.wo);
		Float fp2 = INV_FOURPI * (1 - m_g2_dry*m_g2_dry) / (temp2 * std::sqrt(temp2));
		return m_w1_dry * fp1 + (1 - m_w1_dry)*fp2;
	}

	Float eval(const PhaseFunctionSamplingRecord &pRec, float roughness) const {
		Float temp = 1.0f + roughness*roughness + 2.0f * roughness * dot(pRec.wi, pRec.wo);
		return INV_FOURPI * (1 - roughness*roughness) / (temp * std::sqrt(temp));
	}

	Float eval_wet(const PhaseFunctionSamplingRecord &pRec, Float saturation) const{
		Float temp1 = 1.0f + m_g1_dry*m_g1_dry + 2.0f * m_g1_dry * dot(pRec.wi, pRec.wo);
		Float fp1 = INV_FOURPI * (1 - m_g1_dry*m_g1_dry) / (temp1 * std::sqrt(temp1));

		Float temp2 = 1.0f + m_g2_dry*m_g2_dry + 2.0f * m_g2_dry * dot(pRec.wi, pRec.wo);
		Float fp2 = INV_FOURPI * (1 - m_g2_dry*m_g2_dry) / (temp2 * std::sqrt(temp2));

		Float fp_dry = m_w1_dry * fp1 + (1 - m_w1_dry)*fp2;

		temp1 = 1.0f + m_g1_wet*m_g1_wet + 2.0f * m_g1_wet * dot(pRec.wi, pRec.wo);
		fp1 = INV_FOURPI * (1 - m_g1_wet*m_g1_wet) / (temp1 * std::sqrt(temp1));

		temp2 = 1.0f + m_g2_wet*m_g2_wet + 2.0f * m_g2_wet * dot(pRec.wi, pRec.wo);
		fp2 = INV_FOURPI * (1 - m_g2_wet*m_g2_wet) / (temp2 * std::sqrt(temp2));
		Float fp_wet = m_w1_wet * fp1 + (1 - m_w1_wet)*fp2;

		return fp_dry * (1 - saturation) + saturation * fp_wet;
	}

	// Float sample_wet(PhaseFunctionSamplingRecord &pRec,
	// 	Sampler *sampler, Float &pdf, Float saturation) const{
	// 	//HGPhaseFunction::sample(pRec, sampler);
	// 	//pdf = HGPhaseFunction::eval(pRec);
	// 	//return 1.0f;
	// 	return sample(pRec, pdf, sampler);
	// }

	Float sample_wet(PhaseFunctionSamplingRecord &pRec,
		Sampler *sampler, Float &pdf, Float saturation) const{
		Point2 sample2D = sampler->next2D();
		Float sample1D = sampler->next1D();
		Float chosenG = 0.0f;

		/* 1. 层次化选择 Lobe */
		if (sample1D < (1.0f - saturation)) {
			// 进入 Dry 组 (fp_dry)
			// 这里的随机数映射：将 [0, 1-sat] 重新映射到 [0, 1]
			Float reuse1D = sample1D / (1.0f - saturation);
			
			if (reuse1D < m_w1_dry) {
				chosenG = m_g1_dry;
			} else {
				chosenG = m_g2_dry;
			}
		} else {
			// 进入 Wet 组 (fp_wet)
			// 这里的随机数映射：将 [1-sat, 1] 重新映射到 [0, 1]
			Float reuse1D = (sample1D - (1.0f - saturation)) / saturation;
			
			if (reuse1D < m_w1_wet) {
				chosenG = m_g1_wet;
			} else {
				chosenG = m_g2_wet;
			}
		}

		/* 2. 对选定的 chosenG 执行 HG 采样 (标准公式) */
		Float cosTheta;
		if (std::abs(chosenG) < Epsilon) {
			cosTheta = 1.0f - 2.0f * sample2D.x;
		} else {
			Float sqrTerm = (1.0f - chosenG * chosenG) / 
							(1.0f - chosenG + 2.0f * chosenG * sample2D.x);
			cosTheta = (1.0f + chosenG * chosenG - sqrTerm * sqrTerm) / (2.0f * chosenG);
		}

		Float sinTheta = math::safe_sqrt(1.0f - cosTheta * cosTheta);
		Float sinPhi, cosPhi;
		math::sincos(2.0f * M_PI * sample2D.y, &sinPhi, &cosPhi);

		/* 3. 坐标系转换 */
		// 注意：HG 相位函数采样通常是相对于入射方向 -wi 的偏移
		pRec.wo = Frame(-pRec.wi).toWorld(Vector(
			sinTheta * cosPhi,
			sinTheta * sinPhi,
			cosTheta
		));

		pdf = eval_wet(pRec, saturation);

		/* 4. 返回 PDF 权重 */
		// 对于相位函数采样，如果返回 eval / pdf，
		// 因为是完美的重要性采样 (选择概率完全匹配权重)，所以返回值通常为 1.0
		return 1.0f;
	}

	Float getMeanCosine() const {
		return m_g;
	}

	bool needsDirectionallyVaryingCoefficients() const { return true; }

    Float DsDv() const {
		return 1;
	}

    Float sigmaDir(const Vector &orentation, const Vector &wi) const {
        return 1;
    }

	std::string toString() const {
		std::ostringstream oss;
		oss << "HGPhaseFunction[g=" << m_g << "]";
		return oss.str();
	}

	MTS_DECLARE_CLASS()
private:
	Float m_g;
	Float m_g1_dry;
	Float m_g2_dry;
	Float m_w1_dry;
	Float m_g1_wet;
	Float m_g2_wet;
	Float m_w1_wet;
};

MTS_IMPLEMENT_CLASS_S(HGPhaseFunction, false, PhaseFunction)
MTS_EXPORT_PLUGIN(HGPhaseFunction, "Henyey-Greenstein phase function");
MTS_NAMESPACE_END
