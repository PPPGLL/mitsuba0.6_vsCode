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

/// Generate a few statistics related to the implementation?
// #define SGGX_STATISTICS 1


/// The following file implements the micro-flake distribution
/// for rough fibers
#include "sggx_fiber.h"
#define TINYEXR_IMPLEMENTATION
#include "tinyexr.h"

#define MTS_OPENMP 1
#if defined(MTS_OPENMP)
#include <omp.h>
#endif

MTS_NAMESPACE_BEGIN
//#define SGGXDEBUGEVAL 1
//#define SGGXDEBUGSAMPLE 1
//#define SGGXDEBUG 1
#if defined(SGGX_STATISTICS)
static StatsCounter avgSampleIterations("Sggx model",
		"Average rejection sampling iterations", EAverage);
#endif

/*!\plugin{microflake}{Micro-flake phase function}
 * \parameters{
 *     \parameter{stddev}{\Float}{
 *       Standard deviation of the micro-flake normals. This
 *       specifies the roughness of the fibers in the medium.
 *     }
 * }
 *
 * \renderings{
 *     \rendering{\code{stddev}=0.2}{phase_microflakes_02}
 *     \rendering{\code{stddev}=0.05}{phase_microflakes_005}
 * }
 *
 * This plugin implements the anisotropic micro-flake phase function
 * described in
 * ``A radiative transfer framework for rendering materials with
 * anisotropic structure'' by Wenzel Jakob, Adam Arbree,
 * Jonathan T. Moon, Kavita Bala, and Steve Marschner
 * \cite{Jakob2010Radiative}.
 *
 * The implementation in this plugin is specific to rough fibers
 * and uses a Gaussian-type flake distribution. It is much faster
 * than the spherical harmonics approach proposed in the original
 * paper. This distribution, as well as the implemented sampling
 * method, are described in the paper
 * ``Building Volumetric Appearance Models of Fabric using
 * Micro CT Imaging'' by Shuang Zhao, Wenzel Jakob, Steve Marschner,
 * and Kavita Bala \cite{Zhao2011Building}.
 *
 * Note: this phase function must be used with a medium that specifies
 * the local fiber orientation at different points in space. Please
 * refer to \pluginref{heterogeneous} for details.
 *
 * \vspace{1cm}
 *
 * \renderings{
 *   \fbox{\includegraphics[width=\textwidth]{images/medium_knitpatterns}}
 *   \caption{A range of different knit patterns, rendered using the
 *   \pluginref{heterogeneous} and \pluginref{microflake} plugins. Courtesy
 *   of Yuksel et al. \cite{Yuksel2012Stitch}.}
 * }
 */

# define thetaCount 180
# define phiCount 1
# define phaseCount 64
# define phaseCountTotal 1048576   //32

#define MaxBounce 20//1000
#define SampleCount 100000//100000

template <class T>
inline T randUniform() {
	return rand() / (T)RAND_MAX;
}

int to_1d_index(int i, int j, int k, int l, int a, int b, int c, int d) {
	return i * (b * c * d) + j * (c * d) + k * d + l;
}

void wToIndex(Vector3 w, Float& thetaIndex, Float& phiIndex) {
	double r = std::sqrt(w.x*w.x + w.y*w.y + w.z*w.z); // �����ĳ���
	Float theta, phi;
	if (r == 0) {
		// �����������Ϊ0����ô�Ƕ�û�����壬�����������Ĭ��ֵ�����׳��쳣
		theta = 0;
		phi = 0;
		return;
	}
	Float costheta = w.z / r;
	theta = std::acos(w.z / r); // �����
	phi = std::atan2(w.y, w.x); // �����
	if (phi < 0)
		phi = 2 * M_PI + phi;
	//thetaIndex = theta / M_PI * thetaCount;
	thetaIndex = (costheta + 1) / 2.f * thetaCount;
	phiIndex = phi / (2 * M_PI) * phiCount;
}

void wToIndexPhase(Vector3 w, Float& thetaIndex, Float& phiIndex) {
	double r = std::sqrt(w.x*w.x + w.y*w.y + w.z*w.z); // �����ĳ���
	Float theta, phi;
	if (r == 0) {
		// �����������Ϊ0����ô�Ƕ�û�����壬�����������Ĭ��ֵ�����׳��쳣
		theta = 0;
		phi = 0;
		return;
	}
	Float costheta = w.z / r;
	theta = std::acos(w.z / r); // �����
	phi = std::atan2(w.y, w.x); // �����
	if (phi < 0)
		phi = 2 * M_PI + phi;
	//thetaIndex = theta / M_PI * (phaseCount + 0.0f);
	thetaIndex = (costheta + 1) / 2.f * phaseCount;
	phiIndex = phi / (2 * M_PI) * (phaseCount + 0.0f);
}

Vector3 indexToW(Float thetaIndex, Float phiIndex){
	Float theta, phi;
	//theta = thetaIndex / (thetaCount + 0.f) * M_PI;
	Float costheta = thetaIndex / (thetaCount + 0.f) * 2.f - 1.f;
	theta = std::acos(costheta);
	phi = phiIndex / (phiCount + 0.f) * 2 * M_PI;
	Vector3 w;
	w.x = std::sin(theta) * std::cos(phi); // ����x
	w.y = std::sin(theta) * std::sin(phi); // ����y
	w.z = std::cos(theta);                 // ����z
	return w;
}

Vector3 indexToWPhase(Float thetaIndex, Float phiIndex){
	Float theta, phi;
	//theta = thetaIndex / (phaseCount + 0.f) * M_PI;
	Float costheta = (thetaIndex+0.0) / (phaseCount + 0.f) * 2.f - 1.f;
	theta = std::acos(costheta);
	phi = (phiIndex + 0.0) / (phaseCount + 0.f) * 2 * M_PI;
	Vector3 w;
	w.x = std::sin(theta) * std::cos(phi); // ����x
	w.y = std::sin(theta) * std::sin(phi); // ����y
	w.z = std::cos(theta);                 // ����z
	return w;
}

/*������*/
class ProlateSpheroid{
public:
	ProlateSpheroid(){}
	ProlateSpheroid(const Float &a, const Float &c, const Float &roughness, const Float &intIOR, const Float &extIOR) :
		m_a(a), m_c(c), m_roughness(roughness), m_intIOR(intIOR), m_extIOR(extIOR){}

	bool rayIntersect(const Ray &ray, Float mint, Float maxt, Float &t) const {
		Vector3d o = Vector3d(ray.o);
		Vector3d d(ray.d);
		double inv_a_2 = (1.f / m_a) * (1.f / m_a);
		double inv_c_2 = (1.f / m_c) * (1.f / m_c);
		double A = d.x * d.x * inv_a_2 + d.y * d.y * inv_a_2 + d.z * d.z * inv_c_2;
		double B = 2 * (o.x * d.x * inv_a_2 + o.y * d.y * inv_a_2 + o.z * d.z * inv_c_2);
		double C = o.x * o.x * inv_a_2 + o.y * o.y * inv_a_2 + o.z * o.z * inv_c_2 - 1;

		double nearT, farT;
		if (!solveQuadraticDouble(A, B, C, nearT, farT))
			return false;

		if (!(nearT <= maxt && farT >= mint)) /* NaN-aware conditionals */
			return false;

		if (nearT < mint) {
			if (farT > maxt)
				return false;
			t = (Float)farT;
		}
		else {
			t = (Float)nearT;
		}

		return true;
	}

	Normal getNormal(const Point3 &point){
		return normalize(Normal(2 * point.x / (m_a*m_a), 2 * point.y / (m_a*m_a), 2 * point.z / (m_c*m_c)));
	}

	Point3 samplePosition(const Point2 &sample){
		// Generate random spherical coordinates
		double theta = std::acos(1 - 2 * sample.x); // Polar angle
		double phi = 2 * M_PI * sample.y; // Azimuthal angle

		// Parametric equations for ellipsoid
		Point3 p;
		p.x = m_a * std::sin(theta) * std::cos(phi);
		p.y = m_a * std::sin(theta) * std::sin(phi);
		p.z = m_c * std::cos(theta);
		return p;
	}

	//private:
	Float m_a;
	Float m_c;
	Float m_roughness;
	Float m_intIOR;
	Float m_extIOR;
};

void precomputeMicroPhase(Float data[phaseCount][phaseCount][phaseCount], Float intIOR, Float extIOR, Float sigma, bool fiber, bool mirror){
	//std::string address = "D:/Pan/porousMedium/scenes/exp_aniMedium/phase";
	std::string address = "phaseData";
	std::string filename(std::string("/") + "microphase_i" + std::to_string(intIOR) + "_e" + std::to_string(extIOR) + "_rP" + std::to_string(sigma) + (fiber?"_fiber":"_surface") + ".bin");

	std::ifstream coe(address + filename, std::ios::binary);
	if (coe.is_open())//�ļ��Ѿ������� ����Ҫ�ظ�Ԥ����
	{
		// ��ȡ����
		coe.read(reinterpret_cast<char*>(data), phaseCount * phaseCount * phaseCount * sizeof(float));

		// �ر��ļ�
		coe.close();
	}
	
	//setup
	ProlateSpheroid ps;
	if (!fiber){
		ps.m_a = 1.f;// spherecity*spherecity;
		ps.m_c = sigma;//1.f;
	}
	else{
		ps.m_a = 1.f;// spherecity*spherecity;
		ps.m_c = 1.f / sigma;//1.f;
	}
	//ps.m_roughness = roughness;
	ps.m_intIOR = intIOR;
	ps.m_extIOR = extIOR;
	//Properties props("roughdielectric");
	//Properties props("conductor");
	Properties props("dielectric");
	//props.setString("distribution", "ggx");
	//props.setFloat("alpha", roughness);
	props.setFloat("intIOR", intIOR);
	props.setFloat("extIOR", extIOR);
	BSDF* bsdf = static_cast<BSDF*>(
		PluginManager::getInstance()->createObject(MTS_CLASS(BSDF), props)
		);

	//��ʼ�������������
	Sampler* sampler = static_cast<Sampler*>(
		PluginManager::getInstance()->createObject(MTS_CLASS(Sampler), Properties("independent"))
		);
	sampler->generate(Point2i(0));
	int                   tcount = mts_omp_get_max_threads();
	std::vector<Sampler*> samplers(tcount);
	for (size_t i = 0; i < tcount; ++i) {
		ref<Sampler> clonedSampler = sampler->clone();
		clonedSampler->incRef();
		samplers[i] = clonedSampler.get();
	}
	//std::vector<std::vector<Float>> mpdata(tcount, std::vector<Float>(180, 0.f));

#pragma omp parallel for
	for (int i = 0; i < phaseCount; i++){
		std::vector<std::vector<std::vector<Float>>> tmpData(tcount, std::vector<std::vector<Float>>(phaseCount, std::vector<Float>(phaseCount, 0.f)));
		std::vector<std::vector<Float>> CDFData(phaseCount, std::vector<Float>(phaseCount, 0.f));
		Vector wi_ori = indexToWPhase(i, 0);

#if defined(MTS_OPENMP)
		int tid = mts_omp_get_thread_num();
#else 
		int tid = 0;
#endif
		for (int s = 0; s < SampleCount; s++)
		{
			Float eta = ps.m_intIOR / ps.m_extIOR;
			//sample a wi
			//Vector wi_ori(0, 0, 1);//warp::squareToUniformSphere(samplers[tid]->next2D());
			Vector wi = wi_ori;
			Float weight = 1;
			//sample a origin
			Point2 diskPoint = ps.m_a*warp::squareToUniformDisk(samplers[tid]->next2D());
			Vector3 F_origin(diskPoint.x, diskPoint.y, 2.f);
			//Vector3 F_origin(0,0,10);
			Frame F_wi(wi);
			Point3 origin(F_wi.toWorld(F_origin));
			//Point3 origin(2.f * wi);
			//Point3 origin(diskPoint.x, diskPoint.y, 2);
			int bounce = 0;
			Ray ray;
			ray.mint = Epsilon;
			ray.maxt = std::numeric_limits<Float>::infinity();
			ray.setOrigin(origin);
			ray.setDirection(-wi);

			while (bounce <= MaxBounce || MaxBounce < 0){
				//while (1){
				//setup ray

				//ray intersect
				Float t = 0;
				if (ps.rayIntersect(ray, ray.mint, ray.maxt, t)){
					//if (sphere->rayIntersect(ray, ray.mint, ray.maxt, t, &t)){
					//outT = t;
					//cout << t << endl;
					Point3 intersection = ray.o + ray.d * t;
					Normal surface_n = ps.getNormal(intersection);
					//Normal surface_n = intersection-Point(0,0,0);
					//sample a normal
					Frame surface(surface_n);
					Vector local_wi = surface.toLocal(-ray.d);

					Intersection its;
					its.wi = local_wi;
					BSDFSamplingRecord bRec(its, samplers[tid], ERadiance);
					bRec.bounces = bounce;
					if (mirror)
						bRec.typeMask &= ~BSDF::EBSDFType::EDeltaTransmission;
					Spectrum tmp = bsdf->sample(bRec, samplers[tid]->next2D());
					if (tmp.isZero())
						break;
					Vector wo = surface.toWorld(bRec.wo);
					//Vector wo = reflect(wi, surface_n);
					//wi = -wo;

					ray.setOrigin(intersection);
					ray.setDirection(wo);
				}
				else{
					if (bounce == 0){
						s--;
						break;
					}
					Vector wo = ray.d;
					//Float cosTheta = dot(wi_ori, wo);
					//int index = costhetaToIndex(cosTheta);
					Vector localWo = Frame(wi_ori).toLocal(wo);
					Float thetaIndexO, phiIndexO;

					//world
					wToIndexPhase(wo, thetaIndexO, phiIndexO);

					//local
					//wToIndexPhase(localWo, thetaIndexO, phiIndexO);

					int x2 = math::clamp(static_cast<int>(std::floor(thetaIndexO)), 0, phaseCount - 1);
					int y2 = math::clamp(static_cast<int>(std::floor(phiIndexO)), 0, phaseCount - 1);
					tmpData[tid][x2][y2] += weight;
					//mpdata[tid][index] += weight;

				}
				/**/
				if (bounce >= 10){
					if (samplers[tid]->next1D() >= 0.95)
						break;
					else
						weight /= 0.95;
				}
				bounce++;
			}
		}
		
		Float totalCount = 0.f;
		for (int j = 0; j < phaseCount; j++){
			for (int k = 0; k < phaseCount; k++){
				for (int l = 0; l < tcount; l++){
					CDFData[j][k] += tmpData[l][j][k];
				}
				totalCount += CDFData[j][k];
			}
		}
		Float invTotal = 1.f / totalCount;
		Float interval = 4 * M_PI / phaseCount / phaseCount;
		Float invInterval = 1.f / interval;
		for (int j = 0; j < phaseCount; j++){
			for (int k = 0; k < phaseCount; k++){
				data[i][j][k] = CDFData[j][k] * invTotal * invInterval;
			}
		}

	}

	std::ofstream microPhase(address + filename, std::ios::binary);
	// д������
	microPhase.write(reinterpret_cast<const char*>(data), phaseCount * phaseCount * phaseCount * sizeof(float));

	// �ر��ļ�
	microPhase.close();
}


class SggxPhaseFunction : public PhaseFunction {

public:

	SggxPhaseFunction(const Properties &props) : PhaseFunction(props) {
		/// Standard deviation of the flake distribution
		m_fiber = props.getBoolean("fiber", true);
		m_alpha = props.getFloat("roughness", 0.1);

		m_fit = props.getBoolean("fit", false);
		m_fitAlpha = props.getFloat("fitRoughness", 0.1);

		m_fiberP = props.getBoolean("fiberP", true);
		m_alphaP = props.getFloat("roughnessP", 0.1);
		m_intIOR = props.getFloat("intIOR", 2.0f);
		m_extIOR = props.getFloat("extIOR", 1.0f);
		m_extIOR_wet = props.getFloat("extIOR_wet", 1.0f);
		m_saturation = props.getFloat("saturation", 0.0f);

		m_sample = props.getInteger("samples", 1);
		m_precomputation = props.getBoolean("precomputation", false);

		m_mirror = props.getBoolean("mirror", false);

#pragma omp parallel for
		for (int i = 0; i < thetaCount; i++){
			for (int j = 0; j < phiCount; j++){
				if (i != 0 && i != thetaCount / 2)
					continue;
				Vector3 wi = indexToW(i, j);

				Float result = 0;
				int samples = 1000000;
				for (int i = 0; i < samples; i++){
					Float pdf;
					Vector wm;
					if (m_fiber == false){
						// improtance sampling
						Float alpha = std::max(m_alpha, 0.0f);
						MicrofacetDistribution distr(
							MicrofacetDistribution::EType::EGGX,
							alpha,
							alpha,
							false //sample visible
							);
						pdf = 0.f;
						wm = distr.sample(wi, Point2(randUniform<Float>(), randUniform<Float>()), pdf);
						if (randUniform<Float>() < 0.5){
							wm.z = -wm.z;
						}
						pdf *= 0.5;
					}
					else{
						// uniform sampling
						/**/
						wm = warp::squareToUniformSphere(Point2(randUniform<Float>(), randUniform<Float>()));
						//if (dot(wi, wm) < 0)
						//	continue; // sigmaP = 0;
						pdf = warp::squareToUniformSpherePdf();
					}

					if (pdf < Epsilon)
						continue;
					//result += std::max(0.f, dot(wm, wi)) * D(wm, orentation) / pdf;
					Float Sxx, Sxy, Sxz, Syy, Syz, Szz;
					computeS(Sxx, Sxy, Sxz, Syy, Syz, Szz, wm, m_fiberP, m_alphaP);
					Float sigmaP = sigma(wi, Sxx, Syy, Szz, Sxy, Sxz, Syz);
					result += sigmaP * D(wm, Vector3(0, 0, 1)) / pdf;
				}
				m_sigmaTable[i][j] = result / samples;
			}
		}

		// data output
		if (m_fiber == false){
			m_fitWeight = m_sigmaTable[0][0];
			m_fitSigma = m_sigmaTable[thetaCount / 2][0] / m_fitWeight;
		}
		else{
			m_fitWeight = m_sigmaTable[thetaCount / 2][0];
			m_fitSigma = m_sigmaTable[0][0] / m_fitWeight;
		}
		SLog(EDebug, "The m_fitWeight m_fitSigma is %f, %f", m_fitWeight, m_fitSigma);

		std::string address("D:/Pan/porousMedium/scenes/exp_aniMedium/sigmaData/");
		std::ofstream volume(address + "micro_volume.txt");
		std::ofstream flake(address + "micro_flake.txt");
		for (int u = 0; u < thetaCount; u++){
			volume << u << ' ' << m_sigmaTable[u][0] << endl;
			Vector3 wi = indexToW(u, 0);
			Float Sxx, Sxy, Sxz, Syy, Syz, Szz;
			computeS(Sxx, Sxy, Sxz, Syy, Syz, Szz, Vector3(0, 0, 1), m_fiber, m_fitSigma);

			float ret = sigma(wi, Sxx, Syy, Szz, Sxy, Sxz, Syz);
			flake << u << ' ' << m_fitWeight * ret << endl;
		}
		volume.close();
		flake.close();

		// micro phase function precomputation
		precomputeMicroPhase(m_microPhaseTable, m_intIOR, m_extIOR, m_alphaP, m_fiberP, m_mirror);
		precomputeMicroPhase(m_wetMicroPhaseTable, m_intIOR, m_extIOR_wet, m_alphaP, m_fiberP, m_mirror);
		if (m_precomputation){
			precomputePhase(m_phaseTable, m_microPhaseTable, m_intIOR, m_extIOR, m_alpha, m_alphaP, m_fiber, m_fiberP);
			precomputePhase(m_wetPhaseTable, m_wetMicroPhaseTable, m_intIOR, m_extIOR_wet, m_alpha, m_alphaP, m_fiber, m_fiberP);
		}

		Vector w = indexToWPhase(10, 7);
		SLog(EDebug, "The value is %f, %f, %f", w.x, w.y, w.z);
		Float index1, index2;
		wToIndexPhase(w, index1, index2);
		SLog(EDebug, "The index is %f, %f", index1, index2);

		// normalization
		Float interval = 4 * M_PI / phaseCount / phaseCount;
		Float phaseCDF1 = 0;
		Float phaseCDF2 = 0;
		for (int t = 0; t < phaseCount; t++){
			phaseCDF1 = 0;
			phaseCDF2 = 0;
			for (int i = 0; i < phaseCount; i++){
				for (int j = 0; j < phaseCount; j++){
					phaseCDF1 += m_phaseTable[t][i][j] * interval;
					phaseCDF2 += m_wetPhaseTable[t][i][j] * interval;
				}
			}
			//SLog(EDebug, "The phaseCDF %i is %f", t, phaseCDF1);
			for (int i = 0; i < phaseCount; i++){
				for (int j = 0; j < phaseCount; j++){
					m_phaseTable[t][i][j] /= phaseCDF1;
					m_wetPhaseTable[t][i][j] /= phaseCDF2;
				}
			}
			phaseCDF1 = 0;
			phaseCDF2 = 0;
			for (int i = 0; i < phaseCount; i++){
				for (int j = 0; j < phaseCount; j++){
					phaseCDF1 += m_phaseTable[t][i][j] * interval;
					phaseCDF2 += m_wetPhaseTable[t][i][j] * interval;
				}
			}
			SLog(EDebug, "The phaseCDF %i is %f", t, phaseCDF1);
			SLog(EDebug, "The phaseCDFwet %i is %f", t, phaseCDF2);
		}

		// compute marginal and conditional PDF
		//Float interval = 4 * M_PI / phaseCount / phaseCount;
		for (int t = 0; t < phaseCount; t++){
			for (int i = 0; i < phaseCount; i++){
				Float marPDF = 0;
				Float marPDFwet = 0;
				for (int j = 0; j < phaseCount; j++){
					marPDF += m_phaseTable[t][i][j] * interval;
					marPDFwet += m_wetPhaseTable[t][i][j] * interval;
				}
				m_phaseMar[t][i] = marPDF;
				m_phaseMarWet[t][i] = marPDFwet;
			}
			for (int i = 0; i < phaseCount; i++){
				for (int j = 0; j < phaseCount; j++){
					m_phaseCond[t][i][j] = m_phaseTable[t][i][j] * interval / m_phaseMar[t][i];
				}
			}
		}

		// test
		for (int t = 0; t < phaseCount; t++){
			Float out = 0;
			for (int i = 0; i < phaseCount; i++){
				out += m_phaseMar[t][i];
			}
			SLog(EDebug, "The marPDF %i is %f", t, out);
		}

		//for (int i = 0; i < phaseCount; i++){
		//	SLog(EDebug, "The marPDF 127: %i is %f", i, m_phaseMar[127][i]);
		//}

		for (int i = 0; i < phaseCount; i++){
			Float out = 0;
			for (int j = 0; j < phaseCount; j++){
				out += m_phaseCond[0][i][j];
			}
			SLog(EDebug, "The condPDF %i is %f", i, out);
		}

		for (int t = 0; t < phaseCount; t++){
			Float out = 0;
			for (int i = 0; i < phaseCount; i++){
				out += m_phaseMarWet[t][i];
			}
			SLog(EDebug, "The marPDFwet %i is %f", t, out);
		}


		phaseCDF1 = 0;
		phaseCDF2 = 0;
		Float phaseCDF3 = 0;
		for (int i = 0; i < phaseCount; i++){
			for (int j = 0; j < phaseCount; j++){
				phaseCDF1 += m_microPhaseTable[1][i][j] * interval;
				phaseCDF2 += m_microPhaseTable[32][i][j] * interval;
				phaseCDF3 += m_microPhaseTable[60][i][j] * interval;
			}
		}
		SLog(EDebug, "The micro-phaseCDF is %f, %f, %f", phaseCDF1, phaseCDF2, phaseCDF3);

	}

	SggxPhaseFunction(Stream *stream, InstanceManager *manager)
		: PhaseFunction(stream, manager) {
		m_fiber = stream->readBool();
		m_alpha = stream->readFloat();
		configure();
	}

	virtual ~SggxPhaseFunction() { }

	void configure() {
		PhaseFunction::configure();
		m_type = EAnisotropic | ENonSymmetric;
	}

	void serialize(Stream *stream, InstanceManager *manager) const {
		PhaseFunction::serialize(stream, manager);
		stream->writeBool(m_fiber);
		stream->writeFloat(m_alpha);
	}

	void precomputePhase(Float phase[phaseCount][phaseCount][phaseCount], Float microPhase[phaseCount][phaseCount][phaseCount], Float intIOR, Float extIOR, Float alpha, Float alphaP, bool fiber, bool fiberP){
		// phase function read
		//std::string address = "D:/Pan/porousMedium/scenes/exp_aniMedium/phase";
		std::string address = "phaseData";
		std::string filename(std::string("/") + "phase_i" + std::to_string(intIOR) + "_e" + std::to_string(extIOR) + "_r" + std::to_string(alpha) + "_rP" + std::to_string(alphaP) + (fiber ? "_fiber" : "_surface") + ".bin");
		std::ifstream coe(address + filename, std::ios::binary);
		if (coe.is_open())//�ļ��Ѿ������� ����Ҫ�ظ�Ԥ����
		{
			// ��ȡ����
			coe.read(reinterpret_cast<char*>(phase), phaseCount * phaseCount * phaseCount * sizeof(float));
			// �ر��ļ�
			coe.close();
		}

		// phase function precomputation
#pragma omp parallel for
		for (int i = 0; i <= phaseCount; i++){
			Vector3 wi = indexToWPhase(i, 0);
			//for (int j = 0; j < phaseCount; j++){
			for (int k = 0; k < phaseCount; k++){
				for (int l = 0; l < phaseCount; l++){
					Float result = 0;
					int samples = 2000;// 2000
					//Vector3 wi = indexToWPhase(i, j);
					//Vector3 wo = indexToWPhase(k, l);
					for (int i = 0; i < samples; i++){
						//Vector3 wo = Frame(wi).toWorld(indexToWPhase(k, l));
						Vector3 wo = indexToWPhase(k + randUniform<Float>(), l + randUniform<Float>());

						Vector wm;
						Float pdf;
						if (!fiber){
							// importance sampling
							/**/
							MicrofacetDistribution distr(
								MicrofacetDistribution::EType::EGGX,
								alpha,
								alpha,
								false //sample visible
								);
							pdf = 0.f;
							wm = distr.sample(wi, Point2(randUniform<Float>(), randUniform<Float>()), pdf);
							if (randUniform<Float>() < 0.5){
								wm.z = -wm.z;
							}
							pdf *= 0.5;
						}
						else
						{
							// uniform sampling
							wm = warp::squareToUniformSphere(Point2(randUniform<Float>(), randUniform<Float>()));
							pdf = warp::squareToUniformSpherePdf();
						}

						if (pdf < Epsilon)
							continue;
						Float Sxx, Sxy, Sxz, Syy, Syz, Szz;
						computeS(Sxx, Sxy, Sxz, Syy, Syz, Szz, wm, fiberP, alphaP);
						Float sigmaP = sigma(wi, Sxx, Syy, Szz, Sxy, Sxz, Syz); //abs(dot(wm, pRec.wi));
						//if (dot(wi, wm) < 0)
						//	sigmaP = 0;

						// micro phase 
						//Float microP = std::max(0.f, dot(wm, wo)) / M_PI;
						Frame normal(wm);
						Vector microWi = normal.toLocal(wi);
						Vector microWo = normal.toLocal(wo);

						//local
						//microWo = Frame(microWi).toLocal(microWo);

						Float thetaIndexI, phiIndexI, thetaIndexO, phiIndexO;
						wToIndexPhase(microWi, thetaIndexI, phiIndexI);
						wToIndexPhase(microWo, thetaIndexO, phiIndexO);
						int x1 = math::clamp(static_cast<int>(std::floor(thetaIndexI)), 0, phaseCount - 1);
						int y1 = math::clamp(static_cast<int>(std::floor(phiIndexI)), 0, phaseCount - 1);
						int x2 = math::clamp(static_cast<int>(std::floor(thetaIndexO)), 0, phaseCount - 1);
						int y2 = math::clamp(static_cast<int>(std::floor(phiIndexO)), 0, phaseCount - 1);

						//world
						/**/
						y2 -= y1;
						if (y2 < 0){
							y2 += phaseCount;
						}

						Float microP = microPhase[x1][x2][y2];
						result += microP * sigmaP * D(wm, Vector3(0, 0, 1)) / pdf;
					}
					//m_phaseTable[i][j][k][l] = result / samples / sigmaDir(Vector3(0, 0, 1), wi);
					phase[i][k][l] = result / samples / sigmaDir(Vector3(0, 0, 1), wi);
				}
			}
			//}
		}

		// mean costheta
		Float meanG = 0;
		Vector3 wi = indexToWPhase(phaseCount / 2, 0);
		Float interval = 4 * M_PI / phaseCount / phaseCount;
		for (int i = 0; i < phaseCount; i++){
			for (int j = 0; j < phaseCount; j++){
				Vector3 wo = indexToWPhase(i, j);
				Float costheta = dot(wi, wo);
				meanG += phase[phaseCount / 2][i][j] * costheta * interval;
			}
		}

		std::ofstream phaseFile(address + filename, std::ios::binary);
		// д������
		phaseFile.write(reinterpret_cast<const char*>(phase), (phaseCount+1) * phaseCount * phaseCount * sizeof(float));

		// �ر��ļ�
		phaseFile.close();

	}

	inline float sqr(float x) const
	{
		return x*x;
	}

	float eval_specular(Vector3 wi, Vector3 wo) const
	{
		if ((wi + wo).isZero())
			return 0;

		Vector3 wh = normalize(wi + wo);
		return 0.25f * D(wh) / sigma(wi);
	}

	bool isValid(Vector2 p)
	{
		return p.x*p.x + p.y*p.y < 1.0;
	}

	void computeForWi(std::string output)
	{
	//	printDir(wi);
		const int res = 256;
		std::vector<float> image;
		image.resize(res*res * 3, 0);

		for (int oidx = 0; oidx < res * res; oidx++)
		{
			int io = oidx / res;
			int jo = oidx % res;

			Vector2 p2o = Point2((io + 0.5) / res, (jo + 0.5) / res) * 2 - Point2(1);
			if (isValid(p2o))
			{
				Vector wo = Vector3(p2o.x, p2o.y, sqrt(1 - p2o.x*p2o.x - p2o.y*p2o.y));
				//Spectrum value = eval(bRec, ESolidAngle);

				float value = sigma(wo);//
				//the attenuation along the outgoing ray is not considered.
				image[oidx * 3] = value;// [0];
				image[oidx * 3 + 1] = value;// [1];
				image[oidx * 3 + 2] = value;// [2];

			}
		}

		SaveEXRChannel3(image, res, res, output.c_str());
		cout << "Rendering to Image done!" << endl;
		std::vector<float>().swap(image);
	}	
	
	void computeForWi2(const Vector &orientation, std::string output)
	{
		//	printDir(wi);
		const int res = 256;
		std::vector<float> image;
		image.resize(res*res * 3, 0);
		Float Sxx, Sxy, Sxz, Syy, Syz, Szz;
		computeS(Sxx, Sxy, Sxz, Syy, Syz, Szz, orientation, m_fiber, m_alpha);
		for (int oidx = 0; oidx < res * res; oidx++)
		{
			int io = oidx / res;
			int jo = oidx % res;

			Vector2 p2o = Point2((io + 0.5) / res, (jo + 0.5) / res) * 2 - Point2(1);
			if (isValid(p2o))
			{
				Vector wo = Vector3(p2o.x, p2o.y, sqrt(1 - p2o.x*p2o.x - p2o.y*p2o.y));

				float value = sigma(wo, Sxx, Syy, Szz, Sxy, Sxz, Syz); //sigmaDir(orientation, wo);// 
				//the attenuation along the outgoing ray is not considered.
				image[oidx * 3] = value;// [0];
				image[oidx * 3 + 1] = value;// [1];
				image[oidx * 3 + 2] = value;// [2];

			}
		}

		SaveEXRChannel3(image, res, res, output.c_str());
		cout << "Rendering to Image done!" << endl;
		std::vector<float>().swap(image);
	}

	float sigma(Vector3 wi,
		float S_xx, float S_yy, float S_zz,
		float S_xy, float S_xz, float S_yz) const
	{
		const float sigma_squared = wi.x*wi.x*S_xx + wi.y*wi.y*S_yy + wi.z*wi.z*S_zz
			+ 2.0f * (wi.x*wi.y*S_xy + wi.x*wi.z*S_xz + wi.y*wi.z*S_yz);
		return (sigma_squared > 0.0f) ? sqrtf(sigma_squared) : 0.0f; // conditional to avoid numerical errors
	}

	float D(const Vector3 &wm,
		float S_xx, float S_yy, float S_zz,
		float S_xy, float S_xz, float S_yz)const
	{
		const float detS =
			S_xx*S_yy*S_zz - S_xx*S_yz*S_yz - S_yy*S_xz*S_xz - S_zz*S_xy*S_xy + 2.0f*S_xy*S_xz*S_yz;
		
		const float den = wm.x*wm.x*(S_yy*S_zz - S_yz*S_yz) + wm.y*wm.y*(S_xx*S_zz - S_xz*S_xz) + wm.z*wm.z*(S_xx*S_yy - S_xy*S_xy)
			+ 2.0f*(wm.x*wm.y*(S_xz*S_yz - S_zz*S_xy) + wm.x*wm.z*(S_xy*S_yz - S_yy*S_xz) + wm.y*wm.z*(S_xy*S_xz - S_xx*S_yz));
		const float D = powf(fabsf(detS), 1.50f) / (M_PI*den*den);
		return D;
	}

	float eval_specular(const Vector3 & wi, const Vector3 & wo,
		const float S_xx, const float S_yy, const float S_zz,
		const float S_xy, const float S_xz, const float S_yz)const
	{
		Vector3 wh = normalize(wi + wo);
		return 0.25f * D(wh, S_xx, S_yy, S_zz, S_xy, S_xz, S_yz) / sigma(wi, S_xx, S_yy, S_zz, S_xy, S_xz, S_yz);
	}

	float sigma(const Vector3 &x) const
	{
		if (m_fiber)
		{
			// For an SGGX microflake, sigma(x)^2 = x^T * S * x,
			// where S = diag(1, 1, alpha^2) for a fiber-like distribution.
			float s2 = sqr(x[0]) + sqr(x[1]) + sqr(m_alpha * x[2]);
			return std::sqrt(s2);
		}
		else
		{	// For an SGGX microflake, sigma(x)^2 = x^T * S * x,
			// where S = diag(alpha^2, alpha^2, 1) for a surface-like distribution.
			float s2 = sqr(m_alpha * x[0]) + sqr(m_alpha * x[1]) + sqr(x[2]);
			return std::sqrt(s2);
		}
	}

	float sigma(const Vector3 &x, float roughness) const
	{
		if (m_fiber)
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

	float D(const Vector3 &x) const
	{

		if (m_fiber)
		{
			// For an SGGX microflake, D(x) = 1 / (pi * sqrt |det S| * (x^T S^-1 x)^2),
			// where S = diag(1, 1, alpha^2) for a fiber-like distribution.
			float tmp = sqr(x[0]) + sqr(x[1]) + sqr(x[2] / m_alpha);      // tmp = x^T S^-1 x
			return 1 / (M_PI * m_alpha * sqr(tmp));                       // det S = alpha^2
		}
		else
		{
			// For an SGGX microflake, D(x) = 1 / (pi * sqrt |det S| * (x^T S^-1 x)^2),
			// where S = diag(alpha^2, alpha^2, 1) for a surface-like distribution.
			float tmp = sqr(x[0] / m_alpha) + sqr(x[1] / m_alpha) + sqr(x[2]);      // tmp = x^T S^-1 x
			return 1 / (M_PI * m_alpha * m_alpha* sqr(tmp));                       // det S = alpha^2
		}

	}

	float D(const Vector3 &x, float roughness) const
	{

		if (m_fiber)
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


	float D(const Vector3 &x, const Vector &orientation) const
	{
		Float Sxx, Sxy, Sxz, Syy, Syz, Szz;
		computeS(Sxx, Sxy, Sxz, Syy, Syz, Szz, orientation, m_fiber, m_alpha);

		return D(x, Sxx, Syy, Szz, Sxy, Sxz, Syz);
	}


	float sigma(const Vector3 &x, const Vector &orientation) const
	{
		Float Sxx, Sxy, Sxz, Syy, Syz, Szz;
		computeS(Sxx, Sxy, Sxz, Syy, Syz, Szz, orientation, m_fiber, m_alpha);

		return sigma(x, Sxx, Syy, Szz, Sxy, Sxz, Syz);
	}

	void computeS(Float &Sxx, Float &Sxy, Float &Sxz, Float &Syy, Float &Syz, Float & Szz,
		const Vector &omega3, bool fiber, Float alpha) const
	{
		float x = omega3.x;
		float y = omega3.y;
		float z = omega3.z;

		float roughness = alpha;
		float sigma_2 = roughness * roughness;
		if (fiber)
		{
			Sxx = sigma_2 *x*x + y*y + z*z;

			Sxy = sigma_2*x*y - x*y;
			Sxz = sigma_2*x*z - x*z;
			Syy = sigma_2*y*y + x*x + z*z;
			Syz = sigma_2*y*z - y*z;
			Szz = sigma_2 *z*z + x*x + y*y;
		}
		else
		{
			Sxx = x*x + sigma_2 *(y*y + z*z);
			Sxy = x*y - sigma_2 *x*y;
			Sxz = x*z - sigma_2 * x*z;
			Syy = y*y + sigma_2 *(x*x + z*z);
			Syz = y*z - sigma_2 *y*z;
			Szz = z*z + sigma_2 *(x*x + y*y);
		}
	}

	Float eval(const PhaseFunctionSamplingRecord &pRec) const {
		return eval_wet(pRec, m_saturation);
	}

	Float eval_wet(const PhaseFunctionSamplingRecord &pRec, Float saturation) const{
		
		//return 0.25/M_PI;
		if (pRec.mRec.orientation.isZero())
			return 0.0f;
		if (m_precomputation){
			/*
			Frame orient(pRec.mRec.orientation);
			Vector3 wi = orient.toLocal(pRec.wi);
			Vector3 wo = orient.toLocal(pRec.wo);

			Float thetaIndexI, phiIndexI, thetaIndexO, phiIndexO;
			wToIndexPhase(wi, thetaIndexI, phiIndexI);
			wToIndexPhase(wo, thetaIndexO, phiIndexO);

			int x1 = math::clamp((int)std::floor(thetaIndexI), 0, phaseCount - 2);
			int y1 = math::clamp((int)std::floor(phiIndexI), 0, phaseCount - 1);
			int x2 = math::clamp((int)std::floor(thetaIndexO), 0, phaseCount - 2);
			int y2 = math::clamp((int)std::floor(phiIndexO), 0, phaseCount - 1);

			y2 -= y1;
			if (y2 < 0)
				y2 += phaseCount;

			// ��ֵ����
			Float dx = thetaIndexI - x1;
			Float dy = thetaIndexO - x2;
			Float dz = phiIndexO - std::floor(phiIndexO);

			// ȡ�� 8 ���ڽ����ֵ��ע��߽磡��
			auto lookup = [&](int i, int j, int k) {
				i = std::min(i, phaseCount);
				j = std::min(j, phaseCount - 1);
				k = (k + phaseCount) % phaseCount;  // wrap around phi
				Float wet = m_wetPhaseTable[i][j][k];
				Float dry = m_phaseTable[i][j][k];
				return saturation * wet + (1 - saturation) * dry;
			};

			// 8 ���ǵ�ֵ
			Float c000 = lookup(x1, x2, y2);
			Float c001 = lookup(x1, x2, y2 + 1);
			Float c010 = lookup(x1, x2 + 1, y2);
			Float c011 = lookup(x1, x2 + 1, y2 + 1);
			Float c100 = lookup(x1 + 1, x2, y2);
			Float c101 = lookup(x1 + 1, x2, y2 + 1);
			Float c110 = lookup(x1 + 1, x2 + 1, y2);
			Float c111 = lookup(x1 + 1, x2 + 1, y2 + 1);

			// �����Բ�ֵ
			Float c00 = c000 * (1 - dx) + c100 * dx;
			Float c01 = c001 * (1 - dx) + c101 * dx;
			Float c10 = c010 * (1 - dx) + c110 * dx;
			Float c11 = c011 * (1 - dx) + c111 * dx;

			Float c0 = c00 * (1 - dy) + c10 * dy;
			Float c1 = c01 * (1 - dy) + c11 * dy;
			//return c0;
			Float result = c0 * (1 - dz) + c1 * dz;
			
			return result;*/
			//return INV_FOURPI;
			/*
			Frame orient(pRec.mRec.orientation);
			Vector3 wi = orient.toLocal(pRec.wi);
			Vector3 wo = orient.toLocal(pRec.wo);

			Float thetaIndexI, phiIndexI, thetaIndexO, phiIndexO;
			wToIndexPhase(wi, thetaIndexI, phiIndexI);

			// local
			//wo = Frame(wi).toLocal(wo);

			wToIndexPhase(wo, thetaIndexO, phiIndexO);
			int x1 = math::clamp(static_cast<int>(std::floor(thetaIndexI)), 0, phaseCount - 1);
			int y1 = math::clamp(static_cast<int>(std::floor(phiIndexI)), 0, phaseCount - 1);
			int x2 = math::clamp(static_cast<int>(std::floor(thetaIndexO)), 0, phaseCount - 1);
			int y2 = math::clamp(static_cast<int>(std::floor(phiIndexO)), 0, phaseCount - 1);
			// world
			y2 -= y1;
			if (y2 < 0)
				y2 += phaseCount;

			Float tmp0, tmp1;
			if (x1 + 1 <= phaseCount - 1){
				tmp0 = saturation*m_wetPhaseTable[x1][x2][y2] + (1 - saturation)*m_phaseTable[x1][x2][y2];
				tmp1 = saturation*m_wetPhaseTable[x1 + 1][x2][y2] + (1 - saturation)*m_phaseTable[x1 + 1][x2][y2];
			}
			else{
				tmp0 = saturation*m_wetPhaseTable[x1][x2][y2] + (1 - saturation)*m_phaseTable[x1][x2][y2];
				tmp1 = saturation*m_wetPhaseTable[x1 + 1][x2][y2] + (1 - saturation)*m_phaseTable[x1 + 1][x2][y2];
			}
			//Float tmp0 = saturation*m_wetPhaseTable[x1][x2][y2] + (1 - saturation)*m_phaseTable[x1][x2][y2];
			//Float tmp1 = saturation*m_wetPhaseTable[std::min(x1 + 1, phaseCount - 1)][x2][y2] + (1 - saturation)*m_phaseTable[std::min(x1 + 1, phaseCount - 1)][x2][y2];

			return (1 - thetaIndexI + x1)*tmp0 + (thetaIndexI - x1)*tmp1;// saturation*m_wetPhaseTable[x1][x2][y2] + (1 - saturation)*m_phaseTable[x1][x2][y2];*/
			// 1. 计算浮点数索引
			Frame orient(pRec.mRec.orientation);
			Vector3 wi = orient.toLocal(pRec.wi);
			Vector3 wo = orient.toLocal(pRec.wo);
			
			Float thetaIndexI, phiIndexI, thetaIndexO, phiIndexO;
			wToIndexPhase(wi, thetaIndexI, phiIndexI);
			wToIndexPhase(wo, thetaIndexO, phiIndexO);

			// 2. 处理相对方位角 phi (保持环绕特性)
			Float relPhi = phiIndexO - phiIndexI;
			if (relPhi < 0) relPhi += phaseCount;

			// 3. 确定插值基准点和权重
			int i0 = math::clamp((int)std::floor(thetaIndexI), 0, phaseCount - 2);
			int j0 = math::clamp((int)std::floor(thetaIndexO), 0, phaseCount - 2);
			int k0 = (int)std::floor(relPhi) % phaseCount;

			int i1 = i0 + 1;
			int j1 = j0 + 1;
			int k1 = (k0 + 1) % phaseCount; // Phi 是周期的，0度接360度

			Float alpha = thetaIndexI - i0;
			Float beta  = thetaIndexO - j0;
			Float gamma = relPhi - std::floor(relPhi);

			// 4. 定义查找函数（混合 Wet 和 Dry）
			auto getVal = [&](int i, int j, int k) {
				return saturation * m_wetPhaseTable[i][j][k] + (1 - saturation) * m_phaseTable[i][j][k];
			};

			// 5. 进行三线性插值
			Float res = 
				(1-alpha)*(1-beta)*(1-gamma) * getVal(i0, j0, k0) +
				alpha    *(1-beta)*(1-gamma) * getVal(i1, j0, k0) +
				(1-alpha)*beta    *(1-gamma) * getVal(i0, j1, k0) +
				alpha    *beta    *(1-gamma) * getVal(i1, j1, k0) +
				(1-alpha)*(1-beta)*gamma     * getVal(i0, j0, k1) +
				alpha    *(1-beta)*gamma     * getVal(i1, j0, k1) +
				(1-alpha)*beta    *gamma     * getVal(i0, j1, k1) +
				alpha    *beta    *gamma     * getVal(i1, j1, k1);

			return res;
		}
		else{
			Float result = 0;
			int samples = 100;
			Frame orient(pRec.mRec.orientation);
			Vector3 wi = orient.toLocal(pRec.wi);
			Vector3 wo = orient.toLocal(pRec.wo);
			for (int i = 0; i < samples; i++){
				Vector wm;
				Float pdf;
				if (!m_fiber){
					// importance sampling
					/**/
					MicrofacetDistribution distr(
						MicrofacetDistribution::EType::EGGX,
						m_alpha,
						m_alpha,
						false //sample visible
						);
					pdf = 0.f;
					wm = distr.sample(wi, Point2(randUniform<Float>(), randUniform<Float>()), pdf);
					if (randUniform<Float>() < 0.5){
						wm.z = -wm.z;
					}
					pdf *= 0.5;
				}
				else
				{
					// uniform sampling
					wm = warp::squareToUniformSphere(Point2(randUniform<Float>(), randUniform<Float>()));
					pdf = warp::squareToUniformSpherePdf();
				}
				//Vector3 wm = warp::squareToUniformSphere(Point2(randUniform<Float>(), randUniform<Float>()));
				//Float pdf = warp::squareToUniformSpherePdf();
				if (pdf < Epsilon)
					continue;
				Float Sxx, Sxy, Sxz, Syy, Syz, Szz;
				computeS(Sxx, Sxy, Sxz, Syy, Syz, Szz, wm, m_fiberP, m_alphaP);
				Float sigmaP = sigma(wi, Sxx, Syy, Szz, Sxy, Sxz, Syz); //abs(dot(wm, pRec.wi));
				//if (dot(pRec.wi, wm) < 0)
				//	sigmaP = 0;
				Frame normal(wm);
				Vector microWi = normal.toLocal(wi);
				Vector microWo = normal.toLocal(wo);
				//microWo = Frame(microWi).toLocal(microWo);
				Float thetaIndexI, phiIndexI, thetaIndexO, phiIndexO;
				wToIndexPhase(microWi, thetaIndexI, phiIndexI);
				wToIndexPhase(microWo, thetaIndexO, phiIndexO);
				int x1 = math::clamp(static_cast<int>(std::floor(thetaIndexI)), 0, phaseCount - 1);
				int y1 = math::clamp(static_cast<int>(std::floor(phiIndexI)), 0, phaseCount - 1);
				int x2 = math::clamp(static_cast<int>(std::floor(thetaIndexO)), 0, phaseCount - 1);
				int y2 = math::clamp(static_cast<int>(std::floor(phiIndexO)), 0, phaseCount - 1);
				y2 -= y1;
				if (y2 < 0){
					y2 += phaseCount;
				}
				Float microP = m_microPhaseTable[x1][x2][y2];
				//Float microP = std::max(0.f, dot(wm, pRec.wo)) / M_PI;
				result += microP * sigmaP * D(wm, Vector3(0, 0, 1)) / pdf;
			}

			return result / samples / sigmaDir(Vector3(0, 0, 1), wi);
		}
	}

	inline Float sample(PhaseFunctionSamplingRecord &pRec, Sampler *sampler) const {
		Float temp;
		return sample(pRec, temp, sampler);
		/*
		if (fabs(sample(pRec, sampler)) <1e-6) {
			pdf = 0; return 0.0f;
		}
		pdf = eval(pRec);
		return 1.0f;*/
	}

	void buildOrthonormalBasis(Vector3& omega_1, Vector3& omega_2, const Vector3& omega_3) const
	{
		if (omega_3.z < -0.9999999f)
		{
			omega_1 = Vector3(0.0f, -1.0f, 0.0f);
			omega_2 = Vector3(-1.0f, 0.0f, 0.0f);
		}
		else {
			const float a = 1.0f / (1.0f + omega_3.z);
			const float b = -omega_3.x*omega_3.y*a;
			omega_1 = Vector3(1.0f - omega_3.x*omega_3.x*a, b, -omega_3.x);
			omega_2 = Vector3(b, 1.0f - omega_3.y*omega_3.y*a, -omega_3.y);
		}
	}

	Vector3 sample_VNDF(const Vector3 wi,
		const float S_xx, const float S_yy, const float S_zz,
		const float S_xy, const float S_xz, const float S_yz,
		const float U1, const float U2) const
	{
		// generate sample (u, v, w)
		const float r = sqrtf(U1);
		const float phi = 2.0f*M_PI*U2;
		const float u = r*cosf(phi);
		const float v = r*sinf(phi);
		const float w = sqrtf(1.0f - u*u - v*v);
		// build orthonormal basis
		Vector3 wk, wj;
		buildOrthonormalBasis(wk, wj, wi);
		//float dotaa = dot(wk, wj);
		//float dotab = dot(wk, wi);
		//float dotbc = dot(wi, wj);
		//if (abs(dotaa) > 1e-4 || abs(dotab) > 1e-4 || abs(dotbc) > 1e-4)
		//	int a = 0;
		// project S in this basis
		const float S_kk = wk.x*wk.x*S_xx + wk.y*wk.y*S_yy + wk.z*wk.z*S_zz
			+ 2.0f * (wk.x*wk.y*S_xy + wk.x*wk.z*S_xz + wk.y*wk.z*S_yz);
		const float S_jj = wj.x*wj.x*S_xx + wj.y*wj.y*S_yy + wj.z*wj.z*S_zz
			+ 2.0f * (wj.x*wj.y*S_xy + wj.x*wj.z*S_xz + wj.y*wj.z*S_yz);
		const float S_ii = wi.x*wi.x*S_xx + wi.y*wi.y*S_yy + wi.z*wi.z*S_zz
			+ 2.0f * (wi.x*wi.y*S_xy + wi.x*wi.z*S_xz + wi.y*wi.z*S_yz);
		const float S_kj = wk.x*wj.x*S_xx + wk.y*wj.y*S_yy + wk.z*wj.z*S_zz
			+ (wk.x*wj.y + wk.y*wj.x)*S_xy
			+ (wk.x*wj.z + wk.z*wj.x)*S_xz
			+ (wk.y*wj.z + wk.z*wj.y)*S_yz;
		const float S_ki = wk.x*wi.x*S_xx + wk.y*wi.y*S_yy + wk.z*wi.z*S_zz
			+ (wk.x*wi.y + wk.y*wi.x)*S_xy + (wk.x*wi.z + wk.z*wi.x)*S_xz + (wk.y*wi.z + wk.z*wi.y)*S_yz;
		const float S_ji = wj.x*wi.x*S_xx + wj.y*wi.y*S_yy + wj.z*wi.z*S_zz
			+ (wj.x*wi.y + wj.y*wi.x)*S_xy
			+ (wj.x*wi.z + wj.z*wi.x)*S_xz
			+ (wj.y*wi.z + wj.z*wi.y)*S_yz;
		// compute normal
		float sqrtDetSkji = sqrtf(fabsf(S_kk*S_jj*S_ii - S_kj*S_kj*S_ii - S_ki*S_ki*S_jj - S_ji*S_ji*S_kk + 2.0f*S_kj*S_ki*S_ji));
		float inv_sqrtS_ii = 1.0f / sqrtf(S_ii);
		float tmp = sqrtf(S_jj*S_ii - S_ji*S_ji);
		Vector3 Mk(sqrtDetSkji / tmp, 0.0f, 0.0f);
		Vector3 Mj(-inv_sqrtS_ii*(S_ki*S_ji - S_kj*S_ii) / tmp, inv_sqrtS_ii*tmp, 0);
		Vector3 Mi(inv_sqrtS_ii*S_ki, inv_sqrtS_ii*S_ji, inv_sqrtS_ii*S_ii);
		Vector3 wm_kji = normalize(u*Mk + v*Mj + w*Mi);
		// rotate back to world basis
		Vector3 wm =  wm_kji.x * wk + wm_kji.y * wj + wm_kji.z * wi;

		return normalize(wm);
	}


	Vector3 sample_specular(const Vector3 wi,
		const float S_xx, const float S_yy, const float S_zz,
		const float S_xy, const float S_xz, const float S_yz,
		const float U1, const float U2) const
	{
		// sample VNDF
		const Vector3 wm = sample_VNDF(wi, S_xx, S_yy, S_zz, S_xy, S_xz, S_yz, U1, U2);

		// specular reflection
		const Vector3 wo = reflect(wi, wm);
		//if (wo.z > 5)
		//	int a = 0;
		return wo;
	}

	Float sample(PhaseFunctionSamplingRecord &pRec,
			Float &pdf, Sampler *sampler) const {
		return sample_wet(pRec, sampler, pdf, m_saturation);
	}

	Float sample_wet(PhaseFunctionSamplingRecord &pRec,
		Sampler *sampler, Float &pdf, Float saturation) const{
		if (pRec.mRec.orientation.isZero()) {
			return 0.0f;
		}
		if (m_precomputation){
			//pRec.wo = warp::squareToUniformSphere(sampler->next2D());
			//pdf = warp::squareToUniformSpherePdf();
			//return eval_wet(pRec, saturation) / pdf;

			/**/
			// marginal and conditional sampling
			Frame orient(pRec.mRec.orientation);
			Vector3 wi = orient.toLocal(pRec.wi);
			Float thetaIndexI, phiIndexI;
			wToIndexPhase(wi, thetaIndexI, phiIndexI);
			int x1 = math::clamp(static_cast<int>(std::floor(thetaIndexI)), 0, phaseCount - 1); 
			int y1 = math::clamp(static_cast<int>(std::floor(phiIndexI)), 0, phaseCount - 1);
			Float thetaInd = 0, phiInd = 0;
			Float interval = 4 * M_PI / phaseCount / phaseCount;
			// theta sampling
			int thetaIndex = 0;
			Float rand = sampler->next1D();
			for (int i = 0; i < phaseCount; i++){
				Float cdfTheta = (1 - saturation) * m_phaseMar[x1][i] + saturation * m_phaseMarWet[x1][i];
				if (rand < cdfTheta){
					Float offset = rand / cdfTheta;
					thetaInd = math::clamp(i + offset, 0.f, phaseCount + 0.f);
					thetaIndex = i;
					// phi sampling
					rand = sampler->next1D();
					for (int j = 0; j < phaseCount; j++){
						Float cdfPhi = ((1 - saturation)*m_phaseTable[x1][i][j] + saturation*m_wetPhaseTable[x1][i][j]) * interval / cdfTheta;
						if (rand < cdfPhi){
							Float offset = rand / cdfPhi;
							phiInd = j + offset;

							// wo
							phiInd += phiIndexI;
							phiInd = std::fmod(phiInd + phaseCount, phaseCount); // ȷ���Ǹ�
							pRec.wo = orient.toWorld(indexToWPhase(thetaInd, phiInd));
							pdf = (1 - saturation)*m_phaseTable[x1][i][j] + saturation*m_wetPhaseTable[x1][i][j];
							if (pdf == 0)
								return 0.f;
							return 1.f;// eval_wet(pRec, m_saturation) / pdf;// 1.f;
						}
						else
							rand -= cdfPhi;
					}
				}
				else
					rand -= cdfTheta;
			}
	
			return 0.f;

			/*
			// 1D CDF sampling
			Frame orient(pRec.mRec.orientation);
			Vector3 wi = orient.toLocal(pRec.wi);

			Float thetaIndexI, phiIndexI;
			wToIndexPhase(wi, thetaIndexI, phiIndexI);
			int x1 = math::clamp(static_cast<int>(std::floor(thetaIndexI)), 0, phaseCount - 1);
			int y1 = math::clamp(static_cast<int>(std::floor(phiIndexI)), 0, phaseCount - 1);

			Float interval = 4 * M_PI / phaseCount / phaseCount;
			Float rand = sampler->next1D();
			Point2 rand2 = sampler->next2D();
			for (int i = 0; i < phaseCount; i++){
				for (int j = 0; j < phaseCount; j++){
					Float cdf = (saturation*m_wetPhaseTable[x1][i][j] + (1 - saturation)*m_phaseTable[x1][i][j]) * interval;
					if (rand < cdf){
						// world
						Float index2 = j + phiIndexI;
						if (index2 >= phaseCount)
							index2 -= phaseCount;
						Float offset = rand / cdf;
						pRec.wo = orient.toWorld(indexToWPhase(i, index2));

						//local
						//pRec.wo = orient.toWorld(Frame(wi).toWorld(indexToWPhase(i + rand2.x, j + rand2.y)));

						pdf = (saturation*m_wetPhaseTable[x1][i][j] + (1 - saturation)*m_phaseTable[x1][i][j]);// *interval;
						if (pdf == 0)
							return 0.f;
						return 1.f;
					}
					else
						rand -= cdf;
				}
			}
			return 0.f;*/
		}
		else{
			pRec.wo = warp::squareToUniformSphere(sampler->next2D());
			pdf = warp::squareToUniformSpherePdf();
			return eval_wet(pRec, saturation) / pdf;
		}
	}
	/**/
	Float pdf(const PhaseFunctionSamplingRecord &pRec) const{
		if (m_precomputation){
			//return warp::squareToUniformSpherePdf();
			Float interval = 4 * M_PI / phaseCount / phaseCount;
			return eval_wet(pRec, m_saturation);// *interval;
		}
		else{
			return warp::squareToUniformSpherePdf();
		}
	}

	bool needsDirectionallyVaryingCoefficients() const { return true; }

	Float sigmaDir(const Vector &orentation, const Vector &wi) const {
		//return 4;
		//if (m_precomputation){
			Float Sxx, Sxy, Sxz, Syy, Syz, Szz;
			computeS(Sxx, Sxy, Sxz, Syy, Syz, Szz, orentation, m_fiber, m_fitSigma);

			float ret = sigma(wi, Sxx, Syy, Szz, Sxy, Sxz, Syz);
			return ret * m_fitWeight;
		//}
		

		/*
		if (m_precomputation){
			//if (m_fit){
				
			//}
			Frame orient(orentation);
			Vector3 w = orient.toLocal(wi);

			Float thetaIndex, phiIndex;
			wToIndex(w, thetaIndex, phiIndex);
			int x1 = static_cast<int>(std::floor(thetaIndex));
			int y1 = static_cast<int>(std::floor(phiIndex));
			int x2 = std::min(static_cast<int>(std::ceil(thetaIndex)), thetaCount - 1);
			int y2 = std::min(static_cast<int>(std::ceil(phiIndex)), phiCount - 1);

			double tx = thetaIndex - x1;
			double ty = phiIndex - y1;

			double f11 = m_sigmaTable[x1][y1];
			double f21 = m_sigmaTable[x1][y2];
			double f12 = m_sigmaTable[x2][y1];
			double f22 = m_sigmaTable[x2][y2];

			// ˮƽ��������Բ�ֵ
			double f_x_y1 = f11 * (1 - tx) + f21 * tx;
			double f_x_y2 = f12 * (1 - tx) + f22 * tx;
			// ��ֱ��������Բ�ֵ
			double f_xy = f_x_y1 * (1 - ty) + f_x_y2 * ty;

			return f_xy;
			//return m_sigmaTable[x1][y1];
		}*/
		Float alpha = m_alpha*m_alpha;
		MicrofacetDistribution distr(
			MicrofacetDistribution::EType::EGGX,
			alpha,
			alpha,
			false //sample visible
			);

		Float result = 0;
		int samples = 100;
		for (int i = 0; i < samples; i++){
			Vector3 wm = warp::squareToUniformSphere(Point2(randUniform<Float>(), randUniform<Float>()));
			//if (dot(wi, wm) < 0)
			//	continue; // sigmaP = 0;
			Float pdf = warp::squareToUniformSpherePdf();
			if (pdf < Epsilon)
				continue;
			//result += std::max(0.f, dot(wm, wi)) * D(wm, orentation) / pdf;
			Float Sxx, Sxy, Sxz, Syy, Syz, Szz;
			computeS(Sxx, Sxy, Sxz, Syy, Syz, Szz, wm, m_fiberP, m_alphaP);
			Float sigmaP = sigma(wi, Sxx, Syy, Szz, Sxy, Sxz, Syz);
			result += sigmaP * D(wm, orentation) / pdf;
		} 
		return result / samples;
		/*
#if 1
		
		// Scaled such that replacing an isotropic phase function with an
		// isotropic microflake distribution does not cause changes
		Float Sxx, Sxy, Sxz, Syy, Syz, Szz;
		computeS(Sxx, Sxy, Sxz, Syy, Syz, Szz, orentation, m_fiber, m_alpha);

		float ret = sigma(wi, Sxx, Syy, Szz, Sxy, Sxz, Syz);
		return ret;
#else
		Vector3 wi = Vector3(sqrtf(1 - (cosTheta*cosTheta)), 0, cosTheta);
		float ret = sigma(wi,m_alpha);
		return ret;
#endif*/
	}

	Float sigmaDir(const float cosTheta) const {
#if 1
		// Scaled such that replacing an isotropic phase function with an
		// isotropic microflake distribution does not cause changes
		Float Sxx, Sxy, Sxz, Syy, Syz, Szz;
		Vector3 omega3(0.0, 0.0, 1.0);
		computeS(Sxx, Sxy, Sxz, Syy, Syz, Szz, omega3, m_fiber, m_alpha);

		Vector3 wi = Vector3(sqrtf(1 - (cosTheta*cosTheta)), 0, cosTheta);
		float ret = sigma(wi, Sxx, Syy, Szz, Sxy, Sxz, Syz);
		return ret;
#else
		Vector3 wi = Vector3(sqrtf(1 - (cosTheta*cosTheta)), 0, cosTheta);
		float ret = sigma(wi, m_alpha);
		return ret;
#endif
	}

	Float sigmaDirMax() const {
		return sigmaDir(0);
	}

	Float DsDv() const {
		Float resultS = 0, resultD = 0;
		int samples = 10000;
		for (int i = 0; i < samples; i++){
			Vector wm;
			Float pdf;

			wm = warp::squareToUniformSphere(Point2(randUniform<Float>(), randUniform<Float>()));
			pdf = warp::squareToUniformSpherePdf();
			Float sigma = sigmaDir(Vector3(0, 0, 1), wm);
			resultS += sigma;

			wm = warp::squareToUniformSphere(Point2(randUniform<Float>(), randUniform<Float>()));
			pdf = warp::squareToUniformSpherePdf();
			Float d = D(wm, Vector3(0, 0, 1)) / pdf;
			resultD += d;
		}

		Float avgSigma = resultS / resultD;

		return pow(avgSigma, 0.5) / pow(m_alphaP, 1.0 / 3.0);
	}

	std::string toString() const {
		std::ostringstream oss;
		oss << "SggxPhaseFunction[" << endl
			<< "   roughness = " << indent("0.1") << endl
			<< "]";
		return oss.str();
	}

	MTS_DECLARE_CLASS()
private:

	bool m_fiber;
	Float m_alpha;

	bool m_fit;
	Float m_fitAlpha;
	Float m_fitWeight;
	Float m_fitSigma;

	bool m_fiberP;
	Float m_alphaP;
	Float m_intIOR;
	Float m_extIOR;
	Float m_extIOR_wet;
	Float m_saturation;

	int m_sample;

	Float m_sigmaTable[thetaCount][phiCount];
	Float m_phaseTable[phaseCount+1][phaseCount][phaseCount];
	Float m_wetPhaseTable[phaseCount+1][phaseCount][phaseCount];
	Float m_microPhaseTable[phaseCount][phaseCount][phaseCount];
	Float m_wetMicroPhaseTable[phaseCount][phaseCount][phaseCount];
	Float m_phaseMar[phaseCount][phaseCount];
	Float m_phaseMarWet[phaseCount][phaseCount];
	Float m_phaseCond[phaseCount][phaseCount][phaseCount];

	bool m_precomputation;
	bool m_mirror;
};

MTS_IMPLEMENT_CLASS_S(SggxPhaseFunction, false, PhaseFunction)
MTS_EXPORT_PLUGIN(SggxPhaseFunction, "Sggx phase function");
MTS_NAMESPACE_END
