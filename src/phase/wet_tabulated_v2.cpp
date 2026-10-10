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
#include <mitsuba/core/warp.h>
#include <mitsuba/core/plugin.h>
#include <mitsuba/core/fresolver.h>
#include "../bsdfs/ior.h"
#include "../bsdfs/microfacet.h"
#include <mutex>
#include <mitsuba/render/bsdf.h>
#include <mitsuba/render/scene.h>

#define PI_DIVIDE_180 0.0174532922
#define StepWi 5
#define MaxBounce 10//1000
#define SampleCount 10000000//100000000
#define Offset 0.5
#define Res_cosTheta  64

#define COS 1


#define MTS_OPENMP 1
#if defined(MTS_OPENMP)
#include <omp.h>
#endif

MTS_NAMESPACE_BEGIN

/*!\plugin{isotropic}{Isotropic phase function}
 * \order{1}

 * \renderings{
 *     \rendering{Isotropic}{phase_isotropic}
 *     \rendering{Anisotropic micro-flakes}{phase_microflakes_005}
 *     \caption{Heterogeneous volume renderings of a scarf model
 *     with isotropic and anisotropic phase functions.}
 * }
 *
 *
 * This phase function simulates completely uniform scattering,
 * where all directionality is lost after a single scattering
 * interaction. It does not have any parameters.
 */

Float Gaussian(Float x, Float A1, Float mu1, Float sigma1, Float A2, Float mu2, Float sigma2){
	return (A1 / (sqrt(2 * M_PI) * sigma1) * exp(-(x - mu1)*(x - mu1) / (2 * sigma1 * sigma1)) +
		A2 / (sqrt(2 * M_PI) * sigma2) * exp(-(x - mu2)*(x - mu2) / (2 * sigma2 * sigma2)));
}

Float hg(Float x, Float g1, Float weight1, Float g2, Float weight2){
	Float temp = 1.0f + g1*g1 + 2.0f * g1 * x;
	Float term1 = INV_FOURPI * (1 - g1*g1) / (temp * std::sqrt(temp));
	temp = 1.0f + g2*g2 + 2.0f * g2 * x;
	Float term2 = INV_FOURPI * (1 - g2*g2) / (temp * std::sqrt(temp));
	return weight1*term1 + weight2*term2;
}

template <class T>
inline T randUniform() {
	return rand() / (T)RAND_MAX;
}

int costhetaToIndex(double costheta){
	int index;
	if (COS)
		index = static_cast<int>(((costheta + 1) / 2) * Res_cosTheta);
	else {
		float theta = std::acos(costheta);
		index = math::clamp(static_cast<int>(theta * INV_PI * Res_cosTheta), 0, Res_cosTheta);
	}
	return index;
}


 /*椭球类*/
class ProlateSpheroid{
public:
	ProlateSpheroid(){}
	ProlateSpheroid(const Float &a, const Float &c, const Float &roughness, const Float &intIOR_n, const Float &intIOR_k, const Float &extIOR) :
		m_a(a), m_c(c), m_roughness(roughness), m_intIOR_n(intIOR_n), m_intIOR_k(intIOR_k), m_extIOR(extIOR){}
	
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

	Float getTransmittance(const Float &d, const int &k) const{
		Float sigma = 4 * M_PI * m_intIOR_k / k;
		return exp(-d * sigma);
	}

	//private:
	Float m_a;
	Float m_c;
	Float m_roughness;
	Float m_intIOR_n;
	Float m_intIOR_k;
	Float m_extIOR;
};


void precomputePhase2(std::vector<Float> &data, Float intIOR, Float extIOR, Float spherecity, Float roughness, std::string address, std::string surfaceBSDF){
	//std::string address("D:/Pan/porousMedium/scenes/exp_parameterScan/davidhead/results/data");
	//std::string address("D:/Pan/porousMedium/scenes/exp_multipleNet/up/data");
	std::string filename(std::string("/") + "phase_i" + std::to_string(intIOR) + "_e" + std::to_string(extIOR) + "_s" + std::to_string(spherecity) + "_r" + std::to_string(roughness) + (COS ? "_cosTheta" : "_theta") + "_new.txt");
	std::ifstream coe(address + filename);
	/*
	if (coe.is_open())//文件已经存在了 不需要重复预计算
	{
		Float index, tmp;
		while (!coe.eof()) { // 逐行读取文件
			coe >> index >> tmp; // 将每一行数据添加到 vector 中
			data.push_back(tmp);
		}
		return;
	}*/
	//文件不存在 需要预计算
	std::ofstream myfile(address + filename);

	Float out_data[Res_cosTheta + 1] = {};
	Float out_data_2[Res_cosTheta + 1] = {};
	//for (int i = 0; i < spherecityList.size(); i++){
	//setup
	ProlateSpheroid ps;
	ps.m_a = 1.f;// spherecity*spherecity;
	ps.m_c = 1.f;
	ps.m_roughness = roughness;
	ps.m_intIOR_n = intIOR;
	ps.m_intIOR_k = 0;
	ps.m_extIOR = extIOR;
	//Properties props("roughdielectric");
	//Properties props("conductor");
	BSDF* bsdf;
	if (surfaceBSDF == "refractive" || surfaceBSDF == "mirror"){
		Properties props("dielectric");
		props.setFloat("intIOR", intIOR);
		props.setFloat("extIOR", extIOR);
		bsdf = static_cast<BSDF*>(
			PluginManager::getInstance()->createObject(MTS_CLASS(BSDF), props)
			);
	}
	if (surfaceBSDF == "lambert"){
		Properties props("diffuse");
		props.setSpectrum("reflectance", Spectrum(1.f));
		bsdf = static_cast<BSDF*>(
			PluginManager::getInstance()->createObject(MTS_CLASS(BSDF), props)
			);
	}

	Properties sphereProps("sphere");
	sphereProps.setPoint("center", Point(0, 0, 0));
	sphereProps.setFloat("radius", 1);
	Shape* sphere = static_cast<Shape*>(
		PluginManager::getInstance()->createObject(MTS_CLASS(Shape), sphereProps)
		);

	//Scene* scene()

	//RoughDielectric dielectric();
	//std::vector<std::vector<float>> phaseData(tcount, std::vector<float>(Res_cosTheta + 1));
	Float data2[Res_cosTheta + 1] = {};

	//初始化随机数发生器
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
	std::vector<std::vector<Float>> mpdata(tcount, std::vector<Float>(Res_cosTheta, 0.f));
	float outT = -5;
#pragma omp parallel for
	for (int s = 0; s < SampleCount; s++)
	{
#if defined(MTS_OPENMP)
		int tid = mts_omp_get_thread_num();
#else 
		int tid = 0;
#endif
		Float eta = ps.m_intIOR_n / ps.m_extIOR;
		//sample a wi
		Vector wi_ori(0, 0, 1);//warp::squareToUniformSphere(samplers[tid]->next2D());
		Vector wi = wi_ori;
		Float weight = 1;
		//sample a origin
		Point2 diskPoint = ps.m_c*warp::squareToUniformDisk(samplers[tid]->next2D());
		//Vector3 F_origin(diskPoint.x, diskPoint.y, 10.f);
		//Vector3 F_origin(0,0,10);
		Frame F_wi(wi);
		//Point3 origin(F_wi.toWorld(F_origin));
		//Point3 origin(2.f * wi);
		Point3 origin(diskPoint.x, diskPoint.y, 2);
		int bounce = 0;
		Ray ray;
		ray.mint = Epsilon;
		ray.maxt = std::numeric_limits<Float>::infinity();
		ray.setOrigin(origin);
		ray.setDirection(-wi);

		while (bounce <= MaxBounce|| MaxBounce < 0){
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
				if (surfaceBSDF == "mirror")
					bRec.typeMask &= ~BSDF::EBSDFType::EDeltaTransmission;
				Spectrum tmp = bsdf->sample(bRec, samplers[tid]->next2D());
				//if (tmp.isZero())
				//	break;
				if (surfaceBSDF == "mirror")
					tmp = Spectrum(1.f);
				weight *= tmp[0];
				Vector wo = surface.toWorld(bRec.wo);
				//Vector wo = reflect(wi, surface_n);
				//wi = -wo;
				
				ray.setOrigin(intersection);
				ray.setDirection(wo);
			}
			else{
				if (bounce == 0){
					//s--;
					break;
				}
				Vector wo = ray.d;
				Float cosTheta = dot(wi_ori, wo);
				{
					int index = costhetaToIndex(cosTheta);

					if (!COS){
						Float sinTheta = sqrt(1 - cosTheta*cosTheta);
						if (sinTheta < Epsilon)
							weight = 0;
						else 
							weight /= sinTheta;
					}
					//data2[index] += weight;
					mpdata[tid][index] += weight;/// SampleCount;
					//outT = index;
				}

			}
			/*
			if (bounce >= 10){
				if (samplers[tid]->next1D() >= 0.95)
					break;
				else
					weight /= 0.95;
			}*/
			bounce++;
		}
	}


	//output
	float sum = 0;
	for (int u = 0; u < Res_cosTheta; u++){
		for (int i = 0; i < tcount; i++){
			//sum += phaseData[i][u];
			data2[u] += mpdata[i][u];
		}
		sum += data2[u];
	}
	float inv_sum = 1 / sum;
	float total = 0;
	for (int u = 0; u < Res_cosTheta; u++){
		out_data_2[u] = data2[u] *inv_sum;
		total += out_data_2[u];
	}
	std::cout << "total:" << total << endl;
	for (int u = 0; u < Res_cosTheta; u++){
		data.push_back(out_data_2[u]);
		myfile << u << ' ' << out_data_2[u] << endl;
		//myfile << u << ' ' << outT << endl;
	}
	myfile.close();
}

class WetPhaseFunction : public PhaseFunction {
public:
	WetPhaseFunction(const Properties &props)
		: PhaseFunction(props) {
		Float intIOR = props.getFloat("intIOR", 1),
			extIOR = props.getFloat("extIOR", 1),
			extIOR_wet = props.getFloat("extIOR_wet", 1),
			spherecity = props.getFloat("spherecity", 1),
			roughness = props.getFloat("roughness", 1);
		m_mode = props.getString("mode", "tabulation");
		m_address = props.getString("address", "1");
		m_new = props.getBoolean("newTab", true);
		m_surfaceBSDF = props.getString("surfaceBSDF", "refractive");
		if (m_mode == "tabulation"){
			if (!m_new){
				//precomputePhase(table, intIOR, extIOR, spherecity, roughness, m_address);
				//precomputePhase(table2, intIOR, extIOR_wet, spherecity, roughness, m_address);
			}
			else{
				//precomputePhase2(table, intIOR, extIOR, spherecity, roughness, m_address);
				//precomputePhase2(table2, intIOR, extIOR_wet, spherecity, roughness, m_address);
			}
		}
		precomputePhase2(table, intIOR, extIOR, spherecity, roughness, m_address, m_surfaceBSDF);
		precomputePhase2(table2, intIOR, extIOR_wet, spherecity, roughness, m_address, m_surfaceBSDF);

		resCosTheta = table.size();
		m_saturation = props.getFloat("saturation", 0);

		if (m_mode == "GS"){
			weight1_air = props.getFloat("weight1_air", 1);
			weight2_air = props.getFloat("weight2_air", 1);
			mu1_air = props.getFloat("mu1_air", 1);
			mu2_air = props.getFloat("mu2_air", 1);
			sigma1_air = props.getFloat("sigma1_air", 1);
			sigma2_air = props.getFloat("sigma2_air", 1);
			weight1_wet = props.getFloat("weight1_wet", 1);
			weight2_wet = props.getFloat("weight2_wet", 1);
			mu1_wet = props.getFloat("mu1_wet", 1);
			mu2_wet = props.getFloat("mu2_wet", 1);
			sigma1_wet = props.getFloat("sigma1_wet", 1);
			sigma2_wet = props.getFloat("sigma2_wet", 1);
		}
		if (m_mode == "HG"){
			weight1_air = props.getFloat("weight1_air", 1);
			weight2_air = 1 - weight2_air;
			g1_air = props.getFloat("g1_air", 1);
			g2_air = props.getFloat("g2_air", 1);
			weight1_wet = props.getFloat("weight1_wet", 1);
			weight2_wet = 1 - weight2_wet;
			g1_wet = props.getFloat("g1_wet", 1);
			g2_wet = props.getFloat("g2_wet", 1);
		}
		
		/*// test of mask
		Sampler* sampler = static_cast<Sampler*>(
			PluginManager::getInstance()->createObject(MTS_CLASS(Sampler), Properties("independent"))
			);
		Intersection its;
		BSDFSamplingRecord bRec(its, sampler, ERadiance);
		bool hasReflection = ((bRec.component == -1 || bRec.component == 0)
			&& (bRec.typeMask & 0x00008)),
			hasTransmission = ((bRec.component == -1 || bRec.component == 1)
			&& (bRec.typeMask & 0x00010)),
			sampleReflection = hasReflection;
		int i, j;
		if (hasReflection)
			i = 1;
		if (hasTransmission)
			j = 1;
		SLog(EDebug, "The mask is %i, %i", i, j);*/
	}

	WetPhaseFunction(Stream *stream, InstanceManager *manager)
		: PhaseFunction(stream, manager) {
		configure();
	}

	virtual ~WetPhaseFunction() { }

	void configure() {
		PhaseFunction::configure();
		m_type = EAngleDependence;
	}

	void serialize(Stream *stream, InstanceManager *manager) const {
		PhaseFunction::serialize(stream, manager);
	}

	double indexToTheta(Float index) const{
		// 将整数索引映射回[0, 180]范围内的角度
		double theta = index / Res_cosTheta * M_PI;
		return theta;
		// 将角度映射回[-1, 1]范围内的costheta值
		//double costheta = std::cos(theta);
		//return costheta;
	}

	Float indexToCosTheta(int index) const{
		Float cosTheta = (float)index / Res_cosTheta * 2.f - 1.f;
		return cosTheta;
	}

	Float sample(PhaseFunctionSamplingRecord &pRec,
			Sampler *sampler) const {	
		Float pdf;
		return sample(pRec, pdf, sampler);
	}

	Float sample(PhaseFunctionSamplingRecord &pRec,
			Float &pdf, Sampler *sampler) const {
		return sample_wet(pRec, sampler, pdf, m_saturation);

	}

	Float eval(const PhaseFunctionSamplingRecord &pRec) const {
		return eval_wet(pRec, m_saturation);
	}

	Float eval_wet(const PhaseFunctionSamplingRecord &pRec,Float saturation) const{
		if (m_mode == "GS"){
			Float x = std::acos(dot(pRec.wi, pRec.wo));
			Float wet_P = Gaussian(x, weight1_wet, mu1_wet, sigma1_wet, weight2_wet, mu2_wet, sigma2_wet);
			Float air_P = Gaussian(x, weight1_air, mu1_air, sigma1_air, weight2_air, mu2_air, sigma2_air);
			return  wet_P * saturation + air_P * (1 - saturation);
		}
		if (m_mode == "HG"){
			Float x = dot(pRec.wi, pRec.wo);
			Float wet_P = hg(x, g1_wet, weight1_wet, g2_wet, weight2_wet);
			Float air_P = hg(x, g1_air, weight1_air, g2_air, weight2_air);
			return wet_P * saturation + air_P * (1 - saturation);
		}
		Float costheta = dot(pRec.wi, pRec.wo);
		Float theta = acos(costheta);
		int index = costhetaToIndex(costheta);
		if (index < 0 || index >= Res_cosTheta) return 0;
		Float CDF = (1 - saturation) * table[index] + saturation * table2[index];
		Float interval = M_PI / Res_cosTheta;
		Float fp;
		if (COS)
			fp = CDF * Res_cosTheta / (2 * M_PI * 2);//linear in cosTheta
		else
			//fp = CDF / (2 * M_PI*(-cos((index + 1) / 180.f*M_PI) + cos(index / 180.f*M_PI)));//linear in theta
			fp = CDF / (2 * M_PI*interval*sin(theta));
		return fp;
	}

	Float sample_wet(PhaseFunctionSamplingRecord &pRec,
		Sampler *sampler, Float &pdf, Float saturation) const{
		Float rand = randUniform<Float>();
		int index = 0;
		Float CDF = 0;
		while (1){
			CDF = (1 - saturation) * table[index] + saturation * table2[index];
			if (rand < CDF) break;
			rand -= CDF;
			index++;
			if (index < 0 || index >= Res_cosTheta) return 0;
		}
		Float cosTheta;
		Float theta;
		if (COS){
			Float interval = 2.f / Res_cosTheta;
			cosTheta = indexToCosTheta(index) + interval * rand / CDF;
		}
		else{
			Float interval = M_PI / Res_cosTheta;
			theta = indexToTheta(index) + interval * rand / CDF;
			cosTheta = cos(theta);
		}

		Float sinTheta = math::safe_sqrt(1.0f - cosTheta*cosTheta),
			sinPhi, cosPhi;

		math::sincos(2 * M_PI * randUniform<Float>(), &sinPhi, &cosPhi);

		pRec.wo = Frame(pRec.wi).toWorld(Vector(
			sinTheta * cosPhi,
			sinTheta * sinPhi,
			cosTheta
			));
		//pdf = WetPhaseFunction::eval(pRec);
		//return 1.0f;

		//Float interval = M_PI / resCosTheta;
		if (COS)
			pdf = CDF * Res_cosTheta / (2 * M_PI * 2);
		else
			pdf = CDF * Res_cosTheta / (2 * M_PI * M_PI * sin(theta));
		return 1;// eval_wet(pRec, saturation) / pdf;
	}

	std::string toString() const {
		return "IsotropicPhaseFunction[]";
	}

	Float DsDv() const {
		return 1;
	}

	Float sigmaDir(const Vector &orentation, const Vector &wi) const {
		return 4;
	}

	bool needsDirectionallyVaryingCoefficients() const { return true; }

	Float m_saturation;

	Float weight1_air, weight2_air;
	Float mu1_air, mu2_air;
	Float sigma1_air, sigma2_air;

	Float weight1_wet, weight2_wet;
	Float mu1_wet, mu2_wet;
	Float sigma1_wet, sigma2_wet;

	Float g1_air, g2_air;
	Float g1_wet, g2_wet;

	Float alpha_air, alpha_wet;

	std::vector<Float> table,table2;
	Float resCosTheta;

	std::string m_address;
	std::string m_mode;
	std::string m_surfaceBSDF;

	bool m_new;

	MTS_DECLARE_CLASS()
};


MTS_IMPLEMENT_CLASS_S(WetPhaseFunction, false, PhaseFunction)
MTS_EXPORT_PLUGIN(WetPhaseFunction, "Isotropic phase function");
MTS_NAMESPACE_END
