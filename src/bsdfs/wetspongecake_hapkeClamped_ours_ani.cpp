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

#include "microfacet.h"
#include <mitsuba/core/fresolver.h>
#include <mitsuba/core/plugin.h>
#include <mitsuba/hw/basicshader.h>
#include <mitsuba/render/bsdf.h>
#include <mitsuba/render/medium.h>
#include <mitsuba/render/phase.h>

#include <mitsuba/core/statistics.h>

#include "ior.h"
#include <iomanip>

// #include "roughBeckmann.h"
#define MTS_OPENMP 1
#if defined(MTS_OPENMP)
#include <omp.h>
#endif
#include <mitsuba/core/warp.h>
#define TINYEXR_IMPLEMENTATION
#include "tinyexr.h"

#define M_PI            3.14159265358979323846f /* pi */
#define INV_M_PI        0.31830988618379067153f /* 1/pi */
#define M_PI_2          1.57079632679489661923f /* pi/2 */
#define SQRT_M_PI       1.77245385090551602729f /* sqrt(pi) */
#define SQRT_2          1.41421356237309504880f /* sqrt(2) */
#define INV_SQRT_M_PI   0.56418958354775628694f /* 1/sqrt(pi) */
#define INV_2_SQRT_M_PI 0.28209479177387814347f /* 0.5/sqrt(pi) */
#define INV_SQRT_2_M_PI 0.3989422804014326779f  /* 1/sqrt(2*pi) */
#define INV_SQRT_2      0.7071067811865475244f  /* 1/sqrt(2) */

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

#define MAX_LAYER_COUNT 5
#define DELTA_BSDF      1
#define WHITEFURNACE    0

#define FRESNEL_MULTIPLE 1
#define NEW_MAPPING      1

#define phaseCount 64
#define bsdfCount 64
#define bsdfspp 8196


bool isValid(Vector2 p) { return p.x * p.x + p.y * p.y < 1.0; }

void wToIndexPhase(Vector3 w, Float& thetaIndex, Float& phiIndex) {
	double r = std::sqrt(w.x*w.x + w.y*w.y + w.z*w.z); // 向量的长度
	Float theta, phi;
	if (r == 0) {
		// 如果向量长度为0，那么角度没有意义，这里可以设置默认值或者抛出异常
		theta = 0;
		phi = 0;
		return;
	}
	Float costheta = w.z / r;
	theta = std::acos(w.z / r); // 计算θ
	phi = std::atan2(w.y, w.x); // 计算φ
	if (phi < 0)
		phi = 2 * M_PI + phi;
	//thetaIndex = theta / M_PI * (phaseCount + 0.0f);
	thetaIndex = (costheta + 1) / 2.f * phaseCount;
	phiIndex = phi / (2 * M_PI) * (phaseCount + 0.0f);
}


class PrecomputedMultipleTableSlice {
public:
    PrecomputedMultipleTableSlice(int res) {
        m_res = res;

#if 1
        for (int k = 0; k < 2; k++) {
            m_data[k].resize(res * res, Color3(0.0f));
        }
#endif
    }
    Vector idx2dir(const Point2i& idx, bool up) const {
        int i = idx.x;
        int j = idx.y;
#if NEW_MAPPING
        float r1    = (i + 0.5) / m_res;
        float r2    = (j + 0.5) / m_res;
        float theta = 2.f * M_PI * r1;
        float z     = r2;
        float r     = sqrtf(1.f - z * z);
        float x, y;
        math::sincos(theta, &y, &x);
        x *= r;
        y *= r;
        return up ? Vector3(x, y, z) : Vector3(x, y, -z);
#else
        Vector2 p2 = Point2((i + 0.5) / m_res, (j + 0.5) / m_res) * 2 - Point2(1);
        if (isValid(p2)) {
            Vector wi = up ? Vector3(p2.x, p2.y, sqrt(1 - p2.x * p2.x - p2.y * p2.y))
                           : Vector3(p2.x, p2.y, -sqrt(1 - p2.x * p2.x - p2.y * p2.y));
            return wi;
        } else
            return Vector(0, 0, 0);
#endif
    }
    Vector idx2dirSpherical(const Point2i& idx, bool up, const int resulution) const {
        int     i  = idx.x;
        int     j  = idx.y;
        Vector2 p2 = Point2((i + 0.5) / resulution, (j + 0.5) / resulution) * 2 - Point2(1);
        if (isValid(p2)) {
            Vector wi = up ? Vector3(p2.x, p2.y, sqrt(1 - p2.x * p2.x - p2.y * p2.y))
                           : Vector3(p2.x, p2.y, -sqrt(1 - p2.x * p2.x - p2.y * p2.y));
            return wi;
        } else
            return Vector(0, 0, 0);
    }
    int dir2idx(const Vector& wi, int idx[], float weight[]) const {
#if NEW_MAPPING

        float phi = std::atan2(wi.y, wi.x); // +M_PI;
        if (phi < 0)
            phi += 2 * M_PI;

        float r1   = phi / (2.f * M_PI);
        float r2   = abs(wi.z);
        float iOff = r1 * m_res;
        float jOff = r2 * m_res;
        int   i_0  = floor(iOff);
        int   j_0  = floor(jOff);
        if (i_0 >= m_res || i_0 < 0 || j_0 >= m_res || j_0 < 0)
            return -1;

        int i_1 = iOff - i_0 > 0.5 ? std::min(m_res - 1, i_0 + 1) : std::max(0, i_0 - 1);
        int j_1 = jOff - j_0 > 0.5 ? std::min(m_res - 1, j_0 + 1) : std::max(0, j_0 - 1);

        float weightX = 1 - abs(iOff - i_0 - 0.5);
        float weightY = 1 - abs(jOff - j_0 - 0.5);

        idx[0] = i_0 * m_res + j_0;
        idx[1] = i_0 * m_res + j_1;
        idx[2] = i_1 * m_res + j_1;
        idx[3] = i_1 * m_res + j_0;

        weight[0] = weightX * weightY;
        weight[1] = weightX * (1 - weightY);
        weight[2] = (1 - weightX) * (1 - weightY);
        ;
        weight[3] = (1 - weightX) * weightY;
        return 1;
#else
        Vector3 pos = (wi + Vector(1.0f)) * 0.5 * m_res - Vector(0.5);
        int     iWo = pos.x;
        int     jWo = pos.y;
        if (iWo >= m_res || iWo < 0 || jWo >= m_res || jWo < 0)
            return -1;

        int index = int(pos.x) * m_res + int(pos.y);
        return index;
#endif
    }


    int dir2idx(const Vector& wi) const {
#if NEW_MAPPING

        float phi = std::atan2(wi.y, wi.x); // +M_PI;
        if (phi < 0)
            phi += 2 * M_PI;

        float r1 = phi / (2.f * M_PI);
        float r2 = abs(wi.z);
        int   i  = floor(r1 * m_res);
        int   j  = floor(r2 * m_res);
        if (i >= m_res || i < 0 || j >= m_res || j < 0)
            return -1;
        int index = i * m_res + j;
        return index;
#else
        Vector3 pos = (wi + Vector(1.0f)) * 0.5 * m_res - Vector(0.5);
        int     iWo = pos.x;
        int     jWo = pos.y;
        if (iWo >= m_res || iWo < 0 || jWo >= m_res || jWo < 0)
            return -1;

        int index = int(pos.x) * m_res + int(pos.y);
        return index;
#endif
    }
    float getKernelSize(const Vector& wo) {
#if NEW_MAPPING
        // float step1 = (2.f * M_PI) / m_res;
        // float step2 = M_PI * 0.5 * abs(sin(wo.z)) / m_res;

        // return step1 * step2;
        return (2.f * M_PI) / (m_res * m_res);

#else
        return 4.0f / (float)m_res * m_res;
#endif
    }
    void put(const Vector& wo, const Color3& value) {
        int j = dir2idx(wo);
        if (j != -1) {
            int idx         = (wo.z < 0 ? 1 : 0);
            m_data[idx][j] += value;
        }
    }

    void set(const Vector& wo, const Color3& value) {
        int j = dir2idx(wo);
        if (j != -1) {
            int idx        = (wo.z < 0 ? 1 : 0);
            m_data[idx][j] = value;
        }
    }
    int getRes() { return m_res; }

    void writeTable(std::string path) {

        const int totalCount = m_res * m_res * 3;
        float*    data       = new float[totalCount];
        memset(data, 0.0f, totalCount);

        FILE* fp = fopen(path.c_str(), "wb");
        fwrite(&m_res, sizeof(float), 1, fp);

        for (int k = 0; k < 2; k++) {
            for (int i = 0; i < m_res * m_res; i++) {
                int index       = (i) * 3;
                data[index]     = m_data[k][i][0];
                data[index + 1] = m_data[k][i][1];
                data[index + 2] = m_data[k][i][2];
            }
            fwrite(&data[0], sizeof(float), totalCount, fp);
        }
        fclose(fp);
        cout << "Done output. \n";
        free(data);
    }
    void readTable(std::string path) {
        cout << "Find the file. \n";
        const int totalCount = m_res * m_res * 3;

        float* data = new float[totalCount];
        memset(data, 0.0f, totalCount);
        FILE* fp = fopen(path.c_str(), "rb");
        int   res;
        fread(&res, sizeof(int), 1, fp);
        SLog(EDebug, "The resolution of the table is %d", res);

        for (int k = 0; k < 2; k++) {
            fread(&data[0], sizeof(float), totalCount, fp);

            for (int i = 0; i < m_res * m_res; i++) {
                m_data[k][i] = Color3(data[i * 3], data[i * 3 + 1], data[i * 3 + 2]);
            }
        }
        fclose(fp);
        free(data);
    }


    Color3 readRadiance(const Vector& wo) const {
        int      index[4];
        float    weight[4];
        int      i = dir2idx(wo, index, weight);
        Spectrum res(0.0f);
        int      idx = (wo.z < 0 ? 1 : 0);
        if (i != -1) {
            for (int k = 0; k < 4; k++) {
                res += weight[k] * m_data[idx][index[k]];
            }
        }
        return res;
    }

    // void visTable(const Vector &wi, std::string output, bool up)
    //{
    //	int ires = dir2idx(wi);
    //	if (ires == -1) return;

    //	std::vector<float> image;
    //	int totalCount = m_res * m_res;
    //	image.resize(totalCount * 3);

    //	for (int i = 0; i <totalCount; i++)
    //	{

    //		std::vector<Color3> &data = up ? (wi.z > 0 ? m_data[0][ires] : m_data[2][ires]) :
    //			(wi.z > 0 ? m_data[1][ires] : m_data[3][ires]);
    //		image[i * 3] = data[i][0];
    //		image[i * 3 + 1] = data[i][1];
    //		image[i * 3 + 2] = data[i][2];
    //	}

    //	SaveEXRChannel3(image, m_res, m_res, output.c_str());
    //	cout << "Rendering to Image done!" << endl;
    //	std::vector<float>().swap(image);
    //}

    void visTableWithSphere(std::string output, bool up, const int resolution) {
        std::vector<float> image;
        int                totalCount = resolution * resolution;
        image.resize(totalCount * 3);

        for (int i = 0; i < resolution; i++) {
            for (int j = 0; j < resolution; j++) {
                Vector wo                           = idx2dirSpherical(Point2i(i, j), up, resolution);
                Color3 value                        = readRadiance(wo);
                image[(i * resolution + j) * 3]     = value[0];
                image[(i * resolution + j) * 3 + 1] = value[1];
                image[(i * resolution + j) * 3 + 2] = value[2];
            }
        }

        SaveEXRChannel3(image, resolution, resolution, output.c_str());
        cout << "Rendering to Image done!" << endl;
        std::vector<float>().swap(image);
    }


private:
    std::vector<Color3> m_data[2]; // up up
    int                 m_res;
};


class PrecomputedMultipleTable {
public:
    PrecomputedMultipleTable(int res) {
        m_res          = res;
        const int res2 = res * res;

#if 1
        for (int k = 0; k < 4; k++) {
            m_data[k].resize(res2);
            for (int i = 0; i < res2; i++) {
                m_data[k][i].resize(res2, Color3(0.0f));
            }
        }
#endif
    }
    Vector idx2dir(const Point2i& idx, bool up) const {
        int i = idx.x;
        int j = idx.y;
#if NEW_MAPPING
        float r1    = (i + 0.5) / m_res;
        float r2    = (j + 0.5) / m_res;
        float theta = 2.f * M_PI * r1;
        float z     = r2;
        float r     = sqrtf(1.f - z * z);
        float x, y;
        math::sincos(theta, &y, &x);
        x *= r;
        y *= r;
        return up ? Vector3(x, y, z) : Vector3(x, y, -z);
#else
        Vector2 p2 = Point2((i + 0.5) / m_res, (j + 0.5) / m_res) * 2 - Point2(1);
        if (isValid(p2)) {
            Vector wi = up ? Vector3(p2.x, p2.y, sqrt(1 - p2.x * p2.x - p2.y * p2.y))
                           : Vector3(p2.x, p2.y, -sqrt(1 - p2.x * p2.x - p2.y * p2.y));
            return wi;
        } else
            return Vector(0, 0, 0);
#endif
    }

    Vector idx2dir(int idx_, bool up) const {
        Point2i idx = Point2i(idx_ / m_res, idx_ % m_res);
        int     i   = idx.x;
        int     j   = idx.y;
#if NEW_MAPPING
        float r1    = (i + 0.5) / m_res;
        float r2    = (j + 0.5) / m_res;
        float theta = 2.f * M_PI * r1;
        float z     = r2;
        float r     = sqrtf(1.f - z * z);
        float x, y;
        math::sincos(theta, &y, &x);
        x *= r;
        y *= r;
        return up ? Vector3(x, y, z) : Vector3(x, y, -z);
#else
        Vector2 p2 = Point2((i + 0.5) / m_res, (j + 0.5) / m_res) * 2 - Point2(1);
        if (isValid(p2)) {
            Vector wi = up ? Vector3(p2.x, p2.y, sqrt(1 - p2.x * p2.x - p2.y * p2.y))
                           : Vector3(p2.x, p2.y, -sqrt(1 - p2.x * p2.x - p2.y * p2.y));
            return wi;
        } else
            return Vector(0, 0, 0);
#endif
    }


    Vector idx2dirSpherical(const Point2i& idx, bool up, const int resulution) const {
        int     i  = idx.x;
        int     j  = idx.y;
        Vector2 p2 = Point2((i + 0.5) / resulution, (j + 0.5) / resulution) * 2 - Point2(1);
        if (isValid(p2)) {
            Vector wi = up ? Vector3(p2.x, p2.y, sqrt(1 - p2.x * p2.x - p2.y * p2.y))
                           : Vector3(p2.x, p2.y, -sqrt(1 - p2.x * p2.x - p2.y * p2.y));
            return wi;
        } else
            return Vector(0, 0, 0);
    }
    int dir2idx(const Vector& wi, int idx[], float weight[]) const {
#if NEW_MAPPING

        float phi = std::atan2(wi.y, wi.x); // +M_PI;
        if (phi < 0)
            phi += 2 * M_PI;

        float r1   = phi / (2.f * M_PI);
        float r2   = abs(wi.z);
        float iOff = r1 * m_res;
        float jOff = r2 * m_res;
        int   i_0  = floor(iOff);
        int   j_0  = floor(jOff);
        if (i_0 >= m_res || i_0 < 0 || j_0 >= m_res || j_0 < 0)
            return -1;

        int i_1 = iOff - i_0 > 0.5 ? std::min(m_res - 1, i_0 + 1) : std::max(0, i_0 - 1);
        int j_1 = jOff - j_0 > 0.5 ? std::min(m_res - 1, j_0 + 1) : std::max(0, j_0 - 1);

        float weightX = 1 - abs(iOff - i_0 - 0.5);
        float weightY = 1 - abs(jOff - j_0 - 0.5);

        idx[0] = i_0 * m_res + j_0;
        idx[1] = i_0 * m_res + j_1;
        idx[2] = i_1 * m_res + j_1;
        idx[3] = i_1 * m_res + j_0;

        weight[0] = weightX * weightY;
        weight[1] = weightX * (1 - weightY);
        weight[2] = (1 - weightX) * (1 - weightY);
        ;
        weight[3] = (1 - weightX) * weightY;
        return 1;
#else
        Vector3 pos = (wi + Vector(1.0f)) * 0.5 * m_res - Vector(0.5);
        int     iWo = pos.x;
        int     jWo = pos.y;
        if (iWo >= m_res || iWo < 0 || jWo >= m_res || jWo < 0)
            return -1;

        int index = int(pos.x) * m_res + int(pos.y);
        return index;
#endif
    }


    int dir2idx(const Vector& wi) const {
#if NEW_MAPPING

        float phi = std::atan2(wi.y, wi.x); // +M_PI;
        if (phi < 0)
            phi += 2 * M_PI;

        float r1 = phi / (2.f * M_PI);
        float r2 = abs(wi.z);
        int   i  = floor(r1 * m_res);
        int   j  = floor(r2 * m_res);
        if (i >= m_res || i < 0 || j >= m_res || j < 0)
            return -1;
        int index = i * m_res + j;
        return index;
#else
        Vector3 pos = (wi + Vector(1.0f)) * 0.5 * m_res - Vector(0.5);
        int     iWo = pos.x;
        int     jWo = pos.y;
        if (iWo >= m_res || iWo < 0 || jWo >= m_res || jWo < 0)
            return -1;

        int index = int(pos.x) * m_res + int(pos.y);
        return index;
#endif
    }
    float getKernelSize(const Vector& wo) {
#if NEW_MAPPING
        // float step1 = (2.f * M_PI) / m_res;
        // float step2 = M_PI * 0.5 * abs(sin(wo.z)) / m_res;

        // return step1 * step2;
        return (2.f * M_PI) / (m_res * m_res);

#else
        return 4.0f / (float)m_res * m_res;
#endif
    }
    void put(const Vector& wi, const Vector& wo, const Color3& value) {
        int i = dir2idx(wi);
        int j = dir2idx(wo);
        if (i != -1 && j != -1) {
            int idx            = (wi.z < 0 ? 2 : 0) + (wo.z < 0 ? 1 : 0);
            m_data[idx][i][j] += value;
        }
    }

    void set(const Vector& wi, const Vector& wo, const Color3& value) {
        int i = dir2idx(wi);
        int j = dir2idx(wo);
        if (i != -1 && j != -1) {
            int idx           = (wi.z < 0 ? 2 : 0) + (wo.z < 0 ? 1 : 0);
            m_data[idx][i][j] = value;
        }
    }
    int getRes() { return m_res; }

    void writeTable(std::string path) {
#if 0
		std::ofstream myfile(path);

		for (int i = 0; i < m_res * m_res; i++)
		{
			for (int j = 0; j < m_res * m_res; j++)
			{
				myfile << m_data[i][j][0] << "\n";
				myfile << m_data[i][j][1] << "\n";
				myfile << m_data[i][j][2] << "\n";
			}
		}
		myfile.close();
#else
        const int totalCount = m_res * m_res * m_res * m_res * 3;
        float*    data       = new float[totalCount];
        memset(data, 0.0f, totalCount);

        FILE* fp = fopen(path.c_str(), "wb");
        fwrite(&m_res, sizeof(float), 1, fp);

        for (int k = 0; k < 2; k++) {
            for (int i = 0; i < m_res * m_res; i++) {
                for (int j = 0; j < m_res * m_res; j++) {
                    int index       = (i * m_res * m_res + j) * 3;
                    data[index]     = m_data[k][i][j][0];
                    data[index + 1] = m_data[k][i][j][1];
                    data[index + 2] = m_data[k][i][j][2];
                }
            }
            fwrite(&data[0], sizeof(float), totalCount, fp);
        }
        fclose(fp);
        cout << "Done output. \n";
        free(data);

#endif
    }
    void readTable(std::string path) {
        cout << "Find the file. \n";
        const int totalCount = m_res * m_res * m_res * m_res * 3;

        float* data = new float[totalCount];
        memset(data, 0.0f, totalCount);
        FILE* fp = fopen(path.c_str(), "rb");
        int   res;
        fread(&res, sizeof(int), 1, fp);
        SLog(EDebug, "The resolution of the table is %d", res);

        for (int k = 0; k < 2; k++) {
            fread(&data[0], sizeof(float), totalCount, fp);

            for (int i = 0; i < m_res * m_res; i++) {
                for (int j = 0; j < m_res * m_res; j++) {
                    int index       = i * m_res * m_res + j;
                    m_data[k][i][j] = Color3(data[index * 3], data[index * 3 + 1], data[index * 3 + 2]);
                }
            }
        }
        fclose(fp);
        free(data);
    }


    Color3 readRadiance(const Vector3& wi, const Vector& wo) const {
        if (wi.z <= 0)
            return Color3(0.0f);
#if 1
#if 0
		int i = dir2idx(wi);
		int j = dir2idx(wo);

		if (i == -1 || j == -1)
		{
			return Spectrum(0.0f);
		}
		int idx = (wi.z < 0 ? 2 : 0) + (wo.z < 0 ? 1 : 0);

		return m_data[idx][i][j];

#else
        int      index[4];
        float    weight[4];
        int      i = dir2idx(wi, index, weight);
        int      j = dir2idx(wo);
        Spectrum res(0.0f);
        int      idx = (wi.z < 0 ? 2 : 0) + (wo.z < 0 ? 1 : 0);
        if (i != -1 && j != -1) {
            for (int k = 0; k < 4; k++) {
                res += weight[k] * m_data[idx][index[k]][j];
            }
        }
        return res;

#endif

#else
        Vector3 pos   = (wo + Vector(1.0f)) * 0.5 * m_res - Vector(0.5); // wo * m_res - Vector(0.5);
        int     iWo_0 = floor(pos.x);
        int     jWo_0 = floor(pos.y);

        int   iWo_1   = pos.x - iWo_0 > 0.5 ? std::min(m_res - 1, iWo_0 + 1) : std::max(0, iWo_0 - 1);
        int   jWo_1   = pos.y - jWo_0 > 0.5 ? std::min(m_res - 1, jWo_0 + 1) : std::max(0, jWo_0 - 1);
        float weight1 = abs(pos.x - iWo_0 - 0.5);
        float weight2 = abs(pos.y - jWo_0 - 0.5);

        float weight[] = {
            (1 - weight1) * (1 - weight2), (1 - weight1) * weight2, weight1 * weight2, weight1 * (1 - weight2)
        };

        Point2i posWo[] = {Point2i(iWo_0, jWo_0), Point2i(iWo_0, jWo_1), Point2i(iWo_1, jWo_1), Point2i(iWo_1, jWo_0)};

        Color3 result(0.0f);
        for (int i = 0; i < 4; i++) {
            Point2i woPos = posWo[i];
            if (woPos.x >= m_res || woPos.x < 0 || woPos.y >= m_res || woPos.y < 0)
                continue;

            Vector3 ipos  = (wi + Vector(1.0f)) * 0.5 * m_res - Vector(0.5);
            int     iWi_0 = floor(ipos.x);
            int     jWi_0 = floor(ipos.y);

#if 1
            int iWi_1 = ipos.x - iWi_0 > 0.5 ? std::min(m_res - 1, iWi_0 + 1) : std::max(0, iWi_0 - 1);
            int jWi_1 = ipos.y - jWi_0 > 0.5 ? std::min(m_res - 1, jWi_0 + 1) : std::max(0, jWi_0 - 1);

            float weight1i = abs(ipos.x - iWi_0 - 0.5);
            float weight2i = abs(ipos.y - jWi_0 - 0.5);

            float weighti[] = {
                (1 - weight1i) * (1 - weight2i),
                (1 - weight1i) * weight2i,
                weight1i * weight2i,
                weight1i * (1 - weight2i)
            };

            Point2i posWi[] = {
                Point2i(iWi_0, jWi_0), Point2i(iWi_0, jWi_1), Point2i(iWi_1, jWi_1), Point2i(iWi_1, jWi_0)
            };
            Spectrum resLocal(0.0);
            for (int j = 0; j < 4; j++) {
                Point2i wiPos = posWi[j];
                if (wiPos.x >= m_res || wiPos.x < 0 || wiPos.y >= m_res || wiPos.y < 0) {
                    continue;
                }
                resLocal += weighti[j] * m_data[wiPos.x * m_res + wiPos.y][woPos.x * m_res + woPos.y];
            }


            result += resLocal * weight[i];
#else
            Point2i wiPos = Point2i(iWi_0, jWi_0);
            if (wiPos.x >= m_res || wiPos.x < 0 || wiPos.y >= m_res || wiPos.y < 0) {
                continue;
            }
            result += weight[i] * m_data[iWi_0 * m_res + jWi_0][woPos.x * m_res + woPos.y];
#endif
        }
        return result;
#endif
    }

    void visTable(const Vector& wi, std::string output, bool up) {
        int ires = dir2idx(wi);
        if (ires == -1)
            return;

        std::vector<float> image;
        int                totalCount = m_res * m_res;
        image.resize(totalCount * 3);

        for (int i = 0; i < totalCount; i++) {

            std::vector<Color3>& data =
                up ? (wi.z > 0 ? m_data[0][ires] : m_data[2][ires]) : (wi.z > 0 ? m_data[1][ires] : m_data[3][ires]);
            image[i * 3]     = data[i][0];
            image[i * 3 + 1] = data[i][1];
            image[i * 3 + 2] = data[i][2];
        }

        SaveEXRChannel3(image, m_res, m_res, output.c_str());
        cout << "Rendering to Image done!" << endl;
        std::vector<float>().swap(image);
    }

    void visTableWithSphere(const Vector& wi, std::string output, bool up, const int resolution) {
        int ires = dir2idx(wi);
        if (ires == -1)
            return;

        std::vector<float> image;
        int                totalCount = resolution * resolution;
        image.resize(totalCount * 3);

        for (int i = 0; i < resolution; i++) {
            for (int j = 0; j < resolution; j++) {
                Vector wo                           = idx2dirSpherical(Point2i(i, j), up, resolution);
                Color3 value                        = readRadiance(wi, wo);
                image[(i * resolution + j) * 3]     = value[0];
                image[(i * resolution + j) * 3 + 1] = value[1];
                image[(i * resolution + j) * 3 + 2] = value[2];
            }
        }

        SaveEXRChannel3(image, resolution, resolution, output.c_str());
        cout << "Rendering to Image done!" << endl;
        std::vector<float>().swap(image);
    }

    void visAllData(std::string output, bool up) {

        std::vector<float> image;
        int                totalCount = m_res * m_res * m_res * m_res;
        image.resize(totalCount * 3);

        for (int i = 0; i < m_res * m_res; i++) {
            for (int j = 0; j < m_res * m_res; j++) {
                int idx = i * m_res * m_res + j;

                int    k           = up ? 0 : 1;
                Color3 value       = m_data[k][i][j];
                image[idx * 3]     = value[0];
                image[idx * 3 + 1] = value[1];
                image[idx * 3 + 2] = value[2];
            }
        }

        SaveEXRChannel3(image, m_res * m_res, m_res * m_res, output.c_str());
        cout << "Rendering to Image done!" << endl;
        std::vector<float>().swap(image);
    }

private:
    std::vector<std::vector<Color3>> m_data[4]; // up up
    int                              m_res;
};

#define Power powf
#define Dot dot
#define Abs fabsf
#define Sqrt sqrtf
#define Pi M_PI

// Albedo inversion: map spherical/bond albedo kd to single-scattering albedo c [Hitchhiker's guide approximation 3]
float c_from_kd_chandra(const float kd)
{
	return (-1.0893564101764108 - 2.3011443820100275*kd - 0.8247913150511265*Power(kd, 2) +
		0.9234846594011388*(0.39417098289205227 + kd)*Sqrt(0.5667087129883742 + kd)*
		Sqrt(15.803432476716953 + kd)) / Power(0.5461928294878655 + kd, 2);
}

float c_from_kd_lambert(const float kd)
{
	return (1 - Power(1 - kd, 2.73556)) / (1 - 0.184096*Power(1 - kd, 2.48423));
}

class SheenSingleLayer : public BSDF {
public:
    SheenSingleLayer(const Properties& props) : BSDF(props) {
        m_layerCount  = props.getInteger("LayerCount", 2);
        m_depth       = props.getInteger("MaxDepth", 10);
        m_sampleCount = props.getInteger("sampleCount", 100000);


        for (int l = 0; l < m_layerCount; ++l) {
            std::string index = std::to_string(l);
            std::string name;

            // PGL
            name = std::string("wet_") + index;
            m_flag_wet.push_back(props.getBoolean(name, false));
            // std::cout << "[PGL]: layer " << l + 1 << " : " << m_flag_wet[l] << " wet" << std::endl;

            name = std::string("n_") + index;
            m_float_ns.push_back(props.getFloat(name, 1000));
            // std::cout << "[PGL]: layer " << l + 1 << " : " << m_float_ns[l] << " n" << std::endl;

			name = std::string("minSL_") + index;
			m_float_minSL.push_back(props.getFloat(name, 0));

            name = std::string("porosity_") + index;
            m_float_porositys.push_back(props.getFloat(name, 0.8));
            // std::cout << "[PGL]: layer " << l + 1 << " : " << m_float_porositys[l] << " porosity" << std::endl;

            name = std::string("saturation_") + index;
            m_float_saturations.push_back(props.getFloat(name, 0.f));
            // std::cout << "[PGL]: layer " << l + 1 << " : " << m_float_saturations[l] << " saturation" << std::endl;

            name = std::string("sigmaWet_") + index;
            m_spectrum_sigmaWets.push_back(props.getSpectrum(name, Spectrum(0.0)));
            // std::cout << "[PGL]: layer " << l + 1 << " : " << m_spectrum_sigmaWets[l][0] <<
            // m_spectrum_sigmaWets[l][1] << m_spectrum_sigmaWets[l][2] << " sigmaWet" << std::endl;

            name = std::string("aniso_") + index;
            m_flag_aniso.push_back(props.getBoolean(name, false));
            // if (!m_flag_aniso[l]) std::cout << "[GY]: layer " << l + 1 << " of " << m_nbLayers << " :: medium ::
            // Homogeneous Isotropic" << std::endl; else std::cout << "[GY]: layer " << l + 1 << " of " << m_nbLayers <<
            // " :: medium :: Homogeneous Anisotropic" << std::endl;

            name = std::string("sigmaT_") + index;
            m_spectrum_sigmaTs.push_back(props.getSpectrum(name, Spectrum(1.0)));
            // if (!m_flag_aniso[l]) std::cout << "[GY]: layer " << l + 1 << " of " << m_nbLayers << " :: medium ::
            // sigmaT(Spectrum)" << std::endl;

            m_flag_sigmaTs.push_back(false);

            name = std::string("density_") + index;
            m_float_densities.push_back(props.getFloat(name, 1.0));
            // if (m_flag_aniso[l]) std::cout << "[GY]: layer " << l + 1 << " of " << m_nbLayers << " :: medium ::
            // density(Float)" << std::endl;

            m_flag_densities.push_back(false);

			name = std::string("albedo_") + index;
			std::string aMap = props.getString("albedoMap", "false");
			if (aMap=="chandra"){
				Spectrum oriA = props.getSpectrum(name, Spectrum(1.0));
				Float rgb[3];
				rgb[0] = c_from_kd_chandra(oriA[0]);
				rgb[1] = c_from_kd_chandra(oriA[1]);
				rgb[2] = c_from_kd_chandra(oriA[2]);
				m_spectrum_albedos.push_back(Spectrum(rgb));
			}
			else if (aMap == "lambert"){
				Spectrum oriA = props.getSpectrum(name, Spectrum(1.0));
				Float rgb[3];
				rgb[0] = c_from_kd_lambert(oriA[0]);
				rgb[1] = c_from_kd_lambert(oriA[1]);
				rgb[2] = c_from_kd_lambert(oriA[2]);
				m_spectrum_albedos.push_back(Spectrum(rgb));
			}
			else m_spectrum_albedos.push_back(props.getSpectrum(name, Spectrum(1.0)));
            // std::cout << "[GY]: layer " << l + 1 << " of " << m_nbLayers << " :: medium :: albedo(Spectrum)" <<
            // std::endl;

            m_flag_albedos.push_back(false);

            m_flag_thickness.push_back(false);
            // name = std::string("diffuse_") + index;
            // m_spectrum_diffuse.push_back(props.getSpectrum(name, Spectrum(0.0)));

            // m_flag_diffuse.push_back(false);

            name = std::string("orientation_") + index;
            m_vector_orientations.push_back(normalize(props.getVector(name, normalize(Vector(0, 0, 1)))));
            // if (m_flag_aniso[l]) std::cout << "[GY]: layer " << l + 1 << " of " << m_nbLayers << " :: medium ::
            // orientation(Vector)" << std::endl;

            // PGL
            m_flag_saturation.push_back(false);
            m_flag_orientations.push_back(false);
            m_flag_normals.push_back(false);
            name = std::string("F0_") + index;

            // use 0.1 for top ones
            if (l == 0)
                m_F0[l] = props.getSpectrum(name, Spectrum(0.1f));
            else
                m_F0[l] = props.getSpectrum(name, Spectrum(1.0f));

            name           = std::string("thickness_") + index;
            m_thickness[l] = props.getFloat(name, 1.0f);

            name         = std::string("normalScale_") + index;
            m_scale_n[l] = props.getFloat(name, 1.0f);

            name         = std::string("scale_") + index;
            m_scaling[l] = props.getFloat(name, 1.0f);


            name = std::string("preFile_") + index;
            m_prePathLayers.push_back(props.getString(name, "no"));
            //	name = std::string("fiber_") + index;
            //	m_type[l] = props.getBoolean(name, false);

            m_bsdfList.push_back(NULL);

            name = std::string("multipleIgnore_") + index;
            m_flag_multipleIgnore.push_back(props.getBoolean(name, false));
        }

        m_flag_normals.push_back(false);


        m_phaseFunctions.resize(m_layerCount);


        //	m_transmission = props.getBoolean("T", false);


        m_prePath = props.getString("preFile", "no");

        float res = props.getInteger("res", 32);


        float theta_i = props.getFloat("inputTheta", 0.0);
        cout << "The vis angle is " << theta_i << endl;
        float cosTheta2 = cos(theta_i);
        float dPhi      = M_PI / 2;
        float sinTheta2 = sqrt(1 - cosTheta2 * cosTheta2);
        m_wi            = Vector(sinTheta2 * cos(dPhi), sinTheta2 * sin(dPhi), cosTheta2);
        printDir(m_wi);
        m_analytic = props.getBoolean("Analytic", false);
        //	m_forMultiple = props.getBoolean("forMultiple", false);
        //	m_NEE = props.getBoolean("NEE", false);

        m_visUp      = props.getBoolean("visUp", true);
        m_mapping    = props.getBoolean("mapping", false);
        m_useTwoSide = props.getBoolean("twoside", false);
        if (m_mapping) {
            for (int k = 0; k < 4; k++) {

                m_texture_albedos_m[k].resize(m_layerCount);
                m_texture_F0_m[k].resize(m_layerCount);
                m_roughness_m[k].resize(m_layerCount);
                m_w1w2_m[k].resize(m_layerCount);
                m_bsdfList_m[k].resize(m_layerCount);
                for (int i = 0; i < m_layerCount; i++)
                    m_bsdfList_m[k][i] = NULL;
            }
        }
        m_paramName          = props.getString("mappingPath", "no");
        m_paramNameOthers[0] = props.getString("mappingPath1", "no");
        m_paramNameOthers[1] = props.getString("mappingPath2", "no");
        m_paramNameOthers[2] = props.getString("mappingPath3", "no");

        m_texture_sigmaTs.resize(m_layerCount);
        m_texture_densities.resize(m_layerCount);
        m_texture_albedos.resize(m_layerCount);
        m_texture_orientations.resize(m_layerCount);
        m_texture_normals.resize(m_layerCount + 1);
        m_texture_thickness.resize(m_layerCount);
        m_roughness.resize(m_layerCount);
        // PGL
        m_texture_saturation.resize(m_layerCount);

        { m_table = new PrecomputedMultipleTable(res); }
		{ m_phaseTable = new PrecomputedMultipleTable(res); }

        m_thicknessAcc.resize(m_layerCount + 1);
        m_thicknessAcc[0] = 0;
        for (int i = 1; i < m_layerCount + 1; i++) {
            m_thicknessAcc[i] = m_thicknessAcc[i - 1] + m_thickness[i - 1];
        }

        m_reflectionOnly = props.getBoolean("reflectionOnly", true);
        // this is the test of the thickness
        //	float depth = props.getFloat("queryLayer", 0.0);
        //	int layer = locateLayer(depth);
        //	SLog(EDebug, "The Layer is %i.", layer);
        m_table_slice = new PrecomputedMultipleTableSlice(512);

        // PGL whitefurnace
        m_deltaOnly = props.getBoolean("deltaOnly", false);
        // PGL wetSurface
        m_wetSurface = props.getBoolean("wetSurface", false);
        m_IOR        = props.getFloat("wetIOR", 1);
		if (m_wetSurface){
			Properties props("dielectric");
			props.setFloat("intIOR", m_IOR);
			props.setFloat("extIOR", 1.0);
			wetSurface = static_cast<BSDF*>(
				PluginManager::getInstance()->createObject(MTS_CLASS(BSDF), props)
				);
		}

        // PGL multiple approx
        m_multipleApprox = props.getBoolean("multipleApprox", false);
		m_multipleApproxHapke = props.getBoolean("multipleApproxHapke", false);

		//multipleNet up
		m_net_porosity = props.getFloat("porosityNet", 1);
		m_net_n = props.getFloat("nNet", 1);
		m_net_sat = props.getFloat("satNet", 1);
		m_net_albedo = props.getSpectrum("albedoNet", Spectrum(0.1f));
		m_net_sigmaWet = props.getSpectrum("sigmaWetNet", Spectrum(0.1f));
		m_net_sigmaT = props.getSpectrum("sigmaTNet", Spectrum(0.1f));
		m_net_saturation = props.getFloat("saturationNet", 1);
		m_net_thickness = props.getFloat("thicknessNet", 1);
		m_net_w1 = props.getFloat("w1Net", 1);
		m_net_w2 = props.getFloat("w2Net", 1);
		m_net_albedo2 = props.getSpectrum("albedo2Net", Spectrum(0.1f));
		// GS
		m_net_wg1 = props.getFloat("wg1Net", 1);
		m_net_wg2 = props.getFloat("wg2Net", 1);
		m_net_mu1 = props.getFloat("mu1Net", 1);
		m_net_mu2 = props.getFloat("mu2Net", 1);
		m_net_sigma1 = props.getFloat("sigma1Net", 1);
		m_net_sigma2 = props.getFloat("sigma2Net", 1);

		//multipleNet down
		m_net_porosity_down = props.getFloat("porosityNet_down", 1);
		m_net_sigmaWet_down = props.getSpectrum("sigmaWetNet_down", Spectrum(0.1f));
		m_net_saturation_down = props.getFloat("saturationNet_down", 1);
		m_net_sat_down = props.getFloat("satNet_down", 1);

		m_net_albedo_down = props.getSpectrum("albedoNet_down", Spectrum(0.1f));
		m_net_sigmaT_down = props.getSpectrum("sigmaTNet_down", Spectrum(0.1f));
		m_net_thickness_down = props.getFloat("thicknessNet_down", 1);
		m_net_w1_down = props.getFloat("w1Net_down", 1);
		m_net_w2_down = props.getFloat("w2Net_down", 1);
		m_net_albedo2_down = props.getSpectrum("albedo2Net_down", Spectrum(0.1f));
		// GS
		m_net_wg1_down = props.getFloat("wg1Net_down", 1);
		m_net_wg2_down = props.getFloat("wg2Net_down", 1);
		m_net_mu1_down = props.getFloat("mu1Net_down", 1);
		m_net_mu2_down = props.getFloat("mu2Net_down", 1);
		m_net_sigma1_down = props.getFloat("sigma1Net_down", 1);
		m_net_sigma2_down = props.getFloat("sigma2Net_down", 1);

		// Ani
		m_net_density = props.getFloat("densityNet", 1);
		m_net_roughness = props.getFloat("roughnessNet", 1);
		m_net_roughness = props.getFloat("roughnessNet_down", 1);
		// HG
		//m_net_whg = props.getFloat("whgNet", 1);
		//m_net_g1 = props.getFloat("g1Net", 1);
		//m_net_g2 = props.getFloat("g2Net", 1);
		m_multipleNet = props.getBoolean("multipleNet", false);
		m_multipleIso = props.getBoolean("multipleIso", false); 
		m_multipeTex = props.getBoolean("multipleTex", true);
		m_thetaI = props.getFloat("thetaI", 45.f);

		cout << "read start" << endl;
		//if (m_multipleNet && !m_flag_aniso[0]){
			m_albedo_m.resize(2);
			m_sigmaWet_m.resize(2);
			m_SP_m.resize(2);
			m_TW_m.resize(2);
			m_albedo2_m.resize(2);
			m_lobe1_m.resize(2);
			m_lobe2_m.resize(2);
		//}
		if (m_multipleNet && m_flag_aniso[0]){
			std::string phaseAddress = props.getString("phaseAddressNet", "1");
			std::ifstream coe(phaseAddress, std::ios::binary);
			if (coe.is_open())//文件已经存在
			{
				// 读取数据
				coe.read(reinterpret_cast<char*>(m_multiplePhaseTable), (phaseCount+1) * phaseCount * phaseCount * sizeof(float));
				// 关闭文件
				coe.close();
				cout << "read success" << endl;
			}
			else
				cout << "read fail" << endl;
				//SLog(EDebug, "read fail");
			phaseAddress = props.getString("phaseAddressNet_down", "1");
			std::ifstream coe2(phaseAddress, std::ios::binary);
			if (coe2.is_open())//文件已经存在
			{
				// 读取数据
				coe2.read(reinterpret_cast<char*>(m_multiplePhaseTable_down), (phaseCount + 1) * phaseCount * phaseCount * sizeof(float));
				// 关闭文件
				coe2.close();
				cout << "read success" << endl;
			}
			else
				cout << "read fail" << endl;
		}
		

		m_bsdfdata = props.getBoolean("bsdfdata", false);
		m_visdata = props.getBoolean("visdata", false);
		m_bsdftable = props.getBoolean("bsdftable", false);
		m_bsdftableName = props.getString("bsdftableName", "bsdftable");

        m_semiinf = props.getBoolean("semiinf", false);

    }

    SheenSingleLayer(Stream* stream, InstanceManager* manager) : BSDF(stream, manager) { configure(); }

    void serialize(Stream* stream, InstanceManager* manager) const { BSDF::serialize(stream, manager); }

    void configure() {
		if (m_flag_wet[0])	m_DsDv = m_phaseFunctions[0]->DsDv();
		SLog(EDebug, "The bsdf Ds/Dv is %f", m_DsDv);

		Float porosity = m_float_porositys[0];
		Float EL, K;
		if (porosity == 1)
			K = 1;
		else{
			Float EL = 1.209 * pow((1 - porosity)*m_DsDv, 2.0 / 3.0);
			K = -log(1 - EL) / EL;
		}
		SLog(EDebug, "The bsdf K is %f", K);


        unsigned int extraFlags = 0;

        m_components.clear();
        // this is the window scene, we need this for two side sampling.
        //	m_components.push_back(EGlossyReflection  | EFrontSide | extraFlags);
        m_components.push_back(EGlossyReflection | EGlossyTransmission | EFrontSide | EBackSide | extraFlags);
        //	m_components.push_back(ENull | EFrontSide | EBackSide);

        m_usesRayDifferentials = false;

        // Medium
        m_mediums.resize(m_layerCount);
        //	m_pdfMediums.resize(NUM_COUNTERS, ref_vector<Medium>(m_layerCount - 1));

        Properties interiorIsoHomoProps("homogeneous");
        interiorIsoHomoProps.setSpectrum("sigmaT", Spectrum(0.1));
        interiorIsoHomoProps.setSpectrum("albedo", Spectrum(1.0));

        Properties interiorAniHomoProps("homogeneous_aniso");
        interiorAniHomoProps.setFloat("density", 0.0);
        interiorAniHomoProps.setSpectrum("albedo", Spectrum(1.0));
        interiorAniHomoProps.setVector("orientation", Vector(1.0, 0.0, 0.0));

        // PGL
        Properties interiorHapkeProps("hapke_Ours");
        //interiorHapkeProps.setSpectrum("sigmaT", Spectrum(0.1));
        //interiorHapkeProps.setSpectrum("albedo", Spectrum(1.0));
        //interiorHapkeProps.setFloat("porosity", 0.0);

		// Properties interiorAniHomoWetProps("homogeneous_micro_volume_clamped");
        Properties interiorAniHomoWetProps("homogeneous_micro_volume_clamped");
		//interiorAniHomoWetProps.setFloat("density", 0.0);
		//interiorAniHomoWetProps.setSpectrum("albedo", Spectrum(1.0));
		//interiorAniHomoWetProps.setVector("orientation", Vector(1.0, 0.0, 0.0));

		Vector w = m_vector_orientations[0];
		SLog(EDebug, "The orientation is %f, %f, %f", w.x, w.y, w.z);

        for (int l = 0; l < m_layerCount; ++l) {
            if (m_bsdfList[l] == NULL) {
                // hapke
                if (m_flag_wet[l]) {
					if (m_flag_aniso[l]){
						interiorAniHomoWetProps.setFloat("density", m_float_densities[l]);
						interiorAniHomoWetProps.setSpectrum("albedo", m_spectrum_albedos[l]);
						interiorAniHomoWetProps.setVector("orientation", m_vector_orientations[l]);
						interiorAniHomoWetProps.setFloat("saturation", m_float_saturations[l]);
						interiorAniHomoWetProps.setSpectrum("sigmaWet", m_spectrum_sigmaWets[l]);
						interiorAniHomoWetProps.setFloat("porosity", m_float_porositys[l]);

						m_mediums[l] = static_cast<Medium*>(
							PluginManager::getInstance()->createObject(MTS_CLASS(Medium), interiorAniHomoWetProps)
							);
						m_mediums[l]->addChild(m_phaseFunctions[l]);
						m_mediums[l]->configure();
					}
					else{
						interiorHapkeProps.setSpectrum("sigmaT", m_spectrum_sigmaTs[l]);
						//interiorHapkeProps.setSpectrum("sigmaT", sigmaE);
						interiorHapkeProps.setSpectrum("albedo", m_spectrum_albedos[l]);
						interiorHapkeProps.setFloat("porosity", m_float_porositys[l]);
						interiorHapkeProps.setFloat("saturation", m_float_saturations[l]);
						//interiorHapkeProps.setFloat("minSepLen", m_float_minSL[l]);
						//interiorHapkeProps.setFloat("minSepLen", msl);
						interiorHapkeProps.setSpectrum("sigmaWet", m_spectrum_sigmaWets[l]);

						m_mediums[l] = static_cast<Medium*>(
							PluginManager::getInstance()->createObject(MTS_CLASS(Medium), interiorHapkeProps)
							);
						m_mediums[l]->addChild(m_phaseFunctions[l]);
						m_mediums[l]->configure();
					}
                } else {
                    if (m_flag_aniso[l]) {
                        interiorAniHomoProps.setFloat("density", m_float_densities[l]);
                        interiorAniHomoProps.setSpectrum("albedo", m_spectrum_albedos[l]);
                        interiorAniHomoProps.setVector("orientation", m_vector_orientations[l]);

                        m_mediums[l] = static_cast<Medium*>(
                            PluginManager::getInstance()->createObject(MTS_CLASS(Medium), interiorAniHomoProps)
                        );
                        m_mediums[l]->addChild(m_phaseFunctions[l]);
                        m_mediums[l]->configure();

                        /*	m_pdfMediums[i][l] = static_cast<Medium *> (PluginManager::getInstance()->
                        createObject(MTS_CLASS(Medium), interiorAniHomoProps));
                        m_pdfMediums[i][l]->addChild(m_phaseFunctions[l]);
                        m_pdfMediums[i][l]->configure();*/
                    } else {
                        interiorIsoHomoProps.setSpectrum("sigmaT", m_spectrum_sigmaTs[l]);
                        interiorIsoHomoProps.setSpectrum("albedo", m_spectrum_albedos[l]);

                        m_mediums[l] = static_cast<Medium*>(
                            PluginManager::getInstance()->createObject(MTS_CLASS(Medium), interiorIsoHomoProps)
                        );
                        m_mediums[l]->addChild(m_phaseFunctions[l]);
                        m_mediums[l]->configure();

                        /*m_pdfMediums[i][l] = static_cast<Medium *> (PluginManager::getInstance()->
                        createObject(MTS_CLASS(Medium), interiorIsoHomoProps));
                        m_pdfMediums[i][l]->addChild(m_phaseFunctions[l]);
                        m_pdfMediums[i][l]->configure();*/
                    }
                }

            } else
                m_mediums[l] = NULL;
        }

        {
            // Medium
            m_mediums_runtime.resize(NUM_COUNTERS, ref_vector<Medium>(m_layerCount));

            Properties interiorIsoHomoProps("homogeneous");
            interiorIsoHomoProps.setSpectrum("sigmaT", Spectrum(0.0));
            interiorIsoHomoProps.setSpectrum("albedo", Spectrum(1.0));

            Properties interiorAniHomoProps("homogeneous_aniso");
            interiorAniHomoProps.setFloat("density", 0.0);
            interiorAniHomoProps.setSpectrum("albedo", Spectrum(1.0));
            interiorAniHomoProps.setVector("orientation", Vector(1.0, 0.0, 0.0));

            for (size_t i = 0; i < NUM_COUNTERS; ++i) {
                SLog(EDebug, "Configuring medium for counter %zu", i);
                for (int l = 0; l < m_layerCount; ++l) {
                    if (m_bsdfList[l] == NULL) {
                        // hapke
                        if (m_flag_wet[l]) {
							if (m_flag_aniso[l]) {
								m_mediums_runtime[i][l] = static_cast<Medium*>(
									PluginManager::getInstance()->createObject(MTS_CLASS(Medium), interiorAniHomoWetProps)
									);
								m_mediums_runtime[i][l]->addChild(m_phaseFunctions[l]);
								m_mediums_runtime[i][l]->configure();
							}
							else{
								m_mediums_runtime[i][l] = static_cast<Medium*>(
									PluginManager::getInstance()->createObject(MTS_CLASS(Medium), interiorHapkeProps)
									);
								m_mediums_runtime[i][l]->addChild(m_phaseFunctions[l]);
								m_mediums_runtime[i][l]->configure();
							}
                        } else {
                            if (m_flag_aniso[l]) {
                                m_mediums_runtime[i][l] = static_cast<Medium*>(
                                    PluginManager::getInstance()->createObject(MTS_CLASS(Medium), interiorAniHomoProps)
                                );
                                m_mediums_runtime[i][l]->addChild(m_phaseFunctions[l]);
                                m_mediums_runtime[i][l]->configure();

                            } else {
                                m_mediums_runtime[i][l] = static_cast<Medium*>(
                                    PluginManager::getInstance()->createObject(MTS_CLASS(Medium), interiorIsoHomoProps)
                                );
                                m_mediums_runtime[i][l]->addChild(m_phaseFunctions[l]);
                                m_mediums_runtime[i][l]->configure();
                            }
                        }

                    } else {
                        m_mediums_runtime[i][l] = NULL;
                    }
                }
            }
        }

		if (m_bsdftable){
			Float brdftable[bsdfCount][bsdfCount] = {};
			Float btdftable[bsdfCount][bsdfCount] = {};
			//初始化随机数发生器
			Sampler* sampler = static_cast<Sampler*>(
				PluginManager::getInstance()->createObject(MTS_CLASS(Sampler), Properties("independent"))
				);
			int                   tcount = mts_omp_get_max_threads();
			std::vector<Sampler*> samplers(tcount);
			for (size_t i = 0; i < tcount; ++i) {
				ref<Sampler> clonedSampler = sampler->clone();
				clonedSampler->incRef();
				samplers[i] = clonedSampler.get();
			}
			// 固定入射方向：theta = 45°, phi = 0°
			Float theta_i = m_thetaI * M_PI /180;
			//Float theta_i = M_PI / 4.0f;
			Float phi_i = 0.0f;

			Vector3f wi;
			wi.x = std::sin(theta_i) * std::cos(phi_i);  // sin(π/4) * cos(0) = √2/2
			wi.y = std::sin(theta_i) * std::sin(phi_i);  // sin(π/4) * sin(0) = 0
			wi.z = std::cos(theta_i);
#pragma omp parallel for
			for (int i = 0; i < bsdfCount; i++){
#if defined(MTS_OPENMP)
				int tid = mts_omp_get_thread_num();
#else 
				int tid = 0;
#endif
				//Float theta = (i + 0.5f) / bsdfCount * (M_PI * 0.5f);  // 半球
                Float cosTheta = 1.0f - (i + 0.5f) / bsdfCount;   // cosθ 从 1 → 0 均匀划分
                Float theta = std::acos(cosTheta);

				for (int j = 0; j < 2; j++){
					//Float phi = (j + 0.5f) / bsdfCount * (2.0f * M_PI);
                    Float phi=0;
                    if(j==0){
                        phi = 0.0f;
                    }
                    else{
                        phi = M_PI;
                    }
					// 球坐标转方向向量
					Vector3f wo;
					wo.x = std::sin(theta) * std::cos(phi);
					wo.y = std::sin(theta) * std::sin(phi);
					wo.z = std::cos(theta);

					Intersection its;
					its.uv = Point2(0, 0);
					its.wi = wi;
					BSDFSamplingRecord bRec(its, wi, wo, ERadiance), bRec2(its, wi, wo, ERadiance);
					bRec.sampler = samplers[tid];
					bRec.sampledComponent = 0;
					bRec.sampledType = EGlossyReflection;
					bRec.bounces = 0;
					bRec2.wo[2] = -bRec2.wo[2];
					bRec2.sampler = samplers[tid];
					bRec2.sampledComponent = 1;
					bRec2.sampledType = EGlossyTransmission;
					bRec2.bounces = 0;
					for (int k = 0; k < bsdfspp; k++){
						Spectrum result = this->eval(bRec, ESolidAngle); //evalSingle(bRec);// +msSimulationNEE(bRec, 0); //
						brdftable[i][j] += result[0];
						
						result = this->eval(bRec2, ESolidAngle); //evalSingle(bRec);// +msSimulationNEE(bRec, 0); //
						btdftable[i][j] += result[0];
					}
					brdftable[i][j] /= bsdfspp;
					btdftable[i][j] /= bsdfspp;
				}
			}

			//输出
			Float s = m_float_saturations[0], p = m_float_porositys[0];
			std::ostringstream filename, filename2;
			filename << "bsdftable/brdf_" << m_bsdftableName
				<< ".txt";
			filename2 << "bsdftable/btdf_" << m_bsdftableName
				<< ".txt";

			// 打开文件写入
			std::ofstream fout(filename.str());
			std::ofstream fout2(filename2.str());
			for (int i = 0; i < bsdfCount; ++i) {
				for (int j = 0; j < 2; ++j) {
					fout << std::setw(12) << std::setprecision(6) << brdftable[i][j] << " ";
					fout2 << std::setw(12) << std::setprecision(6) << btdftable[i][j] << " ";
				}
				fout << "\n";
				fout2 << "\n";
			}
			fout.close();
			fout2.close();
		}

        ///////////
        // some test
        printDir(m_wi);
        int idx = m_table->dir2idx(m_wi);
        SLog(EDebug, "The idex is %i", idx);
        int i = idx / m_table->getRes();
        int j = idx % m_table->getRes();

        Vector dir = m_table->idx2dir(Point2i(i, j), true);
        printDir(dir);
        SLog(EDebug, "The idex is %i", m_table->dir2idx(dir));

        ////////

        // std::string name = "single."

		std::string        imageName = "multiple";
		std::ostringstream ss;
		ss << std::setprecision(7);
		if (!m_flag_aniso[0]){
			for (int i = 0; i < m_layerCount; i++) {
				if (m_bsdfList[i] == NULL){
					const Properties& phase = m_phaseFunctions[i]->getProperties();
					float             thickness = 0.0;

					ss << "_sw_" << m_spectrum_sigmaWets[i][0] << "," << m_spectrum_sigmaWets[i][1] << ","
						<< m_spectrum_sigmaWets[i][2];
					ss << "_a_" << m_spectrum_albedos[i][0] << "," << m_spectrum_albedos[i][1] << ","
						<< m_spectrum_albedos[i][2];
					ss << "_st_" << m_spectrum_sigmaTs[i][0] << "," << m_spectrum_sigmaTs[i][1] << ","
						<< m_spectrum_sigmaTs[i][2];
					//ss << "_n_" << m_float_ns[i];
					ss << "_t_" << m_thickness[i];
					ss << "_s_" << m_float_saturations[i];
					ss << "_p_" << m_float_porositys[i];
					ss << "_ii_" << phase.getFloat("intIOR", 0.0);
					ss << "_ei_" << phase.getFloat("extIOR_wet", 0.0);
					ss << "_r_" << phase.getFloat("roughness", 0.0);
				}
			}
		}
		else{
			for (int i = 0; i < m_layerCount; i++) {
				if (m_bsdfList[i] == NULL){
					const Properties& phase = m_phaseFunctions[i]->getProperties();
					float             thickness = 0.0;

					ss << "_sw_" << m_spectrum_sigmaWets[i][0] << "," << m_spectrum_sigmaWets[i][1] << ","
						<< m_spectrum_sigmaWets[i][2];
					ss << "_a_" << m_spectrum_albedos[i][0] << "," << m_spectrum_albedos[i][1] << ","
						<< m_spectrum_albedos[i][2];
					//ss << "_st_" << m_spectrum_sigmaTs[i][0] << "," << m_spectrum_sigmaTs[i][1] << ","
					//	<< m_spectrum_sigmaTs[i][2];
					//ss << "_n_" << m_float_ns[i];
					ss << "_r_" << phase.getFloat("roughness", 0.0);
					ss << "_rP_" << phase.getFloat("roughnessP", 0.0);
					ss << "_d_" << m_float_densities[i];
					ss << "_t_" << m_thickness[i];
					ss << "_s_" << m_float_saturations[i];
					ss << "_p_" << m_float_porositys[i];
					ss << "_ii_" << phase.getFloat("intIOR", 0.0);
					ss << "_ei_" << phase.getFloat("extIOR_wet", 0.0);
					bool fiber = phase.getBoolean("fiber", true);
					if (fiber)
						ss << "_type_fiber";
					else
						ss << "_type_surface";
				}
			}
		}

		

        std::string        imageName1 = "multiple_albedo";
        std::ostringstream ss1;
        float              roughness;
        if (m_bsdfList[0] != NULL)
            roughness = 1.0;
        else {
            if (m_flag_aniso[0])
                roughness = m_phaseFunctions[0]->getProperties().getFloat("roughness", 0.1);
            else
                roughness = m_phaseFunctions[0]->getProperties().getFloat("g", 0.0);
        }
        ss1 << m_spectrum_albedos[0][0] << "_roughness_" << roughness;
        ss1 << "_thickness_" << m_thickness[0];
        ss1 << "_theta_" << m_wi.z;
        imageName1.append(ss1.str());
        std::string outputName = imageName1;
        imageName1.append(".exr");

        imageName.append(ss.str());
#if 1
        if (!m_analytic) {
            //SLog(EDebug, "Start precomputation.");

            if (std::strcmp(m_prePath.c_str(), "no") == 0) {
                //computeMutlipleTable(m_prePath, imageName);
                //cout << "Table precomputation done.\n";

            } else {
                SLog(EDebug, "Find the file.");
                // read the data directly.
                //m_table->readTable(m_prePath);
            }

        } //else
           // computeMutlipleTableAnalytical(m_prePath, imageName);

		if (m_bsdfdata){
			std::string prepath = "F:/AniData/";
			SLog(EDebug, "Start precomputation.");
			//Intersection its;
			//BSDFSamplingRecord bRec(its, Vector(0,0,1), ERadiance);
			//setParameters(bRec, m_mediums);
			computeMutlipleTable(m_prePath, imageName);
			//computeMutlipleTablePhase(m_prePath, imageName);
			if (m_visdata){
				m_table->visAllData(prepath + imageName + "_up_gt.exr", true);
				m_table->visAllData(prepath + imageName + "_down_gt.exr", false);
				//m_phaseTable->visAllData(prepath + imageName + "_up_phase.exr", true);
				//m_phaseTable->visAllData(prepath + imageName + "_down_phase.exr", false);
				//m_table = new PrecomputedMultipleTable(32);
				//computeMutlipleTablePhase(m_prePath, imageName);
				//m_table->visAllData(prepath + imageName + "_up_nee.exr", true);
				//m_table->visAllData(prepath + imageName + "_down_nee.exr", false);
				//computeMutlipleTableAnalytical(m_prePath, imageName);
				//m_table->visAllData(prepath + imageName + "_up_gt_analytical.exr", true);
				//m_table->visAllData(prepath + imageName + "_down_gt_analytical.exr", false);
			}
			cout << "Table precomputation done.\n";
		}

		//m_table->visAllData(imageName + "_up_gt.exr", true);
        //m_table->visAllData(outputName + "_down_gt.exr", false);
        Vector wi_temp = m_table->idx2dir(Point2i(0, 0), true);
        printDir(wi_temp);
        wi_temp = m_table->idx2dir(Point2i(0, 16), true);
        printDir(wi_temp);
        wi_temp = m_table->idx2dir(Point2i(0, 31), true);
        printDir(wi_temp);
#endif

        imageName.append(".exr");

        //SLog(EDebug, "Precomputation done.");
        //	computeForWi(m_wi, 1, imageName);


        //	Vector wi = m_table->idx2dir(Point2i(0, 0), true);
        //	computeForWi(m_wi, m_depth, imageName1);
        //	m_table->visTable(m_wi, "multiple.exr", m_visUp);
        SLog(EDebug, "Vis precomputation done.");
        //	computeForWi(m_wi, m_depth, imageName1);

        // m_table_slice->visTableWithSphere("visSphere_up.exr", true, 256);
        // m_table_slice->visTableWithSphere("visSphere_bottom.exr", false, 256);

        BSDF::configure();

#if 0
		PrecomputedMultipleTable *table = new PrecomputedMultipleTable(64);
		// this is a test to get the histogram of the orienation
		const Vector &wi_temp = table->idx2dir(Point2i(0, 0), true);
		for (int i = 0; i < 32; i++)
		{
			for (int j = 0; j < 32; j++)
			{
				Vector temp = getOrientationFromTexture(m_texture_orientations[1], Point2((i + 0.5f) / 32, (j + 0.5f) / 32), m_scale_n[1]);
				//if (temp.z < 0)
				printDir(temp);
				table->put(wi_temp, temp, Spectrum(1.0));
			}

		}
		table->visTable(wi_temp, "histogram.exr", true);
#endif

        ////this is a test
        // Vector3 s, t;
        // Vector orientation(0.714, 0.675, 0.186);
        // buildOrthonormalBasis(s, t, orientation);
        // printDir(s);
        // printDir(t);

#if 1
        if (m_mapping) {
            // read the parameter
            for (int i = 0; i < m_layerCount; i++) {
                std::string imageName = m_paramName;
                imageName.append("_");
                std::ostringstream ss;
                ss << i;
                std::string s(ss.str());
                imageName.append(s);
                loadLayerData(imageName, i, 0);
            }

            if (m_useTwoSide) {
                for (int k = 0; k < 3; k++) {
                    for (int i = 0; i < m_layerCount; i++) {
                        std::string imageName = m_paramNameOthers[k];
                        imageName.append("_");
                        std::ostringstream ss;
                        ss << i;
                        std::string s(ss.str());
                        imageName.append(s);
                        loadLayerData(imageName, i, k + 1);
                    }
                }
            }

        } else {
            const int paramSize = 1024;
            outTextureMap(paramSize);
        }
#endif
    }

    void loadLayerData(std::string paramName, int iLayer, int index) {

        float*      image;
        int         width, height;
        const char* err = NULL;

        int type;
        if (m_bsdfList[iLayer] != NULL)
            type = 3;
        else if (!m_flag_aniso[iLayer])
            type = 2;
        else {
            const Properties& phase = m_phaseFunctions[iLayer]->getProperties();
            type                    = phase.getBoolean("fiber", false) ? 1 : 0;
        }

        bool notFound = false;
        if (type == 3) {
            // this is a diffuse // creat it

            static_cast<BSDF*>(PluginManager::getInstance()->createObject(MTS_CLASS(BSDF), Properties("diffuse")));

            BSDF* bsdf =
                static_cast<BSDF*>(PluginManager::getInstance()->createObject(MTS_CLASS(BSDF), Properties("diffuse")));

            std::string albedoName = paramName + "_albedo.exr";


            int ret = LoadEXR(&image, &width, &height, albedoName.c_str(), &err);
            if (ret != TINYEXR_SUCCESS) {
                m_texture_albedos_m[index][iLayer] = NULL;
                notFound                           = true;
            } else {

                Properties props("bitmap");
                props.setString("filename", albedoName);
                Texture2D* texture =
                    static_cast<Texture2D*>(PluginManager::getInstance()->createObject(MTS_CLASS(Texture2D), props));

                m_texture_albedos_m[index][iLayer] = texture;

                bsdf->addChild("reflectance", texture);
                m_bsdfList_m[index][iLayer] = bsdf;
            }

        } else {
            std::string F0Name = paramName + "_F0.exr";
            int         ret    = LoadEXR(&image, &width, &height, F0Name.c_str(), &err);
            if (ret != TINYEXR_SUCCESS) {
                m_texture_F0_m[index][iLayer]      = NULL;
                m_texture_albedos_m[index][iLayer] = NULL;
                m_roughness_m[index][iLayer]       = NULL;
                m_w1w2_m[index][iLayer]            = NULL;
            } else {

                Properties props("bitmap");
                props.setString("filename", F0Name);
                m_texture_F0_m[index][iLayer] =
                    static_cast<Texture2D*>(PluginManager::getInstance()->createObject(MTS_CLASS(Texture2D), props));


                std::string albedoName = paramName + "_albedo.exr";
                Properties  propsAlbedo("bitmap");
                propsAlbedo.setString("filename", albedoName);
                m_texture_albedos_m[index][iLayer] = static_cast<Texture2D*>(
                    PluginManager::getInstance()->createObject(MTS_CLASS(Texture2D), propsAlbedo)
                );

                std::string roughnessName = paramName + "_roughness.exr";
                Properties  propsRoughness("bitmap");
                propsRoughness.setString("filename", roughnessName);
                m_roughness_m[index][iLayer] = static_cast<Texture2D*>(
                    PluginManager::getInstance()->createObject(MTS_CLASS(Texture2D), propsRoughness)
                );
                std::cout << "loadRoughnessFile  finished\n";


                std::string w1w2Name = paramName + "_w1w2.exr";
                Properties  propsW1w2("bitmap");
                propsW1w2.setString("filename", w1w2Name);
                m_w1w2_m[index][iLayer] =
                    static_cast<Texture2D*>(PluginManager::getInstance()->createObject(MTS_CLASS(Texture2D), propsW1w2)
                    );
                std::cout << "loadW1W2File  finished\n";
            }
        }
    }

    void store(
        std::vector<float>& image,
        int                 idx,
        const Spectrum&     albedo,
        float               roughness,
        float               thickness,
        const Spectrum&     F0,
        const Vector&       orientation,
        int                 type
    ) {

        image[idx + 0]  = albedo[0];
        image[idx + 1]  = albedo[1];
        image[idx + 2]  = albedo[2];
        image[idx + 3]  = roughness;
        image[idx + 4]  = thickness;
        image[idx + 5]  = F0[0];
        image[idx + 6]  = F0[1];
        image[idx + 7]  = F0[2];
        image[idx + 8]  = orientation[0];
        image[idx + 9]  = orientation[1];
        image[idx + 10] = orientation[2];
        image[idx + 11] = type;
    }
    void outTextureMap(int paramSize) {
        // output the parameters for the network.
        //  roughness, abledo, thickness, F, orientation, type,
        const int CHANNEL = 1 + 3 + 1 + 3 + 3 + 1;

        for (int k = 0; k < m_layerCount; k++) {
            std::vector<float> image(paramSize * paramSize * CHANNEL);
            std::vector<float> imageTemp(paramSize * paramSize * 3);
#pragma omp parallel for
            for (int i = 0; i < paramSize; i++) {
                for (int j = 0; j < paramSize; j++) {
                    int idx = (i * paramSize + j) * CHANNEL;

                    float    thickness = 0;
                    Spectrum abledo(1);
                    Spectrum F0(1.0f);
                    Vector   orientation(0, 0, 1);
                    int      type;
                    float    roughness = 0.0;

                    if (m_bsdfList[k] != NULL) {
                        type   = 3;
                        abledo = m_spectrum_albedos[k];
                        store(image, idx, abledo, roughness, thickness, F0, orientation, type);
                        continue;
                    }

                    Point2 uv = Point2(i + 0.5f, j + 0.5f) / paramSize;
                    abledo = m_flag_albedos[k] ? getColorFromTexture(m_texture_albedos[k], uv) : m_spectrum_albedos[k];

                    F0                                     = m_F0[k];
                    imageTemp[(i * paramSize + j) * 3 + 0] = F0[0];
                    imageTemp[(i * paramSize + j) * 3 + 1] = F0[1];
                    imageTemp[(i * paramSize + j) * 3 + 2] = F0[2];
                    const Properties& phase                = m_phaseFunctions[k]->getProperties();
                    if (m_flag_aniso[k]) {

                        roughness      = phase.getFloat("roughness", 1.0);
                        bool typeFiber = phase.getBoolean("fiber", false);
                        if (typeFiber)
                            type = 1;
                        else
                            type = 0;
						Intersection its;
						its.uv = uv;
                        orientation = m_flag_orientations[k]
							? getOrientationFromTexture(m_texture_orientations[k], its, m_scale_n[k])
                                          : m_vector_orientations[k];
                    } else {
                        type      = 2;
                        roughness = phase.getFloat("g", 0.0);
                    }

                    thickness = m_thickness[k];

                    if (m_flag_thickness[k]) {
                        thickness = getColorFromTexture(m_texture_thickness[k], uv)[0];
                        // thickness *= thickness * thickness;
                        // thickness *= m_scale_n[k];
                        // if (thickness >= 2.5f)
                        //	thickness = 2.5f;
                        thickness = 1;
                        // this is because of the miss aligned UV in the plant scene.
                    }
                    store(image, idx, abledo, roughness, thickness, F0, orientation, type);
                }
            }
            std::string imageName = m_paramName;
            imageName.append("_");
            std::ostringstream ss;
            ss << k;
            std::string s(ss.str());

            imageName.append(s);
            imageName.append(".exr");
            SaveEXRChannel12(image, paramSize, paramSize, imageName.c_str());

            /*		std::string imageNameTemp = m_paramName + "layerTemp";
            imageNameTemp.append("_");
            std::ostringstream ss1;
            ss1 << k;
            std::string s1(ss1.str());

            imageNameTemp.append(s1);
            imageNameTemp.append(".exr");
            SaveEXRChannel3(imageTemp, paramSize, paramSize, imageNameTemp.c_str());*/
        }
    }

    Spectrum exp(const Spectrum& input) { return Color3(std::exp(input[0]), std::exp(input[1]), std::exp(input[2])); }

    void print(const Vector& dir) { SLog(EDebug, "f%,f%,f%", dir.x, dir.y, dir.z); }

    Spectrum computeLayredAttLayers(int iLayer, float z, const Vector& wo, const Point2& uv, const Intersection& its) const {
        Spectrum attenuation(1.0f);

        if (wo.z > 0) {
            for (int i = 0; i < iLayer; i++) {
                Vector orientation = m_vector_orientations[i];
                if (m_flag_orientations[i]) {
					orientation = getOrientationFromTexture(m_texture_orientations[i], its, m_scale_n[i]);
                }
                Float saturation = m_float_saturations[i];
                if (m_flag_saturation[i]) {
                    saturation = getColorFromTexture(m_texture_saturation[i], uv)[0];
                }
                if (m_flag_wet[i])
					attenuation *= computeAttenuationWet(wo, m_thickness[i], i, saturation, m_roughness[i], orientation);
                else
                    attenuation *= computeAttenuation(wo, m_roughness[i], m_thickness[i], i, orientation);
            }

            Vector orientation = m_vector_orientations[iLayer];
            if (m_flag_orientations[iLayer]) {
				orientation = getOrientationFromTexture(m_texture_orientations[iLayer], its, m_scale_n[iLayer]);
            }

            Float saturation = m_float_saturations[iLayer];
            if (m_flag_saturation[iLayer]) {
                saturation = getColorFromTexture(m_texture_saturation[iLayer], uv)[0];
            }
            if (m_flag_wet[iLayer])
				attenuation *= computeAttenuationWet(wo, z - m_thicknessAcc[iLayer], iLayer, saturation, m_roughness[iLayer], orientation);
            // float value = computeAttenuation(wo, z, iLayer);
            else
                attenuation *=
                    computeAttenuation(wo, m_roughness[iLayer], z - m_thicknessAcc[iLayer], iLayer, orientation);
            ;

        } else // go outside from the bottom layer
        {

            for (int i = iLayer + 1; i < m_layerCount; i++) {
                Vector orientation = m_vector_orientations[i];
                if (m_flag_orientations[i]) {
					orientation = getOrientationFromTexture(m_texture_orientations[i], its, m_scale_n[i]);
                }
                Float saturation = m_float_saturations[i];
                if (m_flag_saturation[i]) {
                    saturation = getColorFromTexture(m_texture_saturation[i], uv)[0];
                }
                if (m_flag_wet[i])
					attenuation *= computeAttenuationWet(wo, m_thickness[i], i, saturation, m_roughness[i], orientation);
                else
                    attenuation *= computeAttenuation(wo, m_roughness[i], m_thickness[i], i, orientation);
            }

            Vector orientation = m_vector_orientations[iLayer];
            if (m_flag_orientations[iLayer]) {
				orientation = getOrientationFromTexture(m_texture_orientations[iLayer], its, m_scale_n[iLayer]);
            }
            Float saturation = m_float_saturations[iLayer];
            if (m_flag_saturation[iLayer]) {
                saturation = getColorFromTexture(m_texture_saturation[iLayer], uv)[0];
            }
            if (m_flag_wet[iLayer])
				attenuation *= computeAttenuationWet(wo, m_thicknessAcc[iLayer + 1] - z, iLayer, saturation, m_roughness[iLayer], orientation);
            else
                attenuation *=
                    computeAttenuation(wo, m_roughness[iLayer], m_thicknessAcc[iLayer + 1] - z, iLayer, orientation);
        }

        return attenuation;
    }


    Spectrum computeLayredAtt(int iLayer, float z, const Vector& wo, const Point2& uv, const Intersection& its) const {
        Spectrum attenuation(1.0f);
#if 0
#if 0
		//if the media is isotropic media, then there are sigmaT, otherwise SigmaT is related to wi
		for (int i = 0; i < iLayer - 1; i++)
		{
			float sampledDistance = 1;
			attenuation *= exp(-m_spectrum_sigmaTs[i] * sampledDistance / wo[2]);

		}
		float sampledDistance = z - math::floorToInt(z);
		attenuation *= exp(-m_spectrum_sigmaTs[iLayer] * sampledDistance / wo[2]);
#else
		for (int i = 0; i < iLayer - 1; i++)
		{
			float sampledDistance = 1;
			float sigmaT = m_mediums[i]->lookupSigmaT(wo);
			float value = std::exp(-sigmaT * sampledDistance / wo[2]);
			attenuation *= Spectrum(value);

		}
		float sampledDistance = z - math::floorToInt(z);
		float sigmaT = m_mediums[iLayer]->lookupSigmaT(wo);
		float value = std::exp(-sigmaT * sampledDistance / wo[2]);
		attenuation *= value;
#endif
#else
        Vector orientation = m_vector_orientations[iLayer];
        if (m_flag_orientations[iLayer]) {
			orientation = getOrientationFromTexture(m_texture_orientations[iLayer], its, m_scale_n[iLayer]);
        }
        if (wo.z > 0) {
            if (iLayer != 0)
                SLog(EError, "The exit layer is not 0.");
#if 0
			//go outside from the top layer
			for (int i = 0; i < iLayer - 1; i++)
			{
				attenuation *= computeAttenuation(wo, m_thickness[i], i);

			}
#endif
            // float value = computeAttenuation(wo, z, iLayer);

            attenuation *= computeAttenuation(wo, m_roughness[iLayer], z, iLayer, orientation);
            ;
        } else // go outside from the bottom layer
        {
            if (iLayer != m_layerCount - 1)
                SLog(EError, "The exit layer is not the bottom one.");

            attenuation *=
                computeAttenuation(wo, m_roughness[iLayer], m_thicknessAcc[m_layerCount] - z, iLayer, orientation);
        }
#endif
        return attenuation;
    }

#if 0
	Spectrum generatePathWithNEE(std::vector<float> &image, BSDFSamplingRecord &_bRec, const int maxDepth, float scale)
	{
		const ref_vector<Medium> &mediums = m_mediums;
		int m_nbLayers = m_layerCount;
		Assert(_bRec.sampler);
		Sampler *sampler = _bRec.sampler;

		// Path tracing
		bool flag_incidentDir = _bRec.wi.z > 0;

		Point originPoint = flag_incidentDir ? Point(0.0, 0.0, 0.0) : Point(0.0, 0.0, Float(1 - m_nbLayers));
		Ray ray(originPoint /*+ _bRec.wi*/, -_bRec.wi, 0.0);

		Intersection its;
		MediumSamplingRecord mRec;

		Spectrum throughput(1.0);
		int depth = 0;
		int curLayer;

		while (depth < maxDepth || maxDepth < 0) {

			curLayer = math::floorToInt(-ray(Epsilon).z);
			if (curLayer > m_layerCount - 1) //go though the bottom  //this is fine, if we also consider dielectric
			{
				throughput = Spectrum(0.0);
				break;
			}

			// although we are always in the media, but it's possible the media sampling fails
			if (mediums[curLayer]->sampleDistance(Ray(ray, 0, its.t), mRec, sampler)) {
				if (mRec.p.z > 0) //the ray have been go outside the media, return 
				{
					throughput = Spectrum(0.0);
					break;
					//_bRec.wo = ray.d;
					////we only consider reflection now.
					//_bRec.eta = 1.0;
					//Spectrum att = computeLayredAtt(curLayer, -ray.o.z, ray.d);
					//return throughput * att;
				}

				int newLayer = math::floorToInt(-mRec.p.z);
				if (newLayer > m_layerCount - 1) //go though the bottom  //this is fine, if we also consider dielectric
				{
					throughput = Spectrum(0.0);
					break;
				}
				if (newLayer != curLayer)
				{
					SLog(EDebug, "Sample to another layer!");
					return Spectrum(0.0);
				}

				Spectrum albedo = mRec.sigmaS * mRec.transmittance / mRec.pdfSuccess;
				throughput *= albedo;

				float pdfTemp;
				pdfTemp = mRec.pdfSuccess / std::abs(ray.d.z);
				if (pdfTemp < Epsilon) {
					throughput = Spectrum(0.0);
					break;
				}

				PhaseFunctionSamplingRecord pRec(mRec, -ray.d);
				//update to all wo
				//compute the attenuation
				if (maxDepth == 1 || depth > 0)
					update2Table(image, throughput, pRec, scale, curLayer, -mRec.p.z);

				const Medium* medium = mediums[curLayer].get();
				const PhaseFunction* phase = medium->getPhaseFunction();
				Float phaseVal = phase->sample(pRec, sampler);
				if (std::abs(phaseVal) < Epsilon) {
					throughput = Spectrum(0.0);
					break;
				}
				throughput *= phaseVal;

				pdfTemp = phase->pdf(pRec);
				if (pdfTemp < Epsilon) {
					throughput = Spectrum(0.0);
					break;
				}

				// Trace a ray
				ray = Ray(mRec.p, pRec.wo, 0.0);
				ray.mint = 0.0;

				//the thickness of each layer is 1.
				if (ray(Epsilon).z > 0) //the ray have been go outside the media, return 
				{
					break;
				}
			}
			else
			{
				return Spectrum(0.0);
			}

			depth++;
		}

		return Spectrum(0.0);
	}

	Spectrum generatePathWithNEE(int iidx, BSDFSamplingRecord &_bRec, const int maxDepth, float scale)
	{
		const ref_vector<Medium> &mediums = m_mediums;
		int m_nbLayers = m_layerCount;
		Assert(_bRec.sampler);
		Sampler *sampler = _bRec.sampler;

		// Path tracing
		bool flag_incidentDir = _bRec.wi.z > 0;

		Point originPoint = flag_incidentDir ? Point(0.0, 0.0, 0.0) : Point(0.0, 0.0, Float(1 - m_nbLayers));
		Ray ray(originPoint /*+ _bRec.wi*/, -_bRec.wi, 0.0);

		Intersection its;
		MediumSamplingRecord mRec;

		Spectrum throughput(1.0);
		int depth = 0;
		int curLayer;

		while (depth < maxDepth || maxDepth < 0) {

			curLayer = math::floorToInt(-ray(Epsilon).z);
			if (curLayer > m_layerCount - 1) //go though the bottom  //this is fine, if we also consider dielectric
			{
				throughput = Spectrum(0.0);
				break;
			}

			// although we are always in the media, but it's possible the media sampling fails
			if (mediums[curLayer]->sampleDistance(Ray(ray, 0, its.t), mRec, sampler)) {
				if (mRec.p.z > 0) //the ray have been go outside the media, return 
				{
					throughput = Spectrum(0.0);
					break;
					//_bRec.wo = ray.d;
					////we only consider reflection now.
					//_bRec.eta = 1.0;
					//Spectrum att = computeLayredAtt(curLayer, -ray.o.z, ray.d);
					//return throughput * att;
				}

				int newLayer = math::floorToInt(-mRec.p.z);
				if (newLayer > m_layerCount - 1) //go though the bottom  //this is fine, if we also consider dielectric
				{
					throughput = Spectrum(0.0);
					break;
				}
				if (newLayer != curLayer)
				{
					SLog(EDebug, "Sample to another layer!");
					return Spectrum(0.0);
				}
				const Medium* medium = mediums[curLayer].get();
				const PhaseFunction* phase = medium->getPhaseFunction();

				Spectrum albedo = mRec.sigmaS * mRec.transmittance / mRec.pdfSuccess;
				throughput *= albedo;

				float pdfTemp;
				pdfTemp = mRec.pdfSuccess / std::abs(ray.d.z);
				if (pdfTemp < Epsilon) {
					throughput = Spectrum(0.0);
					break;
				}

				PhaseFunctionSamplingRecord pRec(mRec, -ray.d);
				//update to all wo
				if (maxDepth == 1 || depth > 0)
					update2Table(iidx, throughput, pRec, scale, curLayer, -mRec.p.z);

				Float phaseVal = phase->sample(pRec, sampler);
				if (std::abs(phaseVal) < Epsilon) {
					throughput = Spectrum(0.0);
					break;
				}
				throughput *= phaseVal;

				pdfTemp = phase->pdf(pRec);
				if (pdfTemp < Epsilon) {
					throughput = Spectrum(0.0);
					break;
				}

				// Trace a ray
				ray = Ray(mRec.p, pRec.wo, 0.0);
				ray.mint = 0.0;

				//the thickness of each layer is 1.
				if (ray(Epsilon).z > 0) //the ray have been go outside the media, return 
				{
					break;
				}
			}
			else
			{
				return Spectrum(0.0);
			}

			depth++;
		}
		return Spectrum(0.0);
	}
#endif

    int locateLayer(const std::vector<float>& thickness, const float depth) {
        if (m_layerCount == 1) {
            if (depth <= m_thickness[0]) {
                if (depth >= 0)
                    return 0;
                else
                    return -1;
            } else
                return m_layerCount;
        } else {

            if (depth < 0)
                return 0;
            else if (depth > thickness[m_layerCount])
                return m_layerCount; // go bottom
            else {
                // find the correct layer
                int ilayer = std::lower_bound(thickness.begin(), thickness.end(), depth) - thickness.begin();
                return ilayer - 1;
            }
        }
    }


    int locateLayer(const std::vector<float>& thickness, const float depth) const {
        if (m_layerCount == 1) {
            if (depth <= m_thickness[0]) {
                if (depth >= 0)
                    return 0;
                else
                    return -1;
            } else
                return m_layerCount;
        } else {

            if (depth < 0)
                return 0;
            else if (depth > thickness[m_layerCount])
                return m_layerCount; // go bottom
            else {
                // find the correct layer
                int ilayer = std::lower_bound(thickness.begin(), thickness.end(), depth) - thickness.begin();
                return ilayer - 1;
            }
        }
    }
    Spectrum generatePath(BSDFSamplingRecord& _bRec, const int maxDepth) {
		ref_vector<Medium>& mediums = m_mediums;
		//ref_vector<Medium> mediums;
		//setParameters(_bRec, mediums);
        int                       m_nbLayers = m_layerCount;
        Assert(_bRec.sampler);
        Sampler* sampler = _bRec.sampler;

        // Path tracingco
        bool flag_incidentDir = _bRec.wi.z > 0;

        Point originPoint =
			flag_incidentDir ? Point(0.0, 0.0, -Epsilon*Epsilon) : Point(0.0, 0.0, m_thicknessAcc[m_layerCount]);
        Ray                  ray(originPoint /*+ _bRec.wi*/, -_bRec.wi, 0.0);
        int                  scatteringCount = 0;
        Intersection         its;
        MediumSamplingRecord mRec;

        Spectrum throughput(1.0);
        int      depth = 0;
        int      curLayer;

        int  itest       = 0;
        bool lastSurface = false;
        while (depth <= maxDepth || maxDepth < 0) {
			if (ray.d.z == 0.0)
				 return Spectrum(0.0);
            if (itest > 5)
                SLog(EDebug, "Issue here!");

            float curDepth = -ray.o.z;
            curLayer       = locateLayer(m_thicknessAcc, curDepth); // math::floorToInt(-ray.o.z);

            // although we are always in the media, but it's possible the media sampling fails
            float travDist =
                ray.d.z < 0 ? m_thicknessAcc[curLayer + 1] - curDepth : curDepth - m_thicknessAcc[curLayer];
            travDist /= std::abs(ray.d.z);
            if (travDist == 0)
                return Spectrum(0.0);

            if (m_bsdfList[curLayer] == NULL &&
				((depth == 0) ? mediums[curLayer]->sampleDistance(Ray(ray, 0, travDist), mRec, sampler) 
				: mediums[curLayer]->sampleDistanceC(Ray(ray, 0, travDist), mRec, sampler))) {

                int newLayer = locateLayer(m_thicknessAcc, -mRec.p.z); // math::floorToInt(-mRec.p.z);
                if (newLayer != curLayer) {
                    //	SLog(EError, "Sample to another layer!");
                    return Spectrum(0.0);
                }

                const Medium*        medium = mediums[curLayer].get();
				const PhaseFunction* phase = m_phaseFunctions[curLayer];// medium->getPhaseFunction();


                Spectrum albedo = mRec.sampleWeight;
				
                throughput *= albedo;

                float pdfTemp;
				pdfTemp = mRec.pdfSuccess;// / std::abs(ray.d.z);

                if (pdfTemp < Epsilon) {
                    //throughput = Spectrum(0.0);
                    //break;
                }

                PhaseFunctionSamplingRecord pRec(mRec, -ray.d);
				// pRec.mRec.orientation = m_vector_orientations[0];
				Float phaseVal;
				//PGL
				if (m_flag_wet[curLayer]){
					Float saturation = m_float_saturations[curLayer];
					Float pdf;
					phaseVal = phase->sample_wet(pRec, sampler, pdf, saturation);
					if (pdf < Epsilon) {
						throughput = Spectrum(0.0);
						return Spectrum(0.0);
					}
				}
				else
					phaseVal = phase->sample(pRec, sampler);
#if WHITEFURNACE == 0
                if (std::abs(phaseVal) < Epsilon) {
                    //throughput = Spectrum(0.0);
                    //break;
                }
#endif
                throughput *= phaseVal;

#if FRESNEL_MULTIPLE
				if (!m_flag_wet[curLayer]){
					if ((pRec.wi + pRec.wo).isZero())
						break;
					Vector h = normalize(pRec.wi + pRec.wo);
					throughput *= Specular_Fresnel_Schlick(m_F0[curLayer], std::max(0.0f, dot(h, pRec.wi)));
				}
#endif

                //pdfTemp = phase->pdf(pRec);
#if WHITEFURNACE == 0
                //if (pdfTemp < Epsilon) {
                 //   throughput = Spectrum(0.0);
                 //   break;
               // }
#endif

                scatteringCount++;
                lastSurface = false;
                if (scatteringCount > maxDepth && maxDepth >= 0)
                    return Spectrum(0.0);
                // Trace a ray
                ray      = Ray(mRec.p, pRec.wo, 0.0);
                ray.mint = 0.0;
                depth++;
            } else {
                if (m_bsdfList[curLayer] == NULL) {
                    if (mRec.pdfFailure <= Epsilon)
                        return Spectrum(0.0);
#if WHITEFURNACE == 0
					//Spectrum transmittance = computeLayredAttLayers(curLayer, -mRec.p.z, ray.d, _bRec.its.uv, _bRec.its);
					throughput *= mRec.transmittance / mRec.pdfFailure;

#else
                    throughput *= 1;
#endif
                } else {
                    // if (depth < maxDepth || maxDepth <0)
                    {
                        float              pdf;
                        BSDFSamplingRecord bRec  = _bRec;
                        bRec.wi                  = -ray.d;
                        Spectrum weight          = m_bsdfList[curLayer]->sample(bRec, pdf, bRec.sampler->next2D());
                        throughput              *= weight; // / abs(bRec.wo.z);
                        lastSurface              = true;
                        if (curLayer == 0) {
                            _bRec.wo  = bRec.wo;
                            _bRec.eta = 1.0;
                            return throughput; // / abs(bRec.wo.z);
                        }
                        Point newP = ray.o + (Epsilon)*bRec.wo;
                        ray        = Ray(newP, bRec.wo, 0.0);
                        ray.mint   = 0.0;
                        scatteringCount++;
                        depth++;
                        continue;
                    }
                    /*else
                    {
                    return Spectrum(0.0);
                    }*/
                }

                // if hit a surface
                if (curLayer == 0 && ray.d.z > 0)
                // the ray have been go outside the media, return
                {
                    // if multiple only, use this
                    if ((maxDepth == 1 || depth > 1) && scatteringCount > 0) {
                        _bRec.wo = ray.d;
                        // we only consider reflection now.
                        _bRec.eta = 1.0;
                        //	Spectrum att = computeLayredAtt(curLayer, -ray.o.z, ray.d);
                        //	if (lastSurface)throughput /= abs(_bRec.wo.z);
                        //		if (lastSurface)
                        return throughput; // *att;
                                           //	else return Spectrum(0.0);
                    } else
                        return Spectrum(0.0);
                }
                // go to the bottom of all the layers

                if (curLayer == m_layerCount - 1 && ray.d.z < 0) {
                    if (m_bsdfList[curLayer] != NULL) {
                        if (depth < maxDepth) {
                            float              pdf;
                            BSDFSamplingRecord bRec  = _bRec;
                            bRec.wi                  = -ray.d;
                            Spectrum weight          = m_bsdfList[curLayer]->sample(bRec, pdf, bRec.sampler->next2D());
                            throughput              *= weight; // / abs(bRec.wo.z);
                            lastSurface              = true;
                            Point newP               = ray.o + (travDist)*ray.d;
                            newP                     = newP + (Epsilon)*bRec.wo;
                            ray                      = Ray(newP, bRec.wo, 0.0);
                            ray.mint                 = 0.0;
                            scatteringCount++;
                            depth++;
                            continue;
                        } else {
                            return Spectrum(0.0);
                        }
                    }

                    // if multiple only, use this
                    if (((maxDepth == 1 || depth > 1) && scatteringCount > 0)) {
                        _bRec.wo     = ray.d;
                        _bRec.eta    = 1.0;
                        Spectrum att = computeLayredAtt(curLayer, -ray.o.z, ray.d, Point2(0, 0), _bRec.its);
                        return throughput; // *att;

                    } else
                        return Spectrum(0.0);
                }


                // Trace a ray
                if (ray.d.z < 0) {
                    if (m_bsdfList[curLayer + 1] == NULL) {
                        Point newP = ray.o + (Epsilon + travDist) * ray.d;
                        ray        = Ray(newP, ray.d, 0.0);
                        ray.mint   = 0.0;
                    } else {
                        float              pdf;
                        BSDFSamplingRecord bRec  = _bRec;
                        bRec.wi                  = -ray.d;
                        Spectrum weight          = m_bsdfList[curLayer + 1]->sample(bRec, pdf, bRec.sampler->next2D());
                        throughput              *= weight; // / abs(bRec.wo.z);;
                        Point newP               = ray.o + (travDist)*ray.d;
                        newP                     = newP + (Epsilon)*bRec.wo;
                        ray                      = Ray(newP, bRec.wo, 0.0);
                        ray.mint                 = 0.0;
                        scatteringCount++;
                        depth++;
                        lastSurface = true;
                    }
                } else {
                    if (m_bsdfList[curLayer - 1] == NULL) {
                        Point newP = ray.o + (Epsilon + travDist) * ray.d;
                        ray        = Ray(newP, ray.d, 0.0);
                        ray.mint   = 0.0;
                    } else {
                        SLog(EDebug, "Issue here!");
                        return Spectrum(0.0);
                        /*float pdf;
                        BSDFSamplingRecord bRec = _bRec;
                        bRec.wi = -ray.d;
                        Spectrum weight = m_bsdfList[curLayer - 1]->sample(bRec, pdf, bRec.sampler->next2D());
                        throughput *= weight;
                        Point newP = ray.o + (travDist)* ray.d;
                        newP = newP + (Epsilon)* bRec.wo;
                        ray = Ray(newP, bRec.wo, 0.0);
                        ray.mint = 0.0;
                        scatteringCount++;
                        depth++;*/
                    }
                }
            }
        }
        return Spectrum(0.0);
    }


    void setupMediaRuntime(ref_vector<Medium>& medium, const Point2& uv, const Intersection& its) const {
        // Medium
        medium.resize(m_layerCount);

        Properties interiorIsoHomoProps("homogeneous");
        interiorIsoHomoProps.setSpectrum("sigmaT", Spectrum(0.1));
        interiorIsoHomoProps.setSpectrum("albedo", Spectrum(1.0));

        Properties interiorAniHomoProps("homogeneous_aniso");
        interiorAniHomoProps.setFloat("density", 0.0);
        interiorAniHomoProps.setSpectrum("albedo", Spectrum(1.0));
        interiorAniHomoProps.setVector("orientation", Vector(1.0, 0.0, 0.0));


        for (int l = 0; l < m_layerCount; ++l) {
            if (m_bsdfList[l] == NULL) {
                if (m_flag_aniso[l]) {
                    interiorAniHomoProps.setFloat("density", m_float_densities[l]);

                    Vector orientation = m_vector_orientations[l];
                    if (m_flag_orientations[l]) {
						orientation = getOrientationFromTexture(m_texture_orientations[l], its, m_scale_n[l]);
                    }

                    Spectrum abledo =
                        m_flag_albedos[l] ? getColorFromTexture(m_texture_albedos[l], uv) : m_spectrum_albedos[l];


                    interiorAniHomoProps.setSpectrum("albedo", abledo);
                    interiorAniHomoProps.setVector("orientation", orientation);

                    medium[l] = static_cast<Medium*>(
                        PluginManager::getInstance()->createObject(MTS_CLASS(Medium), interiorAniHomoProps)
                    );
                    //	medium[l]->addChild(m_phaseFunctions[l]);
                    medium[l]->configure();

                } else {
                    interiorIsoHomoProps.setSpectrum("sigmaT", m_spectrum_sigmaTs[l]);
                    interiorIsoHomoProps.setSpectrum("albedo", m_spectrum_albedos[l]);

                    medium[l] = static_cast<Medium*>(
                        PluginManager::getInstance()->createObject(MTS_CLASS(Medium), interiorIsoHomoProps)
                    );
                    //	medium[l]->addChild(m_phaseFunctions[l]);
                    medium[l]->configure();
                }
            } else
                medium[l] = NULL;
        }
    }
    Spectrum msSimulation(BSDFSamplingRecord& bRec, float& pdf) const {
        Spectrum result = generatePathRuntime(bRec, pdf, m_depth);
        return pdf > 1e-7 ? result / pdf : Spectrum(0.0f);
    }

    Spectrum msSimulationNEE(const BSDFSamplingRecord& bRec, int start_depth) const {
        float    pdf;
        Spectrum result = generatePathRuntimeNEE(bRec, pdf, m_depth, start_depth); // *abs(bRec.wo.z);
        return result;
    }

    void setParameters(const BSDFSamplingRecord& _bRec, ref_vector<Medium>& mediums) const {

        Point2 uv        = _bRec.its.uv;
        int    thread_id = Thread::getID() & NUM_COUNTERS_MASK;

        // medium and phase function for specific position.
        mediums = m_mediums_runtime[thread_id];

        for (int l = 0; l < m_layerCount; ++l) {
            if (m_bsdfList[l] == NULL) {
                if (m_flag_aniso[l]) {
                    Float    density = m_flag_densities[l] ? m_texture_densities[l]->eval(uv)[0] * m_float_densities[l]
                                                           : m_float_densities[l];
                    Spectrum albedo  = m_spectrum_albedos[l];
                    if (m_flag_albedos[l])
                        albedo = m_texture_albedos[l]->eval(uv);
                    Vector orientation = m_flag_orientations[l]
						? getOrientationFromTexture(m_texture_orientations[l], _bRec.its, m_scale_n[l])
                                             : m_vector_orientations[l];
					if (m_flag_wet[l]){
						Spectrum sigmaWet = m_spectrum_sigmaWets[l];
						Float saturation = m_flag_saturation[l] ? m_texture_saturation[l]->eval(uv)[0] : m_float_saturations[l];
						mediums[l]->setMediumPropWet(density, albedo, orientation, saturation, sigmaWet);
					}
					else{
						mediums[l]->setMediumProp(density, albedo, orientation);
					}
                } else {
                    Spectrum sigmaT = m_spectrum_sigmaTs[l];
                    if (m_flag_sigmaTs[l])
                        sigmaT = m_texture_sigmaTs[l]->eval(uv);
                    Spectrum albedo = m_spectrum_albedos[l];
                    if (m_flag_albedos[l])
                        albedo = m_texture_albedos[l]->eval(uv);
                    // PGL
                    if (m_flag_wet[l]) {
						Float porosity = m_float_porositys[l];
						Spectrum sigmaWet = m_spectrum_sigmaWets[l];
						Float saturation = m_flag_saturation[l] ? m_texture_saturation[l]->eval(uv)[0] : m_float_saturations[l];
						mediums[l]->setHapkeProp(sigmaT, porosity, albedo, sigmaWet, saturation);
                    }
					else{
						mediums[l]->setSigmaAST(sigmaT, albedo);
					}
                }
            } else
                mediums[l] = NULL;
        }
    }


    Spectrum generatePathRuntime(BSDFSamplingRecord& _bRec, float& pdf, const int maxDepth) const {
        ref_vector<Medium> mediums;
        //	setupMediaRuntime(mediums, _bRec.its.uv);
        setParameters(_bRec, mediums);
        int m_nbLayers = m_layerCount;
        Assert(_bRec.sampler);
        Sampler* sampler = _bRec.sampler;

        // Path tracing
        bool flag_incidentDir = _bRec.wi.z > 0;
        pdf                   = 1.0f;

        Point originPoint =
            flag_incidentDir ? Point(0.0, 0.0, -Epsilon) : Point(0.0, 0.0, -m_thicknessAcc[m_layerCount] + Epsilon);
        Ray                  ray(originPoint /*+ _bRec.wi*/, -_bRec.wi, 0.0);
        int                  scatteringCount = 0;
        Intersection         its;
        MediumSamplingRecord mRec;

        Spectrum throughput(1.0);
        int      depth = 0;
        int      curLayer;

        int  itest       = 0;
        bool lastSurface = false;
        while (depth <= maxDepth || maxDepth < 0) {
            if (ray.d.z == 0.0)
                return Spectrum(0.0);
            if (itest > 5)
                SLog(EDebug, "Issue here!");

            float curDepth = -ray.o.z;
            curLayer       = locateLayer(m_thicknessAcc, curDepth); // math::floorToInt(-ray.o.z);

            // although we are always in the media, but it's possible the media sampling fails
            float travDist =
                ray.d.z < 0 ? m_thicknessAcc[curLayer + 1] - curDepth : curDepth - m_thicknessAcc[curLayer];
            travDist /= std::abs(ray.d.z);
            if (travDist == 0)
                return Spectrum(0.0);

            if (m_bsdfList[curLayer] == NULL &&
                mediums[curLayer]->sampleDistance(Ray(ray, 0, travDist), mRec, sampler)) {

                int newLayer = locateLayer(m_thicknessAcc, -mRec.p.z); // math::floorToInt(-mRec.p.z);
                if (newLayer != curLayer) {
                    //	SLog(EError, "Sample to another layer!");
                    return Spectrum(0.0);
                }

                Medium*              medium = mediums[curLayer];
                const PhaseFunction* phase  = medium->getPhaseFunction();

                Spectrum albedo = mRec.sigmaS * mRec.transmittance / mRec.pdfSuccess;
#if WHITEFURNACE == 0
                throughput *= albedo;
#endif


                float pdfTemp;
                pdfTemp = mRec.pdfSuccess / std::abs(ray.d.z);

#if WHITEFURNACE == 0
                if (pdfTemp < Epsilon) {
                    throughput = Spectrum(0.0);
                    break;
                }

#endif
                //	pdf *= pdfTemp;

                PhaseFunctionSamplingRecord pRec(mRec, -ray.d);
                Float                       phaseVal = phase->sample(pRec, sampler);
#if WHITEFURNACE == 0
                if (std::abs(phaseVal) < Epsilon) {
                    throughput = Spectrum(0.0);
                    break;
                }
#endif

                throughput *= phaseVal;

#if FRESNEL_MULTIPLE
                if ((pRec.wi + pRec.wo).isZero())
                    break;
                Vector h    = normalize(pRec.wi + pRec.wo);
                throughput *= Specular_Fresnel_Schlick(m_F0[curLayer], std::max(0.0f, dot(h, pRec.wi)));
#endif


                pdfTemp = phase->pdf(pRec);
#if WHITEFURNACE == 0
                if (pdfTemp < Epsilon) {
                    throughput = Spectrum(0.0);
                    break;
                }
#endif
                //	pdf *= pdfTemp;
                scatteringCount++;
                lastSurface = false;
                if (scatteringCount > maxDepth)
                    return Spectrum(0.0);
                // Trace a ray
                ray      = Ray(mRec.p, pRec.wo, 0.0);
                ray.mint = 0.0;
                depth++;
            } else {
                if (m_bsdfList[curLayer] == NULL) {
                    if (mRec.pdfFailure == 0)
                        return Spectrum(0.0);

                        //	pdf *= mRec.pdfFailure;
#if WHITEFURNACE == 0
                    throughput *= mRec.transmittance / mRec.pdfFailure;
#else
                    throughput *= 1;
#endif
                } else {
                    if (depth < maxDepth) {
                        float              pdf_;
                        BSDFSamplingRecord bRec  = _bRec;
                        bRec.wi                  = -ray.d;
                        Spectrum weight          = m_bsdfList[curLayer]->sample(bRec, pdf_, bRec.sampler->next2D());
                        throughput              *= weight; // / abs(bRec.wo.z);
                        lastSurface              = true;
                        if (curLayer == 0) {
                            _bRec.wo  = bRec.wo;
                            _bRec.eta = 1.0;
                            return throughput; // / abs(bRec.wo.z);
                        }
                        Point newP = ray.o + (Epsilon)*bRec.wo;
                        ray        = Ray(newP, bRec.wo, 0.0);
                        ray.mint   = 0.0;
                        scatteringCount++;
                        depth++;
                        continue;
                    } else {
                        return Spectrum(0.0);
                    }
                }

                // if hit a surface
                if (curLayer == 0 && ray.d.z > 0) // the ray have been go outside the media, return
                {
                    // if ((maxDepth == 1 || depth > 1) && scatteringCount > 0)
                    if (scatteringCount > 0) // no delta
                    {
                        _bRec.wo = ray.d;
                        // we only consider reflection now.
                        _bRec.eta = 1.0;

                        _bRec.sampledComponent = 0;
                        _bRec.sampledType      = EGlossyReflection;
                        //	Spectrum att = computeLayredAtt(curLayer, -ray.o.z, ray.d);
                        //	if (lastSurface)throughput /= abs(_bRec.wo.z);
                        return throughput; // *att;
                    } else
                        return Spectrum(0.0);
                }
                // go to the bottom of all the layers
                if (curLayer == m_layerCount - 1 && ray.d.z < 0) {
                    if (m_bsdfList[curLayer] != NULL) {
                        if (depth < maxDepth) {
                            float              _pdf;
                            BSDFSamplingRecord bRec  = _bRec;
                            bRec.wi                  = -ray.d;
                            Spectrum weight          = m_bsdfList[curLayer]->sample(bRec, _pdf, bRec.sampler->next2D());
                            throughput              *= weight; // / abs(bRec.wo.z);
                            lastSurface              = true;
                            Point newP               = ray.o + (travDist)*ray.d;
                            newP                     = newP + (Epsilon)*bRec.wo;
                            ray                      = Ray(newP, bRec.wo, 0.0);
                            ray.mint                 = 0.0;
                            scatteringCount++;
                            depth++;
                            continue;
                        } else {
                            return Spectrum(0.0);
                        }
                    }

                    // if (((maxDepth == 1 || depth > 1) && scatteringCount > 0))
                    if (scatteringCount > 0) // no delta
                    {
                        _bRec.wo  = ray.d;
                        _bRec.eta = 1.0;

                        if (scatteringCount == 0) {
                            _bRec.sampledComponent = -2;
                            // this means going straight
                            _bRec.sampledType = EDeltaTransmission;
                        } else {
                            _bRec.sampledComponent = 1;
                            _bRec.sampledType      = EGlossyTransmission;
                        }


                        // Spectrum att = computeLayredAtt(curLayer, -ray.o.z, ray.d);
                        return throughput; // *att;

                    } else
                        return Spectrum(0.0);
                }


                // Trace a ray
                if (ray.d.z < 0) {
                    if (m_bsdfList[curLayer + 1] == NULL) {
                        Point newP = ray.o + (Epsilon + travDist) * ray.d;
                        ray        = Ray(newP, ray.d, 0.0);
                        ray.mint   = 0.0;
                    } else {
                        float              pdf_;
                        BSDFSamplingRecord bRec  = _bRec;
                        bRec.wi                  = -ray.d;
                        Spectrum weight          = m_bsdfList[curLayer + 1]->sample(bRec, pdf_, bRec.sampler->next2D());
                        throughput              *= weight; // / abs(bRec.wo.z);;
                        Point newP               = ray.o + (travDist)*ray.d;
                        newP                     = newP + (Epsilon)*bRec.wo;
                        ray                      = Ray(newP, bRec.wo, 0.0);
                        ray.mint                 = 0.0;
                        scatteringCount++;
                        depth++;
                        lastSurface = true;
                        //	pdf *= pdf_;
                    }
                } else {
                    if (m_bsdfList[curLayer - 1] == NULL) {
                        Point newP = ray.o + (Epsilon + travDist) * ray.d;
                        ray        = Ray(newP, ray.d, 0.0);
                        ray.mint   = 0.0;
                    } else {
                        SLog(EDebug, "Issue here!");
                        return Spectrum(0.0);
                    }
                }
            }
        }


        return Spectrum(0.0);
    }
    Spectrum generatePathRuntimeNEE(const BSDFSamplingRecord& _bRec, float& pdf, const int maxDepth, int start_depth) const {
        ref_vector<Medium> mediums;
        //	setupMediaRuntime(mediums, _bRec.its.uv);
        setParameters(_bRec, mediums);
        
        int m_nbLayers = m_layerCount;
        Assert(_bRec.sampler);
        Sampler* sampler = _bRec.sampler;

        // Path tracing
        bool flag_incidentDir = _bRec.wi.z > 0;
        pdf                   = 0.0f;

		// thickness texture
		std::vector<Float> thicknessList, thicknessAcc;
		thicknessAcc.push_back(0);
		for (int i = 0; i < m_layerCount; i++) {
			Float thickness = m_flag_thickness[i] ? getColorFromTexture(m_texture_thickness[i], _bRec.its.uv)[0]
				: m_thickness[i];
			thicknessList.push_back(thickness);
			thicknessAcc.push_back(thicknessAcc[i] + thickness);
		}

        Point originPoint =
			flag_incidentDir ? Point(0.0, 0.0, -Epsilon*Epsilon) : Point(0.0, 0.0, -thicknessAcc[m_layerCount] + Epsilon*Epsilon);
        // Point originPoint = flag_incidentDir ? Point(0.0, 0.0, -0) : Point(0.0, 0.0, -m_thicknessAcc[m_layerCount] +
        // 0);
        Ray                  ray(originPoint /*+ _bRec.wi*/, -_bRec.wi, 0.0);
        int                  scatteringCount = 0;
        Intersection         its;
        MediumSamplingRecord mRec;

        Spectrum throughput(1.0);
        Spectrum result(0.0f);
        int      depth = 0;
        int      curLayer;

        int  itest       = 0;
        bool lastSurface = false;

        while (depth <= maxDepth || maxDepth < 0) {
            if (ray.d.z == 0.0)
                return result;
            if (itest > 5)
                SLog(EDebug, "Issue here!");

            float curDepth = -ray.o.z;
            curLayer       = locateLayer(thicknessAcc, curDepth); // math::floorToInt(-ray.o.z);

            // although we are always in the media, but it's possible the media sampling fails
            float travDist =
                ray.d.z < 0 ? thicknessAcc[curLayer + 1] - curDepth : curDepth - thicknessAcc[curLayer];
            travDist /= std::abs(ray.d.z);
            if (travDist == 0)
                return result;
            // if (!mediums[curLayer]) {
            //     SLog(EError, "Medium pointer at layer %d is null!", curLayer);
            // }
            //return result;
            if (m_bsdfList[curLayer] == NULL &&
				(depth == 0) ? 
				  mediums[curLayer]->sampleDistance(Ray(ray, 0, travDist), mRec, sampler) //test
				: mediums[curLayer]->sampleDistance(Ray(ray, 0, travDist), mRec, sampler)) {
            // std::vector<ref_vector<Medium>> m_mediums_runtime2;
            // m_mediums_runtime2.resize(1, 1);
            // Properties interiorAniHomoWetProps("homogeneous");
            // int l=0;
            // interiorAniHomoWetProps.setFloat("density", m_float_densities[l]);
            // interiorAniHomoWetProps.setSpectrum("albedo", m_spectrum_albedos[l]);
            // interiorAniHomoWetProps.setVector("orientation", m_vector_orientations[l]);
            // interiorAniHomoWetProps.setFloat("saturation", m_float_saturations[l]);
            // interiorAniHomoWetProps.setSpectrum("sigmaWet", m_spectrum_sigmaWets[l]);
            // interiorAniHomoWetProps.setFloat("porosity", m_float_porositys[l]);
            // m_mediums_runtime2[0][0] = static_cast<Medium*>(
			// 						PluginManager::getInstance()->createObject(MTS_CLASS(Medium), interiorAniHomoWetProps)
			// 						);
            // auto phaseFunction = m_phaseFunctions[l];
            // m_mediums_runtime2[0][0]->addChild(phaseFunction);
            // m_mediums_runtime2[0][0]->configure();
            // if (m_bsdfList[curLayer] == NULL &&
			// 	(depth == 0) ? 
			// 	  m_mediums_runtime2[0][0]->sampleDistance(Ray(ray, 0, travDist), mRec, sampler) //test
			// 	: m_mediums_runtime2[0][0]->sampleDistanceC(Ray(ray, 0, travDist), mRec, sampler)) {
                    
                int newLayer = locateLayer(thicknessAcc, -mRec.p.z); // math::floorToInt(-mRec.p.z);
                if (newLayer != curLayer) {
                    return result;
                }
				
				//Spectrum albedo = mRec.sigmaS / (mRec.sigmaS + mRec.sigmaA) * mRec.sampleWeight;
				Spectrum albedo =  mRec.sampleWeight;

#if WHITEFURNACE == 0
				throughput *= albedo;
#endif
                float pdfTemp;
				pdfTemp = mRec.pdfSuccess;// / std::abs(ray.d.z);

#if WHITEFURNACE == 0
                if (pdfTemp < Epsilon) {
                    //throughput = Spectrum(0.0);
                    return result;
                }

#endif
                
                Medium*                     medium = mediums[curLayer];
                const PhaseFunction*        phase  = medium->getPhaseFunction();
                PhaseFunctionSamplingRecord pRec(mRec, -ray.d);

                // if (maxDepth == 1 || depth > 0)
                if (depth > start_depth)
                // here we decide to scatter,
                // so we connect the current point with the out going direction
                {
                    pRec.wo = _bRec.wo;

                    if (!(pRec.wi + pRec.wo).isZero()) {
                        Vector   h = normalize(pRec.wi + pRec.wo);
                        Spectrum F = Specular_Fresnel_Schlick(m_F0[curLayer], std::max(0.0f, dot(h, pRec.wi)));
                        Float    phaseValue = 0;
#if WHITEFURNACE
                        phaseValue = phase->eval(pRec);
#else
                        if (m_flag_wet[curLayer]) {
                            Float saturation =
                                m_flag_saturation[curLayer]
                                    ? getColorFromTexture(m_texture_saturation[curLayer], _bRec.its.uv)[0]
                                    : m_float_saturations[curLayer];
                            phaseValue = phase->eval_wet(pRec, saturation);
							//if (phaseValue < Epsilon)
								//phaseValue = 0;

                            //phaseValue = phase->eval(pRec);
                        } else
                            phaseValue = phase->eval(pRec);

#endif
                        //{
                        //	Float sigmaT = mediums[curLayer]->lookupSigmaT(pRec.wo);
                        //	float distance = abs(mRec.p.z) / std::abs(pRec.wo.z);
                        //}
                        // this does not work for microflake, but works for SGGX
                        Spectrum _attenation =
							computeLayredAttLayers(curLayer, -mRec.p.z, pRec.wo, _bRec.its.uv, _bRec.its); // need revision
                        //	float _attenation =	math::fastexp(-sigmaT*distance);
                        // if (scatteringCount > 0)
                        // return throughput * _attenation *phaseValue;
                        if (m_flag_wet[curLayer]) {
                            result += throughput * _attenation * phaseValue;
                            // return Spectrum(phaseValue);
                        } else
                            result += throughput * F * _attenation * phaseValue;
                        pdf += 1.0f;
                    }
                }
                Float phaseVal;
                
#if WHITEFURNACE
                phaseVal = phase->sample(pRec, sampler);
#else
                if (m_flag_wet[curLayer]) {
                    Float saturation = m_flag_saturation[curLayer]
                                           ? getColorFromTexture(m_texture_saturation[curLayer], _bRec.its.uv)[0]
                                           : m_float_saturations[curLayer];
                    Float temp;
                    phaseVal = phase->sample_wet(pRec, sampler, temp, saturation);
					//if (temp < Epsilon) {
					//	throughput = Spectrum(0.0);
					//	return result;
					//}
                    //phaseVal = phase->sample(pRec, sampler);
                } else
                    phaseVal = phase->sample(pRec, sampler);
#endif

#if WHITEFURNACE == 0
                if (std::abs(phaseVal) < Epsilon) {
                    //throughput = Spectrum(0.0);
					return result;
                }
#endif
                // if (phaseVal == 0)
                //	return Spectrum(0.5);
                throughput *= phaseVal;

#if FRESNEL_MULTIPLE
                // if ((pRec.wi + pRec.wo).isZero())
                //	break;
                if (!m_flag_wet[curLayer]) {
                    Vector h    = normalize(pRec.wi + pRec.wo);
                    throughput *= Specular_Fresnel_Schlick(m_F0[curLayer], std::max(0.0f, dot(h, pRec.wi)));
                }

#endif

                pdfTemp = phase->pdf(pRec); // need revision
#if WHITEFURNACE == 0
                if (pdfTemp < Epsilon) {
                    //throughput = Spectrum(0.0);
					return result;
                }
#endif
                //	pdf *= pdfTemp;
                scatteringCount++;
                lastSurface = false;
                if (scatteringCount >= maxDepth)
                    return result;
                // Trace a ray
                ray      = Ray(mRec.p, pRec.wo, 0.0);
                ray.mint = 0.0;
                depth++;
            } else {
                if (m_bsdfList[curLayer] == NULL) {
                    if (mRec.pdfFailure == 0)
                        return result;

                        //	pdf *= mRec.pdfFailure;
#if WHITEFURNACE == 0
                    throughput *= mRec.transmittance / mRec.pdfFailure;
#else
                    throughput *= 1;
                    // throughput *= mRec.transmittance / mRec.pdfFailure;//revised
#endif
                }


                // if hit a surface
                if (curLayer == 0 && ray.d.z > 0) // the ray have been go outside the media, return
                {
                    return result;
                }
                // go to the bottom of all the layers
                else if (curLayer == m_layerCount - 1 && ray.d.z < 0) {
                    return result;
                }

                {
                    // Point newP = ray.o + (Epsilon + travDist) * ray.d;
                    Point newP = ray.o + (0 + travDist) * ray.d;
                    ray        = Ray(newP, ray.d, 0.0);
                    ray.mint   = 0.0;
                }
            }
                
        }

        return Spectrum(0.0);
    }

	Spectrum generatePathSample(BSDFSamplingRecord& _bRec, float& pdf, const int maxDepth) const {
		ref_vector<Medium> mediums;
		//	setupMediaRuntime(mediums, _bRec.its.uv);
		setParameters(_bRec, mediums);
		int m_nbLayers = m_layerCount;
		Assert(_bRec.sampler);
		Sampler* sampler = _bRec.sampler;

		// Path tracing
		bool flag_incidentDir = _bRec.wi.z > 0;
		pdf = 1.f;

		Point originPoint =
			flag_incidentDir ? Point(0.0, 0.0, -Epsilon) : Point(0.0, 0.0, -m_thicknessAcc[m_layerCount] + Epsilon);
		// Point originPoint = flag_incidentDir ? Point(0.0, 0.0, -0) : Point(0.0, 0.0, -m_thicknessAcc[m_layerCount] +
		// 0);
		Ray                  ray(originPoint /*+ _bRec.wi*/, -_bRec.wi, 0.0);
		int                  scatteringCount = 0;
		Intersection         its;
		MediumSamplingRecord mRec;

		Spectrum throughput(1.0);
		//Spectrum result(0.0f);
		int      depth = 0;
		int      curLayer;

		int  itest = 0;
		bool lastSurface = false;
		while (depth <= maxDepth || maxDepth < 0) {
			if (ray.d.z == 0.0)
				return throughput;
			if (itest > 5)
				SLog(EDebug, "Issue here!");

			float curDepth = -ray.o.z;
			curLayer = locateLayer(m_thicknessAcc, curDepth); // math::floorToInt(-ray.o.z);

			// although we are always in the media, but it's possible the media sampling fails
			float travDist =
				ray.d.z < 0 ? m_thicknessAcc[curLayer + 1] - curDepth : curDepth - m_thicknessAcc[curLayer];
			travDist /= std::abs(ray.d.z);
			if (travDist == 0)
				return throughput;

			if (m_bsdfList[curLayer] == NULL &&
				(depth == 0) ? mediums[curLayer]->sampleDistance(Ray(ray, 0, travDist), mRec, sampler) : mediums[curLayer]->sampleDistanceC(Ray(ray, 0, travDist), mRec, sampler)) {

				int newLayer = locateLayer(m_thicknessAcc, -mRec.p.z); // math::floorToInt(-mRec.p.z);
				if (newLayer != curLayer) {
					return throughput;
				}
				Float s = mRec.t;
				Spectrum sigmaW = m_spectrum_sigmaWets[newLayer];
				//Float saturation = m_float_saturations[newLayer];
				Float saturation = m_flag_saturation[newLayer] ? getColorFromTexture(m_texture_saturation[newLayer], _bRec.its.uv)[0]
					: m_float_saturations[newLayer];
				//Float msl = m_float_minSL[newLayer];
				Float p = m_float_porositys[newLayer];
				Float n = m_float_ns[newLayer];
				Float v = (1 - p) / n;
				Float msl = power(3.f * v / 4.f / M_PI, 1.f / 3.f);
				Spectrum liquidAbs;
				if (s < msl)
					liquidAbs = Spectrum(1);
				else liquidAbs = exp(-(s - msl)*saturation*sigmaW);

				Spectrum albedo = mRec.sigmaS / (mRec.sigmaS + mRec.sigmaA) * mRec.transmittance / mRec.pdfSuccess*liquidAbs;

				throughput *= albedo;
				pdf *= mRec.pdfSuccess;
				if (pdf < Epsilon)
					return Spectrum(0.0);

				/*
				Medium*                     medium = mediums[curLayer];
				const PhaseFunction*        phase = medium->getPhaseFunction();
				PhaseFunctionSamplingRecord pRec(mRec, -ray.d);

				// if (maxDepth == 1 || depth > 0)
				if (depth > 0)
					// here we decide to scatter,
					// so we connect the current point with the out going direction
				{
					pRec.wo = _bRec.wo;

					if (!(pRec.wi + pRec.wo).isZero()) {
						Vector   h = normalize(pRec.wi + pRec.wo);
						Spectrum F = Specular_Fresnel_Schlick(m_F0[curLayer], std::max(0.0f, dot(h, pRec.wi)));
						Float    phaseValue;

						if (m_flag_wet[curLayer]) {
							Float saturation =
								m_flag_saturation[curLayer]
								? getColorFromTexture(m_texture_saturation[curLayer], _bRec.its.uv)[0]
								: m_float_saturations[curLayer];
							phaseValue = phase->eval_wet(pRec, saturation);
							//phaseValue = phase->eval(pRec);
						}
						else
							phaseValue = phase->eval(pRec);

						//{
						//	Float sigmaT = mediums[curLayer]->lookupSigmaT(pRec.wo);
						//	float distance = abs(mRec.p.z) / std::abs(pRec.wo.z);
						//}
						// this does not work for microflake, but works for SGGX
						Spectrum _attenation =
							computeLayredAttLayers(curLayer, -mRec.p.z, pRec.wo, _bRec.its.uv); // need revision
						//	float _attenation =	math::fastexp(-sigmaT*distance);
						// if (scatteringCount > 0)
						// return throughput * _attenation *phaseValue;
						if (m_flag_wet[curLayer]) {
							result += throughput * _attenation * phaseValue;
							// return Spectrum(phaseValue);
						}
						else
							result += throughput * F * _attenation * phaseValue;
						pdf += 1.0f;
					}
				}*/

				// Sample phase function
				Medium*                     medium = mediums[curLayer];
				const PhaseFunction*        phase = medium->getPhaseFunction();
				PhaseFunctionSamplingRecord pRec(mRec, -ray.d);
				Float phaseVal;
#if WHITEFURNACE
				phaseVal = phase->sample(pRec, sampler);
#else
				if (m_flag_wet[curLayer]) {
					Float saturation = m_flag_saturation[curLayer]
						? getColorFromTexture(m_texture_saturation[curLayer], _bRec.its.uv)[0]
						: m_float_saturations[curLayer];
					Float temp;
					phaseVal = phase->sample_wet(pRec, sampler, temp, saturation);
					//phaseVal = phase->sample(pRec, sampler);
				}
				else
					phaseVal = phase->sample(pRec, sampler);
#endif

				throughput *= phaseVal;
				pdf *= phase->pdf(pRec);
				if (pdf < Epsilon)
					return Spectrum(0.f);

				scatteringCount++;
				lastSurface = false;

				// Trace a ray
				ray = Ray(mRec.p, pRec.wo, 0.0);
				ray.mint = 0.0;
				depth++;
			}
			else {

				Spectrum _attenation =
					computeLayredAttLayers(curLayer, -mRec.p.z, ray.d, _bRec.its.uv, _bRec.its); // need revision
				throughput *= _attenation/mRec.pdfFailure;
				pdf *= mRec.pdfFailure;
				return throughput;
			}
		}

		return Spectrum(0.0);
	}

    void printDir(const Vector3& dir) { SLog(EDebug, "%f, %f, %f", dir.x, dir.y, dir.z); }


    void computeForWi(const Vector& wi, int maxDepth, std::string output) {
        printDir(wi);
        const int res = m_table_slice->getRes();
        // const int maxDepth = m_depth;
        const int          sampleCount = /*m_NEE ? 10000 : */ m_sampleCount * 10;
        std::vector<float> image;
        image.resize(res * res * 3, 0);

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
        std::vector<std::vector<float>> imageTemp;
        imageTemp.resize(tcount);
        for (int i = 0; i < tcount; i++) {
            imageTemp[i].resize(res * res * 3, 0);
        }

        if (!m_analytic) {

#pragma omp parallel for
            for (int k = 0; k < sampleCount; k++) {
#if defined(MTS_OPENMP)
                int tid = mts_omp_get_thread_num();
#else
                int tid = 0;
#endif
                Intersection its;
                its.wi = wi;
                BSDFSamplingRecord bRec(its, wi, ERadiance);
                bRec.sampler = samplers[tid]; // note that: set the sampler here.
                //	if (!m_NEE)
                {

                    Color3 value = generatePath(bRec, maxDepth);

                    Vector3 wo    = bRec.wo;
                    float   scale = 1.0f / (sampleCount * m_table_slice->getKernelSize(wo));

                    //	value *= abs(wo.z) * scale;
                    if (!value.isZero() && wo.z > 0) {
                        int idx = m_table_slice->dir2idx(wo);
                        if (idx != -1) {
                            m_table_slice->put(wo, value * scale /**abs(wo.z)*/);
                            idx                     *= 3;
                            imageTemp[tid][idx]     += value[0];
                            imageTemp[tid][idx + 1] += value[1];
                            imageTemp[tid][idx + 2] += value[2];
                        }
                    }
                }
            }

#pragma omp parallel for
            for (int k = 0; k < res * res * 3; k++) {
                for (int i = 0; i < tcount; i++)
                    image[k] += imageTemp[i][k];
            }
        } else {
            Intersection its;
            its.wi = wi;
            BSDFSamplingRecord bRec(its, wi, ERadiance);

            for (int oidx = 0; oidx < res * res; oidx++) {
                int io = oidx / res;
                int jo = oidx % res;

                Vector wo = m_table_slice->idx2dir(Point2i(io, jo), true);
                if (!wo.isZero()) {
                    bRec.wo        = wo;
                    Spectrum value = evalSingle(bRec);
                    m_table_slice->put(wo, value);
                    image[oidx * 3]     = value[0];
                    image[oidx * 3 + 1] = value[1];
                    image[oidx * 3 + 2] = value[2];
                }
            }
        }

        SaveEXRChannel3(image, res, res, output.c_str());
        cout << "Rendering to Image done!" << endl;
        std::vector<float>().swap(image);
    }

#if 0
	void update2Table(int iidx, const Spectrum &throughput, PhaseFunctionSamplingRecord &_mRec,
		float scale, int iLayer, float z)
	{
		int res = m_table->getRes();
		PhaseFunctionSamplingRecord temRec = _mRec;
		for (int oidx = 0; oidx < res * res; oidx++)
		{
			int io = oidx / res;
			int jo = oidx % res;

			Vector2 p2o = Point2((io + 0.5) / res, (jo + 0.5) / res) * 2 - Point2(1);
			if (isValid(p2o))
			{
				Vector wo = Vector3(p2o.x, p2o.y, sqrt(1 - p2o.x*p2o.x - p2o.y*p2o.y));
				temRec.wo = wo;
				Spectrum attenuation = computeLayredAtt(iLayer, z, wo);
				Spectrum phaseValue = m_phaseFunctions[iLayer]->eval(temRec) * attenuation;
				//the attenuation along the outgoing ray is not considered.
				m_table->put(iidx, wo, throughput*phaseValue*scale);

			}

		}

	}

	void update2Table(std::vector<float> &image, const Spectrum &throughput, PhaseFunctionSamplingRecord &_pRec,
		float scale, int iLayer, float z)
	{
		int res = sqrt(image.size() / 3);
		PhaseFunctionSamplingRecord temRec = _pRec;
		for (int oidx = 0; oidx < res * res; oidx++)
		{
			int io = oidx / res;
			int jo = oidx % res;

			Vector2 p2o = Point2((io + 0.5) / res, (jo + 0.5) / res) * 2 - Point2(1);
			if (isValid(p2o))
			{
				Vector wo = Vector3(p2o.x, p2o.y, sqrt(1 - p2o.x*p2o.x - p2o.y*p2o.y));
				temRec.wo = wo;
				Spectrum attenuation = computeLayredAtt(iLayer, z, wo);
				float phaseValue = m_phaseFunctions[iLayer]->eval(temRec);
				//the attenuation along the outgoing ray is not considered.
				Spectrum value = throughput * phaseValue * scale *attenuation;
				image[oidx * 3] += value[0];
				image[oidx * 3 + 1] += value[1];
				image[oidx * 3 + 2] += value[2];
			}
		}

	}

#endif

    void computeMutlipleTable(std::string path, std::string preName) {

        SLog(EDebug, "Do not find the file.");

        const int res         = m_table->getRes();
        const int maxDepth    = m_depth;
        const int sampleCount = /*m_NEE ? 100 : */ m_sampleCount; // 00;

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

        // for (int k = 0; k < 2;k++)
        {
#pragma omp parallel for
            for (int iidx = 0; iidx < res * res; iidx++) {
#if defined(MTS_OPENMP)
                int tid = mts_omp_get_thread_num();
#else
                int tid = 0;
#endif
                int    i  = iidx / res;
                int    j  = iidx % res;
                bool   up = true;
                Vector wi = m_table->idx2dir(Point2i(i, j), up);
                if (!wi.isZero()) {
                    Intersection its;
                    its.wi = wi;
                    BSDFSamplingRecord bRec(its, wi, ERadiance);
                    bRec.sampler = samplers[tid]; // note that: set the sampler here.
                    //	float scale = (float)res * res / sampleCount / 4.0f;
                    for (int k = 0; k < sampleCount; k++) {
                        // if (!m_NEE)
                        {
                            Color3  value = generatePath(bRec, maxDepth); // *scale;
                            Vector3 wo    = bRec.wo;
                            float   scale = 1.0f / (sampleCount * m_table->getKernelSize(wo));
                            if (!value.isZero()) {
                                // if (wo.z < 0)
                                //{
                                //	m_table->put(wi, wo, Spectrum(1.0));
                                // }
                                // else
                                m_table->put(wi, wo, value * scale /**abs(wo.z)*/);
                            }
                        }
                        // else
                        //	generatePathWithNEE(iidx, bRec, maxDepth, 1.0f / sampleCount);
                    }
					SLog(EDebug, "test.");
                }
            }
        }
		std::string prepath = "bsdfdata/";
        std::string outName = prepath + preName.append(".BSDFData");
		SLog(EDebug, "test2.");
        m_table->writeTable(outName);
		SLog(EDebug, "test3.");
    }

	void computeMutlipleTableMC(std::string path, std::string preName) {
		const int res = m_table->getRes();

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
		// for (int k = 0; k < 2;k++)
		{
#pragma omp parallel for
			for (int iidx = 0; iidx < res * res; iidx++) {
#if defined(MTS_OPENMP)
				int tid = mts_omp_get_thread_num();
#else
				int tid = 0;
#endif
				int    i = iidx / res;
				int    j = iidx % res;
				bool   up = true;
				Vector wi = m_table->idx2dir(Point2i(i, j), up);
				if (wi.isZero())
					continue;
				for (int oidx = 0; oidx < res * res; oidx++) {
					int    oi = oidx / res;
					int    oj = oidx % res;
					bool   up = true;
					Vector wo = m_table->idx2dir(Point2i(oi, oj), up);
					if (wo.isZero())
						continue;
					Intersection its;
					its.wi = wi;
					BSDFSamplingRecord bRec(its, wi, ERadiance);
					bRec.wo = wo;
					bRec.sampler = samplers[tid]; // note that: set the sampler here.
					int samples = 30;
					Spectrum result(0.f);
					for (int i = 0; i < samples; i++){
						result += msSimulationNEE(bRec, 0);// evalSingle(bRec); // eval_sheen_cos(wi, wo, m_vector_orientations[0],
						// m_thickness[0], m_spectrum_albedos[0], 0, !m_visUp);
					}
					result /= samples;
					Color3 cos(0.f);
					cos[0] = abs(wi.z);
					cos[1] = abs(wo.z);
					m_table->set(wi, wo, result);

					up = false;
					wo = m_table->idx2dir(Point2i(oi, oj), up);
					if (wo.isZero())
						continue;
					its.wi = wi;
					BSDFSamplingRecord bRec2(its, wi, ERadiance);
					bRec2.wo = wo;
					bRec2.sampler = samplers[tid];
					Spectrum result2(0.f);
					for (int i = 0; i < samples; i++){
						result2 += msSimulationNEE(bRec2, 0);// evalSingle(bRec); // eval_sheen_cos(wi, wo, m_vector_orientations[0],
						// m_thickness[0], m_spectrum_albedos[0], 0, !m_visUp);
					}
					result2 /= samples;
					//value = msSimulationNEE(bRec2, -1);// evalSingle(bRec2); // eval_sheen_cos(wi, wo, m_vector_orientations[0], m_thickness[0],
					// m_spectrum_albedos[0], 0, !m_visUp);
					//Color3 cos(0.f);
					cos[0] = abs(wi.z);
					cos[1] = abs(wo.z);
					m_table->set(wi, wo, result2);
					/*
					if (value.isZero()) {
					Color3 green(0.f);
					green[1] = 1;
					m_table->set(wi, wo, Spectrum(green));
					}
					else{
					if (value.isNaN())
					{
					Color3 blue(0.f);
					blue[2] = 1;
					m_table->set(wi, wo, Spectrum(blue));
					}
					else{
					//m_table->set(wi, wo, value);
					Color3 cos(0.f);
					cos[0] = abs(wi.z);
					cos[1] = abs(wo.z);
					m_table->set(wi, wo, Spectrum(cos));
					}
					}*/
				}
			}
		}

		std::string prepath = "D:/Pan/porousMedium/scenes/exp_aniMedium/bsdfdata/";
		std::string outName = prepath + preName.append(".BSDFData");
		m_table->writeTable(outName);
	}

	void computeMutlipleTablePhase(std::string path, std::string preName) {
		const int res = m_table->getRes();

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
		// for (int k = 0; k < 2;k++)
		{
#pragma omp parallel for
			for (int iidx = 0; iidx < res * res; iidx++) {
#if defined(MTS_OPENMP)
				int tid = mts_omp_get_thread_num();
#else
				int tid = 0;
#endif
				int    i = iidx / res;
				int    j = iidx % res;
				bool   up = true;
				Vector wi = m_table->idx2dir(Point2i(i, j), up);
				if (wi.isZero())
					continue;
				for (int oidx = 0; oidx < res * res; oidx++) {
					int    oi = oidx / res;
					int    oj = oidx % res;
					bool   up = true;
					Vector wo = m_table->idx2dir(Point2i(oi, oj), up);
					if (wo.isZero())
						continue;
					Intersection its;
					its.wi = wi;
					BSDFSamplingRecord bRec(its, wi, ERadiance);
					bRec.wo = wo;
					bRec.sampler = samplers[tid]; // note that: set the sampler here.
					int samples = 30;
					Spectrum result(0.f);
					for (int i = 0; i < samples; i++){
						result += msSimulationNEE(bRec, 0);// evalSingle(bRec); // eval_sheen_cos(wi, wo, m_vector_orientations[0],
						// m_thickness[0], m_spectrum_albedos[0], 0, !m_visUp);
					}
					result /= samples;
					//Color3 cos(0.f);
					//cos[0] = abs(wi.z);
					//cos[1] = abs(wo.z);
					m_table->set(wi, wo, result);

					MediumSamplingRecord mRec;
					mRec.orientation = Vector(0, 0, 1);
					PhaseFunctionSamplingRecord pRec(mRec, bRec.wi, bRec.wo);
					Float phaseValue = m_mediums[0]->getPhaseFunction()->eval_wet(pRec, m_float_saturations[0]);
					m_phaseTable->set(wi, wo, Spectrum(phaseValue));

					up = false;
					wo = m_table->idx2dir(Point2i(oi, oj), up);
					if (wo.isZero())
						continue;
					its.wi = wi;
					BSDFSamplingRecord bRec2(its, wi, ERadiance);
					bRec2.wo = wo;
					bRec2.sampler = samplers[tid];
					/*Spectrum result2(0.f);
					for (int i = 0; i < samples; i++){
						result2 += msSimulationNEE(bRec2, 0);// evalSingle(bRec); // eval_sheen_cos(wi, wo, m_vector_orientations[0],
						// m_thickness[0], m_spectrum_albedos[0], 0, !m_visUp);
					}
					result2 /= samples;
					//value = msSimulationNEE(bRec2, -1);// evalSingle(bRec2); // eval_sheen_cos(wi, wo, m_vector_orientations[0], m_thickness[0],
					// m_spectrum_albedos[0], 0, !m_visUp);
					//Color3 cos(0.f);
					cos[0] = abs(wi.z);
					cos[1] = abs(wo.z);
					m_table->set(wi, wo, result2);*/

					pRec.wi = wi;
					pRec.wo = wo;
					phaseValue = m_mediums[0]->getPhaseFunction()->eval_wet(pRec, m_float_saturations[0]);
					m_phaseTable->set(wi, wo, Spectrum(phaseValue));
					
				}
			}
		}

		std::string prepath = "bsdfdata/";
		//std::string outName = prepath + preName + ".BSDFData";
		//m_table->writeTable(outName);

		std::string outName = prepath + preName + ".PhaseData";
		m_phaseTable->writeTable(outName);
	}

    void computeMutlipleTableAnalytical(std::string path, std::string preName) {
        const int res = m_table->getRes();

        // for (int k = 0; k < 2;k++)
        {
#pragma omp parallel for
            for (int iidx = 0; iidx < res * res; iidx++) {
#if defined(MTS_OPENMP)
                int tid = mts_omp_get_thread_num();
#else
                int tid = 0;
#endif
                int    i  = iidx / res;
                int    j  = iidx % res;
                bool   up = true;
                Vector wi = m_table->idx2dir(Point2i(i, j), up);
                if (wi.isZero())
                    continue;
                for (int oidx = 0; oidx < res * res; oidx++) {
                    int    oi = oidx / res;
                    int    oj = oidx % res;
                    bool   up = true;
                    Vector wo = m_table->idx2dir(Point2i(oi, oj), up);
                    if (wo.isZero())
                        continue;
                    Intersection its;
                    its.wi = wi;
                    BSDFSamplingRecord bRec(its, wi, ERadiance);
                    bRec.wo        = wo;
					Spectrum value = evalSingle(bRec); // eval_sheen_cos(wi, wo, m_vector_orientations[0],
                                                       // m_thickness[0], m_spectrum_albedos[0], 0, !m_visUp);
					Color3 cos(0.f);
					cos[0] = abs(wi.z);
					cos[1] = abs(wo.z);
					m_table->set(wi, wo, value);
					/*
					if (!value.isZero()) {
                        m_table->set(wi, wo, value);
                    }
					else
						m_table->set(wi, wo, Spectrum(0.f));*/
                    up = false;
                    wo = m_table->idx2dir(Point2i(oi, oj), up);
                    if (wo.isZero())
                        continue;
                    its.wi = wi;
                    BSDFSamplingRecord bRec2(its, wi, ERadiance);
                    bRec2.wo = wo;
                    value    = evalSingle(bRec2); // eval_sheen_cos(wi, wo, m_vector_orientations[0], m_thickness[0],
                                                  // m_spectrum_albedos[0], 0, !m_visUp);
					//Color3 cos(0.f);
					cos[0] = abs(wi.z);
					cos[1] = abs(wo.z);
					m_table->set(wi, wo, value);
					/*
					if (value.isZero()) {
						Color3 green(0.f);
						green[1] = 1;
						m_table->set(wi, wo, Spectrum(green));
                    }
					else{
						if (value.isNaN())
						{
							Color3 blue(0.f);
							blue[2] = 1;
							m_table->set(wi, wo, Spectrum(blue));
						}
						else{
							//m_table->set(wi, wo, value);
							Color3 cos(0.f);
							cos[0] = abs(wi.z);
							cos[1] = abs(wo.z);
							m_table->set(wi, wo, Spectrum(cos));
						}
					}*/
                }
            }
        }

        std::string outName = preName.append("_analytical.BSDFData");
        m_table->writeTable(outName);
    }

    inline float sqr(float x) const { return x * x; }

    Spectrum sigma(const Vector& wi, float roughness, int iLayer) const {
        if (m_flag_aniso[iLayer]) {
            float value = m_phaseFunctions[iLayer]->sigma(wi, roughness);
            return Spectrum(value);
        } else {
            return m_spectrum_sigmaTs[iLayer];
        }
    }

    Spectrum sigmaWithO(const Vector& wi, float roughness, int iLayer, const Vector& orientation) const {
        if (m_flag_aniso[iLayer]) {
            Vector3 s, t;
            buildOrthonormalBasis(s, t, orientation);
            Frame orentation;
            orentation.n = orientation;
            orentation.s = s;
            orentation.t = t;
            Vector wi_   = orentation.toLocal(wi);


            float value = m_phaseFunctions[iLayer]->sigma(wi_, roughness);
            return Spectrum(value);
        } else {
            return m_spectrum_sigmaTs[iLayer];
        }
    }

    Spectrum D(const Vector& wi, const Vector& wo, float roughness, int iLayer) const {
        if (m_flag_aniso[iLayer]) {
            if (!(wi + wo).isZero()) {
                Vector3 half  = normalize(wi + wo);
                float   value = m_phaseFunctions[iLayer]->D(half, roughness);
                return Spectrum(value);
            } else
                return Spectrum(0.0);
        } else {
            MediumSamplingRecord        mRec;
            PhaseFunctionSamplingRecord pRec(mRec, wi);
            pRec.wo     = wo;
            float phase = m_phaseFunctions[iLayer]->eval(pRec, roughness);
            return phase * 4 * m_spectrum_sigmaTs[iLayer];
        }
    }

    Spectrum expm1(const Spectrum& input) const {
        Spectrum result;
        for (int i = 0; i < 3; i++)
            result[i] = std::expm1(input[i]);
        return result;
    }

    Spectrum exp(const Spectrum& input) const {
        Spectrum result;
        for (int i = 0; i < 3; i++)
            result[i] = math::fastexp(input[i]);
        return result;
    }
    double   power(double base, double exponent) const { return std::exp(exponent * std::log(base)); }
    Spectrum   power(Float base, Spectrum exponent) const { return exp(exponent * std::log(base)); }

    Spectrum exp(const Spectrum& base, const Float exponent) const {
        Spectrum result;
        for (int i = 0; i < 3; i++)
            result[i] = power(base[i], exponent);
        return result;
    }

    // PGL
	Spectrum eval_sheen_cos_wet(
		const Vector3&  wi_,
		const Vector3&  wo_,
		float           thickness,
		const Spectrum& albedo,
		int             iLayer,
		bool            transmission,
		float           saturation,
		const Vector&   orientation
    ) const {
		if (!m_flag_aniso[iLayer]){
			MediumSamplingRecord        mRec;
			PhaseFunctionSamplingRecord pRec(mRec, wi_, wo_);
			Float cosWi = abs(wi_.z);
			Float cosWo = abs(wo_.z);
			Float porosity = m_float_porositys[iLayer];
			Spectrum sigmaT = m_spectrum_sigmaTs[iLayer];
			Spectrum sigmaWet = m_spectrum_sigmaWets[iLayer];
			Float fp = m_mediums[iLayer]->getPhaseFunction()->eval_wet(pRec, saturation);
			Float EL, K;
			Float z = thickness;
			if (porosity == 1)
				K = 1;
			else{
				Float EL = 1.209 * pow(1 - porosity, 2.0 / 3.0);
				K = -log(1 - EL) / EL;
			}
			Spectrum SsigmaA, SsigmaS, SsigmaT, Salbedo;
			SsigmaA = sigmaT * (Spectrum(1.f) - albedo) * K + saturation * sigmaWet;
			SsigmaS = sigmaT * albedo * K;
			SsigmaT = SsigmaA + SsigmaS;
			Salbedo = SsigmaS / SsigmaT;

			if (!transmission){
				if ((cosWi + cosWo) == 0 || cosWi*cosWo == 0)
					return Spectrum(0.f);
				Spectrum term1 = Spectrum(1.f) - exp(-(cosWi + cosWo)*SsigmaT*z / (cosWi*cosWo));
				return term1*fp*Salbedo / (cosWi + cosWo)*cosWo;
			}
			else{
				if (cosWi == 0 || cosWo == 0)
					return Spectrum(0.f);
				if (abs(cosWi - cosWo)<Epsilon)
					return exp(-SsigmaT * z / cosWi) * z * fp * SsigmaS / (cosWo);
				Spectrum term1 = exp(-SsigmaT*z / cosWi) - exp(-SsigmaT*z / cosWo);
				return term1*fp*Salbedo / (cosWi - cosWo)*cosWo;
			}
		}
		else{
			//return Spectrum(0.f);
			Float cosWi = abs(wi_.z);
			Float cosWo = abs(wo_.z);
			MediumSamplingRecord        mRec;
			mRec.orientation = orientation;
			PhaseFunctionSamplingRecord pRec(mRec, wi_, wo_);
			Float fp = m_mediums[iLayer]->getPhaseFunction()->eval_wet(pRec, saturation);
			//return Spectrum(fp);
			Spectrum sigmaTI = Spectrum(m_mediums[iLayer]->getPhaseFunction()->sigmaDir(orientation, wi_));
			Spectrum sigmaTO = Spectrum(m_mediums[iLayer]->getPhaseFunction()->sigmaDir(orientation, wo_));

			Spectrum sigmaWet = m_spectrum_sigmaWets[iLayer];
			Float density = m_float_densities[iLayer];
			
			Float porosity = m_float_porositys[iLayer];
			Float EL, K;
			Float z = thickness;
			if (porosity == 1)
				K = 1;
			else{
				Float EL = 1.209 * pow(math::clamp((1 - porosity)* m_DsDv, 0.f, 0.75f), 2.0 / 3.0);
				K = -log(1 - EL) / EL;
			}
			// semi-infinit
            if(m_semiinf){
                if (transmission){
				    return Spectrum(0.f);
                }
                else{
                    /*
                    if ((cosWi + cosWo) == 0 || cosWi*cosWo == 0)
                        return Spectrum(0.f);
                    Float temp = std::min(cosWi, cosWo)*std::min(cosWi, cosWo)*pow(K, 1.f - std::max(cosWi / cosWo, cosWo / cosWi));
                    temp = temp / (cosWi + cosWo) / (1 + log(K));
                    temp = (std::min(cosWi, cosWo) - temp) / cosWi / cosWo;
                    //return albedo * fp / (cosWi + cosWo) * cosWo;
                    return albedo*fp*temp*cosWo;*/

                    Float S = saturation;
                    Float lnK = log(K);
                    Float mu_i = cosWi, mu_o = cosWo;
                    Spectrum alpha = albedo, sigma_l = sigmaWet, sigma_i = sigmaTI, sigma_o = sigmaTO;
                    Spectrum sigma_k_i = Spectrum(K*sigma_i) + S*sigma_l, sigma_k_o = Spectrum(K*sigma_o) + S*sigma_l;
                    Spectrum tmp1 = albedo*sigma_i*K*fp / (mu_i * sigma_k_o * (1 + lnK));
                    Spectrum tmp2 = mu_o * sigma_k_i * power(K, Spectrum(1.f) - mu_i*sigma_k_o / (mu_o*K*sigma_i) + S*sigma_l / (K*sigma_k_o)) / (mu_i*sigma_k_o + mu_o*sigma_k_i);
                    Spectrum tmp3 = (Spectrum(1 + lnK) - tmp2 + S*sigma_l*lnK / (sigma_o*K));

                    Spectrum tmp4 = albedo*sigma_i*K*fp / (mu_i * sigma_k_o * (1 + lnK));
                    Spectrum tmp5 = mu_o * sigma_k_i * power(K, Spectrum(1.f) - mu_i*sigma_k_o / (mu_o*K*sigma_i) + S*sigma_l / (K*sigma_k_o)) / (mu_i*sigma_k_o + mu_o*sigma_k_i);
                    Spectrum tmp6 = (Spectrum(1 + lnK) - tmp2 + S*sigma_l*lnK / (sigma_o*K));
                    Spectrum fr = tmp1*tmp3 + tmp4*tmp6;

                    return fr;
                }
            }
			
			
			
			Float mu_i, mu_o, Z = thickness, sigma_t_o, sigma_t_i, S = saturation;
			Spectrum alpha = albedo, sigma_t_l = sigmaWet;
			/**/
			if (sigmaTO[0] / cosWo < sigmaTI[0] / cosWi){
				mu_i = cosWi;
				mu_o = cosWo;
				sigma_t_o = sigmaTO[0];
				sigma_t_i = sigmaTI[0];
			}
			else{
				mu_i = cosWo;
				mu_o = cosWi;
				sigma_t_o = sigmaTI[0];
				sigma_t_i = sigmaTO[0];
			}

			if (transmission){
				if ((cosWi - cosWo) == 0 || cosWi*cosWo == 0)
					return Spectrum(0.f);
				if (mu_i == 0 || mu_o == 0)
					return Spectrum(0.f);

				Float lnK = log(K);
				Float lnK_KZ = lnK / (K * Z);
				Float sigmai_mui = sigma_t_i / mu_i;
				Float sigmao_muo = sigma_t_o / mu_o;
				Spectrum sSigmalPlusKSigmai = S*sigma_t_l + Spectrum(K*sigma_t_i);
				Spectrum sSigmalPlusKSigmao = S*sigma_t_l + Spectrum(K*sigma_t_o);
				Spectrum sSigmalPlusKSigmaiZ_mui = sSigmalPlusKSigmai*Z / mu_i;
				Spectrum sSigmalPlusKSigmaoZ_muo = sSigmalPlusKSigmao*Z / mu_o;
				Spectrum onePlusSSigmal_KSigmai = Spectrum(1) + S*sigma_t_l / (K*sigma_t_i);
				Spectrum onePlusSSigmal_KSigmao = Spectrum(1) + S*sigma_t_l / (K*sigma_t_o);
				Float case1 = sigma_t_i*sigma_t_o / (mu_i*sigma_t_o + mu_o*sigma_t_i);

				Spectrum A = (alpha * sigma_t_i * fp * K) / (mu_i * mu_o * (1.0 + lnK));

				// part1
				Spectrum term_case1(0.0);

				if (std::max(sigmao_muo, sigmai_mui) <= lnK_KZ) {
					term_case1 = Spectrum(Z);
				}
				else if (lnK_KZ > sigmai_mui && lnK_KZ < sigmao_muo) {
					term_case1 = Spectrum(mu_o*lnK / (K* sigma_t_o));
				}
				else if (lnK_KZ < sigmai_mui && lnK_KZ > sigmao_muo) {
					term_case1 = Spectrum(mu_i*lnK / (K* sigma_t_i));
					//term_case1 = Spectrum(mu_i / ( sigma_t_i));
				}
				else if (lnK_KZ > case1 && lnK_KZ <= std::min(sigmai_mui, sigmao_muo)) {
					term_case1 = Spectrum(lnK / K*(mu_i / sigma_t_i + mu_o / sigma_t_o) - Z);
				}
				else {
					term_case1 = Spectrum(0.0);
				}

				// part2
				Spectrum term_case2(0.0);

				if (lnK_KZ > case1 && lnK_KZ < sigmai_mui) {
					Spectrum tmp1 = Spectrum(1) - exp(-sSigmalPlusKSigmaiZ_mui)*powS(Spectrum(K), onePlusSSigmal_KSigmai);
					term_case2 = mu_i*tmp1 / (sSigmalPlusKSigmai);
				}
				else {
					term_case2 = Spectrum(0.f);
				}

				// part3
				Spectrum term_case3(0.0);

				if (lnK_KZ > case1 && lnK_KZ < sigmao_muo) {
					Spectrum tmp1 = Spectrum(1) - exp(-sSigmalPlusKSigmaoZ_muo)*powS(Spectrum(K), onePlusSSigmal_KSigmao);
					term_case3 = mu_o*tmp1 / (sSigmalPlusKSigmao);
				}
				else {
					term_case3 = Spectrum(0.f);
				}

				// part4
				Spectrum term_case4(0.0);

				if (lnK_KZ < case1) {
					if (mu_i != mu_o){
						Spectrum tmp1 = exp(-sSigmalPlusKSigmaiZ_mui)*powS(Spectrum(K), (mu_i + mu_o)*(sSigmalPlusKSigmai) / (mu_i*K*sigma_t_o));
						Spectrum tmp2 = exp(-sSigmalPlusKSigmaoZ_muo)*powS(Spectrum(K), (mu_i + mu_o)*(sSigmalPlusKSigmao) / (mu_o*K*sigma_t_i));
						term_case4 = mu_i*mu_o*(tmp1 - tmp2) / (mu_i*(sSigmalPlusKSigmao)-mu_o*(sSigmalPlusKSigmai));
					}

					Spectrum exp3 = onePlusSSigmal_KSigmai;
					Spectrum exp4 = (sSigmalPlusKSigmai)*mu_o / (mu_i*K*sigma_t_o);
					Spectrum tmp3 = mu_i*exp(-sSigmalPlusKSigmaiZ_mui + (exp3 + exp4)*lnK)
						- mu_i*exp(-sSigmalPlusKSigmaiZ_mui + exp3*lnK);
					//Spectrum tmp4 = powS(Spectrum(K), (S*sigma_t_l + Spectrum(K*sigma_t_i))*mu_o / (mu_i*K*sigma_t_o)) - Spectrum(1);

					Spectrum exp5 = onePlusSSigmal_KSigmao;
					Spectrum exp6 = (sSigmalPlusKSigmao)*mu_i / (mu_o*K*sigma_t_i);
					Spectrum tmp5 = mu_o*exp(-sSigmalPlusKSigmaoZ_muo + (exp5 + exp6)*lnK)
						- mu_o*exp(-sSigmalPlusKSigmaoZ_muo + exp5*lnK);
					//Spectrum tmp6 = powS(Spectrum(K), (S*sigma_t_l + Spectrum(K*sigma_t_o))*mu_i / (mu_o*K*sigma_t_i)) - Spectrum(1);
					term_case4 += tmp3 / (sSigmalPlusKSigmai)+tmp5 / (sSigmalPlusKSigmao);
				}
				else {
					term_case4 = Spectrum(0.0);
				}

				return A*(term_case1 + term_case2 + term_case3 + term_case4)*cosWo;
				//return Spectrum(0.f);
			}
			else{
				Float lnK = log(K);
				Float lnK_KZ = lnK / (K * Z);
				Float sigmai_mui = sigma_t_i / mu_i;
				Float sigmao_muo = sigma_t_o / mu_o;
				Spectrum sSigmalPlusKSigmai = S*sigma_t_l + Spectrum(K*sigma_t_i);
				Spectrum sSigmalPlusKSigmao = S*sigma_t_l + Spectrum(K*sigma_t_o);

				Spectrum A = (alpha * sigma_t_i * fp * K) / (mu_i * mu_o * (1.0 + lnK));

				// part1
				Spectrum term_case1(0.0);

				if (sigmao_muo < lnK_KZ && lnK_KZ < sigmai_mui) {
					Spectrum tmp = Spectrum(1.f) - exp(-(sSigmalPlusKSigmai)*Z / mu_i)*powS(Spectrum(K), sigma_t_l * S / K / sigma_t_i + Spectrum(1));
					term_case1 = mu_i*tmp / (sSigmalPlusKSigmai);
				}
				else if (lnK_KZ <= sigmao_muo) {
					Spectrum tmp = Spectrum(1.f) - powS(Spectrum(K), (sSigmalPlusKSigmai)*(mu_i*sigma_t_o - mu_o*sigma_t_i) / (mu_i*K*sigma_t_i*sigma_t_o));
					term_case1 = mu_i*tmp / (sSigmalPlusKSigmai);
				}
				else {
					term_case1 = Spectrum(0.0);
				}

				// part2
				Spectrum term_case2(0.0);

				if (lnK_KZ > sigmai_mui) {
					term_case2 = Spectrum(Z);
				}
				else if (lnK_KZ < sigmai_mui) {
					term_case2 = Spectrum(mu_i*lnK / (K*sigma_t_i));
				}
				else {
					term_case2 = Spectrum(0.0);
				}

				// part3
				Spectrum term_case3(0.0);

				if (lnK_KZ < sigmao_muo) {
					Spectrum tmp1 = mu_i*mu_o*powS(Spectrum(K), S*sigma_t_l*(sigma_t_i + sigma_t_o) / (K*sigma_t_i*sigma_t_o) + Spectrum(2.f));
					Spectrum tmp2 = powS(Spectrum(K), -Spectrum(1.f) - (S*sigma_t_l*(sigma_t_i + sigma_t_o) + Spectrum(mu_o*K*sigma_t_i)) / (K*mu_i*sigma_t_o));
					Spectrum tmp3 = exp(-(mu_o*(sSigmalPlusKSigmai)+mu_i*(sSigmalPlusKSigmao))*Z / (mu_o*mu_i));
					Spectrum tmp4 = mu_o*(sSigmalPlusKSigmai)+mu_i*(sSigmalPlusKSigmao);
					term_case3 = tmp1*(tmp2 - tmp3) / tmp4;
				}
				else {
					term_case3 = Spectrum(0.0);
				}

				return A*(term_case1 + term_case2 + term_case3)*cosWo;
			}
		}

		
    }

	Spectrum powS(const Spectrum &a, const Spectrum &b) const{
		Spectrum result(0.f);
		for (int i = 0; i < 3; i++){
			result[i] = pow(a[i], b[i]);
		}
		return result;
	}

    Spectrum eval_sheen_cos(
        const Vector3&  wi_,
        const Vector3&  wo_,
        const Vector&   orientation,
        float           roughness,
        float           thickness,
        const Spectrum& scale,
        int             iLayer,
        bool            transmission,
        const Spectrum& F0
    ) const { // return Color3(1, 0, 0);
        float idotn = wi_[2], rec_idotn = 1.f / (idotn);
        float odotn = wo_[2], rec_odotn = 1.f / (odotn);


        float    currentprod = thickness;
        Spectrum result(0.0f);
        /**/
        Vector3 s, t;
        buildOrthonormalBasis(s, t, orientation);
        Frame orentation;
        orentation.n  = orientation;
        orentation.s  = s;
        orentation.t  = t;
        Vector wi     = orentation.toLocal(wi_);
        Vector wo     = orentation.toLocal(wo_);
        float  scale_ = m_float_densities[0];
        if (!transmission) {
            // if (idotn > 0.f && odotn > 0.f) // reflection
#if 1
            Spectrum tmp =
                sigma(wi, roughness, iLayer) * abs(rec_idotn) + sigma(wo, roughness, iLayer) * abs(rec_odotn);
            Spectrum G     = 1 / (tmp);
            G             *= -expm1(-currentprod * tmp);
            Spectrum brdf  = boost * D(wi, wo, roughness, iLayer) * G * abs(rec_idotn) / 4.f;
            result         = scale * brdf;

#else

            MediumSamplingRecord mRec;
            // mRec.p = bRec.its.p;
            mRec.orientation = orientation;

            PhaseFunctionSamplingRecord pRec(mRec, wi_);
            pRec.wo      = wo_;
            Vector3 half = normalize(wi_ + wo_);
            float   D    = m_phaseFunctions[iLayer]->D(half, orientation);

            Spectrum sigma_i  = Spectrum(m_phaseFunctions[iLayer]->sigma(wi_, orientation));
            Spectrum sigma_o  = Spectrum(m_phaseFunctions[iLayer]->sigma(wo_, orientation));
            Spectrum tmp      = sigma_i * abs(rec_idotn) + sigma_o * abs(rec_odotn);
            Spectrum G        = 1 / tmp;
            G                *= -expm1(-currentprod * tmp);

            Spectrum brdf = boost * D * G * abs(rec_idotn) / 4.f;
            result        = scale * brdf;

#endif
        } else {

            Spectrum tmp =
                sigma(wi, roughness, iLayer) * abs(rec_idotn) - sigma(wo, roughness, iLayer) * abs(rec_odotn);

            Spectrum G(0.0);
            if (!tmp.isZero()) {
                G = 1 / tmp;
#if 0
				G *= -expm1(-currentprod * tmp);
				G *= exp(-currentprod* sigma(wo, roughness, iLayer) * abs(rec_odotn));
#else
                Spectrum temp3  = -currentprod * sigma(wo, roughness, iLayer) * abs(rec_odotn);
                Spectrum temp2  = (temp3[0] < -20) ? Spectrum(0.0f) : exp(temp3);
                G              *= -exp(-currentprod * tmp + temp3) + temp2;
#endif
            }
#if 0
			if (G[0] > 10)
			{
				int a = 0;
				Spectrum tmp1 = sigma(wi, roughness, iLayer) * abs(rec_idotn) - sigma(wo, roughness, iLayer) * abs(rec_odotn);

				Spectrum G1(0.0);
				if (!tmp1.isZero())
				{
					G1 = 1 / tmp1;
					//		G1 *= -expm1(-currentprod * tmp1);
					//		G1 *= exp(-currentprod* sigma(wo, roughness, iLayer) * abs(rec_odotn));
					Spectrum temp2 = exp(-currentprod* sigma(wo, roughness, iLayer)* abs(rec_odotn));
					G1 *= -exp(-currentprod * tmp1 - currentprod* sigma(wo, roughness, iLayer) * abs(rec_odotn)) + temp2;
					result = G1;
				}
			}
#endif
            else {
                G = currentprod * exp(-currentprod * sigma(wo, roughness, iLayer) * abs(rec_odotn));
            }

            Spectrum brdf = boost * D(wi, wo, roughness, iLayer) * G * abs(rec_idotn) / 4.f;
            result        = scale * brdf;
        }

        /* Fresnel term */
        {
            if (!(wi_ + wo_).isZero()) {
                float NdotL  = std::max(0.0f, dot(normalize(wi_ + wo_), wi_));
                result      *= Specular_Fresnel_Schlick(F0, NdotL);
            }
        }

        return result;
    }


    Spectrum eval_single_lambertian(
        const Vector3&  wi_,
        const Vector3&  wo_,
        const Vector&   orientation,
        float           roughness,
        float           thickness,
        const Spectrum& scale,
        int             iLayer,
        bool            transmission,
        const Spectrum& F0,
        float           w1,
        float           w2
    ) const { // return Color3(1, 0, 0);
        float idotn = wi_[2], rec_idotn = 1.f / (idotn);
        float odotn = wo_[2], rec_odotn = 1.f / (odotn);


        float    currentprod = thickness;
        Spectrum result(0.0f);
        /**/
        Vector3 s, t;
        buildOrthonormalBasis(s, t, orientation);
        Frame orentation;
        orentation.n  = orientation;
        orentation.s  = s;
        orentation.t  = t;
        Vector wi     = orentation.toLocal(wi_);
        Vector wo     = orentation.toLocal(wo_);
        float  scale_ = m_float_densities[0];
        if (!transmission) {
            // if (idotn > 0.f && odotn > 0.f) // reflection
#if 1
            Spectrum tmp =
                sigma(wi, roughness, iLayer) * abs(rec_idotn) + sigma(wo, roughness, iLayer) * abs(rec_odotn);
            Spectrum G  = 1 / (tmp);
            G          *= -expm1(-currentprod * tmp);

            Spectrum brdf = boost * D(wi, wo, roughness, iLayer) * G * abs(rec_idotn) / 4.f;
            result        = scale * brdf * w1;


#else

            MediumSamplingRecord mRec;
            // mRec.p = bRec.its.p;
            mRec.orientation = orientation;

            PhaseFunctionSamplingRecord pRec(mRec, wi_);
            pRec.wo      = wo_;
            Vector3 half = normalize(wi_ + wo_);
            float   D    = m_phaseFunctions[iLayer]->D(half, orientation);

            Spectrum sigma_i  = Spectrum(m_phaseFunctions[iLayer]->sigma(wi_, orientation));
            Spectrum sigma_o  = Spectrum(m_phaseFunctions[iLayer]->sigma(wo_, orientation));
            Spectrum tmp      = sigma_i * abs(rec_idotn) + sigma_o * abs(rec_odotn);
            Spectrum G        = 1 / tmp;
            G                *= -expm1(-currentprod * tmp);

            Spectrum brdf = boost * D * G * abs(rec_idotn) / 4.f;
            result        = scale * brdf;

#endif
        } else {

            Spectrum tmp =
                sigma(wi, roughness, iLayer) * abs(rec_idotn) - sigma(wo, roughness, iLayer) * abs(rec_odotn);

            Spectrum G(0.0);
            if (!tmp.isZero()) {
                G = 1 / tmp;
#if 0
				G *= -expm1(-currentprod * tmp);
				G *= exp(-currentprod* sigma(wo, roughness, iLayer) * abs(rec_odotn));
#else
                Spectrum temp3  = -currentprod * sigma(wo, roughness, iLayer) * abs(rec_odotn);
                Spectrum temp2  = (temp3[0] < -20) ? Spectrum(0.0f) : exp(temp3);
                G              *= -exp(-currentprod * tmp + temp3) + temp2;
#endif
            }
#if 0
			if (G[0] > 10)
			{
				int a = 0;
				Spectrum tmp1 = sigma(wi, roughness, iLayer) * abs(rec_idotn) - sigma(wo, roughness, iLayer) * abs(rec_odotn);

				Spectrum G1(0.0);
				if (!tmp1.isZero())
				{
					G1 = 1 / tmp1;
					//		G1 *= -expm1(-currentprod * tmp1);
					//		G1 *= exp(-currentprod* sigma(wo, roughness, iLayer) * abs(rec_odotn));
					Spectrum temp2 = exp(-currentprod* sigma(wo, roughness, iLayer)* abs(rec_odotn));
					G1 *= -exp(-currentprod * tmp1 - currentprod* sigma(wo, roughness, iLayer) * abs(rec_odotn)) + temp2;
					result = G1;
				}
			}
#endif
            else {
                G = currentprod * exp(-currentprod * sigma(wo, roughness, iLayer) * abs(rec_odotn));
            }

            Spectrum brdf = boost * D(wi, wo, roughness, iLayer) * G * abs(rec_idotn) / 4.f;
            result        = scale * brdf * w1;
        }

        /* Fresnel term */
        {
            if (!(wi_ + wo_).isZero()) {
                float NdotL  = std::max(0.0f, dot(normalize(wi_ + wo_), wi_));
                result      *= Specular_Fresnel_Schlick(F0, NdotL);
            }
        }


        /* w2*f2 */
        Spectrum result2(0.0f);
        Spectrum brdf = boost * D(wi, wo, roughness, iLayer) * abs(odotn) / 4.0f;
        result2       = scale * brdf * w2;
        result        = result + result2;


        return result;
    }

    // PGL
    Spectrum
		computeAttenuationWet(const Vector3& wi, const Vector3& wo, float thickness, int iLayer, float saturation, float roughness, const Vector3& orientation) const {
		if (!m_flag_aniso[iLayer]){
			float idotn = wi[2], rec_idotn = 1.f / idotn;
			float odotn = wo[2], rec_odotn = 1.f / odotn;
			Float K, EL;
			if (m_float_porositys[iLayer] == 1)
				K = 1;
			else {
				EL = 1.209 * pow(1 - m_float_porositys[iLayer], 2.0 / 3.0);
				K = -log(1 - EL) / EL;
			}
			// Spectrum sigmaT = Spectrum(EL / pow(m_float_ns[iLayer], -1.0 / 3.0));
			Spectrum sigmaT = m_spectrum_sigmaTs[iLayer];
			Spectrum sigmaWet = m_spectrum_sigmaWets[iLayer];
			Spectrum result = exp(-thickness * (rec_idotn + rec_odotn) * (sigmaT * K + sigmaWet * saturation));

			return result;
		}
		else{
			float idotn = wi[2], rec_idotn = 1.f / idotn;
			float odotn = wo[2], rec_odotn = 1.f / odotn;
			Spectrum sigmaWet = m_spectrum_sigmaWets[iLayer];
			Float porosity = m_float_porositys[iLayer];
			Float K;
			if (porosity == 1)
				K = 1;
			else {
				Float EL = 1.209 * pow(math::clamp((1 - porosity)* m_DsDv, 0.f, 0.75f), 2.0 / 3.0);
				K = -math::fastlog(1 - EL) / EL;
			}
			Spectrum sigmaTI = Spectrum(m_mediums[iLayer]->getPhaseFunction()->sigmaDir(orientation, wi));
			Spectrum sigmaTO = Spectrum(m_mediums[iLayer]->getPhaseFunction()->sigmaDir(orientation, wo));
			Spectrum result(1.f), tmp;
			
			Float msl = math::fastlog(K) / K / sigmaTO[0];
			Float t = thickness * rec_odotn;
			if (t < msl)
				tmp = Spectrum(1.f) - K*sigmaTO*t / (1 + math::fastlog(K));
			else
				tmp = K * exp(-t * Spectrum(sigmaTO) * K) / (1 + math::fastlog(K));
			result *= tmp*exp(-t * saturation * sigmaWet);

			msl = math::fastlog(K) / K / sigmaTI[0];
			t = thickness * rec_idotn;
			if (t < msl)
				tmp = Spectrum(1.f) - K*sigmaTI*t / (1 + math::fastlog(K));
			else
				tmp = K * exp(-t * Spectrum(sigmaTI) * K) / (1 + math::fastlog(K));
			result *= tmp*exp(-t * saturation * sigmaWet);

			return result;
		}
    }

    // PGL
	Spectrum computeAttenuationWet(const Vector3& wo, float thickness, int iLayer, float saturation, float roughness, const Vector3& orientation) const {
		if (!m_flag_aniso[iLayer]){
			float odotn = abs(wo[2]), rec_odotn = 1.f / odotn;
			Float K, EL;
			if (m_float_porositys[iLayer] == 1)
				K = 1;
			else {
				EL = 1.209 * pow(1 - m_float_porositys[iLayer], 2.0 / 3.0);
				K = -log(1 - EL) / EL;
			}
			Spectrum sigmaT = m_spectrum_sigmaTs[iLayer];
			Spectrum sigmaWet = m_spectrum_sigmaWets[iLayer];
			Spectrum result = exp(-thickness * rec_odotn * (sigmaT * K + sigmaWet * saturation));

			return result;
		}
		else{
			//Float DsDv = m_phaseFunctions[iLayer]->DsDv();
			Float porosity = m_float_porositys[iLayer];
			Float K;
			if (porosity == 1)
				K = 1;
			else {
				Float EL = 1.209 * pow(math::clamp((1 - porosity)* m_DsDv, 0.f, 0.75f), 2.0 / 3.0);
				K = -log(1 - EL) / EL;
			}
			float odotn = abs(wo[2]), rec_odotn = 1.f / odotn;
			Float sigmaT = m_mediums[iLayer]->getPhaseFunction()->sigmaDir(orientation, wo);
			Spectrum sigmaWet = m_spectrum_sigmaWets[iLayer];
			Float msl = log(K) / K / sigmaT;
			Spectrum Tc;
			Spectrum Tcl;
			if (thickness * rec_odotn < msl){
				Tc = Spectrum(1.f);
				Tcl = Spectrum(1.f);
			}
			else{
				Tc = Spectrum(K * math::fastexp(- K * sigmaT * thickness * rec_odotn));
				Tcl = Spectrum(exp(-sigmaWet * saturation * (thickness * rec_odotn - msl)));
				//result = exp(-(thickness * rec_odotn - msl) * (Spectrum(sigmaT) * K + sigmaWet * saturation));
			}
			return Tc * Tcl;
		}
    }

	Spectrum computeAttenuationWetUnc(const Vector3& wo, float thickness, int iLayer, float saturation, const Vector3& orientation) const {
		//Float DsDv = m_phaseFunctions[iLayer]->DsDv();
		Float porosity = m_float_porositys[iLayer];
		Float K;
		if (porosity == 1)
			K = 1;
		else {
			Float EL = 1.209 * pow(math::clamp((1 - porosity)* m_DsDv, 0.f, 0.75f), 2.0 / 3.0);
			K = -math::fastlog(1 - EL) / EL;
		}
		float odotn = abs(wo[2]), rec_odotn = 1.f / odotn;
		Float sigmaT = m_mediums[iLayer]->getPhaseFunction()->sigmaDir(orientation, wo);
		Spectrum sigmaWet = m_spectrum_sigmaWets[iLayer];
		Float msl = math::fastlog(K) / K / sigmaT;
		Spectrum result;
		Float t = thickness * rec_odotn;
		if (t < msl)
			result = Spectrum(1.f - K*sigmaT*t / (1 + math::fastlog(K)));
		    //result = Spectrum(1.f);
		else
			result = K * exp(- t * Spectrum(sigmaT) * K) / (1 + math::fastlog(K));
		    //result = Spectrum(K*math::fastexp(-K*sigmaT*t));
		return result*exp(- t * saturation * sigmaWet);
	}

    Spectrum computeAttenuation(
        const Vector3& wi,
        const Vector3& wo,
        float          roughness,
        float          thickness,
        int            iLayer,
        const Vector&  orientation
    ) const {
        float idotn = wi[2], rec_idotn = 1.f / idotn;
        float odotn = wo[2], rec_odotn = 1.f / odotn;

        Spectrum tmp = sigmaWithO(wi, roughness, iLayer, orientation) * abs(rec_idotn) +
                       sigmaWithO(wo, roughness, iLayer, orientation) * abs(rec_odotn);
        Spectrum result = exp(-thickness * tmp);

        return result;
    }

    Spectrum
    computeAttenuation(const Vector3& wo, float roughness, float thickness, int iLayer, const Vector& orientation)
        const {
        float    odotn = wo[2], rec_odotn = 1.f / odotn;
        Spectrum tmp    = sigmaWithO(wo, roughness, iLayer, orientation) * abs(rec_odotn);
        Spectrum result = exp(-thickness * tmp);
        return result;
    }


    void buildOrthonormalBasis(Vector3& omega_1, Vector3& omega_2, const Vector3& omega_3) const {
        if (omega_3.z < -0.9999999f) {
            omega_1 = Vector3(0.0f, -1.0f, 0.0f);
            omega_2 = Vector3(-1.0f, 0.0f, 0.0f);
        } else {
            const float a = 1.0f / (1.0f + omega_3.z);
            const float b = -omega_3.x * omega_3.y * a;
            omega_1       = Vector3(1.0f - omega_3.x * omega_3.x * a, b, -omega_3.x);
            omega_2       = Vector3(b, 1.0f - omega_3.y * omega_3.y * a, -omega_3.y);
        }
    }

    //Vector getOrientationFromTexture(const ref<Texture2D> orien_texture, const Point2& uv, float scale) const {
	Vector getOrientationFromTexture(const ref<Texture> orien_texture, const Intersection& its, float scale) const {
        Vector orien;

        //orien_texture->eval(uv*10).toLinearRGB(orien.x, orien.y, orien.z);//mark
		orien_texture->eval(its).toLinearRGB(orien.x, orien.y, orien.z);//mark

        for (int i = 0; i < 3; ++i) {
            // if (normal[i] > 1 || normal[i] < 0) {
            //	cout << "[GY]: Warning in MultiLayeredBSDF::getNormalFromTexture()" << endl;
            //	cout << "normal: " << normal.toString() << endl;
            // }
            // note this is used for others (leaf, window shade, fiber etc), but not for the wood scene
            orien[i] = Float(2.0) * orien[i] - Float(1.0);
            // orien = normalize(orien);
            orien[i] = orien[i] > Float(1.0) - Epsilon ? Float(1.0) : orien[i];
            orien[i] = orien[i] < Float(-1.0) + Epsilon ? Float(-1.0) : orien[i];
        }
        if (scale != 1.0) {
            orien    /= orien[2];
            orien[0] *= scale;
            orien[1] *= scale;
        }
        return normalize(orien);
    }

    Spectrum getColorFromTexture(const ref<Texture2D> texture, const Point2& uv) const {
        Spectrum albedo = texture->eval(uv);
        return albedo;
    }

	Normal getNormalFromTexture(const ref<Texture2D> texture, const Point2& uv) const {
		Normal normal;
		texture->eval(uv).toLinearRGB(normal.x, normal.y, normal.z);
		return normalize(normal);
	}


    Spectrum Specular_Fresnel_Schlick(const Spectrum& SpecularColor, const float dotLH) const {
        return SpecularColor + (Spectrum(1.0f) - SpecularColor) * pow((1 - dotLH), 5);
    }


    Spectrum evalSingle(const BSDFSamplingRecord& bRec) const {

        Spectrum attenuation(1.0);
        bool     transmission = Frame::cosTheta(bRec.wi) * Frame::cosTheta(bRec.wo) < 0;
        Spectrum result(0.0);
        float    thicknessList[MAX_LAYER_COUNT];
        float    roughnessList[MAX_LAYER_COUNT];
        Vector   orientationList[MAX_LAYER_COUNT];
        Spectrum abledoList[MAX_LAYER_COUNT];
        // PGL
        float saturationList[MAX_LAYER_COUNT];

        // debug
        /*
        if (transmission){
            return Spectrum(0.1f);
        }
        else return Spectrum(1.f);*/

        // texture query
        for (int k = 0; k < m_layerCount; k++) {
            orientationList[k] = m_flag_orientations[k]
                                     ? getOrientationFromTexture(m_texture_orientations[k], bRec.its, m_scale_n[k])
                                     : m_vector_orientations[k];
            roughnessList[k]   = m_roughness[k];
            abledoList[k] =
                m_flag_albedos[k] ? getColorFromTexture(m_texture_albedos[k], bRec.its.uv) : m_spectrum_albedos[k];
            thicknessList[k] =
                m_flag_thickness[k] ? getColorFromTexture(m_texture_thickness[k], bRec.its.uv)[0] : m_thickness[k];
            saturationList[k] = m_flag_saturation[k] ? getColorFromTexture(m_texture_saturation[k], bRec.its.uv)[0]
                                                     : m_float_saturations[k];
        }

        Vector3 wi = bRec.wi;
        Vector3 wo = bRec.wo;

        for (int i = 0; i < m_layerCount; i++) {

            Spectrum layerContribution(0.0f);
            int      iLayer = i;

            if (transmission) {
                attenuation = Spectrum(1.0f);
                // PGL
                for (int k = i + 1; k < m_layerCount; k++) {
                    if (m_flag_wet[iLayer])
						if (m_flag_aniso[iLayer])
							attenuation *= computeAttenuationWetUnc(
							(Frame::cosTheta(wi) < 0) ? wi : wo, m_thickness[k], k, saturationList[k], orientationList[k]
							);
                        else
							attenuation *= computeAttenuationWet(
						(Frame::cosTheta(wi) < 0) ? wi : wo, m_thickness[k], k, saturationList[k], roughnessList[k], orientationList[k]
                        );
                    else
                        attenuation *= computeAttenuation(
                            (Frame::cosTheta(wi) < 0) ? wi : wo, roughnessList[k], m_thickness[k], k, orientationList[k]
                        );
                }
                for (int k = 0; k < i; k++) {
                    if (m_flag_wet[iLayer])
						if (m_flag_aniso[iLayer])
							attenuation *= computeAttenuationWetUnc(
							(Frame::cosTheta(wi) < 0) ? wo : wi, m_thickness[k], k, saturationList[k], orientationList[k]
							);
						else
                        attenuation *= computeAttenuationWet(
						(Frame::cosTheta(wi) < 0) ? wo : wi, m_thickness[k], k, saturationList[k], roughnessList[k], orientationList[k]
                        );
                    else
                        attenuation *= computeAttenuation(
                            (Frame::cosTheta(wi) < 0) ? wo : wi, roughnessList[k], m_thickness[k], k, orientationList[k]
                        );
                }
            }
            if (attenuation.getLuminance() < 1e-4)
                break;

            if (!transmission) {
                if (Frame::cosTheta(bRec.wi) < 0) //(bRec.bounces == -1)
                {
                    iLayer = m_layerCount - 1 - i;
                }
            }

            // this is for leaf
            float         thickness   = thicknessList[iLayer];
            float         roughness   = roughnessList[iLayer];
            const Vector& orientation = orientationList[iLayer];

            if (m_flag_thickness[iLayer]) {
                thickness *= thickness * thickness;
                thickness *= m_scale_n[iLayer];
            }

            if (m_bsdfList[iLayer] == NULL) {
#if WHITEFURNACE
                attenuation = Spectrum(1);
#endif
                // PGL
				if (m_flag_wet[iLayer])
					layerContribution =
					eval_sheen_cos_wet(
					wi, wo, thickness, abledoList[iLayer], iLayer, transmission, saturationList[iLayer], orientation
					) *
					attenuation;
				else
					layerContribution = eval_sheen_cos(
                                            wi,
                                            wo,
                                            orientation,
                                            roughness,
                                            thickness,
                                            abledoList[iLayer],
                                            iLayer,
                                            transmission,
                                            m_F0[iLayer]
                                        ) *
                                        attenuation;

                if (!transmission) {
                    // PGL
					if (m_flag_wet[iLayer])
						attenuation *=
							computeAttenuationWet(bRec.wi, bRec.wo, thickness, iLayer, saturationList[iLayer], roughness, orientation);
                    else
                        attenuation *= computeAttenuation(bRec.wi, bRec.wo, roughness, thickness, iLayer, orientation);
                }
            } else {
                layerContribution = m_bsdfList[iLayer]->eval(bRec) /** abs(bRec.wo.z)*/ * attenuation;
                if (!transmission) {
                    attenuation *= 0;
                }
            }

#if WHITEFURNACE
            result += Spectrum(layerContribution.average());
#else

            result += Spectrum(layerContribution) * m_scaling[iLayer];
#endif
        }
        return result;
    }

    Spectrum evalSingleWithPdf(BSDFSamplingRecord& bRec, float& pdf) const {

        Spectrum attenuation(1.0);
        bool     transmission = Frame::cosTheta(bRec.wi) * Frame::cosTheta(bRec.wo) < 0;
        Spectrum result(0.0);
        float    pdfSum_ = 0.0;
        for (int ii = 0; ii < m_layerCount; ii++) {
            int i = ii;

            Spectrum layerContribution(0.0f);
            int      iLayer    = i;
            Vector3  wi        = bRec.wi;
            Vector3  wo        = bRec.wo;
            float    roughness = m_roughness[iLayer];
            if (transmission) {
                attenuation = Spectrum(1.0f);
                for (int k = i + 1; k < m_layerCount; k++) {
                    Vector orientation = m_vector_orientations[k];
                    if (m_flag_orientations[k]) {
                        orientation = getOrientationFromTexture(m_texture_orientations[k], bRec.its, m_scale_n[k]);
                    }

                    float roughness_k  = m_roughness[k];
                    attenuation       *= computeAttenuation(
                        (Frame::cosTheta(bRec.wi) < 0) ? bRec.wi : bRec.wo, roughness_k, m_thickness[k], k, orientation
                    );
                }
                for (int k = 0; k < i; k++) {
                    Vector orientation = m_vector_orientations[k];
                    if (m_flag_orientations[k]) {
                        orientation = getOrientationFromTexture(m_texture_orientations[k], bRec.its, m_scale_n[k]);
                    }
                    float roughness_k  = m_roughness[k];
                    attenuation       *= computeAttenuation(
                        (Frame::cosTheta(bRec.wi) < 0) ? bRec.wo : bRec.wi, roughness_k, m_thickness[k], k, orientation
                    );
                }
            }
            if (attenuation.getLuminance() < 1e-4)
                break;

            if (!transmission) {
                if (Frame::cosTheta(bRec.wi) < 0) //(bRec.bounces == -1)
                {
                    iLayer = m_layerCount - 1 - i;
                }
            }

            Vector orientation = m_vector_orientations[iLayer];
            if (m_flag_orientations[iLayer]) {
                orientation = getOrientationFromTexture(m_texture_orientations[iLayer], bRec.its, m_scale_n[iLayer]);
            }

            Spectrum abledo = m_flag_albedos[iLayer] ? getColorFromTexture(m_texture_albedos[iLayer], bRec.its.uv)
                                                     : m_spectrum_albedos[iLayer];

            float thickness = m_flag_thickness[iLayer]
                                  ? getColorFromTexture(m_texture_thickness[iLayer], bRec.its.uv)[0]
                                  : m_thickness[iLayer];

            // this is for leaf
            if (m_flag_thickness[iLayer]) {
                thickness *= thickness * thickness;
                thickness *= m_scale_n[iLayer];
            }

            if (m_bsdfList[iLayer] == NULL) {
#if WHITEFURNACE
                attenuation = Spectrum(1);
#endif
                layerContribution =
                    eval_sheen_cos(
                        wi, wo, orientation, roughness, thickness, abledo, iLayer, transmission, m_F0[iLayer]
                    ) *
                    attenuation;

                if (!transmission) {
                    attenuation *= computeAttenuation(bRec.wi, bRec.wo, roughness, thickness, iLayer, orientation);
                }
                MediumSamplingRecord mRec;
                mRec.p           = bRec.its.p;
                mRec.orientation = orientation;
                PhaseFunctionSamplingRecord pRec(mRec, bRec.wi);
                pRec.wo     = bRec.wo;
                float pdf_  = m_phaseFunctions[iLayer]->eval(pRec);
                pdfSum_    += pdf_;

            } else {
                layerContribution = m_bsdfList[iLayer]->eval(bRec) /** abs(bRec.wo.z)*/ * attenuation;
                if (!transmission) {
                    attenuation *= 0;
                }
                float pdf_  = m_bsdfList[iLayer]->pdf(bRec);
                pdfSum_    += pdf_;
            }

#if WHITEFURNACE
            result += Spectrum(layerContribution.average());
#else

            result += Spectrum(layerContribution) * m_scaling[iLayer];
#endif
        }
#if DELTA_BSDF
		Spectrum att = this->attenuation(bRec.wi, bRec.its.uv, bRec.its);

        float _pdf_through = att.getLuminance();
        if (bRec.sampledComponent == -2) {
            if (_pdf_through < 1e-10)
                _pdf_through = 0.0f;
            return att; // _pdf_through;
        }
#else
        float _pdf_through = 0.0f;
#endif
        pdf = pdfSum_ * (1 - _pdf_through) / m_layerCount;
        return result;
    }


    Spectrum evalMappedSingle(const BSDFSamplingRecord& bRec) const {
        // this is used for reflection only!!
        Point2   uv2 = bRec.its.uv;
        Point2   uv(uv2.y, uv2.x);
        Spectrum attenuation(1.0);
        bool     transmission = Frame::cosTheta(bRec.wi) * Frame::cosTheta(bRec.wo) < 0;
        Spectrum result(0.0);

        int idx = 0; // (bRec.wi.z < 0 ? 2 : 0) + (bRec.wo.z < 0 ? 1 : 0);  //

        Vector3 wi = bRec.wi;
        Vector3 wo = bRec.wo;

        for (int i = 0; i < m_layerCount; i++) {
            if (m_texture_albedos_m[idx][i] == NULL)
                continue;

            Spectrum layerContribution(0.0f);
            int      iLayer = i;

            if (transmission) {
                attenuation = Spectrum(1.0f);
                for (int k = i + 1; k < m_layerCount; k++) {
                    if (m_bsdfList_m[idx][k] == NULL) {
                        if (m_texture_albedos_m[idx][k] == NULL)
                            continue;
                        Spectrum rough_thickness = getColorFromTexture(m_roughness_m[idx][k], uv);

                        float  thickness_k = rough_thickness[1];
                        float  roughnessK  = rough_thickness[0];
                        Vector orientation = m_vector_orientations[k];
                        if (m_flag_orientations[k]) {
							orientation = getOrientationFromTexture(m_texture_orientations[k], bRec.its, m_scale_n[k]);
                        }

                        attenuation *= computeAttenuation(
                            (Frame::cosTheta(wi) < 0) ? wi : bRec.wo, roughnessK, thickness_k, k, orientation
                        );
                        //	attenuation *= computeAttenuation(wo, roughnessK, thickness_k, k, orientation);
                    } else
                        attenuation = Spectrum(0.0);
                }
                for (int k = 0; k < i; k++) {
                    if (m_bsdfList_m[idx][k] == NULL) {
                        if (m_texture_albedos_m[idx][k] == NULL)
                            continue;
                        Spectrum rough_thickness = getColorFromTexture(m_roughness_m[idx][k], uv);
                        float    thickness_k     = rough_thickness[1];
                        float    roughnessK      = rough_thickness[0];

                        Vector orientation = m_vector_orientations[k];
                        if (m_flag_orientations[k]) {
							orientation = getOrientationFromTexture(m_texture_orientations[k], bRec.its, m_scale_n[k]);
                        }
                        attenuation *= computeAttenuation(
                            (Frame::cosTheta(wi) < 0) ? wo : wi, roughnessK, thickness_k, k, orientation
                        );
                        //	attenuation *= computeAttenuation(wi, roughnessK, thickness_k, k, orientation);
                    } else
                        attenuation = Spectrum(0.0);
                }
            }

            if (m_bsdfList[iLayer] == NULL) {
                Spectrum w1w2            = getColorFromTexture(m_w1w2_m[idx][iLayer], uv);
                float    w1              = w1w2[0];
                float    w2              = w1w2[1];
                Spectrum rough_thickness = getColorFromTexture(m_roughness_m[idx][iLayer], uv);
                float    thickness       = rough_thickness[1];
                float    roughness       = rough_thickness[0];
                Vector   orientation     = m_vector_orientations[iLayer];
                if (m_flag_orientations[iLayer]) {
                    orientation = getOrientationFromTexture(m_texture_orientations[iLayer], bRec.its, m_scale_n[iLayer]);
                }

                Spectrum f0 = getColorFromTexture(m_texture_F0_m[idx][iLayer], uv);

                Spectrum albedo = getColorFromTexture(m_texture_albedos_m[idx][iLayer], uv);
                layerContribution =
                    eval_single_lambertian(
                        wi, wo, orientation, roughness, thickness, albedo, iLayer, transmission, f0, w1, w2
                    ) *
                    attenuation;

                if (!transmission) {
                    attenuation *= computeAttenuation(bRec.wi, bRec.wo, roughness, thickness, iLayer, orientation);
                }
            } else {
                layerContribution = m_bsdfList_m[idx][iLayer]->eval(bRec) * attenuation;
                if (!transmission) {
                    attenuation *= 0;
                }
            }

            if (!m_flag_multipleIgnore[iLayer])
                result += layerContribution * m_scaling[iLayer];
        }

        return result;
    }

    Spectrum evalMappedSingleTwoSide(const BSDFSamplingRecord& bRec) const {

        Point2   uv2 = bRec.its.uv;
        Point2   uv(uv2.y, uv2.x);
        Spectrum attenuation(1.0);
        bool     transmission = Frame::cosTheta(bRec.wi) * Frame::cosTheta(bRec.wo) < 0;
        Spectrum result(0.0);

        int idx = (bRec.wi.z < 0 ? 2 : 0) + (bRec.wo.z < 0 ? 1 : 0); //

        Vector3 wi = bRec.wi;
        Vector3 wo = bRec.wo;

        float  thicknessList[MAX_LAYER_COUNT];
        float  roughnessList[MAX_LAYER_COUNT];
        Vector orientationList[MAX_LAYER_COUNT];

        for (int k = 0; k < m_layerCount; k++) {
            Spectrum rough_thickness = getColorFromTexture(m_roughness_m[idx][k], uv);
            thicknessList[k]         = rough_thickness[1];
            roughnessList[k]         = rough_thickness[0];
            int k_org                = (idx > 1) ? m_layerCount - 1 - k : k;
            orientationList[k]       = m_flag_orientations[k_org]
				? getOrientationFromTexture(m_texture_orientations[k_org], bRec.its, m_scale_n[k_org])
                                           : m_vector_orientations[k_org];
        }

        if (m_useTwoSide) {
            if (idx > 1) {
                wi = -wi;
                wo = -wo;
            }
        }

        for (int i = 0; i < m_layerCount; i++) {
            if (m_texture_albedos_m[idx][i] == NULL)
                continue;

            Spectrum layerContribution(0.0f);
            int      iLayer = i;

            if (transmission) {
                attenuation = Spectrum(1.0f);
                for (int k = 0; k < m_layerCount; k++) {
                    if (k == i)
                        continue;
                    Vector dir = k < i ? wi : wo;
                    if (m_bsdfList_m[idx][k] == NULL) {
                        if (m_texture_albedos_m[idx][k] == NULL)
                            continue;
                        int k_org = (idx > 1) ? m_layerCount - 1 - k : k;
                        attenuation *=
                            computeAttenuation(dir, roughnessList[k], thicknessList[k], k_org, orientationList[k]);
                    } else
                        attenuation = Spectrum(0.0);
                }
            }

            int k_org = (idx > 1) ? m_layerCount - 1 - iLayer : iLayer;

            if (m_bsdfList[iLayer] == NULL) {

                float         thickness   = thicknessList[iLayer];
                float         roughness   = roughnessList[iLayer];
                const Vector& orientation = orientationList[iLayer];

                Spectrum f0     = getColorFromTexture(m_texture_F0_m[idx][iLayer], uv);
                Spectrum albedo = getColorFromTexture(m_texture_albedos_m[idx][iLayer], uv);
                layerContribution =
                    eval_sheen_cos(wi, wo, orientation, roughness, thickness, albedo, iLayer, transmission, f0) *
                    attenuation;

                if (!transmission) {
                    attenuation *= computeAttenuation(wi, wo, roughness, thickness, k_org, orientation);
                }
            } else {
                layerContribution = m_bsdfList_m[idx][iLayer]->eval(bRec) * attenuation;
                if (!transmission) {
                    attenuation *= 0;
                }
            }

            if (!m_flag_multipleIgnore[k_org])
                result += layerContribution * m_scaling[k_org];
        }

        return result;
    }

    /// Reflection in local coordinates
    inline Vector reflect(const Vector& wi) const { return Vector(-wi.x, -wi.y, wi.z); }

    /// Refraction in local coordinates
    inline Vector refract(const Vector& wi, Float cosThetaT, Float invEta, Float eta) const {
        Float scale = -(cosThetaT < 0 ? invEta : eta);
        return Vector(scale * wi.x, scale * wi.y, cosThetaT);
    }

    Spectrum clamp(Spectrum x, Float lo, Float hi) const {
        Spectrum result;
        for (int i = 0; i < 3; i++) {
            result[i] = math::clamp(x[i], lo, hi);
        }
        return result;
    }

    Spectrum sqrtS(Spectrum x) const {
        Spectrum result;
        for (int i = 0; i < 3; i++) {
            result[i] = sqrt(x[i]);
        }
        return result;
    }

	Float eval_fp_gs(const BSDFSamplingRecord& bRec, Float wg1, Float wg2, Float mu1, Float mu2, Float sigma1, Float sigma2) const{
		Float cosTheta = dot(bRec.wi, bRec.wo);
		Float theta = std::acos(cosTheta);
		return (wg1 / (sqrt(2 * M_PI) * sigma1) * std::exp(-(theta - mu1)*(theta - mu1) / (2 * sigma1*sigma1)) +
			wg2 / (sqrt(2 * M_PI) * sigma2) * std::exp(-(theta - mu2)*(theta - mu2) / (2 * sigma2 *sigma2)));
	}

	Float eval_fp_hg(BSDFSamplingRecord& bRec, Float whg, Float g1, Float g2) const{
		Float cosTheta = dot(bRec.wi, bRec.wo);
		Float temp = 1 + g1 * g1 + 2.0 * g1 * cosTheta;
		Float hg1 = 0.25 / M_PI * (1 - g1*g1) / (temp * sqrt(temp));
		temp = 1 + g2 * g2 + 2.0 * g2 * cosTheta;
		Float hg2 = 0.25 / M_PI * (1 - g2*g2) / (temp * sqrt(temp));
		return whg * hg1 + (1 - whg) * hg2;
	}

	Float evalPhase(PhaseFunctionSamplingRecord& pRec, bool trans) const{
		Frame orient(pRec.mRec.orientation);
		Vector3 wi = orient.toLocal(pRec.wi);
		Vector3 wo = orient.toLocal(pRec.wo);
		/*
		Float thetaIndexI, phiIndexI, thetaIndexO, phiIndexO;
		wToIndexPhase(wi, thetaIndexI, phiIndexI);
		wToIndexPhase(wo, thetaIndexO, phiIndexO);
		int x1 = math::clamp(static_cast<int>(std::floor(thetaIndexI)), 0, phaseCount - 1);
		int y1 = math::clamp(static_cast<int>(std::floor(phiIndexI)), 0, phaseCount - 1);
		int x2 = math::clamp(static_cast<int>(std::floor(thetaIndexO)), 0, phaseCount - 1);
		int y2 = math::clamp(static_cast<int>(std::floor(phiIndexO)), 0, phaseCount - 1);
		// world
		y2 -= y1;
		if (y2 < 0)
			y2 += phaseCount;

		if (trans)
			return m_multiplePhaseTable_down[x1][x2][y2];
		else 
			return m_multiplePhaseTable[x1][x2][y2];
			*/
		Float thetaIndexI, phiIndexI, thetaIndexO, phiIndexO;
		wToIndexPhase(wi, thetaIndexI, phiIndexI);
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
		if (trans){
			tmp0 = m_multiplePhaseTable_down[x1][x2][y2];
			tmp1 = m_multiplePhaseTable_down[x1 + 1][x2][y2];
		}
		else{
			tmp0 = m_multiplePhaseTable[x1][x2][y2];
			tmp1 = m_multiplePhaseTable[x1 + 1][x2][y2];
		}
		return (1 - thetaIndexI + x1)*tmp0 + (thetaIndexI - x1)*tmp1;
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

	float sigma(Vector3 wi,
		float S_xx, float S_yy, float S_zz,
		float S_xy, float S_xz, float S_yz) const
	{
		const float sigma_squared = wi.x*wi.x*S_xx + wi.y*wi.y*S_yy + wi.z*wi.z*S_zz
			+ 2.0f * (wi.x*wi.y*S_xy + wi.x*wi.z*S_xz + wi.y*wi.z*S_yz);
		return (sigma_squared > 0.0f) ? sqrtf(sigma_squared) : 0.0f; // conditional to avoid numerical errors
	}

	Float getSigma(Vector& orientation, Vector& wi, bool trans) const{
		Frame orient(orientation);

		Float Sxx, Sxy, Sxz, Syy, Syz, Szz;
		if (trans)
			computeS(Sxx, Sxy, Sxz, Syy, Syz, Szz, orientation, m_mediums[0]->getPhaseFunction()->getProperties().getBoolean("fiber", false), m_net_roughness_down);
		else
			computeS(Sxx, Sxy, Sxz, Syy, Syz, Szz, orientation, m_mediums[0]->getPhaseFunction()->getProperties().getBoolean("fiber", false), m_net_roughness);

		Float ret = sigma(wi, Sxx, Syy, Szz, Sxy, Sxz, Syz);
		return ret;
	}

	Spectrum multipleNet(const BSDFSamplingRecord& bRec) const {
		bool     transmission = Frame::cosTheta(bRec.wi) * Frame::cosTheta(bRec.wo) < 0;
		Vector wi_ = bRec.wi;
		Vector wo_ = bRec.wo;

		Point2 uv = bRec.its.uv;
		if (!m_flag_aniso[0]){
			MediumSamplingRecord        mRec;
			PhaseFunctionSamplingRecord pRec(mRec, wi_, wo_);
			Float cosWi = abs(wi_.z);
			Float cosWo = abs(wo_.z);
			/**/
			if (!transmission){
				if (cosWi == 0 || cosWo == 0)
					return Spectrum(0.f);
				Spectrum sigmaT = getColorFromTexture(m_sigmaWet_m[0], uv);
				Spectrum albedo = getColorFromTexture(m_albedo_m[0], uv);
				Spectrum TW = getColorFromTexture(m_TW_m[0], uv);
				Spectrum albedo2 = getColorFromTexture(m_albedo2_m[0], uv);
				Spectrum lobe1 = getColorFromTexture(m_lobe1_m[0], uv);
				Spectrum lobe2 = getColorFromTexture(m_lobe2_m[0], uv);
				Float fp = eval_fp_gs(bRec, lobe1[0], lobe1[1], lobe1[2], lobe2[0], lobe2[1], lobe2[2]);
				Float z = TW[0];
				Spectrum fr_lam = albedo2*cosWo / M_PI;
				Spectrum term1 = Spectrum(1.f) - exp(-(cosWi + cosWo)*sigmaT*z / (cosWi*cosWo));
				return TW[1] * term1*fp*albedo / (cosWi + cosWo)*cosWo + TW[2] * fr_lam;
			}
			else{
				if (cosWi == 0 || cosWo == 0)
					return Spectrum(0.f);
				Spectrum sigmaT = getColorFromTexture(m_sigmaWet_m[0], uv);
				Spectrum albedo = getColorFromTexture(m_albedo_m[0], uv);
				Spectrum TW = getColorFromTexture(m_TW_m[0], uv);
				Spectrum albedo2 = getColorFromTexture(m_albedo2_m[0], uv);
				Spectrum lobe1 = getColorFromTexture(m_lobe1_m[0], uv);
				Spectrum lobe2 = getColorFromTexture(m_lobe2_m[0], uv);
				Float fp = eval_fp_gs(bRec, lobe1[0], lobe1[1], lobe1[2], lobe2[0], lobe2[1], lobe2[2]);
				Float z = TW[0];
				//Spectrum fr_lam = albedo2*cosWo / M_PI;
				if (abs(cosWi - cosWo) < Epsilon)
					return TW[1] * exp(-sigmaT * z / cosWi) * z * fp * sigmaT * albedo / (cosWo);// +m_net_w2_down * fr_lam;
				Spectrum term1 = exp(-sigmaT*z / cosWi) - exp(-sigmaT*z / cosWo);
				return TW[1] * term1*fp*albedo / (cosWi - cosWo)*cosWo;
			}
		}
		else{
			Float cosWi = abs(wi_.z);
			Float cosWo = abs(wo_.z);
			
			MediumSamplingRecord        mRec;
			Vector orientation = m_flag_orientations[0]
				? getOrientationFromTexture(m_texture_orientations[0], bRec.its, m_scale_n[0])
				: m_vector_orientations[0];
			mRec.orientation = orientation;
			PhaseFunctionSamplingRecord pRec(mRec, wi_, wo_);

			Float z, porosity, saturation, w1, w2;
			Spectrum fr_lam, albedo, sigmaWet;
			Float fp;
			Spectrum sigmaTI, sigmaTO;
			if (!m_multipleIso)
			{
				if (transmission){
					z = m_net_thickness_down;
					porosity = m_net_porosity_down;
					fr_lam = m_net_albedo2_down*cosWo / M_PI;
					saturation = m_net_saturation_down;
					albedo = m_net_albedo_down;
					sigmaWet = m_net_sigmaWet_down;
					w1 = m_net_w1_down;
					w2 = m_net_w2_down;
				}
				else{
					z = m_net_thickness;
					porosity = m_net_porosity;
					fr_lam = m_net_albedo2*cosWo / M_PI;
					saturation = m_net_saturation;
					albedo = m_net_albedo;
					sigmaWet = m_net_sigmaWet;
					w1 = m_net_w1;
					w2 = m_net_w2;
				}
				fp = std::max(evalPhase(pRec, transmission), 0.f);
				sigmaTI = Spectrum(getSigma(orientation, wi_, transmission));
				sigmaTO = Spectrum(getSigma(orientation, wo_, transmission));
			}
			else{
				if (transmission){
					if (m_multipeTex){
						sigmaWet = getColorFromTexture(m_sigmaWet_m[1], uv);
						albedo = getColorFromTexture(m_albedo_m[1], uv);
						Spectrum TW = getColorFromTexture(m_TW_m[1], uv);
						z = TW[0];
						w1 = TW[1];
						w2 = TW[2];
						Spectrum albedo2 = getColorFromTexture(m_albedo2_m[1], uv);
						fr_lam = albedo2*cosWo / M_PI;
						Spectrum SP = getColorFromTexture(m_SP_m[1], uv);
						saturation = SP[0];
						porosity = SP[1];
						Spectrum lobe1 = getColorFromTexture(m_lobe1_m[1], uv);
						Spectrum lobe2 = getColorFromTexture(m_lobe2_m[1], uv);
						fp = eval_fp_gs(bRec, lobe1[0], lobe1[1], lobe1[2], lobe2[0], lobe2[1], lobe2[2]);
					}
					else{
						z = m_net_thickness_down;
						porosity = m_net_porosity_down;
						fr_lam = m_net_albedo2_down*cosWo / M_PI;
						saturation = m_net_saturation_down;
						albedo = m_net_albedo_down;
						sigmaWet = m_net_sigmaWet_down;
						w1 = m_net_w1_down;
						w2 = m_net_w2_down;
						fr_lam = m_net_albedo2_down*cosWo / M_PI;
						fp = eval_fp_gs(bRec, m_net_wg1_down, m_net_wg2_down, m_net_mu1_down, m_net_mu2_down, m_net_sigma1_down, m_net_sigma2_down);
					}
				}
				else{
					if (m_multipeTex){
						sigmaWet = getColorFromTexture(m_sigmaWet_m[0], uv);
						albedo = getColorFromTexture(m_albedo_m[0], uv);
						Spectrum TW = getColorFromTexture(m_TW_m[0], uv);
						z = TW[0];
						w1 = TW[1];
						w2 = TW[2];
						Spectrum albedo2 = getColorFromTexture(m_albedo2_m[0], uv);
						fr_lam = albedo2*cosWo / M_PI;
						Spectrum SP = getColorFromTexture(m_SP_m[0], uv);
						saturation = SP[0];
						porosity = SP[1];
						Spectrum lobe1 = getColorFromTexture(m_lobe1_m[0], uv);
						Spectrum lobe2 = getColorFromTexture(m_lobe2_m[0], uv);
						fp = eval_fp_gs(bRec, lobe1[0], lobe1[1], lobe1[2], lobe2[0], lobe2[1], lobe2[2]);
					}
					else{
						z = m_net_thickness;
						porosity = m_net_porosity;
						fr_lam = m_net_albedo2*cosWo / M_PI;
						saturation = m_net_saturation;
						albedo = m_net_albedo;
						sigmaWet = m_net_sigmaWet;
						w1 = m_net_w1;
						w2 = m_net_w2;
						fr_lam = m_net_albedo2*cosWo / M_PI;
						fp = eval_fp_gs(bRec, m_net_wg1, m_net_wg2, m_net_mu1, m_net_mu2, m_net_sigma1, m_net_sigma2);
					}
					
				}
				sigmaTI = Spectrum(4);
				sigmaTO = Spectrum(4);
			}

			//Spectrum sigmaS = sigmaTI * albedo;	

			//return Spectrum(fp);			
			Float EL, K;
			
			if (porosity == 1)
				K = 1;
			else{
				Float EL = 1.209 * pow(math::clamp((1 - porosity), 0.f, 0.75f), 2.0 / 3.0);
				K = -log(1 - EL) / EL;
			}

			Float mu_i, mu_o, Z = z, sigma_t_o, sigma_t_i, S = saturation;
			Spectrum alpha = albedo, sigma_t_l = sigmaWet;

            if(m_semiinf){
                if (transmission)
                    return Spectrum(0.f);
                else{
                    Float lnK = log(K);
                    Float mu_i = cosWi, mu_o = cosWo;
                    Spectrum alpha = albedo, sigma_l = sigmaWet, sigma_i = sigmaTI, sigma_o = sigmaTO;
                    Spectrum sigma_k_i = Spectrum(K*sigma_i) + S*sigma_l, sigma_k_o = Spectrum(K*sigma_o) + S*sigma_l;
                    Spectrum tmp1 = albedo*sigma_i*K*fp / (mu_i * sigma_k_o * (1 + lnK));
                    Spectrum tmp2 = mu_o * sigma_k_i * power(K, Spectrum(1.f) - mu_i*sigma_k_o / (mu_o*K*sigma_i) + S*sigma_l / (K*sigma_k_o)) / (mu_i*sigma_k_o + mu_o*sigma_k_i);
                    Spectrum tmp3 = (Spectrum(1 + lnK) - tmp2 + S*sigma_l*lnK / (sigma_o*K));
    
                    Spectrum tmp4 = albedo*sigma_i*K*fp / (mu_i * sigma_k_o * (1 + lnK));
                    Spectrum tmp5 = mu_o * sigma_k_i * power(K, Spectrum(1.f) - mu_i*sigma_k_o / (mu_o*K*sigma_i) + S*sigma_l / (K*sigma_k_o)) / (mu_i*sigma_k_o + mu_o*sigma_k_i);
                    Spectrum tmp6 = (Spectrum(1 + lnK) - tmp2 + S*sigma_l*lnK / (sigma_o*K));
                    Spectrum fr = tmp1*tmp3 + tmp4*tmp6;

                    return fr;
                }
            }
			if (sigmaTO[0] / cosWo < sigmaTI[0] / cosWi){
				mu_i = cosWi;
				mu_o = cosWo;
				sigma_t_o = sigmaTO[0];
				sigma_t_i = sigmaTI[0];
			}
			else{
				mu_i = cosWo;
				mu_o = cosWi;
				sigma_t_o = sigmaTI[0];
				sigma_t_i = sigmaTO[0];
			}

			if (transmission){
				if ((cosWi - cosWo) == 0 || cosWi*cosWo == 0)
					return Spectrum(0.f);
				if (mu_i == 0 || mu_o == 0)
					return Spectrum(0.f);

				Float lnK = log(K);
				Float lnK_KZ = lnK / (K * Z);
				Float sigmai_mui = sigma_t_i / mu_i;
				Float sigmao_muo = sigma_t_o / mu_o;
				Spectrum sSigmalPlusKSigmai = S*sigma_t_l + Spectrum(K*sigma_t_i);
				Spectrum sSigmalPlusKSigmao = S*sigma_t_l + Spectrum(K*sigma_t_o);
				Spectrum sSigmalPlusKSigmaiZ_mui = sSigmalPlusKSigmai*Z / mu_i;
				Spectrum sSigmalPlusKSigmaoZ_muo = sSigmalPlusKSigmao*Z / mu_o;
				Spectrum onePlusSSigmal_KSigmai = Spectrum(1) + S*sigma_t_l / (K*sigma_t_i);
				Spectrum onePlusSSigmal_KSigmao = Spectrum(1) + S*sigma_t_l / (K*sigma_t_o);
				Float case1 = sigma_t_i*sigma_t_o / (mu_i*sigma_t_o + mu_o*sigma_t_i);

				Spectrum A = (alpha * sigma_t_i * fp * K) / (mu_i * mu_o * (1.0 + lnK));

				// part1
				Spectrum term_case1(0.0);

				if (std::max(sigmao_muo, sigmai_mui) <= lnK_KZ) {
					term_case1 = Spectrum(Z);
				}
				else if (lnK_KZ > sigmai_mui && lnK_KZ < sigmao_muo) {
					term_case1 = Spectrum(mu_o*lnK / (K* sigma_t_o));
				}
				else if (lnK_KZ < sigmai_mui && lnK_KZ > sigmao_muo) {
					term_case1 = Spectrum(mu_i*lnK / (K* sigma_t_i));
					//term_case1 = Spectrum(mu_i / ( sigma_t_i));
				}
				else if (lnK_KZ > case1 && lnK_KZ <= std::min(sigmai_mui, sigmao_muo)) {
					term_case1 = Spectrum(lnK / K*(mu_i / sigma_t_i + mu_o / sigma_t_o) - Z);
				}
				else {
					term_case1 = Spectrum(0.0);
				}

				// part2
				Spectrum term_case2(0.0);

				if (lnK_KZ > case1 && lnK_KZ < sigmai_mui) {
					Spectrum tmp1 = Spectrum(1) - exp(-sSigmalPlusKSigmaiZ_mui)*powS(Spectrum(K), onePlusSSigmal_KSigmai);
					term_case2 = mu_i*tmp1 / (sSigmalPlusKSigmai);
				}
				else {
					term_case2 = Spectrum(0.f);
				}

				// part3
				Spectrum term_case3(0.0);

				if (lnK_KZ > case1 && lnK_KZ < sigmao_muo) {
					Spectrum tmp1 = Spectrum(1) - exp(-sSigmalPlusKSigmaoZ_muo)*powS(Spectrum(K), onePlusSSigmal_KSigmao);
					term_case3 = mu_o*tmp1 / (sSigmalPlusKSigmao);
				}
				else {
					term_case3 = Spectrum(0.f);
				}

				// part4
				Spectrum term_case4(0.0);

				if (lnK_KZ < case1) {
					if (mu_i != mu_o){
						Spectrum tmp1 = exp(-sSigmalPlusKSigmaiZ_mui)*powS(Spectrum(K), (mu_i + mu_o)*(sSigmalPlusKSigmai) / (mu_i*K*sigma_t_o));
						Spectrum tmp2 = exp(-sSigmalPlusKSigmaoZ_muo)*powS(Spectrum(K), (mu_i + mu_o)*(sSigmalPlusKSigmao) / (mu_o*K*sigma_t_i));
						term_case4 = mu_i*mu_o*(tmp1 - tmp2) / (mu_i*(sSigmalPlusKSigmao)-mu_o*(sSigmalPlusKSigmai));
					}

					Spectrum exp3 = onePlusSSigmal_KSigmai;
					Spectrum exp4 = (sSigmalPlusKSigmai)*mu_o / (mu_i*K*sigma_t_o);
					Spectrum tmp3 = mu_i*exp(-sSigmalPlusKSigmaiZ_mui + (exp3 + exp4)*lnK)
						- mu_i*exp(-sSigmalPlusKSigmaiZ_mui + exp3*lnK);
					//Spectrum tmp4 = powS(Spectrum(K), (S*sigma_t_l + Spectrum(K*sigma_t_i))*mu_o / (mu_i*K*sigma_t_o)) - Spectrum(1);

					Spectrum exp5 = onePlusSSigmal_KSigmao;
					Spectrum exp6 = (sSigmalPlusKSigmao)*mu_i / (mu_o*K*sigma_t_i);
					Spectrum tmp5 = mu_o*exp(-sSigmalPlusKSigmaoZ_muo + (exp5 + exp6)*lnK)
						- mu_o*exp(-sSigmalPlusKSigmaoZ_muo + exp5*lnK);
					//Spectrum tmp6 = powS(Spectrum(K), (S*sigma_t_l + Spectrum(K*sigma_t_o))*mu_i / (mu_o*K*sigma_t_i)) - Spectrum(1);
					term_case4 += tmp3 / (sSigmalPlusKSigmai)+tmp5 / (sSigmalPlusKSigmao);
				}
				else {
					term_case4 = Spectrum(0.0);
				}

				return w1 * A * (term_case1 + term_case2 + term_case3 + term_case4)*cosWo + w2*fr_lam;
				//return Spectrum(0.f);
			}
			else{
				Float lnK = log(K);
				Float lnK_KZ = lnK / (K * Z);
				Float sigmai_mui = sigma_t_i / mu_i;
				Float sigmao_muo = sigma_t_o / mu_o;
				Spectrum sSigmalPlusKSigmai = S*sigma_t_l + Spectrum(K*sigma_t_i);
				Spectrum sSigmalPlusKSigmao = S*sigma_t_l + Spectrum(K*sigma_t_o);

				Spectrum A = (alpha * sigma_t_i * fp * K) / (mu_i * mu_o * (1.0 + lnK));

				// part1
				Spectrum term_case1(0.0);

				if (sigmao_muo < lnK_KZ && lnK_KZ < sigmai_mui) {
					Spectrum tmp = Spectrum(1.f) - exp(-(sSigmalPlusKSigmai)*Z / mu_i)*powS(Spectrum(K), sigma_t_l * S / K / sigma_t_i + Spectrum(1));
					term_case1 = mu_i*tmp / (sSigmalPlusKSigmai);
				}
				else if (lnK_KZ <= sigmao_muo) {
					Spectrum tmp = Spectrum(1.f) - powS(Spectrum(K), (sSigmalPlusKSigmai)*(mu_i*sigma_t_o - mu_o*sigma_t_i) / (mu_i*K*sigma_t_i*sigma_t_o));
					term_case1 = mu_i*tmp / (sSigmalPlusKSigmai);
				}
				else {
					term_case1 = Spectrum(0.0);
				}

				// part2
				Spectrum term_case2(0.0);

				if (lnK_KZ > sigmai_mui) {
					term_case2 = Spectrum(Z);
				}
				else if (lnK_KZ < sigmai_mui) {
					term_case2 = Spectrum(mu_i*lnK / (K*sigma_t_i));
				}
				else {
					term_case2 = Spectrum(0.0);
				}

				// part3
				Spectrum term_case3(0.0);

				if (lnK_KZ < sigmao_muo) {
					Spectrum tmp1 = mu_i*mu_o*powS(Spectrum(K), S*sigma_t_l*(sigma_t_i + sigma_t_o) / (K*sigma_t_i*sigma_t_o) + Spectrum(2.f));
					Spectrum tmp2 = powS(Spectrum(K), -Spectrum(1.f) - (S*sigma_t_l*(sigma_t_i + sigma_t_o) + Spectrum(mu_o*K*sigma_t_i)) / (K*mu_i*sigma_t_o));
					Spectrum tmp3 = exp(-(mu_o*(sSigmalPlusKSigmai)+mu_i*(sSigmalPlusKSigmao))*Z / (mu_o*mu_i));
					Spectrum tmp4 = mu_o*(sSigmalPlusKSigmai)+mu_i*(sSigmalPlusKSigmao);
					term_case3 = tmp1*(tmp2 - tmp3) / tmp4;
				}
				else {
					term_case3 = Spectrum(0.0);
				}

				return w1 * A*(term_case1 + term_case2 + term_case3)*cosWo + w2*fr_lam;
			}
			/*
			if (transmission){
				if (abs(cosWi - cosWo) <= Epsilon || cosWi*cosWo <= Epsilon)
					return Spectrum(0.f);
				if (mu_i <= Epsilon || mu_o <= Epsilon)
					return Spectrum(0.f);
				Spectrum A = (alpha * sigma_t_i * fp * K) / (mu_i * mu_o * (1.0 + log(K)));

				// part1
				Spectrum term_case1(0.0);

				if (std::max(sigma_t_o / mu_o, sigma_t_i / mu_i) <= log(K) / (K * Z)) {
					term_case1 = Spectrum(Z);
				}
				else if (log(K) / (K * Z) > sigma_t_i / mu_i && log(K) / (K * Z) < sigma_t_o / mu_o) {
					term_case1 = Spectrum(mu_o*log(K) / (K* sigma_t_o));
				}
				else if (log(K) / (K * Z) < sigma_t_i / mu_i && log(K) / (K * Z) > sigma_t_o / mu_o) {
					term_case1 = Spectrum(mu_i*log(K) / (K* sigma_t_i));//problem
				}
				else if (log(K) / (K * Z) > sigma_t_i*sigma_t_o / (mu_i*sigma_t_o + mu_o*sigma_t_i) && log(K) / (K * Z) <= std::min(sigma_t_i / mu_i, sigma_t_o / mu_o)) {
					term_case1 = Spectrum(log(K) / K*(mu_i / sigma_t_i + mu_o / sigma_t_o) - Z);
				}
				else {
					term_case1 = Spectrum(0.0);
				}

				// part2
				Spectrum term_case2(0.0);

				if (log(K) / (K * Z) > sigma_t_i*sigma_t_o / (mu_i*sigma_t_o + mu_o*sigma_t_i) && log(K) / (K * Z) < sigma_t_i / mu_i) {
					Spectrum tmp1 = Spectrum(1) - exp(-(S*sigma_t_l + Spectrum(K*sigma_t_i))*Z / mu_i)*powS(Spectrum(K), Spectrum(1) + S*sigma_t_l / (K*sigma_t_i));
					term_case2 = mu_i*tmp1 / (S*sigma_t_l + Spectrum(K*sigma_t_i));
				}
				else {
					term_case2 = Spectrum(0.f);
				}

				// part3
				Spectrum term_case3(0.0);

				if (log(K) / (K * Z) > sigma_t_i*sigma_t_o / (mu_i*sigma_t_o + mu_o*sigma_t_i) && log(K) / (K * Z) < sigma_t_o / mu_o) {
					Spectrum tmp1 = Spectrum(1) - exp(-(S*sigma_t_l + Spectrum(K*sigma_t_o))*Z / mu_o)*powS(Spectrum(K), Spectrum(1) + S*sigma_t_l / (K*sigma_t_o));
					term_case3 = mu_o*tmp1 / (S*sigma_t_l + Spectrum(K*sigma_t_o));
				}
				else {
					term_case3 = Spectrum(0.f);
				}

				// part4
				Spectrum term_case4(0.0);

				if (log(K) / (K * Z) < sigma_t_i*sigma_t_o / (mu_i*sigma_t_o + mu_o*sigma_t_i)) {
					if (mu_i != mu_o){
						Spectrum exponent1 = (mu_i + mu_o)*(S*sigma_t_l + Spectrum(K*sigma_t_i)) / (mu_i*K*sigma_t_o);
						Spectrum tmp1 = exp(-(S*sigma_t_l + Spectrum(K*sigma_t_i))*Z / mu_i + exponent1*log(K));
						Spectrum exponent2 = (mu_i + mu_o)*(S*sigma_t_l + Spectrum(K*sigma_t_o)) / (mu_o*K*sigma_t_i);
						Spectrum tmp2 = exp(-(S*sigma_t_l + Spectrum(K*sigma_t_o))*Z / mu_o + exponent2*log(K));
						term_case4 = mu_i*mu_o*(tmp1 - tmp2) / (mu_i*(S*sigma_t_l + Spectrum(K*sigma_t_o)) - mu_o*(S*sigma_t_l + Spectrum(K*sigma_t_i)));
						for (int i = 0; i < 3; i++){
							term_case4[i] = std::max(term_case4[i], 0.f);
						}
					}

					Spectrum exp3 = Spectrum(1) + S*sigma_t_l / (K*sigma_t_i);
					Spectrum exp4 = (S*sigma_t_l + Spectrum(K*sigma_t_i))*mu_o / (mu_i*K*sigma_t_o);
					Spectrum tmp3 = mu_i*exp(-(S*sigma_t_l + Spectrum(K*sigma_t_i))*Z / mu_i + (exp3 + exp4)*log(K))
						- mu_i*exp(-(S*sigma_t_l + Spectrum(K*sigma_t_i))*Z / mu_i + exp3*log(K));
					//Spectrum tmp4 = powS(Spectrum(K), (S*sigma_t_l + Spectrum(K*sigma_t_i))*mu_o / (mu_i*K*sigma_t_o)) - Spectrum(1);

					Spectrum exp5 = Spectrum(1) + S*sigma_t_l / (K*sigma_t_o);
					Spectrum exp6 = (S*sigma_t_l + Spectrum(K*sigma_t_o))*mu_i / (mu_o*K*sigma_t_i);
					Spectrum tmp5 = mu_o*exp(-(S*sigma_t_l + Spectrum(K*sigma_t_o))*Z / mu_o + (exp5 + exp6)*log(K))
						- mu_o*exp(-(S*sigma_t_l + Spectrum(K*sigma_t_o))*Z / mu_o + exp5*log(K));
					//Spectrum tmp6 = powS(Spectrum(K), (S*sigma_t_l + Spectrum(K*sigma_t_o))*mu_i / (mu_o*K*sigma_t_i)) - Spectrum(1);
					term_case4 += tmp3 / (S*sigma_t_l + Spectrum(K*sigma_t_i)) + tmp5 / (S*sigma_t_l + Spectrum(K*sigma_t_o));
				}
				else {
					term_case4 = Spectrum(0.0);
				}

				return w1 * A * (term_case1 + term_case2 + term_case3 + term_case4)*cosWo + w2*fr_lam;
				//return Spectrum(0.f);
			}
			else{
				Spectrum A = (alpha * sigma_t_i * fp * K) / (mu_i * mu_o * (1.0 + log(K)));

				// part1
				Spectrum term_case1(0.0);

				if (sigma_t_o / mu_o < log(K) / (K * Z) && log(K) / (K * Z) < sigma_t_i / mu_i) {
					Spectrum tmp = Spectrum(1.f) - exp(-(S*sigma_t_l + Spectrum(K*sigma_t_i))*Z / mu_i)*powS(Spectrum(K), sigma_t_l * S / K / sigma_t_i + Spectrum(1));
					term_case1 = mu_i*tmp / (S*sigma_t_l + Spectrum(K*sigma_t_i));
				}
				else if (log(K) / (K * Z) <= sigma_t_o / mu_o) {
					Spectrum tmp = Spectrum(1.f) - powS(Spectrum(K), (S*sigma_t_l + Spectrum(K*sigma_t_i))*(mu_i*sigma_t_o - mu_o*sigma_t_i) / (mu_i*K*sigma_t_i*sigma_t_o));
					term_case1 = mu_i*tmp / (S*sigma_t_l + Spectrum(K*sigma_t_i));
				}
				else {
					term_case1 = Spectrum(0.0);
				}

				// part2
				Spectrum term_case2(0.0);

				if (log(K) / (K * Z) > sigma_t_i / mu_i) {
					term_case2 = Spectrum(Z);
				}
				else if (log(K) / (K * Z) < sigma_t_i / mu_i) {
					term_case2 = Spectrum(mu_i*log(K) / (K*sigma_t_i));
				}
				else {
					term_case2 = Spectrum(0.0);
				}

				// part3
				Spectrum term_case3(0.0);

				if (log(K) / (K * Z) < sigma_t_o / mu_o) {
					Spectrum tmp1 = mu_i*mu_o*powS(Spectrum(K), S*sigma_t_l*(sigma_t_i + sigma_t_o) / (K*sigma_t_i*sigma_t_o) + Spectrum(2.f));
					Spectrum tmp2 = powS(Spectrum(K), -Spectrum(1.f) - (S*sigma_t_l*(sigma_t_i + sigma_t_o) + Spectrum(mu_o*K*sigma_t_i)) / (K*mu_i*sigma_t_o));
					Spectrum tmp3 = exp(-(mu_o*(S*sigma_t_l + Spectrum(K*sigma_t_i)) + mu_i*(S*sigma_t_l + Spectrum(K*sigma_t_o)))*Z / (mu_o*mu_i));
					Spectrum tmp4 = mu_o*(S*sigma_t_l + Spectrum(K*sigma_t_i)) + mu_i*(S*sigma_t_l + Spectrum(K*sigma_t_o));
					term_case3 = tmp1*(tmp2 - tmp3) / tmp4;
				}
				else {
					term_case3 = Spectrum(0.0);
				}

				return w1 * A*(term_case1 + term_case2 + term_case3)*cosWo + w2*fr_lam;
			}*/
		}


			/*
			if (!transmission){
				//return Spectrum(0.f);
				//if ((cosWi + cosWo) == 0 || cosWi*cosWo == 0)
				//	return Spectrum(0.f);
				if (cosWi == 0 || cosWo == 0)
					return Spectrum(0.f);
				Spectrum tau = (density*sigmaTI + saturation*sigmaWet) / cosWi + (density*sigmaTO + saturation*sigmaWet) / cosWo;
				return (fp*m_net_w1*sigmaS*(Spectrum(1.f) - exp(-thickness*tau)) / tau / cosWi + m_net_w2*fr_lam);

			}
			else{
				if (cosWi == 0 || cosWo == 0)
					return Spectrum(0.f);
				Spectrum sigmaI = density*sigmaTI + saturation*sigmaWet;
				Spectrum sigmaO = density*sigmaTO + saturation*sigmaWet;
				Spectrum term1 = exp(-sigmaI*thickness / cosWi) - exp(-sigmaO*thickness / cosWo);
				if ((cosWi*sigmaO - cosWo*sigmaI).isZero())
					return Spectrum(0.f);
				return (fp*m_net_w1*sigmaS*cosWo*term1 / (cosWi*sigmaO - cosWo*sigmaI) + m_net_w2*fr_lam);
			}*/
		
	}

    Spectrum eval(const BSDFSamplingRecord& bRec, EMeasure measure) const {
        /* Stop if this component was not requested */
        
        // whitefurnace
        if (m_deltaOnly)
            return Spectrum(0.0f);
        if ( m_reflectionOnly && (Frame::cosTheta(bRec.wi) <= 0 || Frame::cosTheta(bRec.wo) <= 0)) {
            return Spectrum(0.0f);
        }

        Spectrum result(0.0);
#if DELTA_BSDF
        if (bRec.sampledComponent == -2) {
            // return Spectrum(1.0);
            if (bRec.bounces == -1)
				return this->attenuation(-bRec.wi, bRec.its.uv, bRec.its);
            else
				return this->attenuation(bRec.wi, bRec.its.uv, bRec.its);
        }
        // return Spectrum(0.0f);
#endif

        // wetSurface
        BSDFSamplingRecord bRecCopy = bRec; // 拷贝一份
        Assert(bRec.sampler);
        Vector wiOri = bRecCopy.wi;
        Vector woOri = bRecCopy.wo;
		if (m_flag_normals[0]){
			Vector normal = getNormalFromTexture(m_texture_normals[0], bRec.its.uv);
			Frame N(normal);
            
			bRecCopy.wi = N.toLocal(bRec.wi);
			bRecCopy.wo = N.toLocal(bRec.wo);
		}
        
        Float    scale = 1;
        Spectrum reflectBRDF(0.f);
		Float saturation = m_flag_saturation[0]
			? getColorFromTexture(m_texture_saturation[0], bRec.its.uv)[0]
			: m_float_saturations[0];
        if (m_wetSurface&&(saturation==1)) {
            Float cosThetaT,
                eta    = m_IOR, // assum extIOR = 1 (air)
                invEta = 1.f / eta;

			Float F = fresnelDielectricExt(Frame::cosTheta(bRecCopy.wi), cosThetaT, eta);
			if (Frame::cosTheta(bRecCopy.wi) * Frame::cosTheta(bRecCopy.wo) >= 0) {
				if (std::abs(dot(reflect(bRecCopy.wi), bRecCopy.wo) - 1) < DeltaEpsilon)
					reflectBRDF = Spectrum(F);
			}

			bool reverse = Frame::cosTheta(bRecCopy.wi) < 0;
			if (reverse) bRecCopy.wi = -bRecCopy.wi;
            F = fresnelDielectricExt(Frame::cosTheta(bRecCopy.wi), cosThetaT, eta);
			bRecCopy.wi = -refract(bRecCopy.wi, cosThetaT, invEta, eta);
			if (reverse) bRecCopy.wi = -bRecCopy.wi;
            Float factor  = (bRecCopy.mode == ERadiance) ? (cosThetaT < 0 ? invEta : eta) : 1.0f;
            scale        *= (1 - F) * factor * factor;

			reverse = Frame::cosTheta(bRecCopy.wo) < 0;
			if (reverse) bRecCopy.wo = -bRecCopy.wo;
            F = fresnelDielectricExt(Frame::cosTheta(bRecCopy.wo), cosThetaT, eta);
			bRecCopy.wo = -refract(bRecCopy.wo, cosThetaT, invEta, eta);
			F = fresnelDielectricExt(Frame::cosTheta(-bRecCopy.wo), cosThetaT, eta);
			if (reverse) bRecCopy.wo = -bRecCopy.wo;
            factor  = (bRecCopy.mode == ERadiance) ? (cosThetaT < 0 ? invEta : eta) : 1.0f;
            scale  *= (1 - F) * factor * factor;
        }/**/
#if 1
        if (m_analytic) {
            Spectrum multiple(0.f);
            if (m_depth > 1) {
                if (m_multipleNet) {
                    multiple = multipleNet(bRecCopy);
                } else
                    multiple = msSimulationNEE(bRecCopy, 0); // 0 means multiple
            }
			result = evalSingle(bRecCopy) + multiple * m_scaling[0]; // Spectrum(0.0f);// evalSingle(bRec);// *m_scaling[0];
            if (m_mapping) {
                if (m_useTwoSide)
                    result += evalMappedSingleTwoSide(bRecCopy); // *m_scaling[0];
                else
                    result += evalMappedSingle(bRecCopy); // *m_scaling[0];
            }
        } else {
            // result += m_table->readRadiance(bRec.wi, bRec.wo);
            result += msSimulationNEE(bRecCopy, -1); //-1 means full solution
            // result += evalSingle(bRec);
            result *= m_scaling[0];
        }
#endif
        // bRec.wi = wiOri;
        // bRec.wo = woOri;

		Spectrum tmp = result * scale + reflectBRDF;
		if (tmp.isValid())
			return tmp; // *m_scaling[0];
		else
			return Spectrum(0.f);
            

        return Spectrum(0.f);
    }

    Spectrum evalWithPdf(BSDFSamplingRecord& bRec, EMeasure measure, float& pdf) const {
        /* Stop if this component was not requested */


#if DELTA_BSDF
        if (bRec.sampledComponent == -2) {
            // return Spectrum(1.0);
            if (bRec.bounces == -1)
				return this->attenuation(-bRec.wi, bRec.its.uv, bRec.its);
            else
				return this->attenuation(bRec.wi, bRec.its.uv, bRec.its);
        }
#endif
        Spectrum result = evalSingleWithPdf(bRec, pdf); // *m_scaling[0];
        if (m_mapping)
            result += evalMappedSingle(bRec); // *m_scaling[0];
        return result;                        // *m_scaling[0];
    }

    Float pdf(const BSDFSamplingRecord& bRec, EMeasure measure) const {
		//return warp::squareToUniformSpherePdf();
		//return (INV_PI * Frame::cosTheta(bRec.wo));//test

        if (/**/ m_reflectionOnly && (Frame::cosTheta(bRec.wi) <= 0 || Frame::cosTheta(bRec.wo) <= 0)) {
            return 0.0f;
        }

#if DELTA_BSDF
		float _pdf_through = 0;
		if (!m_reflectionOnly){
			Spectrum att =
				bRec.bounces == -1 ? this->attenuation(-bRec.wi, bRec.its.uv, bRec.its) : this->attenuation(bRec.wi, bRec.its.uv, bRec.its);

			_pdf_through = att.getLuminance();
			if (bRec.sampledComponent == -2) {
				if (_pdf_through < 1e-10)
					_pdf_through = 0.0f;
				return _pdf_through;
			}
		}
#else
        float _pdf_through = 0.0f;
#endif

        MediumSamplingRecord mRec;
        mRec.p = bRec.its.p;

        float pdfSum_ = 0.0f;

        //for (int iLayer = 0; iLayer < m_layerCount; iLayer++) {
		for (int iLayer = 0; iLayer < 1; iLayer++) {
            if (m_bsdfList[iLayer] == NULL) {
                Vector orientation = m_vector_orientations[iLayer];
                if (m_flag_orientations[iLayer]) {
                    orientation =
                        getOrientationFromTexture(m_texture_orientations[iLayer], bRec.its, m_scale_n[iLayer]);
                }

                mRec.orientation = orientation;
                PhaseFunctionSamplingRecord pRec(mRec, bRec.wi);
                pRec.wo    = bRec.wo;
				float pdf_ = 0;
				if (m_flag_wet[iLayer]){
					Float saturation = m_flag_saturation[iLayer]
						? getColorFromTexture(m_texture_saturation[iLayer], bRec.its.uv)[0]
						: m_float_saturations[iLayer];
					pdf_ = m_phaseFunctions[iLayer]->eval_wet(pRec, saturation);
				}
                else 
					pdf_ = m_phaseFunctions[iLayer]->pdf(pRec); // need to be perfect sample

                /*{
                if (!(pRec.wi + pRec.wo).isZero()){
                float NdotL = std::max(0.0f, dot(normalize(pRec.wi + pRec.wo), pRec.wi));
                pdf_ *= Specular_Fresnel_Schlick(m_F0[iLayer], NdotL).getLuminance();
                }
                }*/


                pdfSum_ += pdf_;
            } else {
                float pdf_  = m_bsdfList[iLayer]->pdf(bRec);
                pdfSum_    += pdf_;
            }
        }

        return pdfSum_ * (1 - _pdf_through) / m_layerCount;
    }

    Spectrum sample(BSDFSamplingRecord& bRec, const Point2& sample) const {
        float pdf1;
        return this->sample(bRec, pdf1, sample);
    }

    Spectrum sample(BSDFSamplingRecord& bRec, Float& pdf, const Point2& sample) const {
		/*bRec.wo = warp::squareToUniformSphere(sample);
		pdf = warp::squareToUniformSpherePdf();
		return eval(bRec, ESolidAngle) / pdf;

		
		bRec.wo = warp::squareToCosineHemisphere(sample);
		bRec.eta = 1.0f;
		bRec.sampledComponent = 0;
		bRec.sampledType = EDiffuseReflection;
		pdf = warp::squareToCosineHemispherePdf(bRec.wo);
		return eval(bRec, ESolidAngle) / pdf;//test*/

# if 0
		return generatePathSample(bRec, pdf, m_depth);
# endif
        if (/**/ m_reflectionOnly && (Frame::cosTheta(bRec.wi) < 0)) {
            return Spectrum(0.0f);
        }
        // wetSurface
        Float F = 0;
		Float saturation = m_flag_saturation[0]
			? getColorFromTexture(m_texture_saturation[0], bRec.its.uv)[0]
			: m_float_saturations[0];
		if (m_wetSurface && (saturation == 1)) {
        //if (m_wetSurface) {
            // Float F = 0;
            Float cosThetaT, eta, invEta;
            eta        = m_IOR, // assum extIOR = 1 (air)
                invEta = 1.f / eta;
            F          = fresnelDielectricExt(Frame::cosTheta(bRec.wi), cosThetaT, eta);
            if (bRec.sampler->next1D() < F) { // reflect
                bRec.wo = reflect(bRec.wi);
                pdf     = F;
                return eval(bRec, ESolidAngle) / pdf;
            } else {
                // bRec.wi = -refract(bRec.wi, cosThetaT, invEta, eta);
                Float factor = (bRec.mode == ERadiance) ? (cosThetaT < 0 ? invEta : eta) : 1.0f;
                pdf          = 1 - F;
            }
        }
        // if (m_analytic)
		{
			// #if 0
#if 0

			Vector d = warp::squareToUniformSphere(sample);
			pdf = warp::squareToUniformSpherePdf();
			bRec.wo = d;
			if (pdf <= 1e-10) return Spectrum(0.0);
			if (bRec.wi.z * bRec.wo.z < 0)
			{
				bRec.sampledComponent = 1;
				bRec.sampledType = EGlossyTransmission;
			}
			else
			{
				bRec.sampledComponent = 0;
				bRec.sampledType = EGlossyReflection;
			}
			bRec.eta = 1.0f;
			return eval(bRec, ESolidAngle) / pdf;

#endif
			//	Spectrum att = bRec.bounces == -1 ? this->attenuation(-bRec.wi, bRec.its.uv) :
			//		this->attenuation(bRec.wi, bRec.its.uv);

#if DELTA_BSDF
			float _pdf_through = 0;
			if(!m_reflectionOnly){
				Spectrum att = /*Frame::cosTheta(bRec.wi) < 0 ? this->attenuation(-bRec.wi, bRec.its.uv) :*/
					this->attenuation(bRec.wi, bRec.its.uv, bRec.its);
				// whitefurnace
				
				if (m_deltaOnly)
					_pdf_through = 1;
				else
					_pdf_through = std::min(att.getLuminance(), 1.f);
				float random = bRec.sampler->next1D();
				if (random < _pdf_through) {
					bRec.wo = -bRec.wi;
					bRec.sampledComponent = -2;
					// this means going straight
					bRec.sampledType = EDeltaTransmission;
					bRec.eta = 1.0f;
					pdf = _pdf_through * (1 - F); // this->pdf(bRec, ESolidAngle);
					return att / pdf; // Spectrum(1.0);// //Spectrum(1.0);// eval(bRec, ESolidAngle) / pdf;
				}
			}
#else
            Spectrum att          = this->attenuation(bRec.wi, bRec.its.uv);
            float    _pdf_through = 0;                         //
            float    random = random = bRec.sampler->next1D(); // +att.getLuminance();// *(1.0 - att.getLuminance());
#endif
#if WHITEFURNACE
            Vector d = warp::squareToUniformSphere(sample);
            pdf      = warp::squareToUniformSpherePdf() * (1 - _pdf_through);
            bRec.wo  = d;
            // if (pdf <= 1e-10) return Spectrum(0.0);
            {
                bRec.sampledComponent = 0;
                bRec.sampledType      = EGlossyReflection;
            }
            bRec.eta = 1.0f;
            return eval(bRec, ESolidAngle) / pdf;
#endif
#if 1
			float random = bRec.sampler->next1D();
            //if (m_layerCount > 1) {
			if (0) {
#if 1
                Spectrum attenuation(1.0);
                int      idLayer = -1;
                if (Frame::cosTheta(bRec.wi) < 0) //(bRec.bounces == -1)
                {
                    // this is the opersite side
                    idLayer = m_layerCount - 1;
                    for (int iLayer = m_layerCount - 1; iLayer >= 0; iLayer--) {
                        Vector orientation = m_vector_orientations[iLayer];
                        if (m_flag_orientations[iLayer]) {
                            orientation = getOrientationFromTexture(
                                m_texture_orientations[iLayer], bRec.its, m_scale_n[iLayer]
                            );
                        }
                        // PGL
                        Float saturation = m_flag_saturation[iLayer]
                                               ? getColorFromTexture(m_texture_saturation[iLayer], bRec.its.uv)[0]
                                               : m_float_saturations[iLayer];
                        // PGL
                        if (m_flag_wet[iLayer])
							attenuation *= computeAttenuationWet(bRec.wi, m_thickness[iLayer], iLayer, saturation, m_roughness[iLayer], orientation);
                        else
                            attenuation *= computeAttenuation(
                                bRec.wi, m_roughness[iLayer], m_thickness[iLayer], iLayer, orientation
                            );
                        if (attenuation.getLuminance() < random) {
                            idLayer = iLayer;
                            break;
                        }
                    }
                } else {
                    idLayer = 0;
                    for (int iLayer = 0; iLayer < m_layerCount; iLayer++) {
                        Vector orientation = m_vector_orientations[iLayer];
                        if (m_flag_orientations[iLayer]) {
                            orientation = getOrientationFromTexture(
                                m_texture_orientations[iLayer], bRec.its, m_scale_n[iLayer]
                            );
                        }
                        // PGL
                        Float saturation = m_flag_saturation[iLayer]
                                               ? getColorFromTexture(m_texture_saturation[iLayer], bRec.its.uv)[0]
                                               : m_float_saturations[iLayer];
                        // attenuationLayers.push_back(attenuation.getLuminance());
                        // PGL
                        if (m_flag_wet[iLayer])
							attenuation *= computeAttenuationWet(bRec.wi, m_thickness[iLayer], iLayer, saturation, m_roughness[iLayer], orientation);
                        else
                            attenuation *= computeAttenuation(
                                bRec.wi, m_roughness[iLayer], m_thickness[iLayer], iLayer, orientation
                            );
                        if (attenuation.getLuminance() < random) {
                            idLayer = iLayer;
                            break;
                        }
                    }
                }
#else
                int idLayer = 1;
#endif
                // sample from this layer
                if (m_bsdfList[idLayer] == NULL) {
                    MediumSamplingRecord mRec;
                    mRec.p           = bRec.its.p;
                    mRec.orientation = m_flag_orientations[idLayer]
                                           ? getOrientationFromTexture(
                                                 m_texture_orientations[idLayer], bRec.its, m_scale_n[idLayer]
                                             )
                                           : m_vector_orientations[idLayer];

                    PhaseFunctionSamplingRecord pRec(mRec, bRec.wi);
                    m_phaseFunctions[idLayer]->sample(pRec, bRec.sampler);
                    bRec.wo = pRec.wo;
                } else {
                    m_bsdfList[idLayer]->sample(bRec, sample);
                }

                pdf = this->pdf(bRec, ESolidAngle);
                if (pdf <= 1e-4)
                    return Spectrum(0.0);

                if (bRec.wi.z * bRec.wo.z < 0) {
                    bRec.sampledComponent = 1;
                    bRec.sampledType      = EGlossyTransmission;
                } else {
                    bRec.sampledComponent = 0;
                    bRec.sampledType      = EGlossyReflection;
                }
                bRec.eta  = 1.0f;
                pdf      *= (1 - F);
                //	Spectrum value = evalWithPdf(bRec, ESolidAngle)
                return eval(bRec, ESolidAngle) / pdf;
            }
#endif
            // single layer
            if (m_bsdfList[0] != NULL) // BSDF case
            {
                Spectrum weight = m_bsdfList[0]->sample(bRec, pdf, sample);
                return weight / (1 - F);
            }
            // medium case
            MediumSamplingRecord mRec;
            mRec.p             = bRec.its.p;
            Vector orientation = m_vector_orientations[0];
            if (m_flag_orientations[0]) {
                orientation = getOrientationFromTexture(m_texture_orientations[0], bRec.its, m_scale_n[0]);
            }

            mRec.orientation = orientation;
            PhaseFunctionSamplingRecord pRec(mRec, bRec.wi);
            if (m_flag_wet[0]) {
                Float saturation = m_flag_saturation[0] ? getColorFromTexture(m_texture_saturation[0], bRec.its.uv)[0]
                                                        : m_float_saturations[0];
                m_phaseFunctions[0]->sample_wet(pRec, bRec.sampler, pdf, saturation);
                //m_phaseFunctions[0]->sample(pRec, pdf, bRec.sampler);
            } else
                m_phaseFunctions[0]->sample(pRec, pdf, bRec.sampler);

            bRec.wo = pRec.wo;

            if (bRec.wi.z * bRec.wo.z < 0) {
                bRec.sampledComponent = 1;
                bRec.sampledType      = EGlossyTransmission;
            } else {
                bRec.sampledComponent = 0;
                bRec.sampledType      = EGlossyReflection;
            }
            bRec.eta  = 1.0f;
            pdf      *= (1 - _pdf_through);
            pdf      *= (1 - F);

			if (pdf <= 1e-4)
				return Spectrum(0.0);
			Spectrum tmp = eval(bRec, ESolidAngle) / pdf;
			tmp = clamp(tmp, 0, 10);
			if (tmp.isValid())
				return tmp; // *m_scaling[0];
			else
				return Spectrum(0.f);
        }
        // else//Guo
        {
            pdf = 1.0f;
            return msSimulation(bRec, pdf) * m_scaling[0];
        }
    }

    Spectrum attenuation(const Vector& wi, const Point2& uv, const Intersection& its) const {
        Spectrum result(1.0f);
#if WHITEFURNACE
        const int count = 1;
#else
        const int count = m_layerCount;
#endif

		// thickness texture
		std::vector<Float> thicknessList, thicknessAcc;
		thicknessAcc.push_back(0);
		for (int i = 0; i < m_layerCount; i++) {
			Float thickness = m_flag_thickness[i] ? getColorFromTexture(m_texture_thickness[i], uv)[0]
				: m_thickness[i];
			thicknessList.push_back(thickness);
			thicknessAcc.push_back(thicknessAcc[i] + thickness);
		}

        for (int i = 0; i < count; i++) {
            Vector orientation = m_vector_orientations[i];
            Float  saturation  = m_float_saturations[i];
            if (m_flag_orientations[i]) {
				orientation = getOrientationFromTexture(m_texture_orientations[i], its, m_scale_n[i]);
            }
            // PGL
            if (m_flag_saturation[i]) {
                saturation = getColorFromTexture(m_texture_saturation[i], uv)[0];
            }
            // PGL
            if (m_flag_wet[i])
				result *= computeAttenuationWetUnc(wi, thicknessList[i], i, saturation, orientation);
				//result *= computeAttenuationWet(wi, thicknessList[i], i, saturation,0, orientation);
            else
				result *= computeAttenuation(wi, m_roughness[i], thicknessList[i], i, orientation);
        }

        return result;
    }

    void addChild(const std::string& name, ConfigurableObject* child) {
        std::string prefix = name.substr(0, name.size() - 2);
        int         index  = atoi(name.substr(name.size() - 1).c_str());
        if (child->getClass()->derivesFrom(MTS_CLASS(Texture))) {

            if (prefix == "orientation") {
                // std::cout << "[GY]: layer " << index + 1 << " of " << m_nbLayers << " :: medium ::
                // orientation(texture)" << std::endl;
                m_flag_orientations[index]    = true;
                //m_texture_orientations[index] = static_cast<Texture2D*>(child);//mark
				m_texture_orientations[index] = static_cast<Texture*>(child);
            } else if (prefix == "normal") {
                // std::cout << "[GY]: layer " << index + 1 << " of " << m_nbLayers << " :: medium ::
                // orientation(texture)" << std::endl;
                m_flag_normals[index]    = true;
                m_texture_normals[index] = static_cast<Texture2D*>(child);
            } else if (prefix == "albedo_tex") {
                // std::cout << "[GY]: layer " << index + 1 << " of " << m_nbLayers << " :: medium ::
                // orientation(texture)" << std::endl;
                m_flag_albedos[index]    = true;
                m_texture_albedos[index] = static_cast<Texture2D*>(child);
            } else if (prefix == "thickness_tex") {
                // std::cout << "[GY]: layer " << index + 1 << " of " << m_nbLayers << " :: medium ::
                // orientation(texture)" << std::endl;
                m_flag_thickness[index]    = true;
                m_texture_thickness[index] = static_cast<Texture2D*>(child);
            }
            // PGL
            else if (prefix == "saturation_tex") {
                // std::cout << "[GY]: layer " << index + 1 << " of " << m_nbLayers << " :: medium ::
                // orientation(texture)" << std::endl;
                m_flag_saturation[index]    = true;
                m_texture_saturation[index] = static_cast<Texture2D*>(child);
            } // multiple net
			else if (prefix == "sigmaWet_m") {
				m_sigmaWet_m[index] = static_cast<Texture2D*>(child);
			}
			else if (prefix == "SP_m") {
				m_SP_m[index] = static_cast<Texture2D*>(child);
			}
			else if (prefix == "albedo_m") {
				m_albedo_m[index] = static_cast<Texture2D*>(child);
			}
			else if (prefix == "TW_m") {
				m_TW_m[index] = static_cast<Texture2D*>(child);
			}
			else if (prefix == "albedo2_m") {
				m_albedo2_m[index] = static_cast<Texture2D*>(child);
			}
			else if (prefix == "lobe1_m") {
				m_lobe1_m[index] = static_cast<Texture2D*>(child);
			}
			else if (prefix == "lobe2_m") {
				m_lobe2_m[index] = static_cast<Texture2D*>(child);
			}
			else
                BSDF::addChild(name, child);
        } else if (child->getClass()->derivesFrom(MTS_CLASS(PhaseFunction))) {

            if (prefix == "phase") {
                // std::cout << "[GY]: layer " << index + 1 << " of " << m_nbLayers << " :: surface :: bsdf" <<
                // std::endl;
                m_phaseFunctions[index] = static_cast<PhaseFunction*>(child);
                //	m_alpha[index] = m_phaseFunctions[index]->getProperties().getFloat("roughness");
                //	m_type[index] = m_phaseFunctions[index]->getProperties().getBoolean("fiber", false);
				m_vector_orientations[index] = normalize(
                    m_phaseFunctions[index]->getProperties().getVector("orientation", normalize(Vector(0, 0, 1))));
            }
        } else if (child->getClass()->derivesFrom(MTS_CLASS(BSDF))) {
            if (prefix == "bsdf") {
                m_bsdfList[index] = static_cast<BSDF*>(child);
                m_spectrum_albedos[index] =
                    m_bsdfList[index]->getProperties().getSpectrum("reflectance", Spectrum(1.0));
                m_reflectionOnly = true;
            }
        } else
            BSDF::addChild(name, child);
    }


    std::string toString() const {
        std::ostringstream oss;
        oss << "RoughConductor[" << endl
            << "  id = \"" << getID() << "\"," << endl
            << "  distribution = "
            << "beckmann"
            << "," << endl
            << "  sampleVisible = " << true << "," << endl
            << "  alphaU = " << 0.01 << "," << endl
            << "  alphaV = " << 0.01 << "," << endl
            << "]";
        return oss.str();
    }

    Shader* createShader(Renderer* renderer) const;
    bool    isFlake() const { return true; }


    MTS_DECLARE_CLASS()
private:
    // Spectrum m_scale[MAX_LAYER_COUNT];
    //	float m_alpha[MAX_LAYER_COUNT];
    ref_vector<BSDF>   m_bsdfList;
    float              m_thickness[MAX_LAYER_COUNT];
    std::vector<float> m_thicknessAcc;

    int m_layerCount;
    //	bool m_type[MAX_LAYER_COUNT];
    ref<const Texture> m_normals[MAX_LAYER_COUNT];
    float              m_scale_n[MAX_LAYER_COUNT];
    float              m_scaling[MAX_LAYER_COUNT];

    Spectrum m_F0[MAX_LAYER_COUNT];

    std::vector<bool> m_flag_orientations;
    std::vector<bool> m_flag_normals;
    // PGL
    std::vector<bool>     m_flag_saturation;
    ref_vector<Texture2D> m_texture_saturation;

    std::vector<bool>         m_flag_aniso;
    ref_vector<PhaseFunction> m_phaseFunctions;
    ref_vector<Medium>        m_mediums;

	Float m_DsDv;

    // PGL
    std::vector<bool>     m_flag_wet;
    std::vector<Float>    m_float_ns;
	std::vector<Float>    m_float_minSL;
    std::vector<Float>    m_float_porositys;
    std::vector<Float>    m_float_saturations;
    std::vector<Spectrum> m_spectrum_sigmaWets;
    bool                  m_deltaOnly;
    bool                  m_wetSurface;
    Float                 m_IOR;
    bool                  m_multipleApprox;
	bool                  m_multipleApproxHapke;
	bool				  m_bsdfdata;
	bool				  m_visdata;
	bool				  m_bsdftable;
	std::string			  m_bsdftableName;
    bool                  m_semiinf;

    PrecomputedMultipleTable*      m_table;
	PrecomputedMultipleTable*      m_phaseTable;
    PrecomputedMultipleTableSlice* m_table_slice;
    // PrecomputedMultipleTable *m_table_bottom;

	//multipleNet
	bool m_multipleNet;
	bool m_multipleIso;
	bool m_multipeTex;
	Float m_thetaI;
	Float m_net_porosity;
	Float m_net_n;
	Spectrum m_net_albedo;
	Spectrum m_net_sigmaWet;
	Float m_net_sat;
	Float m_net_saturation;
	Float m_net_thickness;
	Float m_net_w1;
	Float m_net_w2;
	Spectrum m_net_albedo2;
	Float m_net_density;
	Float m_net_roughness;
	Spectrum m_net_sigmaT;
	//HG
	Float m_net_whg;
	Float m_net_g1;
	Float m_net_g2;
	//GS
	Float m_net_wg1;
	Float m_net_wg2;
	Float m_net_mu1;
	Float m_net_mu2;
	Float m_net_sigma1;
	Float m_net_sigma2;

	//multipleNet down
	Float m_net_porosity_down;
	Float m_net_sat_down;
	Spectrum m_net_sigmaWet_down;
	Float m_net_saturation_down;
	Float m_net_roughness_down;
	Spectrum m_net_albedo_down;
	Spectrum m_net_sigmaT_down;
	Float m_net_thickness_down;
	Float m_net_w1_down;
	Float m_net_w2_down;
	Spectrum m_net_albedo2_down;
	// GS
	Float m_net_wg1_down;
	Float m_net_wg2_down;
	Float m_net_mu1_down;
	Float m_net_mu2_down;
	Float m_net_sigma1_down;
	Float m_net_sigma2_down;

	// wetsurface
	ref<BSDF> wetSurface;

	// multiple phase function
	Float m_multiplePhaseTable[phaseCount+1][phaseCount][phaseCount];
	Float m_multiplePhaseTable_down[phaseCount + 1][phaseCount][phaseCount];

    float                    m_step;
    std::string              m_prePath;
    std::vector<std::string> m_prePathLayers;
    Vector                   m_wi;
    bool                     m_analytic;
    bool                     m_visUp;
    bool                     m_mapping;
    std::string              m_paramName;

    bool        m_useTwoSide;
    std::string m_paramNameOthers[3];

    // bool m_forMultiple;
    // bool m_NEE;
    // bool m_transmission;
    int m_depth;

    std::vector<Spectrum> m_spectrum_sigmaTs;
    ref_vector<Texture2D> m_texture_sigmaTs;
    std::vector<bool>     m_flag_sigmaTs;

    std::vector<Float>    m_float_densities;
    ref_vector<Texture2D> m_texture_densities;
    std::vector<bool>     m_flag_densities;

    std::vector<Spectrum> m_spectrum_albedos;
    ref_vector<Texture2D> m_texture_albedos;
    std::vector<bool>     m_flag_albedos;

    std::vector<Float> m_roughness;

    // std::vector<Spectrum> m_spectrum_diffuse;
    // ref_vector<Texture2D> m_texture_diffuse;
    // std::vector<bool> m_flag_diffuse;

    std::vector<Vector>   m_vector_orientations;
    ref_vector<Texture> m_texture_orientations;
	//ref_vector<Texture2D> m_texture_orientations;//mark

    ref_vector<Texture2D> m_texture_normals;
    std::vector<bool>     m_flag_thickness;
    ref_vector<Texture2D> m_texture_thickness;
    const float           prod  = 1.f; // product of layer thickness and microflake density
    const float           boost = 1.f; // boost the intensity while still preserving energy conservation

    // bool m_seperatePre;
    int m_sampleCount;

    ref_vector<Texture2D> m_texture_albedos_m[4];
    ref_vector<Texture2D> m_texture_F0_m[4];
    ref_vector<Texture2D> m_roughness_m[4];
    ref_vector<Texture2D> m_w1w2_m[4];
    ref_vector<BSDF>      m_bsdfList_m[4];

	// multiple textures
	ref_vector<Texture2D> m_albedo_m;
	ref_vector<Texture2D> m_sigmaWet_m;
	ref_vector<Texture2D> m_SP_m;
	ref_vector<Texture2D> m_TW_m;
	ref_vector<Texture2D> m_albedo2_m;
	ref_vector<Texture2D> m_lobe1_m;
	ref_vector<Texture2D> m_lobe2_m;

    bool              m_reflectionOnly;
    std::vector<bool> m_flag_multipleIgnore;

    std::vector<ref_vector<Medium>> m_mediums_runtime;
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
    RoughConductorShader(
        Renderer*       renderer,
        const Texture*  specularReflectance,
        const Texture*  alphaU,
        const Texture*  alphaV,
        const Spectrum& eta,
        const Spectrum& k
    )
    : Shader(renderer, EBSDFShader), m_specularReflectance(specularReflectance), m_alphaU(alphaU), m_alphaV(alphaV) {
        m_specularReflectanceShader = renderer->registerShaderForResource(m_specularReflectance.get());
        m_alphaUShader              = renderer->registerShaderForResource(m_alphaU.get());
        m_alphaVShader              = renderer->registerShaderForResource(m_alphaV.get());

        /* Compute the reflectance at perpendicular incidence */
        m_R0 = fresnelConductorExact(1.0f, eta, k);
    }

    bool isComplete() const {
        return m_specularReflectanceShader.get() != NULL && m_alphaUShader.get() != NULL &&
               m_alphaVShader.get() != NULL;
    }

    void putDependencies(std::vector<Shader*>& deps) {
        deps.push_back(m_specularReflectanceShader.get());
        deps.push_back(m_alphaUShader.get());
        deps.push_back(m_alphaVShader.get());
    }

    void cleanup(Renderer* renderer) {
        renderer->unregisterShaderForResource(m_specularReflectance.get());
        renderer->unregisterShaderForResource(m_alphaU.get());
        renderer->unregisterShaderForResource(m_alphaV.get());
    }

    void resolve(const GPUProgram* program, const std::string& evalName, std::vector<int>& parameterIDs) const {
        parameterIDs.push_back(program->getParameterID(evalName + "_R0", false));
    }

    void bind(GPUProgram* program, const std::vector<int>& parameterIDs, int& textureUnitOffset) const {
        program->setParameter(parameterIDs[0], m_R0);
    }

    void
    generateCode(std::ostringstream& oss, const std::string& evalName, const std::vector<std::string>& depNames) const {
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
            << "   float D = " << evalName << "_D(H, alphaU, alphaV)"
            << ";" << endl
            << "   float G = " << evalName << "_G(H, wi, wo);" << endl
            << "   vec3 F = " << evalName << "_schlick(1-dot(wi, H));" << endl
            << "   return reflectance * F * (D * G / (4*cosTheta(wi)));" << endl
            << "}" << endl
            << endl
            << "vec3 " << evalName << "_diffuse(vec2 uv, vec3 wi, vec3 wo) {" << endl
            << "    if (cosTheta(wi) < 0.0 || cosTheta(wo) < 0.0)" << endl
            << "    	return vec3(0.0);" << endl
            << "    return " << evalName << "_R0 * inv_pi * inv_pi * cosTheta(wo);" << endl
            << "}" << endl;
    }
    MTS_DECLARE_CLASS()
private:
    ref<const Texture> m_specularReflectance;
    ref<const Texture> m_alphaU;
    ref<const Texture> m_alphaV;
    ref<Shader>        m_specularReflectanceShader;
    ref<Shader>        m_alphaUShader;
    ref<Shader>        m_alphaVShader;
    Spectrum           m_R0;
};

Shader* SheenSingleLayer::createShader(Renderer* renderer) const {
    ref<const Texture> m_specularReflectance = new ConstantSpectrumTexture(Spectrum(1.0f));
    float              roughness;
    if (m_bsdfList[0] != NULL)
        roughness = 1.0;
    else {
        if (m_flag_aniso[0]) {
            roughness = m_phaseFunctions[0]->getProperties().getFloat("roughness", 0.1);
        } else
            roughness = 1.0 - m_phaseFunctions[0]->getProperties().getFloat("g", 0.1);
    }

    ref<const Texture> m_alphaU = new ConstantFloatTexture(roughness);
    ref<const Texture> m_alphaV = new ConstantFloatTexture(roughness);

    Spectrum m_eta = Spectrum(0.0f);
    Spectrum m_k   = Spectrum(1.0f);
    return new RoughConductorShader(renderer, m_specularReflectance.get(), m_alphaU.get(), m_alphaV.get(), m_eta, m_k);
}

MTS_IMPLEMENT_CLASS(RoughConductorShader, false, Shader)
MTS_IMPLEMENT_CLASS_S(SheenSingleLayer, false, BSDF)
MTS_EXPORT_PLUGIN(SheenSingleLayer, "SheenSingleLayer BRDF");
MTS_NAMESPACE_END
