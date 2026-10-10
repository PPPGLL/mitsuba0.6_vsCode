/*
    This file is part of Mitsuba, a physically based rendering system.

    Copyright (c) 2007-2011 by Wenzel Jakob and others.

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

#include <mitsuba/render/shape.h>
#include <cfloat>
#include <limits>
#include <cmath>
#include <fstream>
#include <mitsuba/render/bsdf.h>
#include <mitsuba/render/emitter.h>
#include <mitsuba/render/medium.h>
#include <mitsuba/render/sensor.h>
#include <mitsuba/render/subsurface.h>
#include <mitsuba/render/trimesh.h>
#include <mitsuba/render/texture.h>
#include <mitsuba/core/bitmap.h>
#include <mitsuba/core/fresolver.h>
#include <mitsuba/core/fstream.h>
#include <mitsuba/core/statistics.h>
#include <mitsuba/core/timer.h>

#define MTS_QTREE_MAXDEPTH  50
#define MTS_QTREE_FASTSTART 1

MTS_NAMESPACE_BEGIN

static StatsCounter numTraversals("Height field", "Traversal operations per query", EAverage);

namespace {
    /// Find the smallest t >= 0 such that a*t + b is a multiple of c
    inline Float nextMultiple(Float a, Float b, Float c) {
        Float tmp     = b/c,
              rounded = (a > 0 ? std::ceil(tmp) : std::floor(tmp)) * c,
              diff    = rounded - b;

        if (diff == 0)
            diff = math::signum(a) * c;

        return diff / a;
    }

    /// Temporary storage for patch-ray intersections
    struct PatchIntersectionRecord {
        Point p;
        Point worldP;
        int x, y;
    };

    /// Stack entry for recursive quadtree traversal
    struct StackEntry {
        int level, x, y;
    };
};

/*!\plugin{heightfield}{Height field intersection shape}
 * \order{11}
 * \parameters{
 *     \parameter{shadingNormals}{\Boolean}{
 *       Use linearly interpolated shading normals over the height
 *       field as opposed to discontinuous normals from the underlying
 *       bilinear patches? \default{\code{true}, i.e. interpolate smoothly varying normals}
 *     }
 *     \parameter{flipNormals}{\Boolean}{
 *       Optional flag to flip all normals. \default{\code{false}, i.e.
 *       the normals are left unchanged}.
 *     }
 *     \parameter{toWorld}{\Transform}{
 *        Specifies an optional linear object-to-world transformation.
 *        \default{none, i.e. object space $=$ world space}
 *     }
 *     \parameter{width, height}{\Integer}{
 *       When the nested texture is procedural (see below),
 *       this parameter specifies the resolution at which it should
 *       be rasterized to create a height field made of bilinear patches.
 *     }
 *     \parameter{scale}{\Float}{Scale factor that is applied to the height field
 *       values\default{No scaling, i.e. \code{1}}
 *     }
 *     \parameter{filename}{\String}{
 *       File name of an image file containing height field values. Alternatively,
 *       a nested texture can be provided (see below).
 *     }
 *     \parameter{\Unnamed}{\Texture}{
 *       A nested texture that specifies the height field values. This
 *       could be a bitmap-backed texture or one that is procedurally defined.
 *       In the latter case, it will be rasterized using the resolution specified
 *       by the \code{width} and \code{height} arguments.
 *     }
 * }
 *\vspace{-2mm}
 * \renderings{
 *     \rendering{Heigh field rendering of a mountain, see \lstref{heightfield-bitmap}}
 *         {shape_heightfield}
 * }
 * This plugin implements an efficient height field intersection shape, i.e.
 * a two-dimensional plane that is vertically displaced using height values
 * loaded from a texture.
 * Internally, the height field is represented as a min-max mipmap
 * \cite{Tevs2008Maximum}, allowing cheap storage and efficient ray
 * intersection queries. It is generally preferable to represent
 * height fields using this specialized plugin rather than converting
 * them into triangle meshes.
 *
 * \begin{xml}[caption={Declaring a height field from a monochromatic scaled bitmap texture}, label=lst:heightfield-bitmap]
 * <shape type="heightfield">
 *     <string name="filename" value="mountain_profile.exr"/>
 *     <float name="scale" value="0.5"/>
 * </shape>
 * \end{xml}
 */

class Heightfield : public Shape {
public:
    Heightfield(const Properties &props) : Shape(props), m_data(NULL), m_normals(NULL), m_minmax(NULL) {
        m_sizeHint = Vector2i(
            props.getInteger("width", -1),
            props.getInteger("height", -1)
        );

        m_objectToWorld = props.getTransform("toWorld", Transform());
        m_shadingNormals = props.getBoolean("shadingNormals", true);
        m_flipNormals = props.getBoolean("flipNormals", false);
        m_scale = props.getFloat("scale", 1);

        m_snapshot = props.getString("gpuSnapshot", "");
        m_filename = props.getString("filename", "");
        if (!m_filename.empty())
            m_filename = Thread::getThread()->getFileResolver()->resolve(m_filename);
    }

    Heightfield(Stream *stream, InstanceManager *manager)
        : Shape(stream, manager), m_data(NULL), m_normals(NULL), m_minmax(NULL) {

        m_objectToWorld = Transform(stream);
        m_shadingNormals = stream->readBool();
        m_flipNormals = stream->readBool();
        m_scale = stream->readFloat();
        m_filename = stream->readString();
        m_dataSize = Vector2i(stream);
        size_t size = (size_t) m_dataSize.x * (size_t) m_dataSize.y;
        m_data = (Float *) allocAligned(size * sizeof(Float));
        stream->readFloatArray(m_data, size);
        configure();
    }

    ~Heightfield() {
        if (m_data)
            freeAligned(m_data);
        if (m_minmax) {
            for (int i=0; i<m_levelCount; ++i)
                freeAligned(m_minmax[i]);
            delete[] m_minmax;
            delete[] m_levelSize;
            delete[] m_numChildren;
            delete[] m_blockSize;
            delete[] m_blockSizeF;
        }
        if (m_normals)
            freeAligned(m_normals);
    }

    void serialize(Stream *stream, InstanceManager *manager) const {
        Shape::serialize(stream, manager);
        m_objectToWorld.serialize(stream);
        stream->writeBool(m_shadingNormals);
        stream->writeBool(m_flipNormals);
        stream->writeFloat(m_scale);
        stream->writeString(m_filename.string());
        m_dataSize.serialize(stream);
        stream->writeFloatArray(m_data, (size_t) m_dataSize.x * (size_t) m_dataSize.y);
    }

    AABB getAABB() const {
        AABB result;
        for (int i=0; i<8; ++i)
            result.expandBy(m_objectToWorld(m_dataAABB.getCorner(i)));

        return result;
    }

    Float getSurfaceArea() const {
        return m_surfaceArea; /// XXX transformed surface area? ...
    }

    size_t getPrimitiveCount() const {
        return 1;
    }

    size_t getEffectivePrimitiveCount() const {
        return (size_t) m_levelSize[0].x * (size_t) m_levelSize[0].y;
    }

    inline static int signumToInt(Float value) {
        if (value < 0)
            return -1;
        else if (value > 0)
            return 1;
        else
            return 0;
    }

    // Strict piecewise-bilinear intersection. Traverse real XY cells front-to-back.
    // Double arithmetic avoids cell-coordinate rounding; no Epsilon extension.
    bool rayIntersect(const Ray &_ray, Float mint, Float maxt, Float &t, void *tmp) const {
        const Matrix4x4 &matrix = m_objectToWorld.getInverseMatrix();
        double o[3], d[3];
        for (int k=0; k<3; ++k) {
            o[k]=(double)matrix.m[k][3]; d[k]=0;
            for (int j=0; j<3; ++j) {
                o[k]+=(double)matrix.m[k][j]*(double)_ray.o[j];
                d[k]+=(double)matrix.m[k][j]*(double)_ray.d[j];
            }
            if (!std::isfinite(o[k]) || !std::isfinite(d[k])) return false;
        }
        if (d[0]==0 && d[1]==0 && d[2]==0) return false;
        double near=(double)mint, far=(double)maxt;
        for (int k=0; k<3; ++k) {
            double low=(double)m_dataAABB.min[k], high=(double)m_dataAABB.max[k];
            if (d[k]==0) { if (o[k]<low || o[k]>high) return false; }
            else {
                double a=(low-o[k])/d[k], b=(high-o[k])/d[k];
                if (a>b) std::swap(a,b);
                near=std::max(near,a); far=std::min(far,b);
            }
        }
        if (near>far) return false;
        int nx=m_dataSize.x-1, ny=m_dataSize.y-1;
        double uvTol=64*DBL_EPSILON*(1+std::max(nx,ny));
        int cell[2], step[2];
        for (int k=0; k<2; ++k) {
            double p=o[k]+near*d[k], nearest=std::round(p);
            cell[k]=(int)std::floor(p);
            if (d[k]<0 && std::abs(p-nearest)<=uvTol) cell[k]=(int)nearest-1;
            cell[k]=std::max(0,std::min((k==0?nx:ny)-1,cell[k]));
            step[k]=d[k]>0?1:(d[k]<0?-1:0);
        }
        numTraversals.incrementBase(); size_t traversals=0;
        while (cell[0]>=0 && cell[0]<nx && cell[1]>=0 && cell[1]<ny && near<=far) {
            ++traversals;
            int x=cell[0], y=cell[1], w=m_dataSize.x;
            double crossing[2];
            for (int k=0;k<2;++k)
                crossing[k]=d[k]==0?std::numeric_limits<double>::infinity():
                    ((double)(cell[k]+(step[k]>0?1:0))-o[k])/d[k];
            double end=std::min(far,std::min(crossing[0],crossing[1]));
            double duration=end-near;
            double u=o[0]+near*d[0]-x, v=o[1]+near*d[1]-y;
            double h00=m_data[y*w+x], h10=m_data[y*w+x+1],
                   h01=m_data[(y+1)*w+x], h11=m_data[(y+1)*w+x+1];
            double f[3];
            for (int k=0;k<3;++k) {
                double s=.5*k, us=u+s*duration*d[0], vs=v+s*duration*d[1];
                f[k]=(1-us)*(1-vs)*h00+us*(1-vs)*h10+(1-us)*vs*h01+us*vs*h11
                    -(o[2]+(near+s*duration)*d[2]);
            }
            double A=2*(f[2]+f[0]-2*f[1]), B=f[2]-f[0]-A, C=f[0];
            double roots[2]; int nr=0;
            bool linear=std::abs(A)<=64*DBL_EPSILON*std::max(std::max(std::abs(B),std::abs(C)),1e-300);
            if (duration==0) {
                if (std::abs(f[0])<=32*DBL_EPSILON*(1+std::abs(o[2]))) roots[nr++]=0;
            } else if (linear) {
                if (B!=0) roots[nr++]=-C/B;
                else if (std::abs(C)<=32*DBL_EPSILON*(1+std::abs(o[2]))) roots[nr++]=0;
            } else {
                double discriminant=B*B-4*A*C;
                if (discriminant>=0) {
                    double q=-.5*(B+std::copysign(std::sqrt(discriminant),B));
                    if (q==0) roots[nr++]=-B/(2*A);
                    else { roots[nr++]=q/A; roots[nr++]=C/q; }
                }
            }
            double best=std::numeric_limits<double>::infinity(), bestU=0, bestV=0;
            double tTol=64*DBL_EPSILON*(1+std::abs(near)+std::abs(end));
            for (int k=0;k<nr;++k) {
                double candidate=near+roots[k]*duration;
                double cu=o[0]+candidate*d[0]-x, cv=o[1]+candidate*d[1]-y;
                if (!std::isfinite(candidate) || candidate<(double)mint || candidate>(double)maxt
                    || candidate<near-tTol || candidate>end+tTol
                    || cu<-uvTol || cu>1+uvTol || cv<-uvTol || cv>1+uvTol) continue;
                if (candidate<best) { best=candidate; bestU=cu; bestV=cv; }
            }
            if (std::isfinite(best)) {
                t=(Float)best;
                if (tmp) {
                    PatchIntersectionRecord &record=*((PatchIntersectionRecord *)tmp);
                    record.x=x; record.y=y;
                    record.p=Point((Float)std::max(0.,std::min(1.,bestU)),
                        (Float)std::max(0.,std::min(1.,bestV)),(Float)(o[2]+best*d[2]));
                    for (int k=0;k<3;++k)
                        record.worldP[k]=(Float)((double)_ray.o[k]+best*(double)_ray.d[k]);
                }
                numTraversals+=traversals; return true;
            }
            if (end>=far) break;
            bool advanced=false;
            for (int k=0;k<2;++k) if (step[k] && crossing[k]<=end+tTol) {
                cell[k]+=step[k]; advanced=true;
            }
            if (!advanced) break;
            near=end;
        }
        numTraversals+=traversals; return false;
    }


    void fillIntersectionRecord(const Ray &ray,
            const void *tmp, Intersection &its) const {
        PatchIntersectionRecord &temp = *((PatchIntersectionRecord *) tmp);

        int x = temp.x, y = temp.y, width = m_dataSize.x;
        Float
            f00 = m_data[y     * width + x],
            f01 = m_data[(y+1) * width + x],
            f10 = m_data[y     * width + x + 1],
            f11 = m_data[(y+1) * width + x + 1];

        Point pLocal(temp.p.x + temp.x, temp.p.y + temp.y, temp.p.z);
        its.uv = Point2(pLocal.x * m_invSize.x, pLocal.y * m_invSize.y);
        its.p  = temp.worldP;
        its.dpdu = m_objectToWorld(Vector(1, 0,
            (1.0f - temp.p.y) * (f10 - f00) + temp.p.y * (f11 - f01)) * m_levelSize0f.x);
        its.dpdv = m_objectToWorld(Vector(0, 1,
            (1.0f - temp.p.x) * (f01 - f00) + temp.p.x * (f11 - f10)) * m_levelSize0f.y);

        its.geoFrame.s = normalize(its.dpdu);
        its.geoFrame.t = normalize(its.dpdv - dot(its.dpdv, its.geoFrame.s) * its.geoFrame.s);
        its.geoFrame.n = cross(its.geoFrame.s, its.geoFrame.t);

        if (m_shadingNormals) {
            const Normal
                &n00 = m_normals[y     * width + x],
                &n01 = m_normals[(y+1) * width + x],
                &n10 = m_normals[y     * width + x + 1],
                &n11 = m_normals[(y+1) * width + x + 1];

            its.shFrame.n = normalize(m_objectToWorld(Normal(
                (1 - temp.p.x) * ((1-temp.p.y) * n00 + temp.p.y * n01)
                   + temp.p.x  * ((1-temp.p.y) * n10 + temp.p.y * n11))));
        } else {
            its.shFrame.n = its.geoFrame.n;
        }

        if (m_flipNormals) {
            its.shFrame.n *= -1;
            its.geoFrame.n *= -1;
        }

        its.shape = this;
        its.hasUVPartials = false;
        its.instance = NULL;
        its.time = ray.time;
        its.primIndex = x + y*width;
    }

    bool rayIntersect(const Ray &ray, Float mint, Float maxt) const {
        Float t;
        return rayIntersect(ray, mint, maxt, t, NULL);
    }

    void getNormalDerivative(const Intersection &its,
            Vector &dndu, Vector &dndv, bool shadingFrame) const {
        int width = m_dataSize.x,
            x = its.primIndex % width,
            y = its.primIndex / width;

        Float u = its.uv.x * m_levelSize0f.x - x;
        Float v = its.uv.y * m_levelSize0f.y - y;

        Normal normal;
        if (shadingFrame && m_shadingNormals) {
            /* Derivatives for bilinear patch with interpolated shading normals */
            const Normal
                &n00 = m_normals[y     * width + x],
                &n01 = m_normals[(y+1) * width + x],
                &n10 = m_normals[y     * width + x + 1],
                &n11 = m_normals[(y+1) * width + x + 1];

            normal = m_objectToWorld(Normal(
                (1 - u) * ((1-v) * n00 + v * n01)
                   + u  * ((1-v) * n10 + v * n11)));

            dndu = m_objectToWorld(Normal((1.0f - v) * (n10 - n00) + v * (n11 - n01))) * m_levelSize0f.x;
            dndv = m_objectToWorld(Normal((1.0f - u) * (n01 - n00) + u * (n11 - n10))) * m_levelSize0f.y;
        } else {
            /* Derivatives for bilinear patch with geometric normals */
            Float
                f00 = m_data[y     * width + x],
                f01 = m_data[(y+1) * width + x],
                f10 = m_data[y     * width + x + 1],
                f11 = m_data[(y+1) * width + x + 1];

            normal = m_objectToWorld(
                Normal(f00 - f10 + (f01 + f10 - f00 - f11)*v,
                       f00 - f01 + (f01 + f10 - f00 - f11)*u, 1));

            dndu = m_objectToWorld(Normal(0, f01 + f10 - f00 - f11, 0)) * m_levelSize0f.x;
            dndv = m_objectToWorld(Normal(f01 + f10 - f00 - f11, 0, 0)) * m_levelSize0f.y;
        }

        /* Account for normalization */
        Float invLength = 1/normal.length();

        normal *= invLength;
        dndu *= invLength;
        dndv *= invLength;

        dndu -= dot(normal, dndu) * normal;
        dndv -= dot(normal, dndv) * normal;
    }

    void addChild(const std::string &name, ConfigurableObject *child) {
        const Class *cClass = child->getClass();
        if (cClass->derivesFrom(Texture::m_theClass)) {
            if (m_data != NULL)
                Log(EError, "Attempted to attach multiple textures to a height field shape!");

            m_bitmap = static_cast<Texture *>(child)->getBitmap(m_sizeHint);
        } else if (cClass->derivesFrom(ReconstructionFilter::m_theClass)) {
            if (m_rfilter != NULL)
                Log(EError, "Attempted to attach multiple reconstruction filters to a height field shape!");

            m_rfilter = static_cast<ReconstructionFilter *>(child);
        } else {
            Shape::addChild(name, child);
        }
    }

    void configure() {
        Shape::configure();

        if (m_minmax)
            return;

        if (!m_filename.empty()) {
            if (m_bitmap.get())
                Log(EError, "Cannot specify a file name and a nested texture at the same time!");
            ref<FileStream> fs = new FileStream(m_filename, FileStream::EReadOnly);
            m_bitmap = new Bitmap(Bitmap::EAuto, fs);
        } else if (!m_bitmap.get()) {
            Log(EError, "A height field texture must be specified (either as a nested texture, or using the 'filename' parameter)");
        }

        m_dataSize = m_bitmap->getSize();
        if (m_dataSize.x < 2) m_dataSize.x = 2;
        if (m_dataSize.y < 2) m_dataSize.y = 2;
        if (!math::isPowerOfTwo(m_dataSize.x - 1)) m_dataSize.x = (int) math::roundToPowerOfTwo((uint32_t) m_dataSize.x - 1) + 1;
        if (!math::isPowerOfTwo(m_dataSize.y - 1)) m_dataSize.y = (int) math::roundToPowerOfTwo((uint32_t) m_dataSize.y - 1) + 1;

        if (m_bitmap->getSize() != m_dataSize) {
            m_bitmap = m_bitmap->convert(Bitmap::ELuminance, Bitmap::EFloat);

            Log(EInfo, "Resampling heightfield texture from %ix%i to %ix%i ..",
                m_bitmap->getWidth(), m_bitmap->getHeight(), m_dataSize.x, m_dataSize.y);

            m_bitmap = m_bitmap->resample(m_rfilter, ReconstructionFilter::EClamp,
                ReconstructionFilter::EClamp, m_dataSize,
                -std::numeric_limits<Float>::infinity(),
                std::numeric_limits<Float>::infinity());
        }

        size_t size = (size_t) m_dataSize.x * (size_t) m_dataSize.y * sizeof(Float);
        m_data = (Float *) allocAligned(size);
        m_bitmap->convert(m_data, Bitmap::ELuminance, Bitmap::EFloat, 1.0f, m_scale);

        m_objectToWorld = m_objectToWorld * Transform::translate(Vector(-1, -1, 0)) * Transform::scale(Vector(
            (Float) 2 / (m_dataSize.x-1),
            (Float) 2 / (m_dataSize.y-1), 1));

        m_bitmap = NULL;

        size_t storageSize = (size_t) m_dataSize.x * (size_t) m_dataSize.y * sizeof(Float);
        Log(EInfo, "Building acceleration data structure for %ix%i height field ..", m_dataSize.x, m_dataSize.y);

        ref<Timer> timer = new Timer();
        m_levelCount = (int) std::max(math::log2i((uint32_t) m_dataSize.x-1), math::log2i((uint32_t) m_dataSize.y-1)) + 1;

        m_levelSize = new Vector2i[m_levelCount];
        m_numChildren = new Vector2i[m_levelCount];
        m_blockSize = new Vector2i[m_levelCount];
        m_blockSizeF = new Vector2[m_levelCount];
        m_minmax = new Interval*[m_levelCount];

        m_levelSize[0]  = Vector2i(m_dataSize.x - 1, m_dataSize.y - 1);
        m_levelSize0f  = Vector2(m_levelSize[0]);
        m_blockSize[0] = Vector2i(1, 1);
        m_blockSizeF[0] = Vector2(1, 1);
        m_invSize = Vector2((Float) 1 / m_levelSize[0].x, (Float) 1 / m_levelSize[0].y);
        m_surfaceArea = 0;

        size = (size_t) m_levelSize[0].x * (size_t) m_levelSize[0].y * sizeof(Interval);
        m_minmax[0] = (Interval *) allocAligned(size);
        storageSize += size;

        /* Build the lowest MIP layer directly from the heightfield data */
        Interval *bounds = m_minmax[0];
        for (int y=0; y<m_levelSize[0].y; ++y) {
            for (int x=0; x<m_levelSize[0].x; ++x) {
                Float f00 = m_data[y * m_dataSize.x + x];
                Float f10 = m_data[y * m_dataSize.x + x + 1];
                Float f01 = m_data[(y + 1) * m_dataSize.x + x];
                Float f11 = m_data[(y + 1) * m_dataSize.x + x + 1];
                Float fmin = std::min(std::min(f00, f01), std::min(f10, f11));
                Float fmax = std::max(std::max(f00, f01), std::max(f10, f11));
                *bounds++ = Interval(fmin, fmax);

                /* Estimate the total surface area (this is approximate) */
                Float diff0 = f01-f10, diff1 = f00-f11;
                m_surfaceArea += std::sqrt(1.0f + .5f * (diff0*diff0 + diff1*diff1));
            }
        }

        /* Propagate height bounds upwards to the other layers */
        for (int level=1; level<m_levelCount; ++level) {
            Vector2i &cur  = m_levelSize[level],
                     &prev = m_levelSize[level-1];

            /* Calculate size of this layer */
            cur.x = prev.x > 1 ? (prev.x / 2) : 1;
            cur.y = prev.y > 1 ? (prev.y / 2) : 1;

            m_numChildren[level].x = prev.x > 1 ? 2 : 1;
            m_numChildren[level].y = prev.y > 1 ? 2 : 1;
            m_blockSize[level] = Vector2i(
                m_levelSize[0].x / cur.x,
                m_levelSize[0].y / cur.y
            );
            m_blockSizeF[level] = Vector2(m_blockSize[level]);

            /* Allocate memory for interval data */
            Interval *prevBounds = m_minmax[level-1], *curBounds;
            size_t size = (size_t) cur.x * (size_t) cur.y * sizeof(Interval);
            m_minmax[level] = curBounds = (Interval *) allocAligned(size);
            storageSize += size;

            /* Build by querying the previous layer */
            for (int y=0; y<cur.y; ++y) {
                int y0 = std::min(2*y,   prev.y-1),
                    y1 = std::min(2*y+1, prev.y-1);
                for (int x=0; x<cur.x; ++x) {
                    int x0 = std::min(2*x,   prev.x-1),
                        x1 = std::min(2*x+1, prev.x-1);
                    const Interval &f00 = prevBounds[y0 * prev.x + x0], &f01 = prevBounds[y0 * prev.x + x1];
                    const Interval &f10 = prevBounds[y1 * prev.x + x0], &f11 = prevBounds[y1 * prev.x + x1];
                    Interval combined(f00);
                    combined.expandBy(f01);
                    combined.expandBy(f10);
                    combined.expandBy(f11);
                    *curBounds++ = combined;
                }
            }
        }

        if (m_shadingNormals) {
            Log(EInfo, "Precomputing shading normals ..");
            size_t size = (size_t) m_dataSize.x * (size_t) m_dataSize.y * sizeof(Normal);
            m_normals = (Normal *) allocAligned(size);
            memset(m_normals, 0, size);
            storageSize += size;

            for (int offset=0; offset<2; ++offset) {
                #if defined(MTS_OPENMP)
                    #pragma omp parallel for
                #endif
                for (int y=offset; y<m_levelSize[0].y; y+=2) {
                    for (int x=0; x<m_levelSize[0].x; ++x) {
                            Float f00 = m_data[y * m_dataSize.x + x];
                        Float f10 = m_data[y * m_dataSize.x + x + 1];
                        Float f01 = m_data[(y + 1) * m_dataSize.x + x];
                        Float f11 = m_data[(y + 1) * m_dataSize.x + x + 1];

                        m_normals[y       * m_dataSize.x + x]     += normalize(Normal(f00 - f10, f00 - f01, 1));
                        m_normals[y       * m_dataSize.x + x + 1] += normalize(Normal(f00 - f10, f10 - f11, 1));
                        m_normals[(y + 1) * m_dataSize.x + x]     += normalize(Normal(f01 - f11, f00 - f01, 1));
                        m_normals[(y + 1) * m_dataSize.x + x + 1] += normalize(Normal(f01 - f11, f10 - f11, 1));
                    }
                }
            }

            #if defined(MTS_OPENMP)
                #pragma omp parallel for
            #endif
            for (int y=0; y<m_dataSize.y; ++y) {
                for (int x=0; x<m_dataSize.x; ++x) {
                    Normal &normal = m_normals[x + y * m_dataSize.x];
                    normal /= normal.length();
                }
            }
        }

        Log(EInfo, "Done (took %i ms, uses %s of memory)", timer->getMilliseconds(),
                memString(storageSize).c_str());

        if (!m_snapshot.empty()) {
            std::ofstream out(m_snapshot.c_str(), std::ios::binary);
            int header[2]={m_dataSize.x,m_dataSize.y};
            out.write((const char*)header,sizeof(header));
            size_t count=(size_t)m_dataSize.x*m_dataSize.y;
            out.write((const char*)m_data,count*sizeof(Float));
            out.write((const char*)m_normals,count*sizeof(Normal));
            if(!out.good()) Log(EError,"GPU diagnostic snapshot failed");
        }
        m_dataAABB = AABB(
            Point3(0, 0, m_minmax[m_levelCount-1][0].min),
            Point3(m_levelSize0f.x, m_levelSize0f.y, m_minmax[m_levelCount-1][0].max)
        );
    }

    ref<TriMesh> createTriMesh() {
        Vector2i size = m_dataSize;

        /* Limit the size of the mesh */
        while (size.x > 256 && size.y > 256) {
            size.x = std::max(size.x / 2, 2);
            size.y = std::max(size.y / 2, 2);
        }

        size_t numTris = 2 * (size_t) (size.x-1) * (size_t) (size.y-1);
        size_t numVertices = (size_t) size.x * (size_t) size.y;

        ref<TriMesh> mesh = new TriMesh("Height field approximation",
            numTris, numVertices, false, true, false, false, !m_shadingNormals);

        Point *vertices = mesh->getVertexPositions();
        Point2 *texcoords = mesh->getVertexTexcoords();
        Triangle *triangles = mesh->getTriangles();

        Float dx = (Float) 1 / (size.x - 1);
        Float dy = (Float) 1 / (size.y - 1);
        Float scaleX = (Float) m_dataSize.x / size.x;
        Float scaleY = (Float) m_dataSize.y / size.y;

        uint32_t vertexIdx = 0;
        for (int y=0; y<size.y; ++y) {
            int py = std::min((int) (scaleY * y), m_dataSize.y-1);
            for (int x=0; x<size.x; ++x) {
                int px = std::min((int) (scaleX * x), m_dataSize.x-1);
                texcoords[vertexIdx] = Point2(x*dx, y*dy);
                vertices[vertexIdx++] = m_objectToWorld(Point((Float) px, (Float) py,
                    m_data[px + py*m_dataSize.x]));
            }
        }
        Assert(vertexIdx == numVertices);

        uint32_t triangleIdx = 0;
        for (int y=1; y<size.y; ++y) {
            for (int x=0; x<size.x-1; ++x) {
                uint32_t nextx = x + 1;
                uint32_t idx0 = size.x*y + x;
                uint32_t idx1 = size.x*y + nextx;
                uint32_t idx2 = size.x*(y-1) + x;
                uint32_t idx3 = size.x*(y-1) + nextx;

                triangles[triangleIdx].idx[0] = idx0;
                triangles[triangleIdx].idx[1] = idx2;
                triangles[triangleIdx].idx[2] = idx1;
                triangleIdx++;
                triangles[triangleIdx].idx[0] = idx1;
                triangles[triangleIdx].idx[1] = idx2;
                triangles[triangleIdx].idx[2] = idx3;
                triangleIdx++;
            }
        }
        Assert(triangleIdx == numTris);
        mesh->copyAttachments(this);
        mesh->configure();

        return mesh.get();
    }

    std::string toString() const {
        std::ostringstream oss;
        oss << "HeightField[" << endl
            << "  size = " << m_dataSize.toString() << "," << endl
            << "  shadingNormals = " << m_shadingNormals << "," << endl
            << "  flipNormals = " << m_flipNormals << "," << endl
            << "  objectToWorld = " << indent(m_objectToWorld.toString()) << "," << endl
            << "  aabb = " << indent(getAABB().toString()) << "," << endl
            << "  bsdf = " << indent(m_bsdf.toString()) << "," << endl;
        if (isMediumTransition())
            oss << "  interiorMedium = " << indent(m_interiorMedium.toString()) << "," << endl
                << "  exteriorMedium = " << indent(m_exteriorMedium.toString()) << "," << endl;
        oss << "  emitter = " << indent(m_emitter.toString()) << "," << endl
            << "  sensor = " << indent(m_sensor.toString()) << "," << endl
            << "  subsurface = " << indent(m_subsurface.toString()) << endl
            << "]";
        return oss.str();
    }

    MTS_DECLARE_CLASS()
private:
    std::string m_snapshot;
    ref<ReconstructionFilter> m_rfilter;
    ref<Bitmap> m_bitmap;
    Transform m_objectToWorld;
    Vector2i m_sizeHint;
    AABB m_dataAABB;
    bool m_shadingNormals;
    bool m_flipNormals;
    Float m_scale;
    fs::path m_filename;

    /* Height field data */
    Float *m_data;
    Normal *m_normals;
    Vector2i m_dataSize;
    Vector2 m_invSize;
    Float m_surfaceArea;

    /* Min-max quadtree data */
    int m_levelCount;
    Vector2i *m_levelSize;
    Vector2   m_levelSize0f;
    Vector2i *m_numChildren;
    Vector2i *m_blockSize;
    Vector2 *m_blockSizeF;
    Interval **m_minmax;
};

MTS_IMPLEMENT_CLASS_S(Heightfield, false, Shape)
MTS_EXPORT_PLUGIN(Heightfield, "Height field intersection shape");
MTS_NAMESPACE_END

