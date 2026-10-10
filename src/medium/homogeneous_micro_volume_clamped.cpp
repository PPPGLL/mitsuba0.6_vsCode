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

#include <mitsuba/render/scene.h>
#include <mitsuba/render/volume.h>
#include <mitsuba/core/statistics.h>
#include <boost/algorithm/string.hpp>
#include <algorithm>

MTS_NAMESPACE_BEGIN

class Micro_ellipsoid : public Medium {
public:
    Micro_ellipsoid(const Properties &props) : Medium(props) {
        m_density = props.getFloat("density");
        m_albedo = props.getSpectrum("albedo");
        m_orientation = props.getVector("orientation");

		m_saturation = props.getFloat("saturation", 0.f);
		m_sigmaWet = props.getSpectrum("sigmaWet", Spectrum(0.f));
		m_porosity = props.getFloat("porosity", 1);
		
    }

    /* Unserialize from a binary data stream */
    Micro_ellipsoid(Stream *stream, InstanceManager *manager) : Medium(stream, manager) {
        m_density = stream->readFloat();
        m_albedo = Spectrum(stream);
        m_orientation = Vector(stream);
        configure();
    }

    /* Serialize the volume to a binary data stream */
    void serialize(Stream *stream, InstanceManager *manager) const {
        Medium::serialize(stream, manager);
        stream->writeFloat(m_density);
        m_albedo.serialize(stream);
        m_orientation.serialize(stream);
    }

    void configure() {
        Medium::configure();
        if ( !m_phaseFunction->needsDirectionallyVaryingCoefficients() )
            Log(EError, "Phase function needs to be anisotropic!");
        if ( m_orientation.isZero() )
            Log(EError, "Invalid orientation!");
        else
            m_orientation = normalize(m_orientation);

		Float DsDv = m_phaseFunction->DsDv();
		SLog(EDebug, "The Ds/Dv is %f", DsDv);
		if (m_porosity == 1)
			m_K = 1;
		else {
			Float EL = 1.209 * pow(math::clamp((1 - m_porosity)* DsDv, 0.f, 0.75f), 2.0 / 3.0);
			m_K = -log(1 - EL) / EL;
		}
		SLog(EDebug, "The m_K is %f", m_K);
		SLog(EDebug, "The saturation is %f, density %f, porosity %f, albedo %f", m_saturation, m_density, m_porosity, m_albedo[0]);
    }

    Spectrum evalTransmittance(const Ray &ray, Sampler *sampler) const {
        //Float t = std::max(ray.maxt - ray.mint, static_cast<Float>(0.0));
        //return Spectrum(math::fastexp(-t*lookupDensity(ray.d)));
		/**/
		Float sigmaT = lookupDensity(ray.d);
		Float negLength = ray.mint - ray.maxt;
		Spectrum transmittance;
		Float msl = log(m_K) / (m_K * sigmaT);
		Float Tc = m_K * math::fastexp(m_K * (sigmaT)* negLength);
		for (int i = 0; i < SPECTRUM_SAMPLES; ++i){
			Float Tcl = 1;
			if (-negLength>msl)
				Tcl = math::fastexp(m_saturation * m_sigmaWet[i] * (negLength + msl));
			transmittance[i] = Tc * Tcl;
		}
		return transmittance;
    }

    bool sampleDistance(const Ray &ray, MediumSamplingRecord &mRec, Sampler *sampler) const {
        Float sigmaT = lookupDensity(ray.d);
		Float msl = math::fastlog(m_K) / (m_K * lookupDensity(ray.d));
		if (sigmaT == 0.0f)
			return false;
		Float sampledDistance;
		Float rand = sampler->next1D();
		if (rand < math::fastlog(m_K) / (1 + math::fastlog(m_K)))
			sampledDistance = (1 + math::fastlog(m_K)) / m_K / sigmaT*rand;
		else
			sampledDistance = -1 / m_K / sigmaT*math::fastlog((math::fastlog(m_K) + 1 - rand*(1 + math::fastlog(m_K))) / m_K);

		//Float sampledDistance = -math::fastlog(rand) / sigmaT;
		//Float pdf = sigmaT * math::fastexp(-sigmaT*sampledDistance);
        
        bool success = true;
        if ( sampledDistance < ray.maxt - ray.mint ) {
            mRec.t = sampledDistance + ray.mint;
            mRec.p = ray(mRec.t);
            mRec.sigmaS = m_albedo*sigmaT;
            mRec.sigmaA = Spectrum(sigmaT) - mRec.sigmaS;
            mRec.orientation = m_orientation;

            //Fail if there is no forward progress
            //   (e.g. due to roundoff errors) 
            if ( mRec.p == ray.o ) success = false;
        }
        else {
            sampledDistance = ray.maxt - ray.mint;
            success = false;
        }
		Spectrum Tcl(1.f);
		//for (int i = 0; i < 3; i++){
		//	Tcl[i] = math::fastexp(-m_saturation * m_sigmaWet[i] * sampledDistance);
		//}
		Float T, p;
		if (sampledDistance < msl){
			T = 1 - m_K*sigmaT*sampledDistance / (1 + math::fastlog(m_K));
			p = m_K*sigmaT / (1 + math::fastlog(m_K));
			Tcl = Spectrum(1.0f);
		}
		else{
			T = m_K * math::fastexp(-sigmaT*sampledDistance*m_K) / (1 + math::fastlog(m_K));
			p = m_K * m_K * sigmaT * math::fastexp(-sigmaT*sampledDistance*m_K) / (1 + math::fastlog(m_K));
			for (int i = 0; i < 3; i++){
				Tcl[i] = math::fastexp(-m_saturation * m_sigmaWet[i] * (sampledDistance - msl));
			}
		}
        mRec.pdfFailure = T;
		mRec.transmittance = Spectrum(T)*Tcl;
        mRec.pdfSuccess = mRec.pdfSuccessRev = p;
		mRec.sampleWeight = m_albedo * Tcl;
		//mRec.sampleWeight = p * Tcl / pdf;
        mRec.time = ray.time;
        mRec.medium = this;

        if ( mRec.transmittance.max() < 1e-20 ) mRec.transmittance = Spectrum(0.0f);
        return success;
    }

	bool sampleDistanceC(const Ray &ray, MediumSamplingRecord &mRec, Sampler *sampler) const override{
		Float msl = log(m_K) / (m_K * lookupDensity(ray.d));
		Float sigmaT = lookupDensity(ray.d);
		if (sigmaT == 0.0f)
			return false;

		Float sampledDistance = -math::fastlog(1 - sampler->next1D())/m_K /sigmaT + msl;
		//Float sampledDistance = -math::fastlog(sampler->next1D()) / sigmaT;
		//Float pdf = sigmaT * math::fastexp(-sigmaT*sampledDistance);

		bool success = true;
		if ( sampledDistance < ray.maxt - ray.mint ) {
			mRec.t = sampledDistance + ray.mint;
			mRec.p = ray(mRec.t);
			mRec.sigmaS = m_albedo*sigmaT;
			mRec.sigmaA = Spectrum(sigmaT) - mRec.sigmaS;
			mRec.orientation = m_orientation;

			//Fail if there is no forward progress
			//   (e.g. due to roundoff errors)
			if ( mRec.p == ray.o ) success = false;
		}
		else {
			sampledDistance = ray.maxt - ray.mint;
			success = false;
		}

		Float expVal = std::min(1.f, m_K * math::fastexp(-m_K *sigmaT*sampledDistance));
		Spectrum Tcl(1.f);
		if (sampledDistance>msl){
			for (int i = 0; i < 3; i++){
				Tcl[i] = math::fastexp(-m_saturation * m_sigmaWet[i] * (sampledDistance - msl));
			}
		}
		mRec.pdfFailure = expVal;
		mRec.transmittance = Spectrum(expVal) * Tcl;
		mRec.pdfSuccess = mRec.pdfSuccessRev = m_K *sigmaT*expVal;
		mRec.sampleWeight = m_albedo * Tcl;
		//mRec.sampleWeight = sigmaT*expVal*Tcl / pdf;
		mRec.time = ray.time;
		mRec.medium = this;

		if ( mRec.transmittance.max() < 1e-20 ) mRec.transmittance = Spectrum(0.0f);
		return success;
	}

    void eval(const Ray &ray, MediumSamplingRecord &mRec) const {
		/*
        Float sigmaT = lookupDensity(ray.d);
        Float distance = ray.maxt - ray.mint;

        mRec.sigmaS = m_albedo*sigmaT;
        mRec.sigmaA = Spectrum(sigmaT) - mRec.sigmaS;
        mRec.orientation = m_orientation;

        Float expVal = math::fastexp(-sigmaT*distance);
        mRec.pdfFailure = expVal;
        mRec.transmittance = Spectrum(expVal);
        mRec.pdfSuccess = mRec.pdfSuccessRev = sigmaT*expVal;
        mRec.time = ray.time;
        mRec.medium = this;

        if ( mRec.transmittance.max() < 1e-20 ) mRec.transmittance = Spectrum(0.0f);
*/
		Float sigmaTsggx = m_K * lookupDensity(ray.d);
		Spectrum sigmaT = Spectrum(sigmaTsggx) + m_saturation*m_sigmaWet;
		Float distance = ray.maxt - ray.mint;

		mRec.pdfFailure = 0;
		mRec.pdfSuccess = 0;
		for (int i = 0; i<SPECTRUM_SAMPLES; ++i) {
			Float tmp = math::fastexp(-sigmaT[i] * distance);
			mRec.pdfFailure += tmp;
			mRec.pdfSuccess += sigmaT[i] * tmp;
		}
		mRec.pdfFailure /= SPECTRUM_SAMPLES;
		mRec.pdfSuccess /= SPECTRUM_SAMPLES;


		mRec.transmittance = (sigmaT * (-distance)).exp();
		mRec.pdfSuccessRev = mRec.pdfSuccess;
		mRec.medium = this;
		if (mRec.transmittance.max() < 1e-20)
			mRec.transmittance = Spectrum(0.0f);

		mRec.sigmaS = m_albedo*sigmaTsggx;
		mRec.sigmaA = sigmaT - mRec.sigmaS;
		mRec.orientation = m_orientation;
		mRec.time = ray.time;
		mRec.medium = this;
		
    }

    bool isHomogeneous() const {
        return true;
    }

	void setMediumPropWet(const Float &density, const Spectrum &albedo, const Vector &orientation, const Float &saturation, const Spectrum &sigmaWet) {
		m_density = density;
		m_albedo = albedo;
		m_orientation = normalize(orientation);
		m_saturation = saturation;
		m_sigmaWet = sigmaWet;
	}

	void setMediumProp(const Float &density, const Spectrum &albedo, const Vector &orientation) {
		m_density = density;
		m_albedo = albedo;
		m_orientation = normalize(orientation);
	}

	Float lookupSigmaT(const Vector &d) const {
		return m_density*m_phaseFunction->sigmaDir(m_orientation, d);
	}

    std::string toString() const {
        std::ostringstream oss;
        oss << "Micro_ellipsoid[" << endl
            << "  density = " << m_density << "," << endl
            << "  albedo = " << m_albedo.toString() << "," << endl
            << "  orientation = " << m_orientation.toString() << "," << endl
            << "  phase = " << indent(m_phaseFunction->toString()) << endl
            << "]";
        return oss.str();
    }

    MTS_DECLARE_CLASS()


protected:
    inline Float lookupDensity(const Vector &d) const {
        return m_density*m_phaseFunction->sigmaDir(m_orientation, d);
    }

    Float m_density;
    Spectrum m_albedo;
    Vector m_orientation;

	Float m_saturation;
	Spectrum m_sigmaWet;
	Float m_porosity;
	Float m_K;
};

MTS_IMPLEMENT_CLASS_S(Micro_ellipsoid, false, Medium)
MTS_EXPORT_PLUGIN(Micro_ellipsoid, "Micro ellipsoid");
MTS_NAMESPACE_END
