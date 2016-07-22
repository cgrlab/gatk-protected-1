package org.broadinstitute.hellbender.tools.coveragemodel;

import org.apache.commons.math3.util.FastMath;
import org.broadinstitute.hellbender.utils.param.ParamUtils;

import java.io.Serializable;

/**
 * This class stores the required data for calculating the normalized emission probability according to the
 * probabilistic target coverage model.
 *
 * The stored data is as follows:
 *
 * <p>
 *     The mean of the emission probability distribution calculated for n_{st} = 1, c_{st} = 1:
 *     {@link #neutralMean} = \log(P_{st}) + E[\log(d_s)] + m_t + (W.E[z_s])_t
 * </p>
 *
 * <p>
 *     The precision of the distribution:
 *     {@link #precision} = \Psi_{st}^{-1}
 * </p>
 *
 * <p>
 *     Log read count:
 *     {@link #logReadCount} = \log(n_{st})
 * </p>
 *
 * @author Mehrtash Babadi &lt;mehrtash@broadinstitute.org&gt;
 */
public final class CoverageModelCopyRatioEmissionData implements Serializable {

    private static final long serialVersionUID = -7363264674200250712L;

    private double neutralMean, precision, logReadCount;

    /**
     * A useful quantity to calculate in advance: (1/2) \log(\pi/(2\Lambda)) - 1/(2\Lambda)
     */
    private double partialLogNormalizationFactor;

    /**
     *
     * @param neutralMean
     * @param precision
     * @param logReadCount
     */
    public CoverageModelCopyRatioEmissionData(final double neutralMean, final double precision, final double logReadCount) {
        this.neutralMean = neutralMean;
        this.precision = ParamUtils.isPositive(precision, "Precision must be a positive real number. Bad value: " + precision);
        this.logReadCount = ParamUtils.isPositiveOrZero(logReadCount, "Log read count must be a positive real number. Bad value: " + logReadCount);
        this.partialLogNormalizationFactor = 0.5 * (FastMath.log(FastMath.PI/(2*precision)) - 1.0/precision);
    }

    /**
     *
     * @return
     */
    public double getNeutralMean() {
        return neutralMean;
    }

    /**
     *
     * @return
     */
    public double getPrecision() {
        return precision;
    }

    /**
     *
     * @return
     */
    public double getLogReadCount() {
        return logReadCount;
    }

    /**
     *
     * @return
     */
    public double getPartialLogNormalizationFactor() {
        return partialLogNormalizationFactor;
    }

    /**
     *
     * @return
     */
    public double getCopyRatioMaxLikelihoodEstimate() {
        return FastMath.exp(logReadCount - neutralMean);
    }

    @Override
    public String toString() {
        return "CoverageModelCopyRatioEmissionData{" +
                "neutralMean=" + neutralMean +
                ", precision=" + precision +
                ", logReadCount=" + logReadCount +
                ", partialLogNormalizationFactor=" + partialLogNormalizationFactor +
                '}';
    }
}
