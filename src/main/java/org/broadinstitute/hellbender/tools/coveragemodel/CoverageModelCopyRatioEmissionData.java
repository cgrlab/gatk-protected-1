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
 *     Log multiplicative bias in the neutral copy ratio state:
 *     {@link #mu} = \log(P_{st}) + E[\log(d_s)] + m_t + (W.E[z_s])_t
 * </p>
 *
 * <p>
 *     Unexplained variance (anything not modeled, or not due to Poisson statistical uncertainty):
 *     {@link #psi} = \Psi_t
 * </p>
 *
 * <p>
 *     Raw read count:
 *     {@link #readCount} = n_{st}
 * </p>
 *
 * @author Mehrtash Babadi &lt;mehrtash@broadinstitute.org&gt;
 */
public final class CoverageModelCopyRatioEmissionData implements Serializable {

    private static final long serialVersionUID = -7363264674200250712L;

    /**
     * Log multiplicative bias in the neutral copy ratio state (see above)
     */
    private final double mu;

    /**
     * Unexplained variance (see above)
     */
    private final double psi;

    /**
     * Raw read count
     */
    private final double readCount;

    public CoverageModelCopyRatioEmissionData(final double mu, final double psi, final double readCount) {
        this.mu = mu;
        this.psi = ParamUtils.isPositive(psi, "Unexplained variance must be a positive real number. Bad value: " + psi);
        this.readCount = ParamUtils.isPositive(readCount, "Read count must be a positive real number. Bad value: " + readCount);
    }

    /**
     *
     * @return
     */
    public double getMu() {
        return mu;
    }

    /**
     *
     * @return
     */
    public double getPsi() {
        return psi;
    }

    /**
     *
     * @return
     */
    public double getReadCount() {
        return readCount;
    }

    @Override
    public String toString() {
        return "CoverageModelCopyRatioEmissionData{" +
                "mu=" + mu +
                ", psi=" + psi +
                ", readCount=" + readCount +
                '}';
    }
}
