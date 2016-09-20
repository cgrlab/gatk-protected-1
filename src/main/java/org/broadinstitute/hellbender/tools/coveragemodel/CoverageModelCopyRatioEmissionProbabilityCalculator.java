package org.broadinstitute.hellbender.tools.coveragemodel;

import org.apache.commons.math3.special.Erf;
import org.apache.commons.math3.util.FastMath;
import org.broadinstitute.hellbender.tools.coveragemodel.interfaces.TargetLikelihoodCalculator;
import org.broadinstitute.hellbender.tools.exome.Target;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.io.Serializable;

/**
 * @author Mehrtash Babadi &lt;mehrtash@broadinstitute.org&gt;
 */
public class CoverageModelCopyRatioEmissionProbabilityCalculator implements
        TargetLikelihoodCalculator<CoverageModelCopyRatioEmissionData>, Serializable {

    private static final long serialVersionUID = -6985799468753075235L;

    private static final boolean CHECK_FOR_NANS = true;

    /**
     * Calculate the log emission probability. The parameter {@param target} is not used since
     * {@param emissionData} contains the necessary information.
     *
     * @param emissionData an instance of {@link CoverageModelCopyRatioEmissionData}
     * @param copyRatio copy ratio for which the emission probability is calculated
     * @param target target on which the emission probability is calculated (this parameter is not used)
     * @return emission probability
     */
    @Override
    public double logLikelihood(@Nonnull CoverageModelCopyRatioEmissionData emissionData,
                                   double copyRatio, @Nullable Target target) {
        final double mu = emissionData.getNeutralMean() + FastMath.log(copyRatio);
        final double lambda = emissionData.getPrecision();

        final double res = emissionData.getPartialLogNormalizationFactor() - mu
                - FastMath.log(1 + Erf.erf((1 + lambda * mu) / FastMath.sqrt(2 * lambda)))
                - 0.5 * lambda * FastMath.pow(emissionData.getLogReadCount() - mu, 2);

        if (CHECK_FOR_NANS) {
            if (Double.isNaN(res) || Double.isInfinite(res)) {
                throw new RuntimeException("Something went wrong while calculating the likelihood; the" +
                        " emission data is: " + emissionData.toString());
            }
        }
        return res;
    }
}
