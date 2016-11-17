package org.broadinstitute.hellbender.tools.exome.orientationbiasvariantfilter;


import org.apache.commons.math3.distribution.BinomialDistribution;

public class ArtifactStatisticsScorer {
    
    final static double DEFAULT_BIASQP1=1e8;
    final static double DEFAULT_BIASQP2=0.5;

    /** TODO: Finish docs
     *  Attenuates how many artifacts to cut.
     * @param oxoQ
     * @param biasQP1
     * @param biasQP2
     * @return
     */
    public static double calculateSuppressionFactorFromOxoQ(final double oxoQ, final double biasQP1, final double biasQP2) {
        // From matlab: fQ=1./(1+exp(biasQP2*(biasQ-biasQP1)));
        // biasQ is the oxoQ score
        return 1/(1 + Math.exp(biasQP2*(oxoQ-biasQP1)));
    }

    /**
     * See {@link #calculateSuppressionFactorFromOxoQ(double, double, double) calculateSuppressionFactorFromOxoQ}
     *
     *  Configured to behave exactly like the old OxoG filter.
     *
     * @param oxoQ See {@link #calculateSuppressionFactorFromOxoQ(double, double, double) calculateSuppressionFactorFromOxoQ}
     * @return See {@link #calculateSuppressionFactorFromOxoQ(double, double, double) calculateSuppressionFactorFromOxoQ}
     */
    public static double calculateSuppressionFactorFromOxoQ(final double oxoQ) {
        // From matlab: fQ=1./(1+exp(biasQP2*(biasQ-biasQP1)));
        // biasQ is the oxoQ score
        return 1/(1 + Math.exp(DEFAULT_BIASQP2*(oxoQ-DEFAULT_BIASQP1)));
    }

    /**
     * p-value for being an artifact
     */
    public static double calculateArtifactPValue(final int totalAltAlleleCount, final int artifactAltAlleleCount, final double biasP) {

        return new BinomialDistribution(null, totalAltAlleleCount, biasP).cumulativeProbability(artifactAltAlleleCount);
    }
}
