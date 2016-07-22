package org.broadinstitute.hellbender.tools.coveragemodel;

import org.broadinstitute.hellbender.tools.coveragemodel.annots.CachesRDD;
import org.broadinstitute.hellbender.tools.coveragemodel.annots.EvaluatesRDD;
import org.broadinstitute.hellbender.tools.coveragemodel.annots.UpdatesRDD;

import javax.annotation.Nonnull;

/**
 *
 * @author Mehrtash Babadi &lt;mehrtash@broadinstitute.org&gt;
 */
public final class CoverageModelEMAlgorithmNDArraySparkToggle extends CoverageModelEMAlgorithm {

    private final CoverageModelEMWorkspaceNDArraySparkToggle ws;

    public CoverageModelEMAlgorithmNDArraySparkToggle(@Nonnull final CoverageModelEMParams params,
                                                      @Nonnull final CoverageModelEMWorkspaceNDArraySparkToggle ws) {
        super(params);
        this.ws = ws;
    }

    @Override @EvaluatesRDD @UpdatesRDD @CachesRDD
    public SubroutineSignal updateBiasLatentPosteriorExpectations() {
        return ws.updateBiasLatentPosteriorExpectations();
    }

    @Override @EvaluatesRDD @UpdatesRDD @CachesRDD
    public SubroutineSignal updateReadDepthLatentPosteriorExpectations() {
        return ws.updateReadDepthLatentPosteriorExpectations();
    }

    @Override @EvaluatesRDD @UpdatesRDD @CachesRDD
    public SubroutineSignal updateCopyRatioLatentPosteriorExpectations(final boolean performViterbi) {
        return ws.updateCopyRatioLatentPosteriorExpectations(performViterbi);
    }

    @Override @EvaluatesRDD @UpdatesRDD @CachesRDD
    public SubroutineSignal updateTargetMeanBias() {
        return ws.updateTargetMeanBias();
    }

    @Override @EvaluatesRDD @UpdatesRDD @CachesRDD
    public SubroutineSignal updateTargetUnexplainedVariance() {
        return ws.updateTargetUnexplainedVariance();
    }

    @Override @EvaluatesRDD @UpdatesRDD @CachesRDD
    public SubroutineSignal updatePrincipalLatentTargetMap() {
        return ws.updatePrincipalLatentToTargetMap();
    }

    @Override @EvaluatesRDD @CachesRDD
    public double getLogLikelihood() {
        return ws.getLogLikelihood();
    }

    @Override @EvaluatesRDD @CachesRDD
    public double[] getLogLikelihoodPerSample() {
        return ws.getLogLikelihoodPerSample();
    }
}
