package org.broadinstitute.hellbender.tools.coveragemodel;

import org.broadinstitute.hellbender.tools.coveragemodel.annots.CachesRDD;
import org.broadinstitute.hellbender.tools.coveragemodel.annots.EvaluatesRDD;
import org.broadinstitute.hellbender.tools.coveragemodel.annots.UpdatesRDD;
import org.broadinstitute.hellbender.utils.hmm.interfaces.AlleleMetadataProvider;
import org.broadinstitute.hellbender.utils.hmm.interfaces.CallStringProvider;
import org.broadinstitute.hellbender.utils.hmm.interfaces.ScalarProvider;

import javax.annotation.Nonnull;

/**
 *
 * @author Mehrtash Babadi &lt;mehrtash@broadinstitute.org&gt;
 */
public final class CoverageModelEMAlgorithmNDArraySparkToggle<S extends AlleleMetadataProvider & CallStringProvider &
        ScalarProvider> extends CoverageModelEMAlgorithm {

    private final CoverageModelEMWorkspaceNDArraySparkToggle<S> ws;

    public CoverageModelEMAlgorithmNDArraySparkToggle(@Nonnull final CoverageModelEMParams params,
                                                      @Nonnull final CoverageModelEMWorkspaceNDArraySparkToggle<S> ws) {
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
    public SubroutineSignal updateCopyRatioLatentPosteriorExpectations() {
        return ws.updateCopyRatioLatentPosteriorExpectations();
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
