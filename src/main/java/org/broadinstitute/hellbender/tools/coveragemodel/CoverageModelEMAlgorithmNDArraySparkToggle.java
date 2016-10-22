package org.broadinstitute.hellbender.tools.coveragemodel;

import org.broadinstitute.hellbender.tools.coveragemodel.annots.CachesRDD;
import org.broadinstitute.hellbender.tools.coveragemodel.annots.EvaluatesRDD;
import org.broadinstitute.hellbender.tools.coveragemodel.annots.UpdatesRDD;
import org.broadinstitute.hellbender.utils.hmm.interfaces.AlleleMetadataProvider;
import org.broadinstitute.hellbender.utils.hmm.interfaces.CallStringProvider;
import org.broadinstitute.hellbender.utils.hmm.interfaces.ScalarProvider;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;

/**
 *
 * @author Mehrtash Babadi &lt;mehrtash@broadinstitute.org&gt;
 */
public final class CoverageModelEMAlgorithmNDArraySparkToggle<S extends AlleleMetadataProvider & CallStringProvider &
        ScalarProvider> extends CoverageModelEMAlgorithm<S> {

    private final CoverageModelEMWorkspaceNDArraySparkToggle<S> ws;

    public CoverageModelEMAlgorithmNDArraySparkToggle(@Nonnull final CoverageModelEMParams params,
                                                      @Nullable final String outputAbsolutePath,
                                                      @Nonnull final S neutralState,
                                                      @Nonnull final CoverageModelEMWorkspaceNDArraySparkToggle<S> ws) {
        super(params, outputAbsolutePath, neutralState);
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
    public SubroutineSignal updateSampleUnexplainedVariance() {
        return ws.updateSampleUnexplainedVariance();
    }

    @Override @EvaluatesRDD @UpdatesRDD @CachesRDD
    public SubroutineSignal updateCopyRatioLatentPosteriorExpectations() {
        return ws.updateCopyRatioLatentPosteriorExpectations();
    }

    @Override @EvaluatesRDD @UpdatesRDD @CachesRDD
    public SubroutineSignal updateTargetMeanBias(final boolean neglectPCBias) {
        return ws.updateTargetMeanBias(neglectPCBias);
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

    @Override @EvaluatesRDD
    public void saveModel(final String modelOutputPath) {
        ws.saveModel(modelOutputPath);
    }

    @Override @EvaluatesRDD
    public void savePosteriors(final String posteriorOutputPath, final PosteriorVerbosityLevel verbosity) {
        ws.savePosteriors(neutralState, posteriorOutputPath, verbosity, null);
    }

    @Override @EvaluatesRDD
    public void finalizeIteration() {
        ws.performGarbageCollection();
    }
}
