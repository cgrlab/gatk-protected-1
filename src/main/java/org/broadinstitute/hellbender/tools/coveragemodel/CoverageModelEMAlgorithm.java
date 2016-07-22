package org.broadinstitute.hellbender.tools.coveragemodel;

import org.apache.commons.lang3.StringUtils;
import org.apache.commons.math3.util.FastMath;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.broadinstitute.hellbender.utils.Utils;
import org.nd4j.linalg.api.ndarray.INDArray;

import javax.annotation.Nonnull;
import java.util.Arrays;
import java.util.Collections;
import java.util.function.Function;
import java.util.function.Supplier;

/**
 * Implementation of the maximum likelihood estimator of the probabilistic target coverage model parameters
 * via the EM algorithm (see CNV-methods.pdf for technical details).
 *
 * @author Mehrtash Babadi &lt;mehrtash@broadinstitute.org&gt;
 */

public abstract class CoverageModelEMAlgorithm {

    protected final Logger logger = LogManager.getLogger(CoverageModelEMAlgorithm.class);

    protected final CoverageModelEMParams params;

    protected EMAlgorithmStatus status;

    public enum EMAlgorithmStatus {
        TBD(false, "Status is not determined yet."),
        SUCCESS_LIKELIHOOD_TOL(true, "Success -- converged in likelihood change tolerance."),
        SUCCESS_PARAMS_TOL(true, "Success -- converged in parameters change tolerance."),
        FAILURE_MAX_ITERS_REACHED(false, "Failure -- maximum iterations reached."),
        SUCCESS_POSTERIOR_CONVERGENCE(true, "Success -- converged in posterior and likelihood change tolerance.");

        final boolean success;
        final String message;

        EMAlgorithmStatus(final boolean success, final String message) {
            this.success = success;
            this.message = message;
        }
    }

    private final class IterationInfo {
        double logLikelihood, errorNorm;
        int iter;

        IterationInfo(final double logLikelihood, final double errorNorm, final int iter) {
            this.logLikelihood = logLikelihood;
            this.errorNorm = errorNorm;
            this.iter = iter;
        }

        void increaseIterationCount() {
            iter++;
        }
    }

    public EMAlgorithmStatus getStatus() { return status; }

    public CoverageModelEMAlgorithm(@Nonnull final CoverageModelEMParams params) {
        this.params = Utils.nonNull(params, "Target coverage EM algorithm parameters can not be null.");
        this.status = EMAlgorithmStatus.TBD;
        logger.info("EM algorithm initialized.");
    }

    public void showIterationHeader() {
        final String header = String.format("%-15s%-20s%-20s%-20s%-20s", "Iterations", "Type", "Log Likelihood", "Update Size", "Misc.");
        logger.info(header);
        logger.info(StringUtils.repeat("=", header.length()));
    }

    public void showIterationInfo(final int iter, final String type, final double logLikelihood,
                                  final double updateSize, final String misc) {
        final String header = String.format("%-15d%-20s%-20.6e%-20.6e%-20s", iter, type, logLikelihood, updateSize, misc);
        logger.info(header);
    }

    private void runRoutine(@Nonnull final Supplier<SubroutineSignal> func,
                            @Nonnull final Function<SubroutineSignal, String> miscFactory,
                            @Nonnull final String name,
                            IterationInfo iterInfo) {
        final SubroutineSignal sig = func.get();
        final String misc = miscFactory.apply(sig);
        iterInfo.errorNorm = sig.getDouble("error_norm");
        iterInfo.logLikelihood = getLogLikelihood();
        showIterationInfo(iterInfo.iter, name, iterInfo.logLikelihood, iterInfo.errorNorm, misc);
    }

    public void runExpectationMaximization(final boolean performCopyRatioPosteriorCalling) {
        /* if copy ratio posterior calling is enabled, the first few iterations need to be robust */
        if (!this.params.getPsiSolverType().equals(CoverageModelEMParams.PsiSolverType.PSI_ISOTROPIC_VIA_BRENT) &&
                performCopyRatioPosteriorCalling) {
            this.params.setPsiPsiolverType(CoverageModelEMParams.PsiSolverType.PSI_ISOTROPIC_VIA_BRENT);
            logger.info("Overriding the requested unexplained variance solver to " +
                    CoverageModelEMParams.PsiSolverType.PSI_ISOTROPIC_VIA_BRENT.name());
        }

        showIterationHeader();

        double prevLogLikelihood = Double.NEGATIVE_INFINITY;
        final IterationInfo iterInfo = new IterationInfo(Double.NEGATIVE_INFINITY, 0, 0);
        boolean updateCopyRatioPosteriors = false;
        boolean paramEstimationConverged = false;
        boolean performMStep = true;

        while (iterInfo.iter < params.getMaxIterations()) {

            /* cycle through E-step mean-field equations until they are satisfied to the desired degree */
            double maxPosteriorErrorNorm = 0;
            int iterEStep = 0;

            while (iterEStep < params.getMaxEStepCycles()) {
                double posteriorErrorNormReadDepth = 0, posteriorErrorNormBias = 0, posteriorErrorNormCopyRatio = 0;

                runRoutine(this::updateReadDepthLatentPosteriorExpectations, s -> "N/A", "E_STEP_D", iterInfo);
                posteriorErrorNormReadDepth = iterInfo.errorNorm;

                runRoutine(this::updateBiasLatentPosteriorExpectations, s -> "N/A", "E_STEP_Z", iterInfo);
                posteriorErrorNormBias = iterInfo.errorNorm;

                if (updateCopyRatioPosteriors) {
                    runRoutine(() -> updateCopyRatioLatentPosteriorExpectations(false), s -> "N/A", "E_STEP_C", iterInfo);
                    posteriorErrorNormCopyRatio = iterInfo.errorNorm;
                }

                /* calculate the maximum change of posteriors in this cycle */
                maxPosteriorErrorNorm = Collections.max(Arrays.asList(posteriorErrorNormReadDepth,
                        posteriorErrorNormBias, posteriorErrorNormCopyRatio));

                /* check convergence of the E-step */
                if (maxPosteriorErrorNorm < params.getPosteriorErrorNormTol()) {
                    break;
                }

                iterEStep++;
            }

            /* if the likelihood has increased and the increment is small, start updating copy number posteriors */
            if (performCopyRatioPosteriorCalling &&
                    !updateCopyRatioPosteriors &&
                    (iterInfo.logLikelihood - prevLogLikelihood) > 0 &&
                    (iterInfo.logLikelihood - prevLogLikelihood) < params.getLogLikelihoodTolThresholdCopyRatioCalling()) {
                updateCopyRatioPosteriors = true;
                logger.info("Partial convergence achieved; will start calling copy ratio posteriors after the current" +
                        " iteration; also, will switch to " +
                        CoverageModelEMParams.PsiSolverType.PSI_TARGET_RESOLVED_VIA_BRENT.name());
                params.setPsiPsiolverType(CoverageModelEMParams.PsiSolverType.PSI_TARGET_RESOLVED_VIA_BRENT);
            }

            if (maxPosteriorErrorNorm > params.getPosteriorErrorNormTol()) {
                logger.info("E-step cycles did not fully converge. Increase the maximum number of E-step cycles." +
                        " Continuing...");
            }

            /* parameter estimation if required */
            if (performMStep && !paramEstimationConverged) {
                prevLogLikelihood = iterInfo.logLikelihood; /* log likelihood from the previous E-step */
                int iterMStep = 0;
                double maxParamErrorNorm = 0;

                /* sequential M-steps */
                while (iterMStep < params.getMaxMStepCycles()) {
                    double errorNormMeanTargetBias = 0, errorNormUnexplainedVariance = 0, errorNormPrincipalMap = 0;

                    runRoutine(this::updateTargetMeanBias, s -> "N/A", "M_STEP_M", iterInfo);
                    errorNormMeanTargetBias = iterInfo.errorNorm;

                    runRoutine(this::updateTargetUnexplainedVariance, s -> "iters: " + s.getInteger("iterations"),
                            "M_STEP_PSI", iterInfo);
                    errorNormUnexplainedVariance = iterInfo.errorNorm;

                    if (params.fourierRegularizationEnabled()) {
                        runRoutine(this::updatePrincipalLatentTargetMap, s -> "iters: " + s.getInteger("iterations"),
                                "M_STEP_W", iterInfo);
                    } else {
                        runRoutine(this::updatePrincipalLatentTargetMap, s -> "N/A",
                                "M_STEP_W", iterInfo);
                    }
                    errorNormPrincipalMap = iterInfo.errorNorm;

                    maxParamErrorNorm = Collections.max(Arrays.asList(errorNormMeanTargetBias,
                            errorNormUnexplainedVariance, errorNormPrincipalMap));

                    /* check convergence of parameter estimation */
                    if (updateCopyRatioPosteriors && maxParamErrorNorm < params.getParameterEstimationAbsoluteTolerance()) {
                        status = EMAlgorithmStatus.SUCCESS_PARAMS_TOL;
                        paramEstimationConverged = true;
                    }
                    iterMStep++;
                }

            }

            /* check convergence in log likelihood change */
            if (FastMath.abs(iterInfo.logLikelihood - prevLogLikelihood) < params.getLogLikelihoodTolerance()) {
                if (!performCopyRatioPosteriorCalling || updateCopyRatioPosteriors) {
                    status = EMAlgorithmStatus.SUCCESS_LIKELIHOOD_TOL;
                    break;
                }
            } else if (iterInfo.iter == params.getMaxIterations() - 2) {
                performMStep = false; /* so that we end with the E-step */
            }

            iterInfo.increaseIterationCount();
            prevLogLikelihood = iterInfo.logLikelihood;
        }

        if (iterInfo.iter == params.getMaxIterations()) {
            status = EMAlgorithmStatus.FAILURE_MAX_ITERS_REACHED;
        }

        performPostEMOperations();
    }

    /**
     *
     */
    public void runExpectation(final boolean performCopyRatioPosteriorCalling) {

        showIterationHeader();

        double prevLogLikelihood = Double.NEGATIVE_INFINITY;
        final IterationInfo iterInfo = new IterationInfo(Double.NEGATIVE_INFINITY, 0, 0);
        boolean updateCopyRatioPosteriors = false;

        while (iterInfo.iter < params.getMaxIterations()) {

            /* cycle through E-step mean-field equations until they are satisfied to the desired degree */
            double maxPosteriorErrorNorm;
            double posteriorErrorNormReadDepth, posteriorErrorNormBias, posteriorErrorNormCopyRatio;

            runRoutine(this::updateReadDepthLatentPosteriorExpectations, s -> "N/A", "E_STEP_D", iterInfo);
            posteriorErrorNormReadDepth = iterInfo.errorNorm;

            runRoutine(this::updateBiasLatentPosteriorExpectations, s -> "N/A", "E_STEP_Z", iterInfo);
            posteriorErrorNormBias = iterInfo.errorNorm;

            if (updateCopyRatioPosteriors) {
                runRoutine(() -> updateCopyRatioLatentPosteriorExpectations(false), s -> "N/A", "E_STEP_C", iterInfo);
                posteriorErrorNormCopyRatio = iterInfo.errorNorm;
            } else {
                posteriorErrorNormCopyRatio = 0;
            }

            /* calculate the maximum change of posteriors in this cycle */
            maxPosteriorErrorNorm = Collections.max(Arrays.asList(posteriorErrorNormReadDepth,
                    posteriorErrorNormBias, posteriorErrorNormCopyRatio));

            /* check convergence of the E-step */
            if (maxPosteriorErrorNorm < params.getPosteriorErrorNormTol() &&
                    FastMath.abs(iterInfo.logLikelihood - prevLogLikelihood) < params.getLogLikelihoodTolerance()) {
                status = EMAlgorithmStatus.SUCCESS_POSTERIOR_CONVERGENCE;
                break;
            }

            /* if the likelihood has increased and the increment is small, start updating copy number posteriors */
            if (performCopyRatioPosteriorCalling &&
                    !updateCopyRatioPosteriors &&
                    (iterInfo.logLikelihood - prevLogLikelihood) > 0 &&
                    (iterInfo.logLikelihood - prevLogLikelihood) < params.getLogLikelihoodTolThresholdCopyRatioCalling()) {
                updateCopyRatioPosteriors = true;
                logger.info("Partial convergence achieved; will start calling copy ratio posteriors");
            }

            iterInfo.increaseIterationCount();
            prevLogLikelihood = iterInfo.logLikelihood;
        }

        if (iterInfo.iter == params.getMaxIterations()) {
            status = EMAlgorithmStatus.FAILURE_MAX_ITERS_REACHED;
        }

        performPostEMOperations();
    }

    private void performPostEMOperations() {
        logger.info("EM algorithm status: " + status.message);
    }

    /**
     * E-step -- Update E[z] and E[z z^T] for all samples using the current estimate of model parameters
     */
    public abstract SubroutineSignal updateBiasLatentPosteriorExpectations();

    /**
     * E-step -- Update E[log(d_s)] and E[log(d_s)^2]
     */
    public abstract SubroutineSignal updateReadDepthLatentPosteriorExpectations();

    /**
     * E-step -- Update E[log(c_{st})] and E[log(c_{st})^2]
     * @param performViterbi
     */
    public abstract SubroutineSignal updateCopyRatioLatentPosteriorExpectations(final boolean performViterbi);

    /**
     * M-step -- Update mean bias vector "m"
     */
    public abstract SubroutineSignal updateTargetMeanBias();

    /**
     * M-step -- Update Psi
     */
    public abstract SubroutineSignal updateTargetUnexplainedVariance();

    /**
     * M-step -- Update W
     */
    public abstract SubroutineSignal updatePrincipalLatentTargetMap();

    /**
     * Calculate the log likelihood
     * @return log likelihood
     */
    public abstract double getLogLikelihood();

    /**
     * Calculate the log likelihood per sample
     * @return
     */
    public abstract double[] getLogLikelihoodPerSample();
}
