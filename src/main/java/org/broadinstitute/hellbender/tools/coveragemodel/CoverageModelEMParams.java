package org.broadinstitute.hellbender.tools.coveragemodel;

import org.broadinstitute.hellbender.utils.param.ParamUtils;
import org.nd4j.linalg.api.buffer.DataBuffer;

import javax.annotation.Nonnull;

/**
 * Parameters for {@link CoverageModelEMAlgorithm}.
 *
 * @author Mehrtash Babadi &lt;mehrtash@broadinstitute.org&gt;
 */
public class CoverageModelEMParams {

    public enum PsiSolverType {
//        PSI_TARGET_RESOLVED_VIA_NEWTON,
        PSI_TARGET_RESOLVED_VIA_BRENT,
        PSI_ISOTROPIC_VIA_BRENT
    }

    public enum WSolverType {
        W_SOLVER_LOCAL,
        W_SOLVER_SPARK
    }

    public enum CopyRatioHMMType {
        COPY_RATIO_HMM_LOCAL,
        COPY_RATIO_HMM_SPARK
    }

    public enum CommunicationPolicy {
        BROADCAST_HASH_JOIN,
        RDD_JOIN
    }

    public static final double PSI_BRENT_UPPER_LIMIT = 0.25;
    public static final double PSI_BRENT_MIN_STARTING_POINT = 1e-8;

    private DataBuffer.Type dType = DataBuffer.Type.DOUBLE;

    /* maximum number of EM iterations */
    private int maxIterations = 10;

    /* dimension of the latent space */
    private int numLatents = 10;

    /* stopping criterion w.r.t. change in the model log likelihood */
    private double logLikelihoodTol = 1e-5;

    /* stopping criterion w.r.t. change in the model parameters */
    private double paramAbsTol = 1e-4;

    /* E-step cycle termination threshold */
    private double posteriorErrorNormTol = 1e-2;

    /* number of sequential maximization steps in the M step */
    private int maxMStepCycles = 1;

    /* maximum number of E-step cycles */
    private int maxEStepCycles = 4;

    /* when to start calculating copy ratio posteriors */
    private double logLikelihoodTolThresholdCopyRatioCalling = 5e-2;

    /* use Fourier regularization or not */
    private boolean useFourierRegularization = false;

    /* minimum length of CNV event (for regularization) */
    private int minCNVLength = 10;

    /* maximum length of CNV event (for regularization) */
    private int maxCNVLength = 1000;

    /* zero pad data in FFT and promote size to powers of 2 */
    private boolean zeroPadFFT = false;

    /* Fourier regularization strength */
    private double fourierRegularizationStrength = 10_000;

    /* Psi solver type */
    private PsiSolverType psiSolverType = PsiSolverType.PSI_TARGET_RESOLVED_VIA_BRENT;

    /* W solver type */
    private WSolverType wSolverType = WSolverType.W_SOLVER_SPARK;

    /* calculate copy ratio posteriors local or with spark */
    private CopyRatioHMMType copyRatioHMMType = CopyRatioHMMType.COPY_RATIO_HMM_SPARK;

    /* M-step error tolerance in maximizing w.r.t. Psi */
    private double psiAbsTol = 1e-7;
    private double psiRelTol = 1e-4;

    /* M-step maximum iterations in maximizing w.r.t. Psi */
    private int psiMaxIterations = 50;

    /* M-step error tolerance in maximizing w.r.t. W (if Fourier regularization is enabled) */
    private double wAbsTol = 1e-7;
    private double wRelTol = 1e-3;

    /* M-step maximum iterations in maximizing w.r.t. W (if Fourier regularization is enabled) */
    private int wMaxIterations = 20;

    private boolean checkpointingEnabled = true;

    private int checkpointingInterval = 10;

    private CommunicationPolicy principalMapCommunicationPolicy = CommunicationPolicy.RDD_JOIN;

    private double meanFieldAdmixingRatio = 0.75;

    private boolean orthogonalizeAndSortPrincipalMap = true;

    private int modelSavingInterval = 1;

    /********************************
     * accessor and mutator methods *
     ********************************/

    public CoverageModelEMParams setMaxIterations(final int maxIterations) {
        this.maxIterations = ParamUtils.isPositive(maxIterations, "Maximum EM iterations must be positive.");
        return this;
    }

    public int getMaxIterations() { return maxIterations; }

    public CoverageModelEMParams setNumLatents(final int numLatents) {
        this.numLatents = ParamUtils.isPositive(numLatents, "Number of latent variables must be positive.");
        return this;
    }

    public int getNumLatents() { return numLatents; }

    public CoverageModelEMParams setLogLikelihoodTolerance(final double tol) {
        logLikelihoodTol = ParamUtils.isPositive(tol, "The required tolerance on log likelihood " +
                "must be positive.");
        return this;
    }

    public double getLogLikelihoodTolerance() { return logLikelihoodTol; }

    public CoverageModelEMParams setMaxMStepCycles(final int maxMStepCycles) {
        this.maxMStepCycles = ParamUtils.isPositive(maxMStepCycles, "The number of " +
                "sequential partial maximimization steps must be positive.");
        return this;
    }

    public int getMaxMStepCycles() { return maxMStepCycles; }

    public CoverageModelEMParams enableFourierRegularization() {
        useFourierRegularization = true;
        return this;
    }

    public CoverageModelEMParams disableFourierRegularization() {
        useFourierRegularization = false;
        return this;
    }

    public boolean fourierRegularizationEnabled() { return useFourierRegularization; }

    public CoverageModelEMParams setFourierRegularizationStrength(final double fourierRegularizationStrength) {
        this.fourierRegularizationStrength = ParamUtils.isPositive(fourierRegularizationStrength, "The Fourier " +
                "regularization strength must be positive");
        return this;
    }

    public double getFourierRegularizationStrength() { return fourierRegularizationStrength; }

    public CoverageModelEMParams setPsiAbsoluteTolerance(final double tol) {
        this.psiAbsTol = ParamUtils.isPositive(tol, "The absolute tolerance for maximization of Psi must be positive");
        return this;
    }

    public double getPsiAbsoluteTolerance() { return psiAbsTol; }

    public CoverageModelEMParams setPsiRelativeTolerance(final double tol) {
        this.psiRelTol = ParamUtils.isPositive(tol, "The relative tolerance for maximization of Psi must be positive");
        return this;
    }

    public double getPsiRelativeTolerance() { return psiRelTol; }

    public CoverageModelEMParams setPsiMaxIterations(final int psiMaxIterations) {
        this.psiMaxIterations = ParamUtils.isPositive(psiMaxIterations, "The maximum number of interations for M-step of Psi " +
                "must be positive.");
        return this;
    }

    public int getPsiMaxIterations() { return psiMaxIterations; }

    public CoverageModelEMParams setWAbsoluteTolerance(final double tol) {
        this.wAbsTol = ParamUtils.isPositive(tol, "The absolute tolerance for maximization of Psi must be positive");
        return this;
    }

    public double getWAbsoluteTolerance() { return this.wAbsTol; }

    public CoverageModelEMParams setWRelativeTolerance(final double tol) {
        this.wRelTol = ParamUtils.isPositive(tol, "The relative tolerance for maximization of Psi must be positive");
        return this;
    }

    public double getWRelativeTolerance() { return this.wRelTol; }

    public CoverageModelEMParams setWMaxIterations(final int wMaxIterations) {
        this.wMaxIterations = ParamUtils.isPositive(wMaxIterations, "The maximum number of interations for M-step of W " +
                "must be positive.");
        return this;
    }

    public int getWMaxIterations() { return wMaxIterations; }

    public CoverageModelEMParams enableZeroPadFFT() {
        zeroPadFFT = true;
        return this;
    }

    public CoverageModelEMParams disableZeroPadFFT() {
        zeroPadFFT = false;
        return this;
    }

    public boolean zeroPadFFT() {
        return zeroPadFFT;
    }

    public CoverageModelEMParams setParameterEstimationAbsoluteTolerance(final double val) {
        this.paramAbsTol = ParamUtils.isPositive(paramAbsTol, "The required tolerance on parameter change must be positive.");
        return this;
    }

    public double getParameterEstimationAbsoluteTolerance() { return this.paramAbsTol; }

    public PsiSolverType getPsiSolverType() {
        return psiSolverType;
    }

    public CoverageModelEMParams setPsiPsiolverType(@Nonnull final PsiSolverType psiSolverType) {
        this.psiSolverType = psiSolverType;
        return this;
    }

    public WSolverType getWSolverType() {
        return wSolverType;
    }

    public CoverageModelEMParams setWSolverType(@Nonnull final WSolverType wSolverType) {
        this.wSolverType = wSolverType;
        return this;
    }

    public CoverageModelEMParams setMinimumCNVLength(final int minCNVLength) {
        this.minCNVLength = minCNVLength;
        return this;
    }

    public CoverageModelEMParams setMaximumCNVLength(final int maxCNVLength) {
        this.maxCNVLength = maxCNVLength;
        return this;
    }

    public int getMinimumCNVLength() { return minCNVLength; }

    public int getMaximumCNVLength() { return maxCNVLength; }

    public CoverageModelEMParams setDType(@Nonnull final DataBuffer.Type dType) {
        this.dType = dType;
        return this;
    }

    public DataBuffer.Type getdType() { return dType; }

    public CoverageModelEMParams enableCheckpointing() {
        this.checkpointingEnabled = true;
        return this;
    }

    public CoverageModelEMParams disableCheckpointing() {
        this.checkpointingEnabled = false;
        return this;
    }

    public boolean checkpointingEnabled() {
        return checkpointingEnabled;
    }

    public int getCheckpointingInterval() {
        return checkpointingInterval;
    }

    public CoverageModelEMParams setCheckpointingInterval(final int checkpointingInterval) {
        this.checkpointingInterval = checkpointingInterval;
        return this;
    }

    public CopyRatioHMMType getCopyRatioHMMType() {
        return copyRatioHMMType;
    }

    public CoverageModelEMParams setCopyRatioHMMType(final CopyRatioHMMType copyRatioHMMType) {
        this.copyRatioHMMType = copyRatioHMMType;
        return this;
    }

    public double getLogLikelihoodTolThresholdCopyRatioCalling() {
        return logLikelihoodTolThresholdCopyRatioCalling;
    }

    public CoverageModelEMParams setLogLikelihoodTolThresholdCopyRatioCalling(final double logLikelihoodTolThresholdCopyRatioCalling) {
        this.logLikelihoodTolThresholdCopyRatioCalling = logLikelihoodTolThresholdCopyRatioCalling;
        return this;
    }

    public double getPosteriorErrorNormTol() {
        return posteriorErrorNormTol;
    }

    public CoverageModelEMParams setPosteriorErrorNormTol(final double posteriorErrorNormTol) {
        this.posteriorErrorNormTol = posteriorErrorNormTol;
        return this;
    }

    public int getMaxEStepCycles() {
        return maxEStepCycles;
    }

    public CoverageModelEMParams setMaxEStepCycles(final int maxEStepCycles) {
        this.maxEStepCycles = maxEStepCycles;
        return this;
    }

    public CommunicationPolicy getPrincipalMapCommunicationPolicy() {
        return principalMapCommunicationPolicy;
    }

    public CoverageModelEMParams setPrincipalMapCommunicationPolicy(final CommunicationPolicy principalMapCommunicationPolicy) {
        this.principalMapCommunicationPolicy = principalMapCommunicationPolicy;
        return this;
    }

    public double getMeanFieldAdmixingRatio() {
        return meanFieldAdmixingRatio;
    }

    public CoverageModelEMParams setMeanFieldAdmixingRatio(double meanFieldAdmixingRatio) {
        this.meanFieldAdmixingRatio = meanFieldAdmixingRatio;
        return this;
    }

    public boolean isOrthogonalizeAndSortPrincipalMapEnabled() {
        return orthogonalizeAndSortPrincipalMap;
    }

    public CoverageModelEMParams enableOrthogonalizeAndSortPrincipalMap() {
        orthogonalizeAndSortPrincipalMap = true;
        return this;
    }

    public CoverageModelEMParams disableOrthogonalizeAndSortPrincipalMap() {
        orthogonalizeAndSortPrincipalMap = false;
        return this;
    }

    public int getModelSavingInterval() {
        return modelSavingInterval;
    }

    public CoverageModelEMParams setModelSavingInterval(final int modelSavingInterval) {
        this.modelSavingInterval = modelSavingInterval;
        return this;
    }
}
