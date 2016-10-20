package org.broadinstitute.hellbender.tools.coveragemodel;

import org.apache.commons.lang3.tuple.ImmutablePair;
import org.apache.commons.lang3.tuple.ImmutableTriple;
import org.apache.commons.math3.analysis.UnivariateFunction;
import org.apache.commons.math3.analysis.solvers.AbstractUnivariateSolver;
import org.apache.commons.math3.exception.NoBracketingException;
import org.apache.commons.math3.exception.TooManyEvaluationsException;
import org.apache.commons.math3.util.FastMath;
import org.broadinstitute.hellbender.exceptions.UserException;
import org.broadinstitute.hellbender.tools.coveragemodel.cachemanager.Duplicable;
import org.broadinstitute.hellbender.tools.coveragemodel.cachemanager.DuplicableNDArray;
import org.broadinstitute.hellbender.tools.coveragemodel.cachemanager.DuplicableNumber;
import org.broadinstitute.hellbender.tools.coveragemodel.cachemanager.ImmutableComputableGraph;
import org.broadinstitute.hellbender.tools.coveragemodel.math.RobustBrentSolver;
import org.broadinstitute.hellbender.utils.Utils;
import org.broadinstitute.hellbender.utils.param.ParamUtils;
import org.junit.Assert;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.buffer.util.DataTypeUtil;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.ops.transforms.Transforms;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.util.*;
import java.util.stream.Collectors;
import java.util.stream.IntStream;
import java.util.function.Function;

/**
 * This class represents an immutable block of containers corresponding to a partition of the target space
 * as specified by {@link LinearSpaceBlock}.
 *
 * TODO -- logging in spark mode (log4j is not serializable)
 * TODO -- use instrumentation to measure the dependence of the total memory consumption of this class
 *         on the number of samples, targets, and latent variables (or estimate analytically)
 *
 * Note: the read counts and germline ploidies passed to this class have their "0" entries fixed up to
 * avoid numerical instability; the fix-up value is immaterial since these targets are masked out and
 * do not contribute to the likelihood. The user must be careful is fetching and using these values since
 * they do not represent the original unaltered date.
 *
 * @author Mehrtash Babadi &lt;mehrtash@broadinstitute.org&gt;
 */
public class CoverageModelEMComputeBlockNDArray {

    private final ImmutableComputableGraph icg;

    private final LinearSpaceBlock targetBlock;

    private final int numSamples, numLatents, numTargets;

    /* this member store the latest signal emitted from an M-step update */
    private final SubroutineSignal latestMStepSignal;

    /**
     * Private constructor (for cloners)
     *
     * @param targetBlock target space block
     * @param numSamples number of samples
     * @param numLatents dimension of latent space
     * @param icg an instance of {@link ImmutableComputableGraph}
     */
    private CoverageModelEMComputeBlockNDArray(@Nonnull final LinearSpaceBlock targetBlock,
                                               final int numSamples, final int numLatents,
                                               @Nonnull final ImmutableComputableGraph icg,
                                               @Nullable SubroutineSignal latestMStepSignal) {
        /* Set Nd4j DType to Double in context */
        Nd4j.create(1);
        DataTypeUtil.setDTypeForContext(DataBuffer.Type.DOUBLE);

        this.numSamples = ParamUtils.isPositive(numSamples, "Number of samples must be positive.");
        this.numLatents = ParamUtils.isPositive(numLatents, "Dimension of the latent space must be positive.");
        this.targetBlock = Utils.nonNull(targetBlock, "Target space block identifier can not be null.");
        this.icg = Utils.nonNull(icg, "The immutable computable graph can not be null.");
        this.latestMStepSignal = latestMStepSignal;
        this.numTargets = targetBlock.getNumTargets();
    }

    /**
     * Public constructor
     *
     * @param targetBlock target space block
     * @param numSamples number of samples
     */
    public CoverageModelEMComputeBlockNDArray(@Nonnull final LinearSpaceBlock targetBlock,
                                              final int numSamples, final int numLatents) {
        this(targetBlock, numSamples, numLatents, createEmptyCacheGraph(), SubroutineSignal.builder().build());
    }

    /*************
     * accessors *
     *************/

    /**********************************************************************************************
     * IMPORTANT NOTE: since we do not have an immutable INDArray (or a wrapper for that), the    *
     * accessor methods my ALWAYS return a deep copy for the time being                           *
     **********************************************************************************************/


    public LinearSpaceBlock getTargetSpaceBlock() {
        return targetBlock.clone();
    }

    public SubroutineSignal getLatestMStepSignal() { return latestMStepSignal; }

    /**
     * Fetch the value of a node from the cache manager and cast it to INDArray
     * @param key key of the cache
     *
     * Note: upcasting is not checked
     *
     * @return an INDArray
     */
    public INDArray getINDArrayFromCache(final String key) {
        return ((DuplicableNDArray)icg.getValueDirect(key)).value();
    }

    /**
     * Fetch the value of a node from the cache manager and cast it to primitive double
     * @param key key of the cache
     *
     * Note: upcasting is not checked
     *
     * @return an INDArray
     */
    public double getDoubleFromCache(final String key) {
        return ((DuplicableNumber) icg.getValueDirect(key)).value().doubleValue();
    }

    /**
     * Note: the functions that work on NDArrays must make sure to leave the data in the cache node
     * unmutated. For example, to calculate the A.B.C + D the correct template is first fetch the data
     * from cache:
     *
     * final INDArray A = getINDArrayFromCache("A_string_identifier");
     * final INDArray B = getINDArrayFromCache("B_string_identifier");
     * final INDArray C = getINDArrayFromCache("C_string_identifier");
     * final INDArray D = getINDArrayFromCache("D_string_identifier");
     *
     * and then:
     *
     *      final INDArray result_1 = A.mul(B).muli(C).addi(D); // OK
     *                                ---1--- (a copy is instantiated; call it tmp)
     *                                     ----2---  (tmp is modified in-place)
     *                                         -----3---- (tmp is modified in-place)
     *
     * Here, A and multiplied by B to create a new instance; subsequent operations can be performed in-place
     * to avoid unnecessary garbage generation.
     *
     * It is perhaps even safer to use:
     *
     *      final INDArray result_2 = A.mul(B).mul(C).add(D); // OK
     *                                ---1--- (a copy is instantiated)
     *                                     ----2---  (a copy is instantiated)
     *                                         -----3---- (a copy is instantiated)
     * It is, however, WRONG to do:
     *
     *      final INDArray result_3 = A.muli(B).muli(C).addi(D); // NOT OK
     *                                ---1--- (A is modified in-place!)
     *
     * Here, A is modified in-place which corrupts the data in {@link CoverageModelEMComputeBlockNDArray#icg}
     *
     * The code reviewer must CAREFULLY check all of the matrix computations to ensure that the third
     * case does not occur.
     *
     */





    /*******************************************************************************
     * E-step for bias posterior expectations (generates data for the driver node) *
     *******************************************************************************/

    /**
     * Calculate the contribution of this target block to the calculation of the latent
     * posterior expectations for unregularized model as a pair (contribGMatrix, contribZ)
     *
     *      contribGMatrix_{s\mu,\nu} = \sum_{t \in targetBlock} W_{\mu t} [diag(M_st \Psi_st)] W_{t\nu}
     *
     *      contribZ_{\mu,s} = \sum_{t \in targetBlock} W_{\mu T} [diag(M_st \Psi_st)] (m_{st} - m_t)
     *      (note the order of indices)
     *
     * @return an {@link ImmutablePair} of contribGMatrix (left) and contribZ (right)
     */
    public ImmutablePair<INDArray, INDArray> getBiasLatentPosteriorDataUnregularized() {
        /* fetch data from cache */
        final INDArray M_Psi_inv_st = getINDArrayFromCache("M_Psi_inv_st");
        final INDArray Delta_st = getINDArrayFromCache("Delta_st");
        final INDArray W_tl = getINDArrayFromCache("W_tl");

        /* calculate the required quantities */
        final INDArray contribGMatrix = Nd4j.create(numSamples, numLatents, numLatents);
        final INDArray contribZ = Nd4j.create(numLatents, numSamples);
        IntStream.range(0, numSamples).parallel().forEach(si -> {
            contribGMatrix.get(NDArrayIndex.point(si)).assign(
                    W_tl.transpose().mmul(W_tl.mulColumnVector(M_Psi_inv_st.getRow(si).transpose())));
            contribZ.get(NDArrayIndex.all(), NDArrayIndex.point(si)).assign(
                    W_tl.transpose().mmul(M_Psi_inv_st.getRow(si).mul(Delta_st.getRow(si)).transpose()));
        });

        return new ImmutablePair<>(contribGMatrix, contribZ);
    }

    /**
     * Calculate the contribution of this target block to the calculation of the latent
     * posterior expectations for regularized model as a triple (contribGMatrix, contribZ, contribFilter)
     *
     *      contribFilter = [W]^T [F.W]
     *
     * @return an {@link ImmutableTriple} of contribGMatrix (left), contribZ (middle), contribFilter (right)
     */
    public ImmutableTriple<INDArray, INDArray, INDArray> getBiasLatentPosteriorDataRegularized() {
        final ImmutablePair<INDArray, INDArray> unreg = getBiasLatentPosteriorDataUnregularized();
        final INDArray contribFilter = getINDArrayFromCache("W_tl").transpose().mmul(getINDArrayFromCache("F_W_tl"));
        return new ImmutableTriple<>(unreg.left, unreg.right, contribFilter);
    }





    /*************************************************************************************
     * E-step for read depth posterior expectations (generates data for the driver node) *
     *************************************************************************************/

    /**
     * Performs partial read depth estimation:
     * <dl>
     *     <dt> A_s = \sum_{t \in block} M_{st} [\log(n_{st}/(P_{st} c_{st})) - b_{st}]</dt>
     *     <dt> B_s = \sum_{t \in block} M_{st} </dt>
     * </dl>
     * @return an {@link ImmutablePair} of (A_s, B_s)
     */
    public ImmutablePair<INDArray, INDArray> getSampleReadDepthLatentPosteriorData() {
        /* fetch data from cache */
        final INDArray log_nu_st = getINDArrayFromCache("log_nu_st");
        final INDArray log_c_st = getINDArrayFromCache("log_c_st");
        final INDArray m_t = getINDArrayFromCache("m_t");
        final INDArray Wz_st = getINDArrayFromCache("Wz_st");
        final INDArray M_Psi_inv_st = getINDArrayFromCache("M_Psi_inv_st");

        /* calculate the required quantities */
        final INDArray numerator = log_nu_st.sub(log_c_st).subi(Wz_st).subiRowVector(m_t).muli(M_Psi_inv_st).sum(1);
        final INDArray denominator = M_Psi_inv_st.sum(1);

        return ImmutablePair.of(numerator, denominator);
    }

    /****************************************************************************************
     * E-step for sample-specific unexplained variance (generates data for the driver node) *
     ****************************************************************************************/

    public INDArray getTargetSummedGammaPosteriorArgumentMultiSample(final int[] sampleIndices,
                                                                     final INDArray gamma_s) {
        final INDArray M_st = getINDArrayFromCache("M_st");
        final INDArray Psi_t = getINDArrayFromCache("Psi_t");
        final INDArray Sigma_st = getINDArrayFromCache("Sigma_st");
        final INDArray B_st = getINDArrayFromCache("B_st");

        final int numQueries = sampleIndices.length;
        if (numQueries == 0) {
            throw new IllegalArgumentException("Can not take empty queries");
        }
        if (gamma_s.length() != numQueries) {
            throw new IllegalArgumentException("The argument array must have the same length as the" +
                    " sample index array");
        }
        final INDArray assembled_M_st = Nd4j.create(numQueries, numTargets);
        final INDArray assembled_Sigma_st = Nd4j.create(numQueries, numTargets);
        final INDArray assembled_B_st = Nd4j.create(numQueries, numTargets);
        IntStream.range(0, numQueries)
                .forEach(i -> {
                    assembled_M_st.getRow(i).assign(M_st.getRow(sampleIndices[i]));
                    assembled_Sigma_st.getRow(i).assign(Sigma_st.getRow(sampleIndices[i]));
                    assembled_B_st.getRow(i).assign(B_st.getRow(sampleIndices[i]));
                });
        final INDArray totalMaskedPsiInverse = assembled_M_st.div(
                assembled_Sigma_st.addRowVector(Psi_t).addiColumnVector(gamma_s));
        final INDArray totalMaskedPsiInverseMulB = assembled_B_st.mul(totalMaskedPsiInverse);
        return totalMaskedPsiInverse.mul(assembled_M_st.sub(totalMaskedPsiInverseMulB)).sum(1);
    }

    /*************************************************************************************
     * E-step for copy ratio posterior expectations (generates data for the driver node) *
     *************************************************************************************/

    /**
     * Calculates the contribution of this target block to the data required for calculating the
     * copy ratio posterior expectations (done on the driver node).
     *
     * @return
     */
    public List<List<CoverageModelCopyRatioEmissionData>> getSampleCopyRatioLatentPosteriorData() {
        /* fetch data from cache */
        final INDArray log_nu_st = getINDArrayFromCache("log_nu_st");
        final INDArray M_st = getINDArrayFromCache("M_st");
        final INDArray Psi_t = getINDArrayFromCache("Psi_t");
        final INDArray gamma_s = getINDArrayFromCache("gamma_s");
        final INDArray log_P_st = getINDArrayFromCache("log_P_st");
        final INDArray log_d_s = getINDArrayFromCache("log_d_s");
        final INDArray m_t = getINDArrayFromCache("m_t");
        final INDArray Wz_st = getINDArrayFromCache("Wz_st");

        /* calculate the required quantities */
        final INDArray n_st = Transforms.exp(log_nu_st.add(log_P_st), false);
        final INDArray mu_st = log_P_st.add(Wz_st).addiRowVector(m_t).addiColumnVector(log_d_s);
        final double[] psiArray = Psi_t.dup().data().asDouble();
        final double[] gammaArray = gamma_s.dup().data().asDouble();

        return IntStream.range(0, numSamples)
                .mapToObj(si -> {
                    final double[] currentSampleMaskArray = M_st.getRow(si).dup().data().asDouble();
                    final double[] currentSampleReadCountArray = n_st.getRow(si).dup().data().asDouble();
                    final double[] currentSampleMuArray = mu_st.getRow(si).dup().data().asDouble();
                    return IntStream.range(0, targetBlock.getNumTargets())
                            .mapToObj(ti -> (int)currentSampleMaskArray[ti] == 0
                                    ? null
                                    : new CoverageModelCopyRatioEmissionData(currentSampleMuArray[ti],
                                                psiArray[ti] + gammaArray[si], currentSampleReadCountArray[ti]))
                            .collect(Collectors.toList());
                }).collect(Collectors.toList());
    }

    /*******************************************************************************
     * Orthogonalization of the principal map (generates data for the driver node) *
     *******************************************************************************/

    /**
     * Calculate [W]^T [W] in this target space block. The driver node must simply sum the results from all workers
     * in a reduction step to find the full result.
     *
     * @return [W]^T [W]
     */
    public INDArray getPrincipalLatentTargetMapInnerProduct() {
        final INDArray W_tl = getINDArrayFromCache("W_tl");
        return W_tl.transpose().mmul(W_tl);
    }

    /**
     *
     * @param log_c_st
     * @param var_log_c_st
     * @return
     */
    public CoverageModelEMComputeBlockNDArray updateCopyRatioLatentPosteriors(@Nonnull final INDArray log_c_st,
                                                                              @Nonnull final INDArray var_log_c_st,
                                                                              final double admixingRatio) {
        final INDArray old_log_c_st = getINDArrayFromCache("log_c_st");
        final INDArray old_var_log_c_st = getINDArrayFromCache("var_log_c_st");

        /* admix */
        final INDArray admixed_log_c_st = log_c_st.mul(admixingRatio).addi(old_log_c_st.mul(1.0 - admixingRatio));
        final INDArray admixed_var_log_c_st = var_log_c_st.mul(admixingRatio).addi(old_var_log_c_st.mul(1.0 - admixingRatio));

        final double errNormInfinity = CoverageModelEMWorkspaceNDArrayUtils.getINDArrayNormInfinity(
                old_log_c_st.sub(admixed_log_c_st));
        return cloneWithUpdatedPrimitive("log_c_st", admixed_log_c_st)
                .cloneWithUpdatedPrimitive("var_log_c_st", admixed_var_log_c_st)
                .cloneWithUpdatedSignal(SubroutineSignal.builder().put("error_norm", errNormInfinity).build());
    }


    /*******************************
     * M-step for target mean bias *
     *******************************/

    public CoverageModelEMComputeBlockNDArray updateTargetMeanBias(final boolean neglectPCBias) {
        /* fetch the required caches */
        final INDArray log_nu_st = getINDArrayFromCache("log_nu_st");
        final INDArray log_c_st = getINDArrayFromCache("log_c_st");
        final INDArray log_d_s = getINDArrayFromCache("log_d_s");
        final INDArray M_Psi_inv_st = getINDArrayFromCache("M_Psi_inv_st");
        final INDArray Wz_st = getINDArrayFromCache("Wz_st");

        final INDArray numerator;
        if (neglectPCBias) {
            numerator = M_Psi_inv_st.mul(log_nu_st.sub(log_c_st).subiColumnVector(log_d_s)).sum(0);
        } else {
            numerator = M_Psi_inv_st.mul(log_nu_st.sub(log_c_st).subiColumnVector(log_d_s).sub(Wz_st)).sum(0);
        }
        final INDArray denominator = M_Psi_inv_st.sum(0);
        final INDArray newTargetMeanBias = numerator.divi(denominator);

        double errNormInfinity = CoverageModelEMWorkspaceNDArrayUtils.getINDArrayNormInfinity(
                getINDArrayFromCache("m_t").sub(newTargetMeanBias));

        return cloneWithUpdatedPrimitiveAndSignal("m_t", newTargetMeanBias,
                SubroutineSignal.builder()
                        .put("error_norm", errNormInfinity)
                        .build());
    }

    /******************************************
     * M-step for target unexplained variance *
     ******************************************/

    /**
     * Create a per-target object function for univariate solver
     *
     * @param targetIndex
     * @param M_st
     * @param Sigma_st
     * @param gamma_s
     * @param B_st
     * @return
     */
    private UnivariateFunction createPsiSolverObjectiveFunction(final int targetIndex,
                                                                @Nonnull final INDArray M_st,
                                                                @Nonnull final INDArray Sigma_st,
                                                                @Nonnull final INDArray gamma_s,
                                                                @Nonnull final INDArray B_st) {
        final INDArray M_s = M_st.get(NDArrayIndex.all(), NDArrayIndex.point(targetIndex));
        final INDArray Sigma_s = Sigma_st.get(NDArrayIndex.all(), NDArrayIndex.point(targetIndex));
        final INDArray B_s = B_st.get(NDArrayIndex.all(), NDArrayIndex.point(targetIndex));
        return psi -> {
            final INDArray totalMaskedPsiInverse = M_s.div(Sigma_s.add(gamma_s).add(psi));
            return totalMaskedPsiInverse.mul(M_s.sub(B_s.mul(totalMaskedPsiInverse))).sumNumber().doubleValue();
        };
    }

    private UnivariateFunction createPsiSolverMeritFunction(final int targetIndex,
                                                            @Nonnull final INDArray M_st,
                                                            @Nonnull final INDArray Sigma_st,
                                                            @Nonnull final INDArray gamma_s,
                                                            @Nonnull final INDArray B_st) {
        final INDArray M_s = M_st.get(NDArrayIndex.all(), NDArrayIndex.point(targetIndex));
        final INDArray Sigma_s = Sigma_st.get(NDArrayIndex.all(), NDArrayIndex.point(targetIndex));
        final INDArray B_s = B_st.get(NDArrayIndex.all(), NDArrayIndex.point(targetIndex));
        return psi -> {
            final INDArray totalPsi = Sigma_s.add(gamma_s).add(psi);
            final INDArray totalMaskedPsiInverse = M_s.div(totalPsi);
            final INDArray logTotalPsi = Transforms.log(totalPsi);

            return M_s.mul(logTotalPsi).addi(B_s.mul(totalMaskedPsiInverse)).muli(-0.5).sumNumber().doubleValue();
        };
    }

    private double calculatePsiCurvature(final int targetIndex,
                                         final double psi,
                                         @Nonnull final INDArray M_st,
                                         @Nonnull final INDArray Sigma_st,
                                         @Nonnull final INDArray gamma_s,
                                         @Nonnull final INDArray B_st) {
        final INDArray M_s = M_st.get(NDArrayIndex.all(), NDArrayIndex.point(targetIndex));
        final INDArray Sigma_s = Sigma_st.get(NDArrayIndex.all(), NDArrayIndex.point(targetIndex));
        final INDArray B_s = B_st.get(NDArrayIndex.all(), NDArrayIndex.point(targetIndex));
        final INDArray J = M_s.div(Sigma_s.add(gamma_s).add(psi));
        final INDArray J2 = J.mul(J);
        final INDArray J3 = J.mul(J2);
        return - 0.5 * (J2.sub(B_s.mul(2).mul(J3)).sumNumber().doubleValue());
    }

    /**
     * Helper function for isotropic update of Psi
     * @param psi test value
     * @return
     */
    public double calculateSampleTargetSummedPsiGradient(final double psi) {
        final INDArray M_st = getINDArrayFromCache("M_st");
        final INDArray Sigma_st = getINDArrayFromCache("Sigma_st");
        final INDArray B_st = getINDArrayFromCache("B_st");
        final INDArray gamma_s = getINDArrayFromCache("gamma_s");
        final INDArray totalMaskedPsiInverse = M_st.div(Sigma_st.addColumnVector(gamma_s).addi(psi));
        final INDArray totalMaskedPsiInverseMulB = B_st.mul(totalMaskedPsiInverse);
        return totalMaskedPsiInverse.mul(M_st.sub(totalMaskedPsiInverseMulB)).sumNumber().doubleValue();
    }

    /**
     * Helper function for isotropic update of Psi
     * @param psi test value
     * @return
     */
    public double calculateSampleTargetSummedPsiMerit(final double psi) {
        final INDArray M_st = getINDArrayFromCache("M_st");
        final INDArray Sigma_st = getINDArrayFromCache("Sigma_st");
        final INDArray B_st = getINDArrayFromCache("B_st");
        final INDArray gamma_s = getINDArrayFromCache("gamma_s");

        final INDArray totalPsi = Sigma_st.addColumnVector(gamma_s).addi(psi);
        final INDArray totalMaskedPsiInverse = M_st.div(totalPsi);
        final INDArray logTotalPsi = Transforms.log(totalPsi);

        return M_st.mul(logTotalPsi).addi(B_st.mul(totalMaskedPsiInverse)).muli(-0.5).sumNumber().doubleValue();
    }

    /**
     * Solve the M-step equation for $\Psi_t$ by Newton iterations
     * @return
     */
    public CoverageModelEMComputeBlockNDArray updateTargetUnexplainedVarianceTargetResolved(final int maxIters,
                                                                                            final double psiUpperLimit,
                                                                                            final double absTol,
                                                                                            final double relTol,
                                                                                            final int numBisections,
                                                                                            final int depth) {
        /* fetch the required caches */
        final INDArray Psi_t = getINDArrayFromCache("Psi_t");
        final INDArray M_st = getINDArrayFromCache("M_st");
        final INDArray Sigma_st = getINDArrayFromCache("Sigma_st");
        final INDArray gamma_s = getINDArrayFromCache("gamma_s");
        final INDArray B_st = getINDArrayFromCache("B_st");

        final double psiLowerBound = 0.0;

        /*
         * If we want to leverage from parallelism, we need need to instantiate a new solver for each target and pay
         * the overhead
         */
        final List<ImmutablePair<Double, Integer>> res = IntStream.range(0, numTargets).parallel()
                .mapToObj(ti -> {
                    final UnivariateFunction objFunc = createPsiSolverObjectiveFunction(ti, M_st, Sigma_st, gamma_s, B_st);
                    final UnivariateFunction meritFunc = createPsiSolverMeritFunction(ti, M_st, Sigma_st, gamma_s, B_st);
                    /* TODO */
                    final RobustBrentSolver solver = new RobustBrentSolver(relTol, absTol, 1e-15);
                    double newPsi;
                    try {
                        newPsi = solver.solve(maxIters, objFunc, meritFunc, null, psiLowerBound, psiUpperLimit,
                                numBisections, depth);
//                        if (calculatePsiCurvature(ti, newPsi, M_st, Sigma_st, gamma_s, B_st) > 0) {
//                            /* we have landed on a local maximum -- reject */
//                            newPsi = Psi_t.getDouble(ti);
//                        }
                    } catch (NoBracketingException | TooManyEvaluationsException e) {
                        /* if a solution can not be found, set Psi to its previous value */
                        newPsi = Psi_t.getDouble(ti);
                    }
                    return new ImmutablePair<>(newPsi, solver.getEvaluations());
                })
                .collect(Collectors.toList());

        final INDArray newPsi_t = Nd4j.create(res.stream().mapToDouble(p -> p.left).toArray(), Psi_t.shape());
        final int maxIterations = Collections.max(res.stream().mapToInt(p -> p.right).boxed().collect(Collectors.toList()));
        final double errNormInfinity = CoverageModelEMWorkspaceNDArrayUtils.getINDArrayNormInfinity(newPsi_t.sub(Psi_t));
        return cloneWithUpdatedPrimitiveAndSignal("Psi_t", newPsi_t, SubroutineSignal.builder()
                .put("error_norm", errNormInfinity)
                .put("iterations", maxIterations).build());
    }

    /******************************************
     * M-step for principal latent target map *
     ******************************************/

    /**
     * Update the principal latent target map without regularization
     *
     * @return
     */
    public CoverageModelEMComputeBlockNDArray updatePrincipalLatentTargetMapUnregularized() {
        /* fetch the required caches */
        final INDArray Q_tll = getINDArrayFromCache("Q_tll");
        final INDArray v_tl = getINDArrayFromCache("v_tl");
        final int numTargets = v_tl.shape()[0];
        final int numLatents = v_tl.shape()[1];

        final INDArray newPrincipalLatentTargetMap = Nd4j.create(numTargets, numLatents);
        IntStream.range(0, numTargets).parallel().forEach(ti ->
                newPrincipalLatentTargetMap.get(NDArrayIndex.point(ti), NDArrayIndex.all()).assign(
                        CoverageModelEMWorkspaceNDArrayUtils.linsolve(Q_tll.get(NDArrayIndex.point(ti),
                                NDArrayIndex.all(), NDArrayIndex.all()),
                                v_tl.get(NDArrayIndex.point(ti), NDArrayIndex.all()))));

        final double errNormInfinity = CoverageModelEMWorkspaceNDArrayUtils.getINDArrayNormInfinity(newPrincipalLatentTargetMap
                .sub(getINDArrayFromCache("W_tl")));

        return cloneWithUpdatedPrimitiveAndSignal("W_tl", newPrincipalLatentTargetMap,
                SubroutineSignal.builder().put("error_norm", errNormInfinity).build());
    }

    /***********
     * cloners *
     ***********/

    /**********************************************************************************************
     * IMPORTANT NOTE: cloners are NOT allowed to mutate any member variables; they must return a *
     * new instance with reference to unchanged and persistent members                            *
     **********************************************************************************************/

    /**
     * Initialize read count caches based on the provided {@code rawReadCountBlock}
     * and {@code germlineCopyNumberBlock} corresponding to Fortran-ordered raveled arrays of S x T read counts
     * and germline copy numbers in the target space block corresponding to {@code targetBlock}
     *
     * @param rawReadCountBlock Fortran-order raveled array of raw read counts
     * @param germlineCopyNumberBlock Fortran-order raveled array of germline copy numbers
     * @param maskBlock Fortran-order raveled array of masks
     */
    public CoverageModelEMComputeBlockNDArray cloneWithInitializedReadCountData(@Nonnull final double[] rawReadCountBlock,
                                                                                @Nonnull final double[] germlineCopyNumberBlock,
                                                                                @Nonnull final double[] maskBlock) {
        try {
            Assert.assertEquals(rawReadCountBlock.length, numSamples * numTargets);
            Assert.assertEquals(germlineCopyNumberBlock.length, numSamples * numTargets);
            Assert.assertEquals(maskBlock.length, numSamples * numTargets);
        } catch (final AssertionError e) {
            throw new UserException.BadInput("The provided data blocks do not have the expected length.");
        }

        final INDArray sampleReadCounts = Nd4j.create(rawReadCountBlock, new int[]{numSamples, numTargets}, 'f');
        final INDArray sampleGermlineCopyNumber = Nd4j.create(germlineCopyNumberBlock, new int[]{numSamples, numTargets}, 'f');
        final INDArray mask = Nd4j.create(maskBlock, new int[]{numSamples, numTargets}, 'f');

        return this.cloneWithUpdatedPrimitive("log_nu_st", Transforms.log(sampleReadCounts.div(sampleGermlineCopyNumber), false))
                .cloneWithUpdatedPrimitive("Sigma_st", Nd4j.ones(sampleReadCounts.shape()).divi(sampleReadCounts))
                .cloneWithUpdatedPrimitive("M_st", mask)
                .cloneWithUpdatedPrimitive("log_P_st", Transforms.log(sampleGermlineCopyNumber, true));
    }

    /**
     *
     * @return
     */
    public CoverageModelEMComputeBlockNDArray cloneWithUnityCopyRatio() {
        final ImmutableComputableGraph newICG = icg
                .setValue("log_c_st", new DuplicableNDArray(Nd4j.zeros(numSamples, numTargets)))
                .setValue("var_log_c_st", new DuplicableNDArray(Nd4j.zeros(numSamples, numTargets)));
        return new CoverageModelEMComputeBlockNDArray(targetBlock, numSamples, numLatents, newICG, latestMStepSignal);
    }

    /**
     * TODO duplicates the value -- check every use case and see if it is necessary
     *
     * @param key
     * @param value
     * @return
     */
    public CoverageModelEMComputeBlockNDArray cloneWithUpdatedPrimitive(@Nonnull final String key,
                                                                        @Nullable final INDArray value) {
        if (value == null) {
            return new CoverageModelEMComputeBlockNDArray(targetBlock, numSamples, numLatents,
                    icg.setValue(key, new DuplicableNDArray()), latestMStepSignal);
        } else {
            return new CoverageModelEMComputeBlockNDArray(targetBlock, numSamples, numLatents,
                    icg.setValue(key, new DuplicableNDArray(value.dup())), latestMStepSignal);
        }
    }

    /**
     * TODO duplicates the value -- check every use case and see if it is necessary
     *
     * @param key
     * @param value
     * @param latestMStepSignal
     * @return
     */
    public CoverageModelEMComputeBlockNDArray cloneWithUpdatedPrimitiveAndSignal(@Nonnull final String key,
                                                                                 @Nullable final INDArray value,
                                                                                 @Nonnull final SubroutineSignal latestMStepSignal) {
        if (value == null) {
            return new CoverageModelEMComputeBlockNDArray(targetBlock, numSamples, numLatents,
                    icg.setValue(key, new DuplicableNDArray()), latestMStepSignal);
        } else {
            return new CoverageModelEMComputeBlockNDArray(targetBlock, numSamples, numLatents,
                    icg.setValue(key, new DuplicableNDArray(value.dup())), latestMStepSignal);
        }
    }

    /**
     *
     * @param latestMStepSignal
     * @return
     */
    public CoverageModelEMComputeBlockNDArray cloneWithUpdatedSignal(@Nonnull final SubroutineSignal latestMStepSignal) {
        return new CoverageModelEMComputeBlockNDArray(targetBlock, numSamples, numLatents, icg, latestMStepSignal);
    }

    /**
     *
     * @return
     */
    public CoverageModelEMComputeBlockNDArray cloneWithUpdatedAllCaches() {
        return new CoverageModelEMComputeBlockNDArray(targetBlock, numSamples, numLatents,
                icg.updateAllCaches(), latestMStepSignal);
    }

    /**
     *
     * @param tag
     * @return
     */
    public CoverageModelEMComputeBlockNDArray cloneWithUpdatedCachesByTag(final String tag) {
        return new CoverageModelEMComputeBlockNDArray(targetBlock, numSamples, numLatents,
                icg.updateCachesForTag(tag), latestMStepSignal);
    }

    /**
     * Clones the compute block with rotated latent space (affects bias posteriors and the principal map)
     *
     * @param U an orthogonal transformation
     * @return an instance of {@link CoverageModelEMComputeBlockNDArray}
     */
    public CoverageModelEMComputeBlockNDArray cloneWithRotatedLatentSpace(@Nonnull final INDArray U) {
        /* fetch all affected quantities */
        final INDArray W_tl = getINDArrayFromCache("W_tl");
        INDArray F_W_tl;
        try {
            F_W_tl = getINDArrayFromCache("F_W_tl");
        } catch (final IllegalStateException ex) { /* will be thrown if F[W] is not initialized */
            F_W_tl = null;
        }

        /* rotate [W] and [F][W] */
        final INDArray new_W_tl = (W_tl == null) ? null : W_tl.mmul(U.transpose());
        final INDArray new_F_W_tl = (F_W_tl == null) ? null : F_W_tl.mmul(U.transpose());

        /* rotate bias latent variables */
        final INDArray z_sl = getINDArrayFromCache("z_sl");
        final INDArray zz_sll = getINDArrayFromCache("zz_sll");

        /* rotate E[z_s] and E[z_s z_s^T] */
        final INDArray new_z_sl, new_zz_sll;
        if (z_sl == null || zz_sll == null) {
            new_z_sl = null;
            new_zz_sll = null;
        } else {
            new_z_sl = Nd4j.zeros(z_sl.shape());
            new_zz_sll = Nd4j.zeros(zz_sll.shape());
            IntStream.range(0, numSamples).parallel().forEach(si -> {
                new_z_sl.get(NDArrayIndex.point(si), NDArrayIndex.all()).assign(
                        U.mmul(z_sl.get(NDArrayIndex.point(si), NDArrayIndex.all()).transpose()).transpose());
                new_zz_sll.get(NDArrayIndex.point(si), NDArrayIndex.all(), NDArrayIndex.all()).assign(
                        U.mmul(zz_sll.get(NDArrayIndex.point(si), NDArrayIndex.all(), NDArrayIndex.all())).mmul(U.transpose()));
            });
        }

        return cloneWithUpdatedPrimitive("W_tl", new_W_tl)
                .cloneWithUpdatedPrimitive("F_W_tl", new_F_W_tl)
                .cloneWithUpdatedPrimitive("z_sl", new_z_sl)
                .cloneWithUpdatedPrimitive("zz_sll", new_zz_sll);
    }

    /**
     * This method creates an empty instance of {@link ImmutableComputableGraph} that automatically
     * performs most of the internal calculations of this compute block
     *
     * @return an instance of {@link ImmutableComputableGraph}
     */
    private static ImmutableComputableGraph createEmptyCacheGraph() {
        final ImmutableComputableGraph.ImmutableComputableGraphBuilder cgbuilder =
                ImmutableComputableGraph.builder();

        /**
         * Data nodes
         */
        cgbuilder.addPrimitiveNode("log_nu_st", /* \log(n_{st}/P_{st}) */
                        new String[]{},
                        new DuplicableNDArray())
                .addPrimitiveNode("Sigma_st", /* \Sigma_{st} = 1/n_{st} */
                        new String[]{},
                        new DuplicableNDArray())
                .addPrimitiveNode("M_st", /* mask */
                        new String[]{},
                        new DuplicableNDArray())
                .addPrimitiveNode("log_P_st", /* \log(P_{st}) */
                        new String[]{},
                        new DuplicableNDArray());

        /**
         * Model parameters
         */
        cgbuilder.addPrimitiveNode("m_t", /* model mean target bias */
                        new String[]{},
                        new DuplicableNDArray())
                .addPrimitiveNode("Psi_t", /* model unexplained variance */
                        new String[]{},
                        new DuplicableNDArray())
                .addPrimitiveNode("W_tl", /* model principal map */
                        new String[]{},
                        new DuplicableNDArray());

        /**
         * Externally determined computable nodes (all of latent posterior expectations + etc)
         */
        cgbuilder.addComputableNode("log_c_st", /* E[log(c_{st})] */
                        new String[]{},
                        new String[]{},
                        null, true)
                .addComputableNode("var_log_c_st", /* var[log(c_{st})] */
                        new String[]{},
                        new String[]{},
                        null, true)
                .addComputableNode("log_d_s", /* E[log(d_s)] */
                        new String[]{},
                        new String[]{},
                        null, true)
                .addComputableNode("var_log_d_s", /* var[log(d_s)] */
                        new String[]{},
                        new String[]{},
                        null, true)
                .addComputableNode("gamma_s", /* E[\gamma_s] */
                        new String[]{},
                        new String[]{},
                        null, true)
                .addComputableNode("z_sl", /* E[z_{sm}] */
                        new String[]{},
                        new String[]{},
                        null, true)
                .addComputableNode("zz_sll", /* E[z_{sm} z_{sn}] */
                        new String[]{},
                        new String[]{},
                        null, true)
                .addComputableNode("F_W_tl", /* F [W] */
                        new String[]{},
                        new String[]{"W_tl"},
                        null, true);

        /**
         * Automatic computable nodes
         */
        cgbuilder.addComputableNode("sum_M_t", /* \sum_s M_{st} */
                        new String[]{},
                        new String[]{"M_st"},
                        calculate_sum_M_t, true)
                .addComputableNode("sum_M_s", /* \sum_t M_{st} */
                        new String[]{"LOGLIKE_UNREG", "LOGLIKE_REG"},
                        new String[]{"M_st"},
                        calculate_sum_M_s, true);

        cgbuilder.addComputableNode("Wz_st", /* W E[z_s] */
                        new String[]{"M_STEP_M", "E_STEP_D", "E_STEP_C"},
                        new String[]{"W_tl", "z_sl"},
                        calculate_Wz_st, true)
                .addComputableNode("WzzWT_st", /* (W E[z_s z_s^T] W^T)_{tt} */
                        new String[]{},
                        new String[]{"W_tl", "zz_sll"},
                        calculate_WzzWT_st, true);

        cgbuilder.addComputableNode("tot_Psi_st", /* \Psi_{st} = \Psi_t + \Sigma_{st} + E[\gamma_s] */
                        new String[]{},
                        new String[]{"Sigma_st", "Psi_t", "gamma_s"},
                        calculate_tot_Psi_st, true)
                .addComputableNode("Delta_st", /* log(n_{st}/P_{st}) - E[log(c_{st})] - E[log(d_s)] - m_t */
                        new String[]{"E_STEP_Z"},
                        new String[]{"log_nu_st", "log_c_st", "log_d_s", "m_t"},
                        calculate_Delta_st, true)
                .addComputableNode("M_log_Psi_s", /* \sum_{t} M_{st} \log(\Psi_{st}) */
                        new String[]{},
                        new String[]{"M_st", "tot_Psi_st"},
                        calculate_M_log_Psi_s, true)
                .addComputableNode("M_Psi_inv_st", /* M_{st} \Psi_{st}^{-1} */
                        new String[]{"M_STEP_W_REG", "M_STEP_M", "E_STEP_Z", "E_STEP_D", "E_STEP_C"},
                        new String[]{"M_st", "tot_Psi_st"},
                        calculate_M_Psi_inv_st, true);

        cgbuilder.addComputableNode("v_tl", /* v_{t\mu} */
                        new String[]{"M_STEP_W_REG", "M_STEP_W_UNREG"},
                        new String[]{"M_Psi_inv_st", "Delta_st", "z_sl"},
                        calculate_v_tl, true)
                .addComputableNode("Q_tll", /* Q_{t\mu\nu} */
                        new String[]{"M_STEP_W_REG", "M_STEP_W_UNREG"},
                        new String[]{"M_Psi_inv_st", "zz_sll"},
                        calculate_Q_tll, true)
                .addComputableNode("sum_Q_ll", /* \sum_t Q_{t\mu\nu} */
                        new String[]{"M_STEP_W_REG"},
                        new String[]{"Q_tll"},
                        calculate_sum_Q_ll, true)
                .addComputableNode("B_st", /* B_{st} */
                        new String[]{"M_STEP_PSI", "E_STEP_GAMMA"},
                        new String[]{"Delta_st", "var_log_c_st", "var_log_d_s", "WzzWT_st", "Wz_st"},
                        calculate_B_st, true);

        cgbuilder.addComputableNode("loglike_unreg", /* non-normalized log likelihood without regularization per sample */
                        new String[]{"LOGLIKE_UNREG"},
                        new String[]{"B_st", "M_Psi_inv_st", "M_log_Psi_s"},
                        calculate_loglike_unreg, true)
                .addComputableNode("loglike_reg", /* non-normalized log likelihood with regularization per sample */
                        new String[]{"LOGLIKE_REG"},
                        new String[]{"B_st", "M_Psi_inv_st", "M_log_Psi_s", "W_tl", "F_W_tl", "zz_sll"},
                        calculate_loglike_reg, true);

        return cgbuilder.build();
    }

    /********************************
     * helper computational methods *
     ********************************/

    /**
     *
     * @param key
     * @param dat
     * @return
     */
    private static INDArray getINDArrayFromMap(final String key, Map<String, ? extends Duplicable> dat) {
        return ((DuplicableNDArray)dat.get(key)).value();
    }

    /**
     *
     * @param key
     * @param dat
     * @return
     */
    @SuppressWarnings("unchecked")
    private static double getDoubleFromMap(final String key, Map<String, ? extends Duplicable> dat) {
        return ((DuplicableNumber<Double>)dat.get(key)).value();
    }

    /********************
     * persistent nodes *
     ********************/

    /* dependents: [M_st] */
    private static final Function<Map<String, ? extends Duplicable>, ? extends Duplicable> calculate_sum_M_t =
            new Function<Map<String, ? extends Duplicable>, Duplicable>() {
                @Override
                public Duplicable apply(Map<String, ? extends Duplicable> dat) {
                    return new DuplicableNDArray(getINDArrayFromMap("M_st", dat).sum(0));
                }
            };

    /* dependents: [M_st] */
    private static final Function<Map<String, ? extends Duplicable>, ? extends Duplicable> calculate_sum_M_s =
            new Function<Map<String, ? extends Duplicable>, Duplicable>() {
                @Override
                public Duplicable apply(Map<String, ? extends Duplicable> dat) {
                    return new DuplicableNDArray(getINDArrayFromMap("M_st", dat).sum(1));
                }
            };

    /******************************
     * automatic computable nodes *
     ******************************/

    /**
     * Calculates $\Delta_{st} = \log(n_{st}/(P_{st}) - E[log(c_{st}) - E[log(d_s)] - m_t$
     *
     * Note: it is assumed that the argument of the log is proper, i.e. n_{st} != 0, P_{st} != 0, c_{st} !=0
     * and d_s != 0
     *
     * dependents: ["log_nu_st", "log_c_st", "log_d_s", "m_t"]
     */
    private static final Function<Map<String, ? extends Duplicable>, ? extends Duplicable> calculate_Delta_st =
            new Function<Map<String, ? extends Duplicable>, Duplicable>() {
                @Override
                public Duplicable apply(Map<String, ? extends Duplicable> dat) {
                    final INDArray log_nu_st = getINDArrayFromMap("log_nu_st", dat);
                    final INDArray log_c_st = getINDArrayFromMap("log_c_st", dat);
                    final INDArray log_d_s = getINDArrayFromMap("log_d_s", dat);
                    final INDArray m_t = getINDArrayFromMap("m_t", dat);
                    return new DuplicableNDArray(log_nu_st.sub(log_c_st).subiColumnVector(log_d_s).subiRowVector(m_t));
                }
            };

    /* dependents: [W_tl, z_sl] */
    private static final Function<Map<String, ? extends Duplicable>, ? extends Duplicable> calculate_Wz_st =
            new Function<Map<String, ? extends Duplicable>, Duplicable>() {
                @Override
                public Duplicable apply(Map<String, ? extends Duplicable> dat) {
                    final INDArray W_tl = getINDArrayFromMap("W_tl", dat);
                    final INDArray z_sl = getINDArrayFromMap("z_sl", dat);
                    return new DuplicableNDArray(W_tl.mmul(z_sl.transpose()).transpose());
                }
            };

    /* dependents: [W_tl, zz_sll] */
    private static final Function<Map<String, ? extends Duplicable>, ? extends Duplicable> calculate_WzzWT_st =
            new Function<Map<String, ? extends Duplicable>, Duplicable>() {
                @Override
                public Duplicable apply(Map<String, ? extends Duplicable> dat) {
                    final INDArray W_tl = getINDArrayFromMap("W_tl", dat);
                    final INDArray zz_sll = getINDArrayFromMap("zz_sll", dat);
                    final int numSamples = zz_sll.shape()[0];
                    final int numTargets = W_tl.shape()[0];
                    final INDArray WzzWT_st = Nd4j.create(numSamples, numTargets);
                    IntStream.range(0, numSamples).parallel().forEach(si ->
                            WzzWT_st.get(NDArrayIndex.point(si)).assign(
                                    W_tl.mmul(zz_sll.get(NDArrayIndex.point(si))).mul(W_tl).sum(1).transpose()));
                    return new DuplicableNDArray(WzzWT_st);
                }
            };

    /* dependents: ["Sigma_st", "Psi_t"] */
    private static final Function<Map<String, ? extends Duplicable>, ? extends Duplicable> calculate_tot_Psi_st =
            new Function<Map<String, ? extends Duplicable>, Duplicable>() {
                @Override
                public Duplicable apply(Map<String, ? extends Duplicable> dat) {
                    final INDArray Sigma_st = getINDArrayFromMap("Sigma_st", dat);
                    final INDArray Psi_t = getINDArrayFromMap("Psi_t", dat);
                    final INDArray gamma_s = getINDArrayFromMap("gamma_s", dat);
                    return new DuplicableNDArray(Sigma_st.addRowVector(Psi_t).addiColumnVector(gamma_s));
                }
            };

    /* dependents: ["M_st", "tot_Psi_st"] */
    private static final Function<Map<String, ? extends Duplicable>, ? extends Duplicable> calculate_M_log_Psi_s =
            new Function<Map<String, ? extends Duplicable>, Duplicable>() {
                @Override
                public Duplicable apply(Map<String, ? extends Duplicable> dat) {
                    final INDArray M_st = getINDArrayFromMap("M_st", dat);
                    final INDArray tot_Psi_st = getINDArrayFromMap("tot_Psi_st", dat);
                    final INDArray M_log_Psi_s = Transforms.log(tot_Psi_st, true).muli(M_st).sum(1);
                    return new DuplicableNDArray(M_log_Psi_s);
                }
            };

    /* dependents: ["M_st", "tot_Psi_st"] */
    private static final Function<Map<String, ? extends Duplicable>, ? extends Duplicable> calculate_M_Psi_inv_st =
            new Function<Map<String, ? extends Duplicable>, Duplicable>() {
                @Override
                public Duplicable apply(Map<String, ? extends Duplicable> dat) {
                    return new DuplicableNDArray(getINDArrayFromMap("M_st", dat).div(getINDArrayFromMap("tot_Psi_st", dat)));
                }
            };

    /* dependents: ["M_Psi_inv_st", "Delta_st", "z_sl"] */
    private static final Function<Map<String, ? extends Duplicable>, ? extends Duplicable> calculate_v_tl =
            new Function<Map<String, ? extends Duplicable>, Duplicable>() {
                @Override
                public Duplicable apply(Map<String, ? extends Duplicable> dat) {
                    final INDArray M_Psi_inv_st = getINDArrayFromMap("M_Psi_inv_st", dat);
                    final INDArray Delta_st = getINDArrayFromMap("Delta_st", dat);
                    final INDArray z_sl = getINDArrayFromMap("z_sl", dat);
                    return new DuplicableNDArray(M_Psi_inv_st.mul(Delta_st).transpose().mmul(z_sl));
                }
            };

    /* dependents: ["M_Psi_inv_st", "zz_sll"] */
    private static final Function<Map<String, ? extends Duplicable>, ? extends Duplicable> calculate_Q_tll =
            new Function<Map<String, ? extends Duplicable>, Duplicable>() {
                @Override
                public Duplicable apply(Map<String, ? extends Duplicable> dat) {
                    final INDArray M_Psi_inv_st_trans = getINDArrayFromMap("M_Psi_inv_st", dat).transpose();
                    final INDArray zz_sll = getINDArrayFromMap("zz_sll", dat);
                    final int numTargets = M_Psi_inv_st_trans.shape()[0];
                    final int numLatents = zz_sll.shape()[1];
                    final INDArray res = Nd4j.create(numTargets, numLatents, numLatents);
                    IntStream.range(0, numLatents).parallel().forEach(li ->
                            res.get(NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.point(li)).assign(
                                    M_Psi_inv_st_trans.mmul(zz_sll.get(NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.point(li)))));
                    return new DuplicableNDArray(res);
                }
            };

    /* dependents: ["Q_tll"] */
    private static final Function<Map<String, ? extends Duplicable>, ? extends Duplicable> calculate_sum_Q_ll =
            new Function<Map<String, ? extends Duplicable>, Duplicable>() {
                @Override
                public Duplicable apply(Map<String, ? extends Duplicable> dat) {
                    return new DuplicableNDArray(getINDArrayFromMap("Q_tll", dat).sum(0));
                }
            };

    /* dependents: ["Delta_st", "var_log_c_st", "var_log_d_s", "WzzWT_st", "Wz_st"] */
    private static final Function<Map<String, ? extends Duplicable>, ? extends Duplicable> calculate_B_st =
            new Function<Map<String, ? extends Duplicable>, Duplicable>() {
                @Override
                public Duplicable apply(Map<String, ? extends Duplicable> dat) {
                    final INDArray Delta_st = getINDArrayFromMap("Delta_st", dat);
                    final INDArray var_log_c_st = getINDArrayFromMap("var_log_c_st", dat);
                    final INDArray var_log_d_s = getINDArrayFromMap("var_log_d_s", dat);
                    final INDArray WzzWT_st = getINDArrayFromMap("WzzWT_st", dat);
                    final INDArray Wz_st = getINDArrayFromMap("Wz_st", dat);
                    return new DuplicableNDArray(
                            Delta_st.mul(Delta_st)
                                    .addi(var_log_c_st)
                                    .addiColumnVector(var_log_d_s)
                                    .addi(WzzWT_st)
                                    .subi(Delta_st.mul(Wz_st.mul(2))));
                }
            };

    /************************
     * log likelihood nodes *
     ************************/

    /* dependents: ["B_st", "M_Psi_inv_st", "M_log_Psi_s"] */
    private static final Function<Map<String, ? extends Duplicable>, ? extends Duplicable> calculate_loglike_unreg =
            new Function<Map<String, ? extends Duplicable>, Duplicable>() {
                @Override
                public Duplicable apply(Map<String, ? extends Duplicable> dat) {
                    final INDArray B_st = getINDArrayFromMap("B_st", dat);
                    final INDArray M_Psi_inv_st = getINDArrayFromMap("M_Psi_inv_st", dat);
                    final INDArray M_log_Psi_s = getINDArrayFromMap("M_log_Psi_s", dat);
                    return new DuplicableNDArray(B_st.mul(M_Psi_inv_st).sum(1)
                            .addi(M_log_Psi_s)
                            .muli(-0.5));
                }
            };

    /* TODO: the filter contribution part could be cached */
    /* dependents: ["B_st", "M_Psi_inv_st", "M_log_Psi_s", "W_tl", "F_W_tl", "zz_sll"] */
    private static final Function<Map<String, ? extends Duplicable>, ? extends Duplicable> calculate_loglike_reg =
            new Function<Map<String, ? extends Duplicable>, Duplicable>() {
                @Override
                public Duplicable apply(Map<String, ? extends Duplicable> dat) {
                    final INDArray B_st = getINDArrayFromMap("B_st", dat);
                    final INDArray M_Psi_inv_st = getINDArrayFromMap("M_Psi_inv_st", dat);
                    final INDArray M_log_Psi_s = getINDArrayFromMap("M_log_Psi_s", dat);
                    final INDArray regularPart = B_st.mul(M_Psi_inv_st).sum(1)
                            .addi(M_log_Psi_s)
                            .muli(-0.5);

                    final INDArray W_tl = getINDArrayFromMap("W_tl", dat);
                    final INDArray F_W_tl = getINDArrayFromMap("F_W_tl", dat);
                    final INDArray zz_sll = getINDArrayFromMap("zz_sll", dat);
                    final INDArray WFW_ll = W_tl.transpose().mmul(F_W_tl).muli(-0.5);
                    final int numSamples = B_st.shape()[0];
                    final INDArray filterPart = Nd4j.create(new int[]{numSamples, 1},
                            IntStream.range(0, numSamples).mapToDouble(si ->
                                    zz_sll.get(NDArrayIndex.point(si)).mul(WFW_ll).sumNumber().doubleValue()).toArray());

                    return new DuplicableNDArray(regularPart.addi(filterPart));
                }
            };

}
