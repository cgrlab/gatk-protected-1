package org.broadinstitute.hellbender.tools.coveragemodel;

import com.google.common.annotations.VisibleForTesting;
import org.apache.commons.lang.ArrayUtils;
import org.apache.commons.lang3.tuple.ImmutablePair;
import org.apache.commons.lang3.tuple.ImmutableTriple;
import org.apache.commons.lang3.tuple.Pair;
import org.apache.commons.math3.analysis.UnivariateFunction;
import org.apache.commons.math3.analysis.solvers.BrentSolver;
import org.apache.commons.math3.exception.NoBracketingException;
import org.apache.commons.math3.exception.TooManyEvaluationsException;
import org.apache.commons.math3.util.FastMath;
import org.apache.spark.HashPartitioner;
import org.apache.spark.Partitioner;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.api.java.function.Function2;
import org.apache.spark.broadcast.Broadcast;
import org.apache.spark.storage.StorageLevel;
import org.broadinstitute.hellbender.exceptions.UserException;
import org.broadinstitute.hellbender.tools.coveragemodel.annots.CachesRDD;
import org.broadinstitute.hellbender.tools.coveragemodel.annots.EvaluatesRDD;
import org.broadinstitute.hellbender.tools.coveragemodel.annots.UpdatesRDD;
import org.broadinstitute.hellbender.tools.coveragemodel.interfaces.CopyRatioPosteriorCalculator;
import org.broadinstitute.hellbender.tools.coveragemodel.linalg.FourierLinearOperator;
import org.broadinstitute.hellbender.tools.coveragemodel.linalg.FourierLinearOperatorNDArray;
import org.broadinstitute.hellbender.tools.coveragemodel.linalg.GeneralLinearOperator;
import org.broadinstitute.hellbender.tools.coveragemodel.linalg.IterativeLinearSolverNDArray;
import org.broadinstitute.hellbender.tools.coveragemodel.linalg.IterativeLinearSolverNDArray.ExitStatus;
import org.broadinstitute.hellbender.tools.exome.ReadCountCollection;
import org.broadinstitute.hellbender.tools.exome.ReadCountRecord;
import org.broadinstitute.hellbender.tools.exome.Target;
import org.broadinstitute.hellbender.tools.exome.sexgenotyper.GermlinePloidyAnnotatedTargetCollection;
import org.broadinstitute.hellbender.tools.exome.sexgenotyper.SexGenotypeDataCollection;
import org.broadinstitute.hellbender.utils.hmm.interfaces.AlleleMetadataProvider;
import org.broadinstitute.hellbender.utils.hmm.interfaces.CallStringProvider;
import org.broadinstitute.hellbender.utils.hmm.interfaces.ScalarProvider;
import org.broadinstitute.hellbender.utils.param.ParamUtils;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.buffer.util.DataTypeUtil;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;
import scala.Tuple2;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.util.*;
import java.util.stream.Collectors;
import java.util.stream.IntStream;
import java.util.stream.Stream;

import static org.broadinstitute.hellbender.tools.coveragemodel.CoverageModelEMParams.CopyRatioHMMType.*;

/**
 * This class implements a local memory version of {@link CoverageModelEMWorkspace}
 *
 * @param <S> hidden state type
 *
 * @author Mehrtash Babadi &lt;mehrtash@broadinstitute.org&gt;
 */

public final class CoverageModelEMWorkspaceNDArraySparkToggle<S extends AlleleMetadataProvider & CallStringProvider &
        ScalarProvider> extends CoverageModelEMWorkspace<INDArray, INDArray, S> {

    /**
     * Targets with zero read count will be promoted to the following -- this value is arbitrary and immaterial
     * since such targets are masked out and do not contribute to the likelihood function
     */
    private static final int READ_COUNT_ON_UNCOVERED_TARGETS_FIXUP_VALUE = 1;

    /**
     * Zero ploidy targets will be promoted to the following -- this value is arbitrary and immaterial
     * since such targets are masked out and do not contribute to the likelihood function
     * */
    private static final int PLOIDY_ON_UNCOVERED_TARGETS_FIXUP_VALUE = 1;

    /**
     * Minimum size of a target block
     */
    private static final int DEFAULT_MIN_TARGET_BLOCK_SIZE = 5;

    /**
     *
     */
    private static final double INITIAL_TARGET_UNEXPLAINED_VARIANCE = 0.05;

    /**
     *
     */
    private static final double INITIAL_PRINCIPAL_MAP_DIAGONAL = 1.0;

    /**************
     * spark mode *
     **************/

    private final boolean sparkContextIsAvailable;

    private final JavaSparkContext ctx;

    private JavaPairRDD<LinearSpaceBlock, CoverageModelEMComputeBlockNDArray> computeRDD;
    private JavaPairRDD<LinearSpaceBlock, CoverageModelEMComputeBlockNDArray> prevCheckpointedComputeRDD;
    private Deque<JavaPairRDD<LinearSpaceBlock, CoverageModelEMComputeBlockNDArray>> prevCachedComputeRDDDeque = new LinkedList<>();
    private int cacheCallCounter;

    /***********************
     * single-machine mode *
     ***********************/

    private CoverageModelEMComputeBlockNDArray localComputeBlock;

    /****************************
     * related to target blocks *
     ****************************/

    /* number of target space blocks */
    private final int numTargetBlocks;

    /* list of target space blocks */
    private List<LinearSpaceBlock> targetBlocks;

    /*****************************************************
     * driver node copy of latent posterior expectations *
     *****************************************************/

    /* $E[log(d_s)]$ */
    private final INDArray sampleMeanLogReadDepths;

    /* $var[log(d_s)]$ */
    private final INDArray sampleVarLogReadDepths;

    /* $E[z_{sm}]$ */
    private final INDArray sampleBiasLatentPosteriorFirstMoments;

    /* $E[z_{sm} z_{sn}]$ */
    private final INDArray sampleBiasLatentPosteriorSecondMoments;

    /* $E[\gamma_s]$ */
    private final INDArray sampleUnexplainedVariance;

    /**
     * Regularizer Fourier factors
     */

    private final double[] fourierFactors;

    /**
     * Public constructor
     *
     * @param rawReadCounts an instance of {@link ReadCountCollection} containing raw read counts
     * @param ploidyAnnots an instance of {@link GermlinePloidyAnnotatedTargetCollection} for obtaining target ploidies
     *                     for different sex genotypes
     * @param sexGenotypeData an instance of {@link SexGenotypeDataCollection} for obtaining sample sex genotypes
     * @param params EM algorithm parameter oracle
     * @param numTargetBlocks number of target space partitions (will be ignored if the Spark context is null)
     * @param ctx the Spark context
     */
    @UpdatesRDD @CachesRDD @EvaluatesRDD
    public CoverageModelEMWorkspaceNDArraySparkToggle(@Nonnull final ReadCountCollection rawReadCounts,
                                                      @Nonnull final GermlinePloidyAnnotatedTargetCollection ploidyAnnots,
                                                      @Nonnull final SexGenotypeDataCollection sexGenotypeData,
                                                      @Nonnull final CopyRatioPosteriorCalculator<CoverageModelCopyRatioEmissionData, S> copyRatioPosteriorCalculator,
                                                      @Nonnull final CoverageModelEMParams params,
                                                      @Nullable final CoverageModelParametersNDArray model,
                                                      final int numTargetBlocks,
                                                      @Nullable final JavaSparkContext ctx) {
        /* the super constructor takes care of filtering the reads and fetching germline ploidies */
        super(rawReadCounts, ploidyAnnots, sexGenotypeData, copyRatioPosteriorCalculator, params, model);

        this.ctx = ctx;
        if (ctx == null) {
            sparkContextIsAvailable = false;
            this.numTargetBlocks = 1;
        } else {
            sparkContextIsAvailable = true;
            this.numTargetBlocks = ParamUtils.inRange(numTargetBlocks, 1, numTargets, "Number of target blocks must be " +
                    "between 1 and the size of target space.");
        }

        /* Set Nd4j DType to Double in context */
        Nd4j.create(1);
        DataTypeUtil.setDTypeForContext(DataBuffer.Type.DOUBLE);

        /**
         * Allocate memory for latent posterior expectations (these objects live on the driver node
         * Note: a copy is also sent to the compute nodes when needed
         */
        sampleMeanLogReadDepths = Nd4j.zeros(numSamples, 1);
        sampleVarLogReadDepths = Nd4j.zeros(numSamples, 1);
        sampleBiasLatentPosteriorFirstMoments = Nd4j.zeros(numSamples, numLatents);
        sampleBiasLatentPosteriorSecondMoments = Nd4j.zeros(numSamples, numLatents, numLatents);
        sampleUnexplainedVariance = Nd4j.zeros(numSamples, 1);

        /* initialize the regularizer filter operator
         * note: the fourier operator is pre-multiplied by the filter strength */
        if (params.fourierRegularizationEnabled()) {
            fourierFactors = FourierLinearOperator.getMidpassFilterFourierFactors(numTargets,
                    numTargets/params.getMaximumCNVLength(), numTargets/params.getMinimumCNVLength());
        } else {
            fourierFactors = null;
        }

        /* initialize target blocks */
        initializeTargetSpacePartitions();

        /* create RDDs */
        instantiateWorkers();

        /* read $n_st$ and $P_st$ and push to RDDs */
        initializeComputeBlocksWithReadCounts();

        if (processedModel == null) {
            /* initialize model parameters and posterior expectations to default initial values */
            initializeModelParametersToDefaultValues();
        } else {
            initializeModelParametersFromGivenModel(processedModel);
        }
    }

    /**
     * Partitions the target space into {@link #numTargetBlocks} contiguous blocks
     */
    private void initializeTargetSpacePartitions() {
        targetBlocks = CoverageModelSparkUtils.createLinearSpaceBlocks(numTargets, numTargetBlocks,
                DEFAULT_MIN_TARGET_BLOCK_SIZE);
        logger.debug("Target space blocks: " + targetBlocks.stream().map(LinearSpaceBlock::toString)
                .reduce((L, R) -> L + "\t" + R).orElse("None"));
    }

    /**
     * Pushes the read count data to compute block(s)
     */
    @UpdatesRDD
    private void initializeComputeBlocksWithReadCounts() {
        /* parse reads and initialize containers */
        final List<ReadCountRecord> recs = processedReadCounts.records();

        logger.info("Pushing read count data to the worker(s)");
        final List<Tuple2<LinearSpaceBlock, ImmutableTriple<double[], double[], double[]>>> readCountKeyValuePairList =
                new ArrayList<>();
        targetBlockStream().forEach(tb -> {
            /* take a contiguous (target chuck) x numSamples chunk from the read count collection */
            double[] rawReadCountBlock = IntStream.range(tb.getBegIndex(), tb.getEndIndex())
                    .mapToObj(ti -> Arrays.asList(ArrayUtils.toObject(recs.get(ti).getDoubleCounts())))
                    .flatMap(List::stream)
                    .mapToDouble(d -> (double)FastMath.round(d)) /* round to integer values */
                    .toArray();

            /* fetch the germline copy numbers within the same contiguous block of reads */
            double[] germlinePloidyBlock = IntStream.range(tb.getBegIndex(), tb.getEndIndex())
                    .mapToObj(ti -> sampleGermlinePloidies.stream()
                            .mapToInt(intArray -> intArray[ti])
                            .boxed()
                            .collect(Collectors.toList()))
                    .flatMap(Collection::stream)
                    .mapToDouble(Integer::doubleValue)
                    .toArray();

            /* create the mask for ploidy and read counts */
            double[] maskBlock = IntStream.range(0, rawReadCountBlock.length)
                    .mapToDouble(idx -> (int)rawReadCountBlock[idx] == 0 || (int)germlinePloidyBlock[idx] == 0 ? 0 : 1)
                    .toArray();

            /**
             * if a target has zero read count, set it to a finite value to avoid generating NaNs and infinities
             */
            IntStream.range(0, rawReadCountBlock.length)
                    .filter(idx -> (int)rawReadCountBlock[idx] == 0)
                    .forEach(idx -> rawReadCountBlock[idx] = READ_COUNT_ON_UNCOVERED_TARGETS_FIXUP_VALUE);

            /**
             * if a target has zero ploidy, set it to a finite value to avoid generating NaNs and infinities
             */
            IntStream.range(0, rawReadCountBlock.length)
                    .filter(idx -> (int)germlinePloidyBlock[idx] == 0)
                    .forEach(idx -> germlinePloidyBlock[idx] = PLOIDY_ON_UNCOVERED_TARGETS_FIXUP_VALUE);

            /* add the block to list */
            readCountKeyValuePairList.add(new Tuple2<>(tb, ImmutableTriple.of(rawReadCountBlock,
                    germlinePloidyBlock, maskBlock)));
        });

        /* push to RDD */
        joinWithWorkersAndMap(readCountKeyValuePairList,
                p -> p._1.cloneWithInitializedReadCountData(p._2.left, p._2.middle, p._2.right));
    }

    /**
     * Initialize the model to default values
     */
    @UpdatesRDD
    private void initializeModelParametersToDefaultValues() {
        /* make local references for Lambdas */
        final CoverageModelEMParams params = this.params;
        final int numSamples = this.numSamples;
        final int numLatents = params.getNumLatents();

        /* set $W_{tm}$ to a zero-padded D X D identity matrix */
        /* set $m_t$ to zero */
        /* set $\Psi_t$ to zero */
        /* set $z_sl$ and $zz_sll$ to zero */
        mapWorkers(cb -> {
            /* find the part of the truncated identity that belongs to the compute block */
            final LinearSpaceBlock tb = cb.getTargetSpaceBlock();
            final INDArray newPrincipalLinearMap = Nd4j.zeros(tb.getNumTargets(), numLatents);
            if (tb.getBegIndex() < numLatents) {
                IntStream.range(tb.getBegIndex(), FastMath.min(tb.getEndIndex(), numLatents)).forEach(ti ->
                        newPrincipalLinearMap.getRow(ti).assign(Nd4j.zeros(1, numLatents).putScalar(0, ti,
                                INITIAL_PRINCIPAL_MAP_DIAGONAL)));
            }
            return cb.cloneWithUpdatedPrimitive("W_tl", newPrincipalLinearMap)
                    .cloneWithUpdatedPrimitive("m_t", Nd4j.zeros(1, cb.getTargetSpaceBlock().getNumTargets()))
                    .cloneWithUpdatedPrimitive("Psi_t", Nd4j.ones(1, cb.getTargetSpaceBlock().getNumTargets())
                            .mul(INITIAL_TARGET_UNEXPLAINED_VARIANCE))
                    .cloneWithUpdatedPrimitive("z_sl", Nd4j.zeros(numSamples, numLatents))
                    .cloneWithUpdatedPrimitive("zz_sll", Nd4j.zeros(numSamples, numLatents, numLatents))
                    .cloneWithUpdatedPrimitive("gamma_s", Nd4j.zeros(numSamples, 1))
                    .cloneWithUnityCopyRatio();
        });
        if (params.fourierRegularizationEnabled()) {
            mapWorkers(cb -> cb.cloneWithUpdatedCachesByTag("M_STEP_W_REG"));
            updateFilteredPrincipalLatentToTargetMap();
        }
    }

    /**
     * Initialize the model to default values
     */
    @UpdatesRDD
    private void initializeModelParametersFromGivenModel(@Nonnull final CoverageModelParametersNDArray model) {
        /* make local references for Lambdas */
        final CoverageModelEMParams params = this.params;
        final int numSamples = this.numSamples;
        final int numLatents = params.getNumLatents();

        if (sparkContextIsAvailable) {
            final Broadcast<CoverageModelParametersNDArray> broadcastedModel = ctx.broadcast(model);
            mapWorkers(cb -> {
                final LinearSpaceBlock tb = cb.getTargetSpaceBlock();
                return cb.cloneWithUpdatedPrimitive("W_tl", broadcastedModel.getValue().getPrincipalLatentToTargetMapOnTargetBlock(tb))
                        .cloneWithUpdatedPrimitive("m_t", broadcastedModel.getValue().getTargetMeanBiasOnTargetBlock(tb))
                        .cloneWithUpdatedPrimitive("Psi_t", broadcastedModel.getValue().getTargetUnexplainedVarianceOnTargetBlock(tb))
                        .cloneWithUpdatedPrimitive("z_sl", Nd4j.zeros(numSamples, numLatents))
                        .cloneWithUpdatedPrimitive("zz_sll", Nd4j.zeros(numSamples, numLatents, numLatents))
                        .cloneWithUpdatedPrimitive("gamma_s", Nd4j.zeros(numSamples, 1))
                        .cloneWithUnityCopyRatio();
            });
        } else {
            mapWorkers(cb -> {
                final LinearSpaceBlock tb = cb.getTargetSpaceBlock();
                return cb.cloneWithUpdatedPrimitive("W_tl", model.getPrincipalLatentToTargetMapOnTargetBlock(tb))
                        .cloneWithUpdatedPrimitive("m_t", model.getTargetMeanBiasOnTargetBlock(tb))
                        .cloneWithUpdatedPrimitive("Psi_t", model.getTargetUnexplainedVarianceOnTargetBlock(tb))
                        .cloneWithUpdatedPrimitive("z_sl", Nd4j.zeros(numSamples, numLatents))
                        .cloneWithUpdatedPrimitive("zz_sll", Nd4j.zeros(numSamples, numLatents, numLatents))
                        .cloneWithUpdatedPrimitive("gamma_s", Nd4j.zeros(numSamples, 1))
                        .cloneWithUnityCopyRatio();
            });
        }

        if (params.fourierRegularizationEnabled()) {
            mapWorkers(cb -> cb.cloneWithUpdatedCachesByTag("M_STEP_W_REG"));
            updateFilteredPrincipalLatentToTargetMap();
        }
    }

    /**
     * This method fetches the principal latent to target map ($W_st$) from the workers, applies the Fourier filter
     * on it, partitions the result, and pushes it to the compute block(s)
     */
    @EvaluatesRDD @UpdatesRDD
    private void updateFilteredPrincipalLatentToTargetMap() {
        updateFilteredPrincipalLatentToTargetMap(fetchFromWorkers("W_tl", 0));
    }

    /**
     * This method applies the Fourier filter on a given {@param principalLatentTargetMap}, applies the Fourier filter on it,
     * partitions the result, and handles it to the compute block(s)
     *
     * @param newPrincipalLatentTargetMap any principal target latent map
     */
    @UpdatesRDD
    private void updateFilteredPrincipalLatentToTargetMap(@Nonnull final INDArray newPrincipalLatentTargetMap) {
        final INDArray filteredPrincipalLatentTargetMap = Nd4j.create(newPrincipalLatentTargetMap.shape());

        /* instantiate the Fourier filter */
        final FourierLinearOperatorNDArray regularizerFourierLinearOperator = createRegularizerFourierLinearOperator();

        /* FFT by resolving W_tl on l */
        IntStream.range(0, numLatents).parallel()
                .forEach(li -> {
                    final INDArrayIndex[] slice = {NDArrayIndex.all(), NDArrayIndex.point(li)};
                    filteredPrincipalLatentTargetMap.get(slice).assign(
                            regularizerFourierLinearOperator.operate(newPrincipalLatentTargetMap.get(slice)));
                });

        /* sent the new W to workers */
        switch (params.getPrincipalMapCommunicationPolicy()) {
            case BROADCAST_HASH_JOIN:
                pushToWorkers(mapINDArrayToBlocks(filteredPrincipalLatentTargetMap),
                        (W, cb) -> cb.cloneWithUpdatedPrimitive("F_W_tl", W.get(cb.getTargetSpaceBlock())));
                break;

            case RDD_JOIN:
                joinWithWorkersAndMap(chopINDArrayToBlocks(filteredPrincipalLatentTargetMap),
                        p -> p._1.cloneWithUpdatedPrimitive("F_W_tl", p._2));
                break;
        }
    }

    @EvaluatesRDD
    private FourierLinearOperatorNDArray createRegularizerFourierLinearOperator() {
        final double psiAverage = mapWorkersAndReduce(cb -> cb.getINDArrayFromCache("M_Psi_inv_st")
                .sumNumber().doubleValue(), (d1, d2) -> d1 + d2) / (numTargets * numSamples);
        final double fact = params.getFourierRegularizationStrength() * psiAverage;
        return new FourierLinearOperatorNDArray(numTargets,
                Arrays.stream(fourierFactors).map(f -> f * fact).toArray(), params.zeroPadFFT());
    }


    /*******************
     * E-step routines *
     *******************/

    /**
     * Update the first (E[z]) and second (E[z z^T]) posterior moments of the bias underlying latent variable (z)
     *
     * Note: the operations done on the driver node has low complexity only if D is small:
     *
     *     (a) G_s = (I + [contribGMatrix])^{-1} for each sample \sim O(S x D^3)
     *     (b) E[z_s] = G_s [contribZ_s] for each sample \sim O(S x D^3)
     *     (c) E[z_s z_s^T] = G_s + E[z_s] E[z_s^T] for each sample \sim O(S x D^2)
     */
    /* TODO storing GMatrix */
    @EvaluatesRDD @UpdatesRDD @CachesRDD
    public SubroutineSignal updateBiasLatentPosteriorExpectations() {
        /* update the E_step caches */
        mapWorkers(cb -> cb.cloneWithUpdatedCachesByTag("E_STEP_Z"));
        cacheWorkers("after E-step for bias initialization");

        final ImmutableTriple<INDArray, INDArray, INDArray> latentPosteriorExpectationsData =
                fetchBiasLatentPosteriorExpectationsDataFromWorkers();
        final INDArray contribGMatrix = latentPosteriorExpectationsData.left;
        final INDArray contribZ = latentPosteriorExpectationsData.middle;
        final INDArray sharedPart = latentPosteriorExpectationsData.right;

        /* calculate G_{s\mu\nu} = (sharedPart + W^T M \Psi^{-1} W)^{-1} by doing sample-wise matrix inversion */
        final INDArray sampleGTensor = Nd4j.create(numSamples, numLatents, numLatents);
        sampleIndexStream().forEach(si -> sampleGTensor.get(NDArrayIndex.point(si)).assign(
                CoverageModelEMWorkspaceNDArrayUtils.minv(sharedPart.add(contribGMatrix.get(NDArrayIndex.point(si))))));

        final INDArray newSampleBiasLatentPosteriorFirstMoments = Nd4j.create(numSamples, numLatents);
        final INDArray newSampleBiasLatentPosteriorSecondMoments = Nd4j.create(numSamples, numLatents, numLatents);

        sampleIndexStream().forEach(si -> {
            final INDArray sampleGMatrix = sampleGTensor.get(NDArrayIndex.point(si));

            /* E[z_s] = G_s W^T M_{st} \Psi_{st}^{-1} (m_{st} - m_t) */
            newSampleBiasLatentPosteriorFirstMoments.get(NDArrayIndex.point(si)).assign(sampleGMatrix.mmul(
                    contribZ.get(NDArrayIndex.all(), NDArrayIndex.point(si))).transpose());

            /* E[z_s z_s^T] = G_s + E[z_s] E[z_s^T] */
            newSampleBiasLatentPosteriorSecondMoments.get(NDArrayIndex.point(si)).assign(sampleGMatrix.add(
                    newSampleBiasLatentPosteriorFirstMoments.get(NDArrayIndex.point(si)).transpose()
                            .mmul(newSampleBiasLatentPosteriorFirstMoments.get(NDArrayIndex.point(si)))));
        });

        /* admix */
        final INDArray newSampleBiasLatentPosteriorFirstMomentsAdmixed = newSampleBiasLatentPosteriorFirstMoments
                .mul(params.getMeanFieldAdmixingRatio()).addi(sampleBiasLatentPosteriorFirstMoments
                        .mul(1.0 - params.getMeanFieldAdmixingRatio()));
        final INDArray newSampleBiasLatentPosteriorSecondMomentsAdmixed = newSampleBiasLatentPosteriorSecondMoments
                .mul(params.getMeanFieldAdmixingRatio()).addi(sampleBiasLatentPosteriorSecondMoments
                        .mul(1.0 - params.getMeanFieldAdmixingRatio()));

        /* calculate the error from the change in E[z_s] */
        final double errorNormInfinity = CoverageModelEMWorkspaceNDArrayUtils.getINDArrayNormInfinity(
                newSampleBiasLatentPosteriorFirstMomentsAdmixed.sub(sampleBiasLatentPosteriorFirstMoments));

        /* update the local values */
        sampleBiasLatentPosteriorFirstMoments.assign(newSampleBiasLatentPosteriorFirstMomentsAdmixed);
        sampleBiasLatentPosteriorSecondMoments.assign(newSampleBiasLatentPosteriorSecondMomentsAdmixed);

        /* broadcast the latent posterior expectations */
        pushToWorkers(ImmutablePair.of(newSampleBiasLatentPosteriorFirstMoments, newSampleBiasLatentPosteriorSecondMoments),
                (dat, cb) -> cb.cloneWithUpdatedPrimitive("z_sl", dat.left)
                        .cloneWithUpdatedPrimitive("zz_sll", dat.right));

        return SubroutineSignal.builder().put("error_norm", errorNormInfinity).build();
   }

    /**
     * Fetches the data required for calculating latent posterior expectations from the workers
     * The return result is an {@link ImmutableTriple} of (contribGMatrix, contribZ, sharedPart)
     *
     * For the definition of the first two elements, refer to the javadoc of
     * {@link CoverageModelEMComputeBlockNDArray#getBiasLatentPosteriorDataUnregularized}
     *
     * If the Fourier regularizer is enabled, shared part is [I] + W^T \lambda [F] [W]
     * If the Fourier regularizer is disabled, shared part is [I]
     *
     * @return an {@link ImmutableTriple}
     */
    private ImmutableTriple<INDArray, INDArray, INDArray> fetchBiasLatentPosteriorExpectationsDataFromWorkers() {
        /*
         * ask the workers to calculate latent posterior data (they all involve partial summations in their
         * respective target blocks), and reduce by pairwise in-place addition
         */
        if (params.fourierRegularizationEnabled()) {
            final ImmutableTriple<INDArray, INDArray, INDArray> data =
                    mapWorkersAndReduce(CoverageModelEMComputeBlockNDArray::getBiasLatentPosteriorDataRegularized,
                            (t1, t2) -> ImmutableTriple.of(t1.left.add(t2.left), t1.middle.add(t2.middle),
                                    t1.right.add(t2.right)));
            /* sharedPart = I + W^T [\lambda F W] */
            final INDArray contribFilter = data.right;
            final INDArray sharedPart = Nd4j.eye(numLatents).addi(contribFilter);
            return ImmutableTriple.of(data.left, data.middle, sharedPart);
        } else {
            final ImmutablePair<INDArray, INDArray> data =
                    mapWorkersAndReduce(CoverageModelEMComputeBlockNDArray::getBiasLatentPosteriorDataUnregularized,
                            (p1, p2) -> ImmutablePair.of(p1.left.add(p2.left), p1.right.add(p2.right)));
            /* no filter => just I */
            final INDArray sharedPart = Nd4j.eye(numLatents);
            return ImmutableTriple.of(data.left, data.right, sharedPart);
        }
    }

    /**
     * M-step for read depth ($d_s$)
     *
     * @return a {@link SubroutineSignal} object containing the update size
     */
    @EvaluatesRDD @UpdatesRDD @CachesRDD
    public SubroutineSignal updateReadDepthLatentPosteriorExpectations() {
        mapWorkers(cb -> cb.cloneWithUpdatedCachesByTag("E_STEP_D"));
        cacheWorkers("after E-step for read depth initialization");
        /**
         * map each worker to their read depth estimation factors (a triple of rank-1 INDArray's with
         * S elements, with partial target summations in the respective target block of the worker),
         * reduce by pairwise addition.
         */
        final ImmutablePair<INDArray, INDArray> factors = mapWorkersAndReduce(
                CoverageModelEMComputeBlockNDArray::getSampleReadDepthLatentPosteriorData,
                (p1, p2) -> ImmutablePair.of(p1.left.add(p2.left), p1.right.add(p2.right)));

        /* put together */
        final INDArray numerator = factors.left;
        final INDArray denominator = factors.right;

        final INDArray newSampleMeanLogReadDepths = numerator.div(denominator);
        final INDArray newSampleVarLogReadDepths = Nd4j.ones(denominator.shape()).div(denominator);

        /* admix */
        final INDArray newSampleMeanLogReadDepthsAdmixed = newSampleMeanLogReadDepths
                .mul(params.getMeanFieldAdmixingRatio())
                .addi(sampleMeanLogReadDepths.mul(1.0 - params.getMeanFieldAdmixingRatio()));
        final INDArray newSampleVarLogReadDepthsAdmixed = newSampleVarLogReadDepths
                .mul(params.getMeanFieldAdmixingRatio())
                .addi(sampleVarLogReadDepths.mul(1.0 - params.getMeanFieldAdmixingRatio()));

        /* calculate the error only using the change in the mean */
        final double errorNormInfinity = CoverageModelEMWorkspaceNDArrayUtils.getINDArrayNormInfinity(
                newSampleMeanLogReadDepthsAdmixed.sub(sampleMeanLogReadDepths));

        /* update local copies of E[log(d_s)] and var[log(d_s)] */
        sampleMeanLogReadDepths.assign(newSampleMeanLogReadDepthsAdmixed);
        sampleVarLogReadDepths.assign(newSampleVarLogReadDepthsAdmixed);

        /* push E[log(d_s)] and var[log(d_s)] to all workers; they all need to have a copy */
        pushToWorkers(ImmutablePair.of(newSampleMeanLogReadDepths, newSampleVarLogReadDepths),
                (dat, cb) -> cb.cloneWithUpdatedPrimitive("log_d_s", dat.left)
                        .cloneWithUpdatedPrimitive("var_log_d_s", dat.right));

        return SubroutineSignal.builder().put("error_norm", errorNormInfinity).build();
    }



    @EvaluatesRDD @UpdatesRDD @CachesRDD
    public SubroutineSignal updateSampleUnexplainedVariance() {
        mapWorkers(cb -> cb.cloneWithUpdatedCachesByTag("E_STEP_GAMMA"));
        cacheWorkers("after E-step for sample unexplained variance initialization");

        final double minPsi = Nd4j.min(fetchFromWorkers("Psi_t", 1)).getDouble(0);

        final double[] newGammaArray = IntStream.range(0, numSamples).mapToDouble(si -> {
            final UnivariateFunction objFunc = gamma ->
                    mapWorkersAndReduce(cb -> cb.getTargetSummedGammaPosteriorArgumentSingleSample(si, gamma),
                            (a, b) -> a + b);
            final BrentSolver solver = new BrentSolver(1e-5, 1e-5);
            double newGamma;
            try {
                newGamma = solver.solve(100, objFunc, -minPsi, 0.5);
            } catch (NoBracketingException e) {
                newGamma = sampleUnexplainedVariance.getDouble(si);
            } catch (TooManyEvaluationsException e) {
                throw new RuntimeException("Increase the number of Brent iterations for E-step of gamma.");
            }
            return newGamma;
        }).toArray();

        /* admix */
        final INDArray newSampleUnexplainedVarianceAdmixed = Nd4j.create(newGammaArray, new int[]{numSamples, 1})
                .muli(params.getMeanFieldAdmixingRatio())
                .addi(sampleUnexplainedVariance.mul(1 - params.getMeanFieldAdmixingRatio()));

        /* calculate the error */
        final double errorNormInfinity = CoverageModelEMWorkspaceNDArrayUtils.getINDArrayNormInfinity(
                newSampleUnexplainedVarianceAdmixed.sub(sampleUnexplainedVariance));

        /* update local copy */
        sampleUnexplainedVariance.assign(newSampleUnexplainedVarianceAdmixed);

        /* TODO DEBUG */
        System.out.println(sampleUnexplainedVariance);

        /* push to workers */
        pushToWorkers(newSampleUnexplainedVarianceAdmixed, (arr, cb) -> cb.cloneWithUpdatedPrimitive("gamma_s",
                newSampleUnexplainedVarianceAdmixed));

        return SubroutineSignal.builder().put("error_norm", errorNormInfinity).build();
    }



    @EvaluatesRDD @UpdatesRDD @CachesRDD
    public SubroutineSignal updateCopyRatioLatentPosteriorExpectations() {
        mapWorkers(cb -> cb.cloneWithUpdatedCachesByTag("E_STEP_C"));
        cacheWorkers("after E-step for copy ratio initialization");

        /* calculate posteriors */
        logger.debug("Copy ratio HMM type: " + params.getCopyRatioHMMType().name());
        long startTime = System.nanoTime();
        final SubroutineSignal sig;
        if (params.getCopyRatioHMMType().equals(COPY_RATIO_HMM_LOCAL) || !sparkContextIsAvailable) {
            /* local mode */
            sig = updateCopyRatioLatentPosteriorExpectationsLocal();
        } else {
            /* spark mode */
            sig = updateCopyRatioLatentPosteriorExpectationsSpark();
        }
        long endTime = System.nanoTime();
        logger.debug("Copy ratio posteriors calculation time: " + (double)(endTime - startTime)/1000000 + " ms");
        return sig;
    }

    @Override @EvaluatesRDD @UpdatesRDD @CachesRDD
    public List<CopyRatioHiddenMarkovModelResults<CoverageModelCopyRatioEmissionData, S>> getCopyRatioHiddenMarkovModelResults() {
        mapWorkers(cb -> cb.cloneWithUpdatedCachesByTag("E_STEP_C"));
        cacheWorkers("after E-step for copy ratio HMM result generation");
        final List<CopyRatioHiddenMarkovModelResults<CoverageModelCopyRatioEmissionData, S>> result;
        /* calculate posteriors */
        logger.debug("Copy ratio HMM type: " + params.getCopyRatioHMMType().name());
        long startTime = System.nanoTime();
        if (params.getCopyRatioHMMType().equals(COPY_RATIO_HMM_LOCAL) || !sparkContextIsAvailable) {
            /* local mode */
            result = getCopyRatioHiddenMarkovModelResultsLocal();
        } else {
            /* spark mode */
            result = getCopyRatioHiddenMarkovModelResultsSpark();
        }
        long endTime = System.nanoTime();
        logger.debug("Copy ratio HMM result genereation time: " + (double)(endTime - startTime)/1000000 + " ms");
        return result;
    }

    /******************************************
     * Local copy ratio posterior calculation *
     ******************************************/

    public SubroutineSignal updateCopyRatioLatentPosteriorExpectationsLocal() {

        /* step 1. fetch copy ratio emission data */
        final List<ImmutableTriple<List<Target>, int[], List<CoverageModelCopyRatioEmissionData>>> copyRatioEmissionData =
                fetchCopyRatioEmissionDataLocal();

        /* step 2. run the forward-backward algorithm and calculate copy ratio posteriors */
        final List<CopyRatioPosteriorResults> copyRatioPosteriorResults = copyRatioEmissionData.stream()
                .map(p -> copyRatioPosteriorCalculator.getCopyRatioPosteriorResults(p.left, p.middle, p.right))
                .collect(Collectors.toList());

        /* sent the results back to workers */
        final ImmutablePair<INDArray, INDArray> copyRatioPosteriorDataPair =
                convertCopyRatioLatentPosteriorExpectationsToNDArray(copyRatioPosteriorResults);
        final INDArray log_c_st = copyRatioPosteriorDataPair.left;
        final INDArray var_log_c_st = copyRatioPosteriorDataPair.right;

        final double meanFieldAdmixingRatio = params.getMeanFieldAdmixingRatio();

        /* partition the pair of (log_c_st, var_log_c_st), sent the result to workers via broadcast-hash-map */
        pushToWorkers(mapINDArrayPairToBlocks(log_c_st.transpose(), var_log_c_st.transpose()),
                (p, cb) -> cb.updateCopyRatioLatentPosteriors(
                        p.get(cb.getTargetSpaceBlock()).left.transpose(),
                        p.get(cb.getTargetSpaceBlock()).right.transpose(),
                        meanFieldAdmixingRatio));
        cacheWorkers("after E-step for copy ratio update");

        /* collect subroutine signals */
        final List<SubroutineSignal> sigs =
                mapWorkersAndCollect(CoverageModelEMComputeBlockNDArray::getLatestMStepSignal);

        final double errorNormInfinity = Collections.max(sigs.stream()
                .map(sig -> sig.getDouble("error_norm"))
                .collect(Collectors.toList()));

        return SubroutineSignal.builder()
                .put("error_norm", errorNormInfinity)
                .build();
    }

    /**
     * Interrogates workers about copy ratio emission data and creates an RDD ready for throwing at a HMM
     * @return a list containing the data required for running an HMM
     */
    private List<ImmutableTriple<List<Target>, int[], List<CoverageModelCopyRatioEmissionData>>> fetchCopyRatioEmissionDataLocal() {
        /* fetch data from workers */
        final List<ImmutablePair<LinearSpaceBlock, List<List<CoverageModelCopyRatioEmissionData>>>> collectedCopyRatioData =
                mapWorkersAndCollect(cb -> ImmutablePair.of(cb.getTargetSpaceBlock(), cb.getSampleCopyRatioLatentPosteriorData()));
        /* assemble and return */
        return IntStream.range(0, numSamples).parallel()
                        .mapToObj(si -> assembleTargetCoverageCopyRatioEmissionDataList(
                                collectedCopyRatioData.stream()
                                        .map(p -> ImmutablePair.of(p.getKey(), p.getValue().get(si)))
                                        .collect(Collectors.toList())))
                        .collect(Collectors.toList());
    }

    /**
     * Converts a list of copy ratio posterior expectation results into a triple of (log_c_st, var_log_c_st,
     * viterbi_c_st) to send back to workers
     *
     * Note: on missing targets, log_c_st is set to 0.0 (neutral state), var_log_c_st is set to 0.0 (no variance)
     *
     * @param copyRatioPosteriorResultsList a list of {@link CopyRatioPosteriorResults}
     * @return a triple of (log_c_st, var_log_c_st)
     */
    private ImmutablePair<INDArray, INDArray> convertCopyRatioLatentPosteriorExpectationsToNDArray(
            @Nonnull final List<CopyRatioPosteriorResults> copyRatioPosteriorResultsList) {

        final INDArray log_c_st = Nd4j.create(numSamples, numTargets);
        final INDArray var_log_c_st = Nd4j.create(numSamples, numTargets);

        sampleIndexStream().forEach(si -> {
            final CopyRatioPosteriorResults res = copyRatioPosteriorResultsList.get(si);
            log_c_st.getRow(si).assign(fillMissingTargets(res.getActiveTargetIndices(),
                    res.getLogCopyRatioPosteriorMeansOnActiveTargets(),
                    CopyRatioPosteriorResults.MEAN_LOG_COPY_RATIO_ON_MISSING_TARGETS));
            var_log_c_st.getRow(si).assign(fillMissingTargets(res.getActiveTargetIndices(),
                    res.getLogCopyRatioPosteriorVariancesOnActiveTargets(),
                    CopyRatioPosteriorResults.VAR_LOG_COPY_RATIO_ON_MISSING_TARGETS));
        });
        return ImmutablePair.of(log_c_st, var_log_c_st);
    }

    /**
     * Takes a list of active targets and corresponding values; returns a INDArray row vector
     * with missing targets replaced with {@param missingTargetValuePlaceholder}
     *
     * @param activeTargetIndices indices of active targets
     * @param values values at active targets
     * @param missingTargetValuePlaceholder the placeholder value for missing targets in the returned row vector
     * @return an INDArray row vector
     */
    private INDArray fillMissingTargets(@Nonnull final int[] activeTargetIndices,
                                        @Nonnull final double[] values,
                                        final double missingTargetValuePlaceholder) {
        final double[] result = IntStream.range(0, numTargets)
                .mapToDouble(i -> missingTargetValuePlaceholder).toArray();
        IntStream.range(0, activeTargetIndices.length)
                .forEach(idx -> result[activeTargetIndices[idx]] = values[idx]);
        return Nd4j.create(new int[] {1, numTargets}, result);
    }


    /**
     * Concatenates lists of {@link CoverageModelCopyRatioEmissionData} in the order of increasing target
     * position and returns the pair: (list of active unmasked targets, list of emission data)
     *
     * @param data a list of ({@link LinearSpaceBlock}, {@link List<CoverageModelCopyRatioEmissionData>}) pairs
     * @return a pair of (list of active unmasked targets, list of emission data)
     */
    private ImmutableTriple<List<Target>, int[], List<CoverageModelCopyRatioEmissionData>> assembleTargetCoverageCopyRatioEmissionDataList(
            @Nonnull final Collection<? extends Pair<LinearSpaceBlock, List<CoverageModelCopyRatioEmissionData>>> data) {

        /* sort by block index, combine, and collect */
        final List<CoverageModelCopyRatioEmissionData> assembledEmissionDataList = data.stream()
                .sorted((Lp, Rp) -> Lp.getKey().getBegIndex() - Rp.getKey().getBegIndex())
                .map(Pair<LinearSpaceBlock, List<CoverageModelCopyRatioEmissionData>>::getValue)
                .flatMap(List::stream)
                .collect(Collectors.toList());

        /* build the indices of targets on which we have emission data ("active targets") */
        final int[] activeTargetIndices = IntStream.range(0, numTargets)
                .filter(ti -> assembledEmissionDataList.get(ti) != null)
                .toArray();

        /* build the list of targets on which we have emission data ("active targets") */
        final List<Target> activeTargetList = Arrays.stream(activeTargetIndices)
                .mapToObj(processedTargetList::get)
                .collect(Collectors.toList());


        /* filter out null emission data elements (by construction, null elements correspond to masked targets) */
        final List<CoverageModelCopyRatioEmissionData> nonNullEmissionDataList = assembledEmissionDataList.stream()
                .filter(dat -> dat != null)
                .collect(Collectors.toList());

        return ImmutableTriple.of(activeTargetList, activeTargetIndices, nonNullEmissionDataList);
    }

    /**
     *
     * @return
     */
    private List<CopyRatioHiddenMarkovModelResults<CoverageModelCopyRatioEmissionData, S>> getCopyRatioHiddenMarkovModelResultsLocal() {
        final List<ImmutableTriple<List<Target>, int[], List<CoverageModelCopyRatioEmissionData>>> copyRatioEmissionData =
                fetchCopyRatioEmissionDataLocal();
        return copyRatioEmissionData.stream()
                .map(p -> copyRatioPosteriorCalculator.getCopyRatioHiddenMarkovModelResults(p.left, p.right))
                .collect(Collectors.toList());
    }

    /******************************************
     * Spark copy ratio posterior calculation *
     ******************************************/

    /**
     * Wall of functional code ahead, not for the faint hearted! ;)
     *
     * @return
     */
    @EvaluatesRDD @UpdatesRDD @CachesRDD
    private SubroutineSignal updateCopyRatioLatentPosteriorExpectationsSpark() {

        /* local final member variables for lambda capture */
        final List<LinearSpaceBlock> targetBlocks = new ArrayList<>();
        targetBlocks.addAll(this.targetBlocks);
        final int numTargetBlocks = targetBlocks.size();
        final CopyRatioPosteriorCalculator<CoverageModelCopyRatioEmissionData, S> calculator =
                this.copyRatioPosteriorCalculator;

        /* step 1. make an RDD of copy ratio results */
        final JavaPairRDD<Integer, CopyRatioPosteriorResults> copyRatioPosteriorResultsPairRDD =
                fetchCopyRatioEmissionDataSpark().mapValues(p -> /* run the HMM on workers */
                        calculator.getCopyRatioPosteriorResults(p.left, p.middle, p.right));

        /* step 2. blockify in target space and repartition */
        final JavaPairRDD<LinearSpaceBlock, ImmutablePair<INDArray, INDArray>>
                blockifiedCopyRatioPosteriorResultsPairRDD = copyRatioPosteriorResultsPairRDD
                .flatMapToPair(tuple -> targetBlocks.stream() /* flat map posterior results to values on target blocks + sample index */
                        .map(tb -> new Tuple2<>(tb, new Tuple2<>(tuple._1, ImmutablePair.of(
                                tuple._2.getLogCopyRatioPosteriorMeansOnTargetBlock(tb),
                                tuple._2.getLogCopyRatioPosteriorVariancesOnTargetBlock(tb)))))
                        .collect(Collectors.toList())
                ).combineByKey( /* combine the 1-to-many map from target blocks to pieces to a 1-to-1 map and repartition */
                        Collections::singletonList,
                        (list, element) -> Stream.concat(list.stream(), Collections.singletonList(element).stream())
                                .collect(Collectors.toList()), /* add an element to the list */
                        (list1, list2) -> Stream.concat(list1.stream(), list2.stream()).collect(Collectors.toList()), /* concatenate two lists */
                        new HashPartitioner(numTargetBlocks) /* repartition with respect to sample indices */
                ).mapValues(list -> list.stream()
                        .sorted((Lp, Rp) -> Lp._1 - Rp._1) /* sort by sample index */
                        .map(p -> p._2) /* remove sample label */
                        .map(t -> ImmutablePair.of(Nd4j.create(t.left), Nd4j.create(t.right))) /* convert double[] to INDArray */
                        .collect(Collectors.toList())
                ).mapValues(CoverageModelEMWorkspaceNDArraySparkToggle::stackCopyRatioPosteriorDataForAllSamples);

        /* step 3. merge with computeRDD and update */
        final double admixingRatio = params.getMeanFieldAdmixingRatio();
        computeRDD = computeRDD.join(blockifiedCopyRatioPosteriorResultsPairRDD)
                .mapValues(t -> t._1.updateCopyRatioLatentPosteriors(t._2.left, t._2.right, admixingRatio));
        cacheWorkers("after E-step for copy ratio update");

        /* collect subroutine signals */
        final List<SubroutineSignal> sigs =
                mapWorkersAndCollect(CoverageModelEMComputeBlockNDArray::getLatestMStepSignal);

        final double errorNormInfinity = Collections.max(sigs.stream()
                .map(sig -> sig.getDouble("error_norm"))
                .collect(Collectors.toList()));

        return SubroutineSignal.builder()
                .put("error_norm", errorNormInfinity)
                .build();
    }

    /**
     * Interrogates workers about copy ratio emission data and creates an RDD ready for throwing at a HMM
     * @return an RDD containing the data required for running an HMM
     */
    private JavaPairRDD<Integer, ImmutableTriple<List<Target>, int[],
            List<CoverageModelCopyRatioEmissionData>>> fetchCopyRatioEmissionDataSpark() {

        final int numSamples = this.numSamples;
        final int numTargets = this.numTargets;
        final List<Target> processedTargetList = new ArrayList<>();
        processedTargetList.addAll(this.processedTargetList);

        return computeRDD.flatMapToPair(tuple -> {
            final LinearSpaceBlock targetBlock = tuple._1;
            final CoverageModelEMComputeBlockNDArray computeBlock = tuple._2;
            final List<List<CoverageModelCopyRatioEmissionData>> emissionDataBlock =
                    computeBlock.getSampleCopyRatioLatentPosteriorData();
            return IntStream.range(0, numSamples)
                    .mapToObj(sampleIndex -> new Tuple2<>(sampleIndex,
                            new Tuple2<>(targetBlock, emissionDataBlock.get(sampleIndex))))
                    .collect(Collectors.toList());
        }).combineByKey( /* combine the 1-to-many map to a 1-to-1 map and repartition */
                Collections::singletonList, /* create a new list */
                (list, element) -> Stream.concat(list.stream(), Collections.singletonList(element).stream())
                        .collect(Collectors.toList()), /* add an element to the list */
                (list1, list2) -> Stream.concat(list1.stream(), list2.stream()).collect(Collectors.toList()), /* concatenate two lists */
                new HashPartitioner(numSamples) /* repartition with respect to sample indices */
        ).mapValues(list -> list.stream() /* for each partition ... */
                .sorted((Lp, Rp) -> Lp._1.getBegIndex() - Rp._1.getBegIndex()) /* sort the data blocks */
                .map(p -> p._2) /* remove the LinearSpaceBlock keys from the sorted list */
                .flatMap(List::stream) /* flatten */
                .collect(Collectors.toList()) /* collect as a single list of emission data entries for all targets */
        ).mapValues(emissionDataList -> ImmutableTriple.of( /* make a triple of ... */
                IntStream.range(0, numTargets) /* (1) list of active targets */
                        .filter(targetIndex -> emissionDataList.get(targetIndex) != null)
                        .mapToObj(processedTargetList::get)
                        .collect(Collectors.toList()),
                IntStream.range(0, numTargets) /* (2) indices of active targets in the full target list */
                        .filter(targetIndex -> emissionDataList.get(targetIndex) != null)
                        .toArray(),
                emissionDataList.stream() /* (3) list of non-null emission data entries */
                        .filter(emissionDataEntry -> emissionDataEntry != null)
                        .collect(Collectors.toList())));
    }

    /**
     * stack all samples to make S x T matrices
     * @param perSampleData
     * @return
     */
    private static ImmutablePair<INDArray, INDArray> stackCopyRatioPosteriorDataForAllSamples(
            final List<ImmutablePair<INDArray, INDArray>> perSampleData) {
        return ImmutablePair.of(
                Nd4j.vstack((Collection<INDArray>)perSampleData.stream().map(p -> p.left).collect(Collectors.toList())),
                Nd4j.vstack((Collection<INDArray>)perSampleData.stream().map(p -> p.right).collect(Collectors.toList())));
    }

    /**
     *
     * @return
     */
    private List<CopyRatioHiddenMarkovModelResults<CoverageModelCopyRatioEmissionData, S>> getCopyRatioHiddenMarkovModelResultsSpark() {
        final CopyRatioPosteriorCalculator<CoverageModelCopyRatioEmissionData, S> calculator =
                this.copyRatioPosteriorCalculator;
        return fetchCopyRatioEmissionDataSpark().mapValues(t -> /* run the HMM on workers */
                        calculator.getCopyRatioHiddenMarkovModelResults(t.left, t.right))
                .collect().stream()
                .sorted((Lp, Rp) -> Lp._1 - Rp._1)
                .map(t -> t._2)
                .collect(Collectors.toList());
    }

    /**********
     * M-step *
     **********/

    /**
     * M-step for target mean bias ($m_t$)
     *
     * @return a {@link SubroutineSignal} object containing the update size
     */
    @UpdatesRDD @CachesRDD
    public SubroutineSignal updateTargetMeanBias(final boolean neglectPCBias) {
        mapWorkers(cb -> cb.cloneWithUpdatedCachesByTag("M_STEP_M").updateTargetMeanBias(neglectPCBias));
        cacheWorkers("after M-step for target mean bias");
        /* accumulate error from all nodes */
        final double errorNormInfinity = Collections.max(
                mapWorkersAndCollect(CoverageModelEMComputeBlockNDArray::getLatestMStepSignal)
                        .stream().map(sig -> sig.getDouble("error_norm")).collect(Collectors.toList()));

        return SubroutineSignal.builder().put("error_norm", errorNormInfinity).build();
    }

    /**
     * M-step for target unexplained variance ($\Psi_t$)
     *
     * @return a {@link SubroutineSignal} object containing information about the solution
     */
    @UpdatesRDD @CachesRDD
    public SubroutineSignal updateTargetUnexplainedVariance() {
        final int maxIters = params.getPsiMaxIterations();
        final double absTol = params.getPsiAbsoluteTolerance();
        final double relTol = params.getPsiRelativeTolerance();
        logger.debug("Psi solver type: " + params.getPsiSolverType().name());
        switch (params.getPsiSolverType()) {
//            case PSI_TARGET_RESOLVED_VIA_NEWTON: /* done on the compute blocks */
//                mapWorkers(cb -> cb.cloneWithUpdatedCachesByTag("M_STEP_PSI")
//                        .updateTargetUnexplainedVarianceTargetResolvedNewton(maxIters, absTol));
//                break;

            case PSI_TARGET_RESOLVED_VIA_BRENT: /* done on the compute blocks */
                mapWorkers(cb -> cb.cloneWithUpdatedCachesByTag("M_STEP_PSI")
                        .updateTargetUnexplainedVarianceTargetResolvedBrent(maxIters, absTol, relTol));
                break;

            case PSI_ISOTROPIC_VIA_BRENT: /* done on the driver node */
                return updateTargetUnexplainedVarianceIsotropicBrent();

            default:
                throw new RuntimeException("Illegal Psi solver type.");
        }

        cacheWorkers("after M-step for target unexplained variance");

        /* accumulate error from all workers */
        final List<SubroutineSignal> signalList = mapWorkersAndCollect(CoverageModelEMComputeBlockNDArray::getLatestMStepSignal);
        final double errorNormInfinity = Collections.max(signalList.stream().map(sig -> sig.getDouble("error_norm"))
                .collect(Collectors.toList()));
        final int maxIterations = Collections.max(signalList.stream().map(sig -> sig.getInteger("iterations"))
                .collect(Collectors.toList()));
        final int minIterations = Collections.min(signalList.stream().map(sig -> sig.getInteger("iterations"))
                .collect(Collectors.toList()));
        return SubroutineSignal.builder()
                .put("error_norm", errorNormInfinity)
                .put("min_iterations", minIterations)
                .put("max_iterations", maxIterations)
                .put("iterations", maxIterations) /* for uniformity */
                .build();
    }

    /**
     * M-step for isotropic unexplained variance using Brent method
     * @return a {@link SubroutineSignal} object containing information about the solution
     */
    @UpdatesRDD @CachesRDD
    private SubroutineSignal updateTargetUnexplainedVarianceIsotropicBrent() {
        mapWorkers(cb -> cb.cloneWithUpdatedCachesByTag("M_STEP_PSI"));
        cacheWorkers("after M-step for isotropic unexplained variance initialization");

        final double oldIsotropicPsi = fetchFromWorkers("Psi_t", 1).meanNumber().doubleValue();
        final double psiLowerBound = - Nd4j.min(sampleUnexplainedVariance).getDouble(0);
        final UnivariateFunction objFunc = psi -> mapWorkersAndReduce(cb -> cb.calculateTargetSummedPsiGradient(psi), (a, b) -> a + b);

        final BrentSolver solver = new BrentSolver(params.getPsiRelativeTolerance(), params.getPsiAbsoluteTolerance());
        double newIsotropicPsi;
        try {
            newIsotropicPsi = solver.solve(params.getPsiMaxIterations(), objFunc,
                    psiLowerBound, CoverageModelEMParams.PSI_BRENT_UPPER_LIMIT,
                    FastMath.max(psiLowerBound + CoverageModelEMParams.PSI_BRENT_MIN_STARTING_POINT,
                            FastMath.min(oldIsotropicPsi, 0.5 * CoverageModelEMParams.PSI_BRENT_UPPER_LIMIT)));
        } catch (NoBracketingException e) {
            logger.warn("Root of M-step for Psi stationarity equation could be bracketed.");
            newIsotropicPsi = 0.0;
        } catch (TooManyEvaluationsException e) {
            throw new RuntimeException("Increase the number of Brent iterations for M-step of Psi.");
        }

        /* update the compute block(s) */
        final double errNormInfinity = FastMath.abs(newIsotropicPsi - oldIsotropicPsi);
        final int maxIterations = solver.getEvaluations();
        final double finalizedNewIsotropicPsi = newIsotropicPsi;
        mapWorkers(cb -> cb.cloneWithUpdatedPrimitive("Psi_t",
                Nd4j.ones(1, cb.getTargetSpaceBlock().getNumTargets()).muli(finalizedNewIsotropicPsi)));
        return SubroutineSignal.builder()
                .put("error_norm", errNormInfinity)
                .put("iterations", maxIterations).build();
    }

    /**
     * M-step for principal latent to target map
     *
     * @return a {@link SubroutineSignal} object containing information about the solution
     */
    @UpdatesRDD @CachesRDD
    public SubroutineSignal updatePrincipalLatentToTargetMap() {
        /* perform the M-step update */
        final SubroutineSignal sig;
        if (!params.fourierRegularizationEnabled()) {
            sig = updatePrincipalLatentToTargetMapUnregularized();
        } else {
            sig = updatePrincipalLatentToTargetMapRegularized();
        }
        return sig;
    }

    /**
     * M-step for principal latent target map (unregularized)
     *
     * @return a {@link SubroutineSignal} object containing information about the solution
     */
    @UpdatesRDD @CachesRDD
    private SubroutineSignal updatePrincipalLatentToTargetMapUnregularized() {
        mapWorkers(cb -> cb.cloneWithUpdatedCachesByTag("M_STEP_W_UNREG")
                .updatePrincipalLatentTargetMapUnregularized());
        cacheWorkers("after M-step for principal latent target map (unregularized)");

        /* perform orthogonalization if required */
        if (params.isOrthogonalizeAndSortPrincipalMapEnabled()) {
            orthogonalizeAndSortPrincipalMap();
            cacheWorkers("after orthogonalization of principal latent target map");
        }

        /* accumulate error from all nodes */
        final double errorNormInfinity = Collections.max(
                mapWorkersAndCollect(CoverageModelEMComputeBlockNDArray::getLatestMStepSignal)
                        .stream().map(sig -> sig.getDouble("error_norm")).collect(Collectors.toList()));
        return SubroutineSignal.builder().put("error_norm", errorNormInfinity).build();
    }

    /**
     * M-step for principal latent target map (regularized, computations on the drive node)
     *
     * @return a {@link SubroutineSignal} object containing information about the solution
     */
    @UpdatesRDD @EvaluatesRDD @CachesRDD
    private SubroutineSignal updatePrincipalLatentToTargetMapRegularized() {
        mapWorkers(cb -> cb.cloneWithUpdatedCachesByTag("M_STEP_W_REG"));
        cacheWorkers("after M-step for principal latent target map initialization (regularized)");

        final INDArray W_tl_old = fetchFromWorkers("W_tl", 0);
        final INDArray v_tl = fetchFromWorkers("v_tl", 0);

        /* initialize the linear operators */
        final GeneralLinearOperator<INDArray> linop, precond;
        final ImmutablePair<GeneralLinearOperator<INDArray>, GeneralLinearOperator<INDArray>> ops =
                getPrincipalLatentToTargetMapRegularizedLinearOperators();
        linop = ops.left;
        precond = ops.right;

        /* initialize the iterative solver */
        final IterativeLinearSolverNDArray iterSolver = new IterativeLinearSolverNDArray(linop, v_tl, precond,
                params.getWAbsoluteTolerance(), params.getWRelativeTolerance(), params.getWMaxIterations(),
                x -> x.normmaxNumber().doubleValue(), /* norm */
                (x, y) -> x.mul(y).sumNumber().doubleValue(), /* inner product */
                true);

        /* solve */
        long startTime = System.nanoTime();
        final SubroutineSignal sig = iterSolver.cg(W_tl_old);
        linop.cleanupAfter();
        precond.cleanupAfter();
        long endTime = System.nanoTime();
        logger.debug("CG execuation time for solving regularized M-step(W): " +
                (double)(endTime - startTime)/1000000 + " ms");

        /* check the exit status of the solver and push the new W to workers */
        final ExitStatus exitStatus = (ExitStatus)sig.getObject("status");
        if (exitStatus == ExitStatus.FAIL_MAX_ITERS) {
            logger.info("CG iterations for M-step(W) did not converge. Increase maximum iterations" +
                    " and/or decrease absolute/relative error tolerances. Continuing...");
        }
        final int iters = sig.getInteger("iterations");
        final INDArray W_tl_new = sig.getINDArray("x");

        switch (params.getPrincipalMapCommunicationPolicy()) {
            case BROADCAST_HASH_JOIN:
                pushToWorkers(mapINDArrayToBlocks(W_tl_new), (W, cb) ->
                        cb.cloneWithUpdatedPrimitive("W_tl", W.get(cb.getTargetSpaceBlock())));
                break;

            case RDD_JOIN:
                joinWithWorkersAndMap(chopINDArrayToBlocks(W_tl_new),
                        p -> p._1.cloneWithUpdatedPrimitive("W_tl", p._2));
                break;
        }

        /* update F[W] */
        updateFilteredPrincipalLatentToTargetMap(W_tl_new);

        /* perform orthogonalization if required */
        final INDArray rot_W_tl_new;
        if (params.isOrthogonalizeAndSortPrincipalMapEnabled()) {
            orthogonalizeAndSortPrincipalMap(W_tl_new.transpose().mmul(W_tl_new));
            cacheWorkers("after orthogonalization of principal latent target map");
            rot_W_tl_new = fetchFromWorkers("W_tl", 0);
        } else {
            rot_W_tl_new = W_tl_new;
        }

        final double errorNormInfinity = CoverageModelEMWorkspaceNDArrayUtils.getINDArrayNormInfinity(
                rot_W_tl_new.sub(W_tl_old));

        /* send the signal to workers for consistency */
        final SubroutineSignal newSig = SubroutineSignal.builder().put("error_norm", errorNormInfinity).put("iterations", iters).build();
        mapWorkers(cb -> cb.cloneWithUpdatedSignal(newSig));
        return newSig;
    }

    /**
     *
     * @return
     */
    @EvaluatesRDD
    private ImmutablePair<GeneralLinearOperator<INDArray>, GeneralLinearOperator<INDArray>> getPrincipalLatentToTargetMapRegularizedLinearOperators() {
        final INDArray Q_ll, Q_tll, Z_ll;
        final GeneralLinearOperator<INDArray> linop, precond;
        final FourierLinearOperatorNDArray regularizerFourierLinearOperator = createRegularizerFourierLinearOperator();

        switch (params.getWSolverType()) {
            case W_SOLVER_LOCAL:

                /* fetch the required INDArrays */
                Q_ll = mapWorkersAndReduce(cb -> cb.getINDArrayFromCache("sum_Q_ll"), INDArray::add).div(numTargets);
                Q_tll = fetchFromWorkers("Q_tll", 0);
                Z_ll = sampleBiasLatentPosteriorSecondMoments.sum(0);

                /* instantiate the local implementation of linear operators */
                linop = new CoverageModelWLinearOperatorNDArrayLocal(Q_tll, Z_ll, regularizerFourierLinearOperator);
                precond = new CoverageModelWPreconditionerNDArrayLocal(Q_ll, Z_ll, regularizerFourierLinearOperator, numTargets);

                return ImmutablePair.of(linop, precond);

            case W_SOLVER_SPARK:

                if (!sparkContextIsAvailable) {
                    throw new UserException("The Spark W solver is only available in the Spark mode");
                }
                /* fetch the required INDArrays */
                Q_ll = mapWorkersAndReduce(cb -> cb.getINDArrayFromCache("sum_Q_ll"), INDArray::add).div(numTargets);
                Z_ll = sampleBiasLatentPosteriorSecondMoments.sum(0);

                /* instantiate the spark implementation of linear operators */
                linop = new CoverageModelWLinearOperatorNDArraySpark(Z_ll, regularizerFourierLinearOperator,
                        numTargets, ctx, computeRDD, targetBlocks);
                precond = new CoverageModelWPreconditionerNDArraySpark(Q_ll, Z_ll, regularizerFourierLinearOperator,
                        numTargets, ctx, numTargetBlocks);

                return ImmutablePair.of(linop, precond);

            default:

                throw new UserException("W solver type is not properly set");
        }
    }

    /**
     * Orthogonalize the principal map and sort the diagonal covariance entries in descending order by performing
     * a rotation in the latent space. This transformation affects:
     *
     *   - worker node copies of E[z_s] and E[z_s z_s^T], and their blocks of W and F[W]
     *   - driver node copies of E[z_s] and E[z_s z_s^T]
     *
     * @param WTW [W]^T [W]
     */
    private void orthogonalizeAndSortPrincipalMap(@Nonnull final INDArray WTW) {
        final INDArray U = CoverageModelEMWorkspaceNDArrayUtils.getOrthogonalizerAndSorterTransformation(WTW, true, logger);

        /* update workers */
        pushToWorkers(U, (rot, cb) -> cb.cloneWithRotatedLatentSpace(rot));

        /* update driver node */
        IntStream.range(0, numSamples).parallel().forEach(si -> {
            sampleBiasLatentPosteriorFirstMoments.get(NDArrayIndex.point(si)).assign(
                    U.mmul(sampleBiasLatentPosteriorFirstMoments.get(NDArrayIndex.point(si)).transpose()).transpose());
            sampleBiasLatentPosteriorSecondMoments.get(NDArrayIndex.point(si)).assign(
                    U.mmul(sampleBiasLatentPosteriorSecondMoments.get(NDArrayIndex.point(si))).mmul(U.transpose()));
        });
    }

    /**
     * Same as {@link #orthogonalizeAndSortPrincipalMap(INDArray)} but fetches [W]^T [W] from the workers
     */
    private void orthogonalizeAndSortPrincipalMap() {
        final INDArray WTW = mapWorkersAndReduce(CoverageModelEMComputeBlockNDArray::getPrincipalLatentTargetMapInnerProduct,
                INDArray::add);
        orthogonalizeAndSortPrincipalMap(WTW);
    }

    /**
     * Fetch the log likelihood from compute block(s)
     * @return log likelihood normalized per sample per target
     */
    @Override @EvaluatesRDD
    public double getLogLikelihood() {
        return Arrays.stream(getLogLikelihoodPerSample()).reduce((a, b) -> a + b).orElse(Double.NaN) / numSamples;
    }

    /**
     * Fetch the log likelihood from compute block(s)
     * @return log likelihood normalized per sample per target
     */
    @Override @EvaluatesRDD @CachesRDD
    public double[] getLogLikelihoodPerSample() {
        updateLogLikelihoodCaches();

        final INDArray biasPriorContrib_s = sampleBiasLatentPosteriorFirstMoments
                .mul(sampleBiasLatentPosteriorFirstMoments).muli(-0.5).sum(1);

        final String key;
        if (params.fourierRegularizationEnabled()) {
            key = "loglike_reg";
        } else {
            key = "loglike_unreg";
        }
        final INDArray restContrib_s = mapWorkersAndReduce(cb -> cb.getINDArrayFromCache(key), INDArray::add);
        final INDArray sum_M_s = mapWorkersAndReduce(cb -> cb.getINDArrayFromCache("sum_M_s"), INDArray::add);
        return restContrib_s.addi(biasPriorContrib_s).divi(sum_M_s).data().asDouble();
    }

    /**
     * Updates log likelihood caches
     */
    @UpdatesRDD @CachesRDD
    public void updateLogLikelihoodCaches() {
        if (params.fourierRegularizationEnabled()) {
            mapWorkers(cb -> cb.cloneWithUpdatedCachesByTag("LOGLIKE_REG"));
        } else {
            mapWorkers(cb -> cb.cloneWithUpdatedCachesByTag("LOGLIKE_UNREG"));
        }
        cacheWorkers("after updating loglikelihood caches");
    }

    /******************************
     * RDD-related helper methods *
     ******************************/

    /**
     * Instatiate compute block(s). If Spark is disabled, a single {@link CoverageModelEMComputeBlockNDArray} is
     * instantiated. Otherwise, a {@link JavaPairRDD} of compute nodes will be created.
     */
    private void instantiateWorkers() {
        if (sparkContextIsAvailable) {
            /* initialize the RDD */
            logger.info("Intializing the RDD of compute blocks");
            computeRDD = ctx.parallelizePairs(targetBlockStream()
                    .map(tb -> new Tuple2<>(tb, new CoverageModelEMComputeBlockNDArray(tb, numSamples, numLatents)))
                    .collect(Collectors.toList()), numTargetBlocks)
                    .partitionBy(new HashPartitioner(numTargetBlocks))
                    .cache();
        } else {
            logger.info("Intializing the local compute block");
            localComputeBlock = new CoverageModelEMComputeBlockNDArray(targetBlocks.get(0), numSamples, numLatents);
        }
        prevCheckpointedComputeRDD = null;
        cacheCallCounter = 0;
    }

    /**
     * A generic function for handling a distributed list of objects to their corresponding compute nodes
     *
     * If Spark is enabled:
     *
     *      Joins an instance of {@code List<Tuple2<LinearSpaceBlock, V>>} with {@link #computeRDD}, calls the provided
     *      map {@code mapper} on the RDD, and the reference to the old RDD will be replaced with the new RDD.
     *
     * If Spark is disabled:
     *
     *      Only a single target space block is assumed, such that {@code secondary} is a singleton. The map function
     *      {@code mapper} will be called on the value contained in {@code seconday} and {@link #localComputeBlock}, and
     *      the old instace of {@link CoverageModelEMComputeBlockNDArray} is replaced with the new instance returned
     *      by {@code mapper.}
     *
     * @param data the list to joined and mapped together with the compute block(s)
     * @param mapper a mapper binary function that takes a compute block together with an object of type {@code V} and
     *               returns a new compute block
     * @param <V> the type of the object to the broadcasted
     */
    @UpdatesRDD
    private <V> void joinWithWorkersAndMap(@Nonnull final List<Tuple2<LinearSpaceBlock, V>> data,
                                           @Nonnull final Function<Tuple2<CoverageModelEMComputeBlockNDArray, V>, CoverageModelEMComputeBlockNDArray> mapper) {
        if (sparkContextIsAvailable) {
            final JavaPairRDD<LinearSpaceBlock, V> newRDD =
                    ctx.parallelizePairs(data, numTargetBlocks).partitionBy(new HashPartitioner(numTargetBlocks));
            computeRDD = computeRDD.join(newRDD).mapValues(mapper);
        } else {
            try {
                localComputeBlock = mapper.call(new Tuple2<>(localComputeBlock, data.get(0)._2));
            } catch (Exception e) {
                throw new RuntimeException("Can not apply the map function to the local compute block: " + e.getMessage());
            }
        }
    }

    /**
     * Calls a map function on the compute block(s)
     *
     * If Spark is enabled:
     *
     *      The map is applied on the values of {@link #computeRDD}, and the reference to the old RDD will be replaced
     *      by the new RDD
     *
     * If Spark is disabled:
     *
     *      Only a single target space block is assumed; the map is applied to {@link #localComputeBlock} and the
     *      reference is updated accordingly
     *
     * @param mapper a map from {@link CoverageModelEMComputeBlockNDArray} onto itself
     */
    @UpdatesRDD
    private void mapWorkers(@Nonnull final Function<CoverageModelEMComputeBlockNDArray, CoverageModelEMComputeBlockNDArray> mapper) {
        if (sparkContextIsAvailable) {
            computeRDD = computeRDD.mapValues(mapper);
        } else {
            try {
                localComputeBlock = mapper.call(localComputeBlock);
            } catch (final Exception ex) {
                ex.printStackTrace();
                throw new RuntimeException("Can not apply the map function to the local compute block: " + ex.getMessage());
            }
        }
    }

    /**
     * Calls a map function on the compute block(s) and collects the values to a {@link List}
     *
     * If Spark is enabled:
     *
     *      The size of the list is the same as the number of elements in the RDD
     *
     * If Spark is disabled:
     *
     *      The list will be singleton
     *
     * @param mapper a map function from {@link CoverageModelEMComputeBlockNDArray} to a generic type
     * @param <V> the return type of the map function
     * @return a list of collected mapped values
     */
    @EvaluatesRDD
    private <V> List<V> mapWorkersAndCollect(@Nonnull final Function<CoverageModelEMComputeBlockNDArray, V> mapper) {
        if (sparkContextIsAvailable) {
            return computeRDD.values().map(mapper).collect();
        } else {
            try {
                return Collections.singletonList(mapper.call(localComputeBlock));
            } catch (final Exception ex) {
                ex.printStackTrace();
                throw new RuntimeException("Can not apply the map function to the local compute block: " + ex.getMessage());
            }
        }
    }

    /**
     * A generic map and reduce step on the compute block(s)
     *
     * If Spark is enabled:
     *
     *      Map the RDD by {@code mapper} and reduce by {@code reducer}
     *
     * If Spark is disabled:
     *
     *      Call on the map and reduce on {@link #localComputeBlock}
     *
     * @param mapper a map from {@link CoverageModelEMComputeBlockNDArray} to a generic type
     * @param reducer a generic symmetric reducer binary function from (V, V) -> V
     * @param <V> the type of reduced value
     * @return the result of map-reduce
     */
    @EvaluatesRDD
    private <V> V mapWorkersAndReduce(@Nonnull final Function<CoverageModelEMComputeBlockNDArray, V> mapper,
                                      @Nonnull final Function2<V, V, V> reducer) {
        if (sparkContextIsAvailable) {
            return computeRDD.values().map(mapper).reduce(reducer);
        } else {
            try {
                return mapper.call(localComputeBlock);
            } catch (final Exception ex) {
                ex.printStackTrace();
                throw new RuntimeException("Can not apply the map function to the local compute block: " + ex.getMessage());
            }

        }
    }

    /**
     * A generic function for broadcasting an object to all compute blocks
     *
     * If Spark is enabled:
     *
     *      A {@link Broadcast} will be created from {@param obj} and will be "received" by the compute nodes by calling
     *      {@param pusher}. A reference to the updated RDD will replace the old RDD.
     *
     * If Spark is disabled:
     *
     *      The {@param pusher} function will be called together with {@param obj} and {@link #localComputeBlock}
     *
     * @param obj te object to broadcast
     * @param pusher a map from (V, CoverageModelEMComputeBlockNDArray) -> CoverageModelEMComputeBlockNDArray that
     *               updates the compute block with the broadcasted value
     * @param <V> the type of the broadcasted object
     */
    @UpdatesRDD
    private <V> void pushToWorkers(@Nonnull final V obj,
                                   @Nonnull final Function2<V, CoverageModelEMComputeBlockNDArray, CoverageModelEMComputeBlockNDArray> pusher) {
        if (sparkContextIsAvailable) {
            final Broadcast<V> broadcastedObj = ctx.broadcast(obj);
            final Function<CoverageModelEMComputeBlockNDArray, CoverageModelEMComputeBlockNDArray> mapper =
                    cb -> pusher.call(broadcastedObj.value(), cb);
            mapWorkers(mapper);
        } else {
            try {
                localComputeBlock = pusher.call(obj, localComputeBlock);
            } catch (final Exception ex) {
                ex.printStackTrace();
                throw new RuntimeException("Can not apply the map function to the local compute block: " + ex.getMessage());
            }
        }
    }

    /**
     * If Spark is enabled, caches the RDD of compute blocks. Otherwise, it does nothing.
     *
     * @param where a message provided by the method that calls this function
     */
    @CachesRDD
    public void cacheWorkers(final String where) {
        if (sparkContextIsAvailable) {
            logger.debug("RDD caching requested (" + where + ")");
            computeRDD.persist(StorageLevel.MEMORY_ONLY_SER());
            cacheCallCounter++;
            if (!prevCachedComputeRDDDeque.isEmpty()) {
                prevCachedComputeRDDDeque.removeFirst().unpersist(true);
                prevCachedComputeRDDDeque.addLast(computeRDD);
            }
            if (params.checkpointingEnabled()) {
                if (cacheCallCounter == params.getCheckpointingInterval()) {
                    logger.debug("Checkpointing compute RDD...");
                    computeRDD.checkpoint();
                    if (prevCheckpointedComputeRDD != null) {
                        prevCheckpointedComputeRDD.unpersist(true);
                        prevCheckpointedComputeRDD = computeRDD;
                    }
                    cacheCallCounter = 0;
                }
            }
        }
    }

    /**
     * If Spark is enabled, fetches the blocks of a target-distributed {@link INDArray} of shape
     * ({@link #numTargets}, ...) and assembles them together by concatenating along {@param axis}.
     *
     * If Spark is disabled, just fetches the {@link INDArray} from {@link #localComputeBlock}.
     *
     * @param key key (name) of the array
     * @param axis axis to stack along
     * @return assembled array
     */
    @EvaluatesRDD @VisibleForTesting
    public INDArray fetchFromWorkers(final String key, final int axis) {
        if (sparkContextIsAvailable) {
            return CoverageModelSparkUtils.assembleINDArrayBlocksFromRDD(computeRDD.mapValues(cb -> cb.getINDArrayFromCache(key)), axis);
        } else {
            return localComputeBlock.getINDArrayFromCache(key);
        }
    }

    /**
     * Partition an {@link INDArray} on its first dimension and make a key-value {@link List} of the blocks
     * @param arr the input array
     * @return list of key-value blocks
     */
    private List<Tuple2<LinearSpaceBlock, INDArray>> chopINDArrayToBlocks(final INDArray arr) {
        if (sparkContextIsAvailable) {
            return CoverageModelSparkUtils.chopINDArrayToBlocks(targetBlocks, arr);
        } else {
            return Collections.singletonList(new Tuple2<>(targetBlocks.get(0), arr));
        }
    }

    /**
     * Partition an {@link INDArray} on its first dimension and make a key-value {@link List} of the blocks
     * @param arr the input array
     * @return list of key-value blocks
     */
    private Map<LinearSpaceBlock, INDArray> mapINDArrayToBlocks(final INDArray arr) {
        if (sparkContextIsAvailable) {
            return CoverageModelSparkUtils.mapINDArrayToBlocks(targetBlocks, arr);
        } else {
            return Collections.singletonMap(targetBlocks.get(0), arr);
        }
    }

    private Map<LinearSpaceBlock, ImmutablePair<INDArray, INDArray>> mapINDArrayPairToBlocks(final INDArray arr1,
                                                                                             final INDArray arr2) {
        if (sparkContextIsAvailable) {
            final Map<LinearSpaceBlock, INDArray> map1 =
                    CoverageModelSparkUtils.mapINDArrayToBlocks(targetBlocks, arr1);
            final Map<LinearSpaceBlock, INDArray> map2 =
                    CoverageModelSparkUtils.mapINDArrayToBlocks(targetBlocks, arr2);
            final Map<LinearSpaceBlock, ImmutablePair<INDArray, INDArray>> res = new HashMap<>();
            targetBlockStream().forEach(tb -> res.put(tb, ImmutablePair.of(map1.get(tb), map2.get(tb))));
            return res;
        } else {
            return Collections.singletonMap(targetBlocks.get(0), ImmutablePair.of(arr1, arr2));
        }
    }

    private Map<LinearSpaceBlock, ImmutableTriple<INDArray, INDArray, INDArray>> mapINDArrayTripleToBlocks(final INDArray arr1,
                                                                                                           final INDArray arr2,
                                                                                                           final INDArray arr3) {
        if (sparkContextIsAvailable) {
            final Map<LinearSpaceBlock, INDArray> map1 =
                    CoverageModelSparkUtils.mapINDArrayToBlocks(targetBlocks, arr1);
            final Map<LinearSpaceBlock, INDArray> map2 =
                    CoverageModelSparkUtils.mapINDArrayToBlocks(targetBlocks, arr2);
            final Map<LinearSpaceBlock, INDArray> map3 =
                    CoverageModelSparkUtils.mapINDArrayToBlocks(targetBlocks, arr3);
            final Map<LinearSpaceBlock, ImmutableTriple<INDArray, INDArray, INDArray>> res = new HashMap<>();
            targetBlockStream().forEach(tb -> res.put(tb, ImmutableTriple.of(map1.get(tb), map2.get(tb), map3.get(tb))));
            return res;
        } else {
            return Collections.singletonMap(targetBlocks.get(0), ImmutableTriple.of(arr1, arr2, arr3));
        }
    }

    /**
     * (shorthand helper method) returns an {@link IntStream} of sample indices
     * @return {@link IntStream}
     */
    private IntStream sampleIndexStream() { return IntStream.range(0, numSamples); }

    /**
     * (shorthand helper method) returns a {@link Stream<LinearSpaceBlock>} of target blocks
     * @return {@link Stream<LinearSpaceBlock>}
     */
    private Stream<LinearSpaceBlock> targetBlockStream() { return targetBlocks.stream(); }

    public Partitioner getTargetSpacePartitioner() {
        return new HashPartitioner(numTargetBlocks);
    }

    @Override
    public INDArray fetchTargetMeanBias() {
        return fetchFromWorkers("m_t", 1);
    }

    public INDArray fetchTotalUnexplainedVariance() {
        return fetchFromWorkers("tot_Psi_st", 1);
    }

    public INDArray fetchTotalNoise() {
        return fetchFromWorkers("Wz_st", 1);
    };

    @Override
    public INDArray fetchTargetUnexplainedVariance() {
        return fetchFromWorkers("Psi_t", 1);
    }

    @Override
    public INDArray fetchPrincipalLatentToTargetMap() {
        return fetchFromWorkers("W_tl", 0);
    }

    @Override
    public ImmutablePair<INDArray, INDArray> fetchCopyRatioMaxLikelihoodResults() {

        final INDArray M_Psi_inv_st = fetchFromWorkers("M_Psi_inv_st", 1);
        final INDArray log_nu_st = fetchFromWorkers("log_nu_st", 1);
        final INDArray m_t = fetchFromWorkers("m_t", 1);
        final INDArray Wz_st = fetchFromWorkers("Wz_st", 1);
        final INDArray log_d_s = sampleMeanLogReadDepths;

        /* calculate the required quantities */
        return ImmutablePair.of(log_nu_st.sub(Wz_st).subiRowVector(m_t).subiColumnVector(log_d_s),
                M_Psi_inv_st);
    }


    @Override
    public INDArray fetchSampleMeanLogReadDepths() {
        return sampleMeanLogReadDepths;
    }

    @Override
    public INDArray fetchSampleVarLogReadDepths() {
        return sampleVarLogReadDepths;
    }

    @Override
    protected double[] vectorToArray(final INDArray vec) {
        final int dim = vec.length();
        return IntStream.range(0, dim).mapToDouble(vec::getDouble).toArray();
    }

    @Override
    protected double[] getMatrixRow(final INDArray mat, final int rowIndex) {
        final INDArray row = mat.getRow(rowIndex);
        final int dim = row.length();
        return IntStream.range(0, dim).mapToDouble(row::getDouble).toArray();

    }

    @Override
    protected double[] getMatrixColumn(final INDArray mat, final int colIndex) {
        final INDArray col = mat.getColumn(colIndex);
        final int dim = col.length();
        return IntStream.range(0, dim).mapToDouble(col::getDouble).toArray();
    }

    @Override
    public void saveModel(@Nonnull final String outputPath) {
        logger.info("Saving the model to disk...");
        CoverageModelParametersNDArray.write(new CoverageModelParametersNDArray(processedTargetList,
                fetchTargetMeanBias(), fetchTargetUnexplainedVariance(), fetchPrincipalLatentToTargetMap()), outputPath);
    }

}