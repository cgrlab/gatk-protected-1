package org.broadinstitute.hellbender.tools.coveragemodel;

import htsjdk.variant.variantcontext.writer.VariantContextWriter;
import org.apache.commons.lang3.tuple.ImmutablePair;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.broadinstitute.hellbender.exceptions.UserException;
import org.broadinstitute.hellbender.tools.coveragemodel.interfaces.CopyRatioPosteriorCalculator;
import org.broadinstitute.hellbender.tools.exome.*;
import org.broadinstitute.hellbender.tools.exome.sexgenotyper.GermlinePloidyAnnotatedTargetCollection;
import org.broadinstitute.hellbender.tools.exome.sexgenotyper.SexGenotypeDataCollection;
import org.broadinstitute.hellbender.utils.IntervalUtils;
import org.broadinstitute.hellbender.utils.SimpleInterval;
import org.broadinstitute.hellbender.utils.Utils;
import org.broadinstitute.hellbender.utils.hmm.interfaces.AlleleMetadataProvider;
import org.broadinstitute.hellbender.utils.hmm.interfaces.CallStringProvider;
import org.broadinstitute.hellbender.utils.hmm.interfaces.ScalarProvider;
import org.broadinstitute.hellbender.utils.hmm.segmentation.HiddenMarkovModelPostProcessor;
import org.broadinstitute.hellbender.utils.hmm.segmentation.HiddenStateSegmentRecordWriter;
import org.broadinstitute.hellbender.utils.tsv.DataLine;
import org.broadinstitute.hellbender.utils.tsv.TableColumnCollection;
import org.broadinstitute.hellbender.utils.tsv.TableWriter;
import org.broadinstitute.hellbender.utils.variant.GATKVariantContextUtils;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.io.Writer;
import java.util.*;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

/**
 * This abstract class provides the basic workspace structure for {@link CoverageModelEMAlgorithm},
 * Explicit implementations may use local or distributed memory allocation and computation.
 *
 * @param <V> vector type
 * @param <M> matrix type
 * @param <S> copy ratio hidden state type
 *
 * @author Mehrtash Babadi &lt;mehrtash@broadinstitute.org&gt;
 */

public abstract class CoverageModelEMWorkspace<V, M, S extends AlleleMetadataProvider & CallStringProvider & ScalarProvider> {

    public static final String COPY_RATIO_MLE_FILENAME = "copy_ratio_MLE.tsv";
    public static final String COPY_RATIO_PRECISION_FILENAME = "copy_ratio_precision.tsv";
    public static final String COPY_RATIO_VITERBI_FILENAME = "copy_ratio_Viterbi.tsv";
    public static final String SAMPLE_READ_DEPTH_POSTERIOS_FILENAME = "sample_read_depth_posteriors.tsv";
    public static final String SAMPLE_LOG_LIKELIHOODS_FILENAME = "sample_log_likelihoods.tsv";
    public static final String SAMPLE_UNEXPLAINED_VARIANCE_FILENAME = "sample_unexplained_variance.tsv";
    public static final String SAMPLE_BIAS_LATENT_POSTERIORS_FILENAME = "sample_bias_latent_posteriors.tsv";
    public static final String TOTAL_UNEXPLAINED_VARIANCE_FILENAME = "copy_ratio_Psi.tsv";
    public static final String TOTAL_REMOVED_BIAS_FILENAME = "copy_ratio_Wz.tsv";
    public static final String COPY_RATIO_SEGMENTS_FILENAME = "copy_ratio_segments.seg";
    public static final String COPY_RATIO_GENOTYPES_FILENAME = "copy_ratio_genotypes.vcf";

    protected final Logger logger = LogManager.getLogger(CoverageModelEMWorkspace.class);

    protected final CoverageModelEMParams params;

    protected final ReadCountCollection processedReadCounts;

    protected final CopyRatioPosteriorCalculator<CoverageModelCopyRatioEmissionData, S> copyRatioPosteriorCalculator;

    /* useful elements to fetch from the processed read count collection */
    protected final List<Target> processedTargetList;
    protected final Map<Target, Integer> processedTargetIndexMap;
    protected final List<String> processedSampleList;

    protected final CoverageModelParametersNDArray processedModel;

    /**
     * List of sample germline ploidies in the same order as {@code processedSampleList}
     */
    protected final List<int[]> sampleGermlinePloidies;

    protected final int numSamples, numTargets, numLatents;

    /**
     * Basic constructor -- does the following tasks:
     *
     * <dl>
     *     <dt> processes the raw read counts and populates {@code processedReadCounts} </dt>
     *     <dt> fetches the target germline ploidy for each sample </dt>
     * </dl>
     *
     * @param rawReadCounts not {@code null} instance of {@link ReadCountCollection}
     * @param ploidyAnnots an instance of {@link GermlinePloidyAnnotatedTargetCollection} for obtaining target ploidies
     *                     for different sex genotypes
     * @param sexGenotypeData an instance of {@link SexGenotypeDataCollection} for obtaining sample sex genotypes
     * @param copyRatioPosteriorCalculator an implementation of {@link CopyRatioPosteriorCalculator} for obtaining copy ratio posterios
     * @param params not {@code null} instance of {@link CoverageModelEMParams}
     */
    protected CoverageModelEMWorkspace(@Nonnull final ReadCountCollection rawReadCounts,
                                       @Nonnull final GermlinePloidyAnnotatedTargetCollection ploidyAnnots,
                                       @Nonnull final SexGenotypeDataCollection sexGenotypeData,
                                       @Nonnull final CopyRatioPosteriorCalculator<CoverageModelCopyRatioEmissionData, S> copyRatioPosteriorCalculator,
                                       @Nonnull final CoverageModelEMParams params,
                                       @Nullable final CoverageModelParametersNDArray model) {
        this.params = params;
        this.copyRatioPosteriorCalculator = copyRatioPosteriorCalculator;

        /* foremost check -- targets are lexicographically sorted */
        final List<Target> originalTargetList = rawReadCounts.targets();
        final ReadCountCollection targetSortedRawReadCounts;
        if (!IntStream.range(0, originalTargetList.size()-1)
                .allMatch(ti -> IntervalUtils.LEXICOGRAPHICAL_ORDER_COMPARATOR
                        .compare(originalTargetList.get(ti + 1), originalTargetList.get(ti)) > 0)) {
            final List<Target> sortedTargetList =  originalTargetList.stream()
                    .sorted(IntervalUtils.LEXICOGRAPHICAL_ORDER_COMPARATOR).collect(Collectors.toList());
            targetSortedRawReadCounts = rawReadCounts.arrangeTargets(sortedTargetList);
        } else {
            targetSortedRawReadCounts = rawReadCounts;
        }

        if (model != null) {
            final ReadCountCollection intermediateReadCounts = processReadCountCollection(targetSortedRawReadCounts,
                    params, logger);
            /* adapt model and read counts */
            final ImmutablePair<CoverageModelParametersNDArray, ReadCountCollection> modelReadCountsPair =
                CoverageModelParametersNDArray.adaptModelToReadCountCollection(model, intermediateReadCounts, logger);
            processedModel = modelReadCountsPair.left;
            processedReadCounts = modelReadCountsPair.right;
            numLatents = processedModel.getNumLatents();
            if (params.getNumLatents() != processedModel.getNumLatents()) {
                logger.info("Changing number of latent variables to " + processedModel.getNumLatents() + " based" +
                        " on the provided model; requested value was: " + params.getNumLatents());
                params.setNumLatents(processedModel.getNumLatents());
            }
        } else {
            processedModel = null;
            processedReadCounts = processReadCountCollection(targetSortedRawReadCounts, params, logger);
            numLatents = params.getNumLatents();
        }

        /* ... and then populate these */
        numSamples = processedReadCounts.columnNames().size();
        numTargets = processedReadCounts.targets().size();
        processedTargetList = processedReadCounts.targets();
        processedSampleList = processedReadCounts.columnNames();
        processedTargetIndexMap = new HashMap<>();
        IntStream.range(0, numTargets).forEach(ti -> processedTargetIndexMap.put(processedTargetList.get(ti), ti));

        /* populate sample germline ploidies */
        sampleGermlinePloidies = Collections.unmodifiableList(
                processedSampleList.stream().map(sampleName -> {
                    final String sampleSexGenotypeIdentifier = sexGenotypeData.getSampleSexGenotypeData(sampleName).getSexGenotype();
                    return processedTargetList.stream().map(t -> ploidyAnnots.getTargetGermlinePloidyByGenotype(t, sampleSexGenotypeIdentifier))
                            .mapToInt(Integer::intValue).toArray();
                }).collect(Collectors.toList()));

        logger.info(String.format("Number of samples before and after pre-processing read counts: (%d, %d)",
                rawReadCounts.columnNames().size(), numSamples));
        logger.info(String.format("Number of targets before and after pre-processing read counts: (%d, %d)",
                rawReadCounts.targets().size(), numTargets));
    }

    /**
     * Process raw read counts and filter bad targets and/or samples:
     *
     * <dl>
     *     <dt> Remove totally uncovered targets </dt>
     * </dl>
     *
     * TODO add more filters?
     *
     * - remove targets with very high and very low GC content
     * - remove targets with lots of repeats
     * - remove targets that have extremely high or low coverage (not sure)
     * -
     *
     * @param rawReadCounts raw read counts
     * @return processed read counts
     */
    public static ReadCountCollection processReadCountCollection(@Nonnull final ReadCountCollection rawReadCounts,
                                                                 @Nonnull final CoverageModelEMParams params,
                                                                 @Nonnull final Logger logger) {
        ReadCountCollection processedReadCounts;
        processedReadCounts = ReadCountCollectionUtils.removeColumnsWithBadValues(rawReadCounts, logger);
        processedReadCounts = ReadCountCollectionUtils.removeTotallyUncoveredTargets(processedReadCounts, logger);

        return processedReadCounts;
    }

    /****************
     * save to disk *
     ****************/

    public abstract void saveModel(final String outputPath);

    protected abstract void saveCopyRatioMLE(final String outputPath);

    protected abstract void saveReadDepthPosteriors(final String outputPath);

    protected abstract void saveLogLikelihoodPosteriors(final String outputPath);

    protected abstract void saveGammaPosteriors(final String outputPath);

    protected abstract void saveBiasLatentPosteriors(final String outputPath);

    protected abstract void saveCopyRatioPosteriors(final String outputPath, final S referenceState, final String commandLine);

    protected abstract void saveExtendedPosteriors(final String outputPath);

    public void savePosteriors(final S referenceState, final String outputPath, final PosteriorVerbosityLevel verbosityLevel,
                               @Nullable final String commandLine) {
        /* create output directory if it doesn't exist */
        createOutputPath(outputPath);

        saveReadDepthPosteriors(outputPath);
        saveLogLikelihoodPosteriors(outputPath);
        saveGammaPosteriors(outputPath);
        saveBiasLatentPosteriors(outputPath);

        if (verbosityLevel.equals(PosteriorVerbosityLevel.FULL)) {
            saveCopyRatioPosteriors(outputPath, referenceState, commandLine);
            if (params.extendedPosteriorOutputEnabled()) {
                saveCopyRatioMLE(outputPath);
                saveExtendedPosteriors(outputPath);
            }
        }
    }

    private void createOutputPath(final String outputPath) {
        final File outputPathFile = new File(outputPath);
        if (!outputPathFile.exists()) {
            if (!outputPathFile.mkdirs()) {
                throw new UserException.CouldNotCreateOutputFile(outputPathFile, "Could not create the output directory");
            }
        }
    }

}