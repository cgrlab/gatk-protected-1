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
                rawReadCounts.targets().size(), getNumTargets()));
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

    /************
     * acessors *
     ************/

    public int getNumSamples() { return numSamples; }

    public int getNumTargets() { return numTargets; }

    public abstract double getLogLikelihood();

    public abstract double[] getLogLikelihoodPerSample();

    public abstract double[] fetchSampleUnexplainedVariance();

    public abstract V fetchTargetMeanBias();

    public abstract V fetchTargetUnexplainedVariance();

    public abstract M fetchPrincipalLatentToTargetMap();

    public abstract M fetchTotalUnexplainedVariance();

    public abstract M fetchTotalNoise();

    public abstract ImmutablePair<M, M> fetchCopyRatioMaxLikelihoodResults();

    public abstract V fetchSampleMeanLogReadDepths();

    public abstract V fetchSampleVarLogReadDepths();

    protected abstract double[] vectorToArray(final V vec);

    protected abstract double[] getMatrixRow(final M mat, final int rowIndex);

    protected abstract double[] getMatrixColumn(final M mat, final int colIndex);

    protected abstract List<CopyRatioHiddenMarkovModelResults<CoverageModelCopyRatioEmissionData, S>> getCopyRatioHiddenMarkovModelResults();

    /****************
     * save to disk *
     ****************/

    public abstract void saveModel(final String outputPath);

    public void savePosteriors(final S referenceState, final String outputPath, @Nullable final String commandLine) {
        /* create output directory if it doesn't exist */
        createOutputPath(outputPath);

        final List<String> sampleNames = processedReadCounts.columnNames();

        /* write copy ratio MLE results to file */
        final ImmutablePair<M, M> copyRatioMLEData = fetchCopyRatioMaxLikelihoodResults();

        final File copyRatioMLEFile = new File(outputPath, "copy_ratio_MLE.tsv");
        try (final TableWriter<TargetDoubleRecord> copyRatioRecordTableWriter = getTargetDoubleRecordTableWriter(new FileWriter(copyRatioMLEFile),
                processedReadCounts.columnNames())) {
            for (int targetIndex = 0; targetIndex < numTargets; targetIndex++) {
                copyRatioRecordTableWriter.writeRecord(new TargetDoubleRecord(processedTargetList.get(targetIndex),
                        getMatrixColumn(copyRatioMLEData.left, targetIndex)));
            }
        } catch (final IOException ex) {
            throw new UserException.CouldNotCreateOutputFile(copyRatioMLEFile, "Could not save copy ratio MLE results");
        }

        final File copyRatioPrecisionFile = new File(outputPath, "copy_ratio_precision.tsv");
        try (final TableWriter<TargetDoubleRecord> copyRatioPrecisionRecordTableWriter = getTargetDoubleRecordTableWriter(new FileWriter(copyRatioPrecisionFile),
                processedReadCounts.columnNames())) {
            for (int targetIndex = 0; targetIndex < numTargets; targetIndex++) {
                copyRatioPrecisionRecordTableWriter.writeRecord(new TargetDoubleRecord(processedTargetList.get(targetIndex),
                        getMatrixColumn(copyRatioMLEData.right, targetIndex)));
            }
        } catch (final IOException ex) {
            throw new UserException.CouldNotCreateOutputFile(copyRatioPrecisionFile, "Could not save copy ratio precision results");
        }

        /* write read depth posteriors to file */
        final File sampleReadDepthPosteriorsFile = new File(outputPath, "sample_read_depth_posteriors.tsv");
        final double[] sampleMeanLogReadDepthsArray = vectorToArray(fetchSampleMeanLogReadDepths());
        final double[] sampleVarLogReadDepthsArray = vectorToArray(fetchSampleVarLogReadDepths());
        try (final TableWriter<SampleReadDepthPosteriorRecord> sampleReadDepthPosteriorsTableWriter =
                     getSampleLogReadDepthTableWriter(new FileWriter(sampleReadDepthPosteriorsFile))) {
            for (int sampleIndex = 0; sampleIndex < numSamples; sampleIndex++) {
                sampleReadDepthPosteriorsTableWriter.writeRecord(new SampleReadDepthPosteriorRecord(sampleNames.get(sampleIndex),
                        sampleMeanLogReadDepthsArray[sampleIndex], sampleVarLogReadDepthsArray[sampleIndex]));
            }
        } catch (final IOException ex) {
            throw new UserException.CouldNotCreateOutputFile(sampleReadDepthPosteriorsFile, "Could not save sample read depth posterior results");
        }

        /* write log likelihood per sample to file */
        final File sampleLogLikelihoodsFile = new File(outputPath, "sample_log_likelihoods.tsv");
        final double[] sampleLogLikelihoods = getLogLikelihoodPerSample();
        try (final TableWriter<SampleDoubleRecord> sampleLogLikelihoodRecordTableWriter =
                     getSampleDoubleRecordTableWriter(new FileWriter(sampleLogLikelihoodsFile), "LOG_LIKELIHOOD")) {
            for (int sampleIndex = 0; sampleIndex < numSamples; sampleIndex++) {
                sampleLogLikelihoodRecordTableWriter.writeRecord(new SampleDoubleRecord(sampleNames.get(sampleIndex),
                        sampleLogLikelihoods[sampleIndex]));
            }
        } catch (final IOException ex) {
            throw new UserException.CouldNotCreateOutputFile(sampleLogLikelihoodsFile, "Could not save sample log likelihood results");
        }

        /* write sample-specific unexplained variance to file */
        final File sampleUnexplainedVarianceFile = new File(outputPath, "sample_unexplained_variance.tsv");
        final double[] sampleUnexplainedVariance = fetchSampleUnexplainedVariance();
        try (final TableWriter<SampleDoubleRecord> sampleUnexplainedVarianceTableWriter =
                     getSampleDoubleRecordTableWriter(new FileWriter(sampleUnexplainedVarianceFile), "SAMPLE_UNEXPLAINED_VARIANCE")) {
            for (int sampleIndex = 0; sampleIndex < numSamples; sampleIndex++) {
                sampleUnexplainedVarianceTableWriter.writeRecord(new SampleDoubleRecord(sampleNames.get(sampleIndex),
                        sampleUnexplainedVariance[sampleIndex]));
            }
        } catch (final IOException ex) {
            throw new UserException.CouldNotCreateOutputFile(sampleUnexplainedVarianceFile, "Could not save sample unexplained variance results");
        }

        /* segmentation, vcf creation */
        final List<CopyRatioHiddenMarkovModelResults<CoverageModelCopyRatioEmissionData, S>> copyRatioHMMResult =
                getCopyRatioHiddenMarkovModelResults();
        final HiddenMarkovModelPostProcessor<CoverageModelCopyRatioEmissionData, S, Target> copyRatioProcessor =
                new HiddenMarkovModelPostProcessor<>(
                        sampleNames,
                        copyRatioHMMResult.stream()
                                .map(CopyRatioHiddenMarkovModelResults::getTargetList)
                                .map(HashedListTargetCollection::new)
                                .collect(Collectors.toList()),
                        copyRatioHMMResult.stream()
                                .map(CopyRatioHiddenMarkovModelResults::getForwardBackwardResult)
                                .collect(Collectors.toList()),
                        copyRatioHMMResult.stream()
                                .map(CopyRatioHiddenMarkovModelResults::getViterbiResult)
                                .collect(Collectors.toList()),
                        referenceState);

        final File segmentsFile = new File(outputPath, "copy_ratio_segments.seg");
        final File vcfFile = new File(outputPath, "copy_ratio_genotypes.vcf");
        try (final HiddenStateSegmentRecordWriter<S, Target> segWriter = new HiddenStateSegmentRecordWriter<>(segmentsFile);
             final VariantContextWriter VCFWriter  = GATKVariantContextUtils.createVCFWriter(vcfFile, null, false)) {
            copyRatioProcessor.writeSegmentsToTableWriter(segWriter);
            copyRatioProcessor.writeVariantsToVCFWriter(VCFWriter, "CNV", commandLine);
        } catch (final IOException ex) {
            throw new UserException.CouldNotCreateOutputFile(segmentsFile, "Could not create copy ratio segments file");
        }

        /* also, save Viterbi as a matrix */
        final File copyRatioViterbiFile = new File(outputPath, "copy_ratio_Viterbi.tsv");
        try (final TableWriter<TargetDoubleRecord> copyRatioRecordTableWriter = getTargetDoubleRecordTableWriter(new FileWriter(copyRatioViterbiFile),
                processedReadCounts.columnNames())) {
            final List<TargetCollection<Target>> sampleTargetCollections = new ArrayList<>(numSamples);
            for (int sampleIndex = 0; sampleIndex < numSamples; sampleIndex++) {
                sampleTargetCollections.add(new HashedListTargetCollection<>(copyRatioHMMResult.get(sampleIndex).getTargetList()));
            }
            for (int targetIndex = 0; targetIndex < numTargets; targetIndex++) {
                copyRatioRecordTableWriter.writeRecord(new TargetDoubleRecord(processedTargetList.get(targetIndex),
                        getViterbiAsDoubleArray(copyRatioHMMResult, sampleTargetCollections, targetIndex)));
            }
        } catch (final IOException ex) {
            throw new UserException.CouldNotCreateOutputFile(copyRatioViterbiFile, "Could not save copy ratio Viterbi results");
        }

        /* save total explained variance as a matrix */
        final File totalExplainedVarianceFile = new File(outputPath, "copy_ratio_Psi.tsv");
        final M totalExplainedVarianceMatrix = fetchTotalUnexplainedVariance();
        try (final TableWriter<TargetDoubleRecord> totalExplainedVarianceTableWriter = getTargetDoubleRecordTableWriter(
                new FileWriter(totalExplainedVarianceFile), processedReadCounts.columnNames())) {
            for (int targetIndex = 0; targetIndex < numTargets; targetIndex++) {
                totalExplainedVarianceTableWriter.writeRecord(new TargetDoubleRecord(processedTargetList.get(targetIndex),
                        getMatrixColumn(totalExplainedVarianceMatrix, targetIndex)));
            }
        } catch (final IOException ex) {
            throw new UserException.CouldNotCreateOutputFile(totalExplainedVarianceFile, "Could not save total unexplained variance results");
        }

        /* save total noise as a matrix */
        final File totalNoiseFile = new File(outputPath, "copy_ratio_Wz.tsv");
        final M totalNoiseMatrix = fetchTotalNoise();
        try (final TableWriter<TargetDoubleRecord> totalNoiseTableWriter = getTargetDoubleRecordTableWriter(
                new FileWriter(totalNoiseFile), processedReadCounts.columnNames())) {
            for (int targetIndex = 0; targetIndex < numTargets; targetIndex++) {
                totalNoiseTableWriter.writeRecord(new TargetDoubleRecord(processedTargetList.get(targetIndex),
                        getMatrixColumn(totalNoiseMatrix, targetIndex)));
            }
        } catch (final IOException ex) {
            throw new UserException.CouldNotCreateOutputFile(totalNoiseFile, "Could not save total noise results");
        }
    }

    private double[] getViterbiAsDoubleArray(final List<CopyRatioHiddenMarkovModelResults<CoverageModelCopyRatioEmissionData, S>> copyRatioHMMResult,
                                             final List<TargetCollection<Target>> sampleTargetCollections,
                                             final int targetIndex) {
        final Target target = processedTargetList.get(targetIndex);
        final double[] res = new double[numSamples];
        for (int si = 0; si < numSamples; si++) {
            final TargetCollection<Target> sampleTargets = sampleTargetCollections.get(si);
            final List<S> sampleCalls = copyRatioHMMResult.get(si).getViterbiResult();
            final int sampleTargetIndex = sampleTargets.index(target);
            if (sampleTargetIndex >= 0) {
                res[si] = sampleCalls.get(sampleTargetIndex).getScalar();
            } else {
                res[si] = 0.0;
            }
        }
        return res;
    }

    private void createOutputPath(final String outputPath) {
        final File outputPathFile = new File(outputPath);
        if (!outputPathFile.exists()) {
            if (!outputPathFile.mkdirs()) {
                throw new UserException.CouldNotCreateOutputFile(outputPathFile, "Could not create the output directory");
            }
        }
    }

    private static final class TargetDoubleRecord {
        private final Target target;
        private final double[] data;

        public TargetDoubleRecord(final Target target, final double[] data) {
            this.target = target;
            this.data = data;
        }

        final Target getTarget() {
            return target;
        }

        public void appendDataTo(final DataLine dataLine) {
            Utils.nonNull(dataLine);
            dataLine.append(data);
        }
    }

    private static TableWriter<TargetDoubleRecord> getTargetDoubleRecordTableWriter(final Writer writer, final List<String> countColumnNames) throws IOException {
        final List<String> columnNames = new ArrayList<>();

        columnNames.add(TargetTableColumn.CONTIG.toString());
        columnNames.add(TargetTableColumn.START.toString());
        columnNames.add(TargetTableColumn.END.toString());
        columnNames.add(TargetTableColumn.NAME.toString());
        columnNames.addAll(Utils.nonNull(countColumnNames));
        final TableColumnCollection columns = new TableColumnCollection(columnNames);

        return new TableWriter<TargetDoubleRecord>(writer, columns) {
            @Override
            protected void composeLine(final TargetDoubleRecord record, final DataLine dataLine) {
                final SimpleInterval interval = record.getTarget().getInterval();
                if (interval == null) {
                    throw new IllegalStateException("invalid combination of targets with and without intervals defined");
                }
                dataLine.append(interval.getContig())
                        .append(interval.getStart())
                        .append(interval.getEnd())
                        .append(record.getTarget().getName());
                record.appendDataTo(dataLine);
            }
        };
    }

    private static final class SampleReadDepthPosteriorRecord {
        private final String sampleName;
        private final double meanLogReadDepth;
        private final double varLogReadDepth;

        public SampleReadDepthPosteriorRecord(final String sampleName, final double meanLogReadDepth, final double varLogReadDepth) {
            this.sampleName = sampleName;
            this.meanLogReadDepth = meanLogReadDepth;
            this.varLogReadDepth = varLogReadDepth;
        }

        public void composeDataLine(final DataLine dataLine) {
            dataLine.append(sampleName);
            dataLine.append(meanLogReadDepth);
            dataLine.append(varLogReadDepth);
        }
    }

    private static final class SampleDoubleRecord {
        private final String sampleName;
        private final double value;

        public SampleDoubleRecord(final String sampleName, final double value) {
            this.sampleName = sampleName;
            this.value = value;
        }

        public void composeDataLine(final DataLine dataLine) {
            dataLine.append(sampleName);
            dataLine.append(value);
        }
    }

    private static TableWriter<SampleReadDepthPosteriorRecord> getSampleLogReadDepthTableWriter(final Writer writer) throws IOException {
        final List<String> columnNames = new ArrayList<>();

        columnNames.add("SAMPLE_NAME");
        columnNames.add("MEAN_LOG_READ_DEPTH_POSTERIOR");
        columnNames.add("VAR_LOG_READ_DEPTH_POSTERIOR");
        final TableColumnCollection columns = new TableColumnCollection(columnNames);

        return new TableWriter<SampleReadDepthPosteriorRecord>(writer, columns) {
            @Override
            protected void composeLine(final SampleReadDepthPosteriorRecord record, final DataLine dataLine) {
                record.composeDataLine(dataLine);
            }
        };
    }

    private static TableWriter<SampleDoubleRecord> getSampleDoubleRecordTableWriter(final Writer writer,
                                                                                    final String valueColumnName) throws IOException {
        final List<String> columnNames = new ArrayList<>();

        columnNames.add("SAMPLE_NAME");
        columnNames.add(valueColumnName);
        final TableColumnCollection columns = new TableColumnCollection(columnNames);

        return new TableWriter<SampleDoubleRecord>(writer, columns) {
            @Override
            protected void composeLine(final SampleDoubleRecord record, final DataLine dataLine) {
                record.composeDataLine(dataLine);
            }
        };
    }
}