package org.broadinstitute.hellbender.tools.coveragemodel;

import com.google.cloud.genomics.dataflow.utils.GCSOptions;
import com.google.common.collect.ImmutableMap;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.apache.spark.api.java.JavaSparkContext;
import org.broadinstitute.hellbender.cmdline.*;
import org.broadinstitute.hellbender.cmdline.programgroups.CopyNumberProgramGroup;
import org.broadinstitute.hellbender.engine.spark.SparkContextFactory;
import org.broadinstitute.hellbender.exceptions.UserException;
import org.broadinstitute.hellbender.tools.coveragemodel.interfaces.CopyRatioPosteriorCalculator;
import org.broadinstitute.hellbender.tools.exome.ReadCountCollection;
import org.broadinstitute.hellbender.tools.exome.ReadCountCollectionUtils;
import org.broadinstitute.hellbender.tools.exome.germlinehmm.CopyNumberTriState;
import org.broadinstitute.hellbender.tools.exome.sexgenotyper.ContigGermlinePloidyAnnotationTableReader;
import org.broadinstitute.hellbender.tools.exome.sexgenotyper.GermlinePloidyAnnotatedTargetCollection;
import org.broadinstitute.hellbender.tools.exome.sexgenotyper.SexGenotypeDataCollection;
import org.broadinstitute.hellbender.utils.SparkToggleCommandLineProgram;
import org.broadinstitute.hellbender.utils.gcs.BucketUtils;

import javax.annotation.Nonnull;
import java.io.*;
import java.util.Map;

/**
 * @author Mehrtash Babadi &lt;mehrtash@broadinstitute.org&gt;
 */
@CommandLineProgramProperties(
        summary = "todo",
        oneLineSummary = "todo",
        programGroup = CopyNumberProgramGroup.class
)
public final class CoverageModellerSparkToggle extends SparkToggleCommandLineProgram {

    private static final long serialVersionUID = 7864459447058892367L;

    private final Logger logger = LogManager.getLogger(CoverageModellerSparkToggle.class);

    private static final String FINAL_MODEL_PATHNAME = "model_final";

    private static final String CONTIG_PLOIDY_ANNOTATIONS_TABLE_LONG_NAME = "contigAnnotationsTable";
    private static final String CONTIG_PLOIDY_ANNOTATIONS_TABLE_SHORT_NAME = "annots";

    private static final String SAMPLE_SEX_GENOTYPE_TABLE_LONG_NAME = "sexGenotypeTable";
    private static final String SAMPLE_SEX_GENOTYPE_TABLE_SHORT_NAME = "gen";

    private static final String TARGET_SPACE_PARTITIONS_LONG_NAME = "targetSpacePartitions";
    private static final String TARGET_SPACE_PARTITIONS_SHORT_NAME = "partitions";

    public static final String EVENT_START_PROBABILITY_LONG_NAME = "eventStartProbability";
    public static final String EVENT_START_PROBABILITY_SHORT_NAME = "eventProb";

    public static final String MEAN_EVENT_SIZE_LONG_NAME = "meanEventSize";
    public static final String MEAN_EVENT_SIZE_SHORT_NAME = "eventSize";

    public static final String OUTPUT_PATH_LONG_NAME = "outputPath";
    public static final String OUTPUT_PATH_SHORT_NAME = "O";

    public static final String MODEL_PATH_LONG_NAME = "modelPath";
    public static final String MODEL_PATH_SHORT_NAME = "model";

    @Argument(
            doc = "Combined read count collection URI",
            fullName = StandardArgumentDefinitions.INPUT_LONG_NAME,
            shortName = StandardArgumentDefinitions.INPUT_SHORT_NAME,
            optional = false
    )
    protected String readCountsURI;

    @Argument(
            doc = "Contig ploidy annotations URI",
            fullName = CONTIG_PLOIDY_ANNOTATIONS_TABLE_LONG_NAME,
            shortName = CONTIG_PLOIDY_ANNOTATIONS_TABLE_SHORT_NAME,
            optional = false
    )
    protected String contigPloidyAnnotationsURI;

    @Argument(
            doc = "Sample sex genotypes URI",
            fullName = SAMPLE_SEX_GENOTYPE_TABLE_LONG_NAME,
            shortName = SAMPLE_SEX_GENOTYPE_TABLE_SHORT_NAME,
            optional = false
    )
    protected String sampleSexGenotypesURI;

    @Argument(doc = "Probability that a base in a copy-neutral segment is followed by a base belonging to a CNV (for" +
            " germline CNV calling)",
            fullName = EVENT_START_PROBABILITY_LONG_NAME,
            shortName = EVENT_START_PROBABILITY_SHORT_NAME,
            optional = true)
    public double eventStartProbability = 1.0e-8;

    @Argument(doc = "Estimated mean size of non-copy-neutral segments, in base-pairs (for germline CNV calling)",
            fullName = MEAN_EVENT_SIZE_LONG_NAME,
            shortName = MEAN_EVENT_SIZE_SHORT_NAME,
            optional = true)
    public double meanEventSize = 70_000;

    @Argument(
            doc = "Number of target space partitions (for Spark mode)",
            fullName = TARGET_SPACE_PARTITIONS_LONG_NAME,
            shortName = TARGET_SPACE_PARTITIONS_SHORT_NAME,
            optional = true
    )
    protected int targetSpacePartitions = 10;

    @Argument(
            doc = "Output path for saving the results",
            fullName = OUTPUT_PATH_LONG_NAME,
            shortName = OUTPUT_PATH_SHORT_NAME,
            optional = false
    )
    protected String outputPath;

    @Argument(
            doc = "Input model path",
            fullName = MODEL_PATH_LONG_NAME,
            shortName = MODEL_PATH_SHORT_NAME,
            optional = true
    )
    protected String modelPath = null;

    @ArgumentCollection
    protected final CoverageModelEMParams params = new CoverageModelEMParams();

    /* Use custom Nd4j Kryo serializer */
    private static final Map<String, String> nd4jSparkProperties = ImmutableMap.<String,String>builder()
            .put("spark.kryo.registrator", "org.nd4j.Nd4jRegistrator")
            .build();

    /**
     * Override doWork to inject custom nd4j serializer and set a temporary checkpointing path
     * @return
     */
    @Override
    protected Object doWork() {
        /* validate parameters */
        params.validate();

        JavaSparkContext ctx = null;
        if (!isDisableSpark) {
            /* create the spark context */
            final Map<String, String> sparkProerties = sparkArgs.getSparkProperties();
            sparkProerties.putAll(nd4jSparkProperties);
            ctx = SparkContextFactory.getSparkContext(getProgramName(), sparkProerties, sparkArgs.getSparkMaster());
            ctx.setCheckpointDir(params.getRDDCheckpointingPath());
        } else {
            logger.info("Spark disabled.  sparkMaster option (" + sparkArgs.getSparkMaster() + ") ignored.");
        }

        try {
            runPipeline(ctx);
            return null;
        } finally {
            afterPipeline(ctx);
        }
    }

    @Override
    protected void runPipeline(JavaSparkContext ctx) {
        logger.info("Parsing the read counts table...");
        final ReadCountCollection readCounts;
        try (final Reader readCountsReader = getReaderFromURI(readCountsURI)) {
            readCounts = ReadCountCollectionUtils.parse(readCountsReader, readCountsURI);
        } catch (final IOException ex) {
            ex.printStackTrace();
            throw new UserException.CouldNotReadInputFile("Could not parse the read count collection");
        }

        logger.info("Parsing the sample sex genotypes data table...");
        final SexGenotypeDataCollection sexGenotypeDataCollection;
        try (final Reader sexGenotypeDataCollectionReader = getReaderFromURI(sampleSexGenotypesURI)) {
            sexGenotypeDataCollection = new SexGenotypeDataCollection(sexGenotypeDataCollectionReader,
                    sampleSexGenotypesURI);
        } catch (final IOException ex) {
            ex.printStackTrace();
            throw new UserException.CouldNotReadInputFile("Could not parse the input sample sex genotypes data");
        }

        logger.info("Parsing the input contig ploidy annotation table...");
        final GermlinePloidyAnnotatedTargetCollection ploidyAnnotatedTargetCollection;
        try (final Reader ploidyAnnotationsReader = getReaderFromURI(contigPloidyAnnotationsURI)) {
            ploidyAnnotatedTargetCollection = new GermlinePloidyAnnotatedTargetCollection(ContigGermlinePloidyAnnotationTableReader
                    .readContigGermlinePloidyAnnotationsFromReader(contigPloidyAnnotationsURI, ploidyAnnotationsReader),
                    readCounts.targets());
        } catch (final IOException ex) {
            ex.printStackTrace();
            throw new UserException.CouldNotReadInputFile("Could not parse the input sample sex genotypes data");
        }

        logger.info("Initializing the copy ratio posterior calculator...");
        final CopyRatioPosteriorCalculator<CoverageModelCopyRatioEmissionData, CopyNumberTriState> copyNumberPosteriorCalculator =
                new CoverageModelGermlineCopyNumberPosteriorCalculator(eventStartProbability, meanEventSize);
        copyNumberPosteriorCalculator.initializeCaches(readCounts.targets());

        final CoverageModelParametersNDArray model;
        if (modelPath != null) {
            logger.info("Loading model parameters...");
            model = CoverageModelParametersNDArray.read(modelPath);
        } else {
            model = null;
        }

        logger.info("Initializing the EM algorithm workspace...");
        final CoverageModelEMWorkspaceNDArraySparkToggle<CopyNumberTriState> ws = new CoverageModelEMWorkspaceNDArraySparkToggle<>(
                readCounts, ploidyAnnotatedTargetCollection, sexGenotypeDataCollection, copyNumberPosteriorCalculator,
                params, model, targetSpacePartitions, ctx);

        final CoverageModelEMAlgorithmNDArraySparkToggle<CopyNumberTriState> algo =
                new CoverageModelEMAlgorithmNDArraySparkToggle<>(params,outputPath, CopyNumberTriState.NEUTRAL, ws);
        if (model == null) {
            algo.runExpectationMaximization();
            logger.info("Saving the model to disk...");
            ws.saveModel(new File(outputPath, FINAL_MODEL_PATHNAME).getAbsolutePath());
        } else {
            algo.runExpectation();
        }

        logger.info("Saving posteriors to disk...");
        ws.savePosteriors(CopyNumberTriState.NEUTRAL,
                new File(outputPath, "posteriors_final").getAbsolutePath(), PosteriorVerbosityLevel.FULL, this.getCommandLine());
    }

    private Reader getReaderFromURI(@Nonnull final String inputURI) throws IOException {
        final String inputAbsolutePath = getFilePathAbsolute(inputURI);
        if (!BucketUtils.isCloudStorageUrl(inputAbsolutePath) && !BucketUtils.isHadoopUrl(inputAbsolutePath)) {
            return new FileReader(inputAbsolutePath);
        } else { /* read from HDFS or GC */
            if (BucketUtils.isCloudStorageUrl(inputAbsolutePath) || BucketUtils.isHadoopUrl(inputAbsolutePath)) {
                final GCSOptions popts = getAuthenticatedGCSOptions();
                final InputStream inputStream = BucketUtils.openFile(inputAbsolutePath, popts);
                return new BufferedReader(new InputStreamReader(inputStream));
            } else {
                throw new UserException("Malformed input URI \"" + inputURI + "\". Please specify" +
                        " the URI either as a canonical path, or as \"hdfs://[...]\", or as \"gs://[...]\"");
            }
        }
    }

    private String getFilePathAbsolute(final String path) {
        if (BucketUtils.isCloudStorageUrl(path) || BucketUtils.isHadoopUrl(path) || BucketUtils.isFileUrl(path)) {
            return path;
        } else {
            return new File(path).getAbsolutePath();
        }
    }
}
