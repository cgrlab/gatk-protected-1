package org.broadinstitute.hellbender.tools.coveragemodel;

import com.google.common.collect.ImmutableMap;
import htsjdk.samtools.util.Log;
import org.apache.spark.api.java.JavaSparkContext;
import org.broadinstitute.hellbender.engine.spark.SparkContextFactory;
import org.broadinstitute.hellbender.tools.exome.ReadCountCollection;
import org.broadinstitute.hellbender.tools.exome.ReadCountCollectionUtils;
import org.broadinstitute.hellbender.tools.exome.sexgenotyper.ContigPloidyAnnotationTableReader;
import org.broadinstitute.hellbender.tools.exome.sexgenotyper.PloidyAnnotatedTargetCollection;
import org.broadinstitute.hellbender.tools.exome.sexgenotyper.SexGenotypeDataCollection;
import org.broadinstitute.hellbender.utils.LoggingUtils;
import org.broadinstitute.hellbender.utils.test.BaseTest;
import org.testng.annotations.BeforeClass;
import org.testng.annotations.BeforeSuite;
import org.testng.annotations.DataProvider;
import org.testng.annotations.Test;

import javax.annotation.Nonnull;
import java.io.File;
import java.io.IOException;
import java.util.Map;

/**
 * @author Mehrtash Babadi &lt;mehrtash@broadinstitute.org&gt;
 */
public class CoverageModelEMAlgorithmUnitTest extends BaseTest {
    private static final String TEST_SUB_DIR = publicTestDir + "org/broadinstitute/hellbender/tools/coveragemodel";
    private static final File TEST_RCC_FILE = new File(TEST_SUB_DIR, "synthetic_1000.tsv");
    private static final File TEST_CONTIG_ANNOTS_SAME_PLOIDY_FILE =
            new File(TEST_SUB_DIR, "synthetic_contig_annots_same.tsv");
    private static final File TEST_CONTIG_ANNOTS_DIFFERENT_PLOIDY_FILE =
            new File(TEST_SUB_DIR, "synthetic_contig_annots_different.tsv");
    private static final File TEST_SAMPLE_SEX_GENOTYPES_FILE =
            new File(TEST_SUB_DIR, "synthetic_sample_sex_genotypes.tsv");

    private static final int NUMBER_OF_PARTITIONS = 7;

    private static double CNV_EVENT_PROBABILITY = 0.0001;
    private static double CNV_EVENT_MEAN_SIZE = 50;

    private static ReadCountCollection testReadCounts;

    private static CoverageModelEMParams params;
    private static CoverageModelEMWorkspaceNDArraySparkToggle ws;
    private static CoverageModelEMAlgorithmNDArraySparkToggle algo;
    private static SexGenotypeDataCollection sexGenotypeData;
    private static PloidyAnnotatedTargetCollection samePloidyAnnots, differentPloidyAnnots;
    private static CoverageModelGermlineCopyNumberPosteriorCalculator copyNumberPosteriorCalculator;

    private static final Map<String, String> nd4jSparkProperties = ImmutableMap.<String,String>builder()
            .put("spark.kryo.registrator", "org.broadinstitute.hellbender.tools.coveragemodel.nd4jutils.Nd4jRegistrator")
            .build();

    @BeforeSuite @Override
    public void setTestVerbosity(){
        LoggingUtils.setLoggingLevel(Log.LogLevel.INFO);
    }

    @BeforeClass
    public static void init() throws IOException {
        testReadCounts = ReadCountCollectionUtils.parse(TEST_RCC_FILE);
        sexGenotypeData = new SexGenotypeDataCollection(TEST_SAMPLE_SEX_GENOTYPES_FILE);
        samePloidyAnnots = new PloidyAnnotatedTargetCollection(ContigPloidyAnnotationTableReader
                .readContigPloidyAnnotationsFromFile(TEST_CONTIG_ANNOTS_SAME_PLOIDY_FILE),
                testReadCounts.targets());
        differentPloidyAnnots = new PloidyAnnotatedTargetCollection(ContigPloidyAnnotationTableReader
                .readContigPloidyAnnotationsFromFile(TEST_CONTIG_ANNOTS_DIFFERENT_PLOIDY_FILE),
                testReadCounts.targets());
        copyNumberPosteriorCalculator = new CoverageModelGermlineCopyNumberPosteriorCalculator(
                CNV_EVENT_PROBABILITY, CNV_EVENT_MEAN_SIZE);
        copyNumberPosteriorCalculator.initializeCaches(testReadCounts.targets());
    }

    @Test(dataProvider = "ploidyAnnotsDataProvider")
    public void localTest(@Nonnull final PloidyAnnotatedTargetCollection ploidyAnnots) {
        params = new CoverageModelEMParams();
                //.enableFourierRegularization()
                //.setFourierFactors(FourierLinearOperator.getMidpassFilterFourierFactors(1000, 0, 100));
        params.setWSolverType(CoverageModelEMParams.WSolverType.W_SOLVER_LOCAL);
        ws = new CoverageModelEMWorkspaceNDArraySparkToggle(testReadCounts, ploidyAnnots,
                sexGenotypeData, copyNumberPosteriorCalculator, params, null,
                1, null);
        algo = new CoverageModelEMAlgorithmNDArraySparkToggle(params, ws);
        algo.runExpectationMaximization(true);
        ws.saveModel("/Users/mehrtash/Data/Genome/PPCA/out/blah");
        ws.savePosteriors("/Users/mehrtash/Data/Genome/PPCA/out/blah", true);
    }

//    @Test(dataProvider = "ploidyAnnotsDataProvider")
//    public void somaticTestLocal(@Nonnull final PloidyAnnotatedTargetCollection ploidyAnnots) {
//        params = new CoverageModelEMParams();
//        final CoverageModelParametersNDArray model =
//                CoverageModelParametersNDArray.read("/Users/mehrtash/Data/Genome/PPCA/out/blah");
//        model.arePrincipalComponenetsOrthogonal(1e-4, true, logger);
//        params.setWSolverType(CoverageModelEMParams.WSolverType.W_SOLVER_LOCAL);
//        ws = new CoverageModelEMWorkspaceNDArraySparkToggle(testReadCounts, ploidyAnnots,
//                sexGenotypeData, copyNumberPosteriorCalculator, params, model,
//                1, null);
//        algo = new CoverageModelEMAlgorithmNDArraySparkToggle(params, ws);
//        algo.runExpectation(true);
//        ws.savePosteriors("/Users/mehrtash/Data/Genome/PPCA/out/blah2", true);
//    }

//
//    @Test(dataProvider = "ploidyAnnotsDataProvider")
//    public void somaticTestSpark(@Nonnull final PloidyAnnotatedTargetCollection ploidyAnnots) {
//        params = new CoverageModelEMParams();
//        final CoverageModelParametersNDArray model = CoverageModelParametersNDArray.read("/Users/mehrtash/Data/Genome/PPCA/out/blah");
//        params.setWSolverType(CoverageModelEMParams.WSolverType.W_SOLVER_LOCAL);
//        final JavaSparkContext ctx = SparkContextFactory.getTestSparkContext(nd4jSparkProperties);
//        final String checkpointingPath = createTempDir("coverage_model_spark_checkpoint").getAbsolutePath();
//        ctx.setCheckpointDir(checkpointingPath);
//        ws = new CoverageModelEMWorkspaceNDArraySparkToggle(testReadCounts, ploidyAnnots,
//                sexGenotypeData, copyNumberPosteriorCalculator, params, model,
//                NUMBER_OF_PARTITIONS, ctx);
//        algo = new CoverageModelEMAlgorithmNDArraySparkToggle(params, ws);
//        algo.runExpectation(true);
//        ws.savePosteriors("/Users/mehrtash/Data/Genome/PPCA/out/blah2", true);
//    }

//    @Test(dataProvider = "ploidyAnnotsDataProvider")
//    public void sparkTest(@Nonnull final PloidyAnnotatedTargetCollection ploidyAnnots) {
//        params = new CoverageModelEMParams();
//                //.enableFourierRegularization()
//                //.setFourierFactors(FourierLinearOperator.getMidpassFilterFourierFactors(1000, 0, 100));
//        params.setWSolverType(CoverageModelEMParams.WSolverType.W_SOLVER_LOCAL);
//        final JavaSparkContext ctx = SparkContextFactory.getTestSparkContext(nd4jSparkProperties);
//        final String checkpointingPath = createTempDir("coverage_model_spark_checkpoint").getAbsolutePath();
//        ctx.setCheckpointDir(checkpointingPath);
//        ws = new CoverageModelEMWorkspaceNDArraySparkToggle(testReadCounts, ploidyAnnots,
//                sexGenotypeData, copyNumberPosteriorCalculator, params, null,
//                NUMBER_OF_PARTITIONS, ctx);
//        algo = new CoverageModelEMAlgorithmNDArraySparkToggle(params, ws);
//        algo.runExpectationMaximization(true);
//    }

    @DataProvider(name = "ploidyAnnotsDataProvider")
    public Object[][] ploidyAnnotsDataProvider() {
//        return new Object[][] {{samePloidyAnnots}, {differentPloidyAnnots}};
        return new Object[][] {{samePloidyAnnots}};
    }

}
