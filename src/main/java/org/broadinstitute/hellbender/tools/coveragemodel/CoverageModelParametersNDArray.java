package org.broadinstitute.hellbender.tools.coveragemodel;

import com.google.cloud.dataflow.sdk.repackaged.com.google.common.collect.Sets;
import org.apache.commons.lang3.tuple.ImmutablePair;
import org.apache.commons.math3.util.FastMath;
import org.apache.logging.log4j.Logger;
import org.broadinstitute.hellbender.exceptions.UserException;
import org.broadinstitute.hellbender.tools.coveragemodel.nd4jutils.Nd4jIOUtils;
import org.broadinstitute.hellbender.tools.exome.*;
import org.broadinstitute.hellbender.utils.param.ParamUtils;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.buffer.util.DataTypeUtil;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndexAll;

import javax.annotation.Nonnull;
import java.io.*;
import java.util.HashSet;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Set;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

/**
 *
 * @author Mehrtash Babadi &lt;mehrtash@broadinstitute.org&gt;
 */
public class CoverageModelParametersNDArray implements Serializable {

    private static final long serialVersionUID = -4350342293001054849L;

    public static final String TARGET_MEAN_BIAS_OUTPUT_FILE = "target_mean_bias.nd4j";
    public static final String TARGET_UNEXPLAINED_VARIANCE_OUTPUT_FILE = "target_unexplained_variance.nd4j";
    public static final String PRINCIPAL_LATENT_TO_TARGET_MAP_OUTPUT_FILE = "principal_map.nd4j";
    public static final String TARGET_LIST_OUTPUT_FILE = "targets.tsv";

    private final List<Target> targetList;

    /* 1 x T */
    private final INDArray targetMeanBias;

    /* 1 x T */
    private final INDArray targetUnexplainedVariance;

    /* T x L */
    private final INDArray principalLatentToTargetMap;

    private final int numTargets, numLatents;

    /**
     *
     * @param targetMeanBias
     * @param targetUnexplainedVariance
     * @param principalLatentToTargetMap
     */
    public CoverageModelParametersNDArray(@Nonnull final List<Target> targetList,
                                          @Nonnull final INDArray targetMeanBias,
                                          @Nonnull final INDArray targetUnexplainedVariance,
                                          @Nonnull final INDArray principalLatentToTargetMap) {
        this.targetList = targetList;
        this.targetMeanBias = targetMeanBias;
        this.targetUnexplainedVariance = targetUnexplainedVariance;
        this.principalLatentToTargetMap = principalLatentToTargetMap;

        /* check the dimensions */
        this.numTargets = targetList.size();
        if (numTargets != targetMeanBias.size(1) ||
                numTargets != targetUnexplainedVariance.size(1) ||
                numTargets != principalLatentToTargetMap.size(0)) {
            throw new UserException.BadInput("The dimension of target space (expected: " + numTargets + ") does not " +
                    "match with the dimensions of the provided model data: " +
                    "target mean bias = " + targetMeanBias.shapeInfoToString() + ", " +
                    "target unexplained variance = " + targetUnexplainedVariance.shapeInfoToString() + ", " +
                    "principal latent to target map = " + principalLatentToTargetMap.shapeInfoToString());
        }
        this.numLatents = principalLatentToTargetMap.size(1);
    }

    public INDArray getTargetMeanBias() {
        return targetMeanBias;
    }

    public INDArray getTargetUnexplainedVariance() {
        return targetUnexplainedVariance;
    }

    public INDArray getPrincipalLatentToTargetMap() {
        return principalLatentToTargetMap;
    }

    public INDArray getTargetMeanBiasOnTargetBlock(@Nonnull final LinearSpaceBlock tb) {
        checkTargetBlock(tb);
        return targetMeanBias.get(NDArrayIndex.all(),
                NDArrayIndex.interval(tb.getBegIndex(), tb.getEndIndex()));
    }

    private void checkTargetBlock(@Nonnull LinearSpaceBlock tb) {
        ParamUtils.inRange(tb.getBegIndex(), 0, numTargets, "The begin index of target block is out of range");
        ParamUtils.inRange(tb.getEndIndex(), 0, numTargets, "The begin index of target block is out of range");
    }

    public INDArray getTargetUnexplainedVarianceOnTargetBlock(@Nonnull final LinearSpaceBlock tb) {
        checkTargetBlock(tb);
        return targetUnexplainedVariance.get(NDArrayIndex.all(),
                NDArrayIndex.interval(tb.getBegIndex(), tb.getEndIndex()));
    }

    public INDArray getPrincipalLatentToTargetMapOnTargetBlock(@Nonnull final LinearSpaceBlock tb) {
        checkTargetBlock(tb);
        return principalLatentToTargetMap.get(NDArrayIndex.interval(tb.getBegIndex(), tb.getEndIndex()),
                NDArrayIndex.all());
    }

    private static void createOutputPath(final String outputPath) {
        final File outputPathFile = new File(outputPath);
        if (!outputPathFile.exists()) {
            if (!outputPathFile.mkdirs()) {
                throw new UserException.CouldNotCreateOutputFile(outputPathFile, "Could not create the output directory");
            }
        }
    }

    public int getNumTargets() {
        return numTargets;
    }

    public int getNumLatents() {
        return numLatents;
    }

    public List<Target> getTargetList() {
        return targetList;
    }

    /**
     * Check whether the principal components are orthogonal to each other or not.
     *
     * @param tol orthogonality tolerance
     * @param logResults print results or not
     */
    public boolean arePrincipalComponenetsOrthogonal(final double tol, final boolean logResults, @Nonnull final Logger logger) {
        ParamUtils.isPositive(tol, "Orthogonality tolerance must be positive");
        boolean orthogonal = true;
        for (int mu = 0; mu < numLatents; mu++) {
            for (int nu = mu; nu < numLatents; nu++) {
                final double innerProd = principalLatentToTargetMap.get(NDArrayIndex.all(),
                        NDArrayIndex.point(mu)).mul(principalLatentToTargetMap.get(NDArrayIndex.all(),
                        NDArrayIndex.point(nu))).sumNumber().doubleValue();
                if (mu != nu && FastMath.abs(innerProd) > tol) {
                    orthogonal = false;
                    logger.info("Inner product test failed on (" + mu + ", " + nu + ")");
                }
                if (logResults) {
                    logger.info("Inner product of (" + mu + ", " + nu + "): " + innerProd);
                } /* no need to continue */
                if (!logResults && !orthogonal) {
                    break;
                }
            }
        }
        return orthogonal;
    }

    /**
     * Read from disk
     *
     * @param modelPath
     * @return
     */
    public static CoverageModelParametersNDArray read(@Nonnull final String modelPath) {
        final File modelPathFile = new File(modelPath);
        if (!modelPathFile.exists()) {
            throw new UserException.BadInput("The model path does not exist: " + modelPathFile.getAbsolutePath());
        }

        Nd4j.create(1);
        DataTypeUtil.setDTypeForContext(DataBuffer.Type.DOUBLE);

        final File targetListFile = new File(modelPath, TARGET_LIST_OUTPUT_FILE);
        final List<Target> targetList;
        try (final Reader reader = new FileReader(targetListFile)) {
            targetList = TargetTableReader.readTargetFromReader(targetListFile.getAbsolutePath(), reader);
        } catch (final IOException ex) {
            throw new UserException.CouldNotCreateOutputFile(targetListFile, "Could not read targets interval list");
        }

        final File targetMeanBiasFile = new File(modelPath, TARGET_MEAN_BIAS_OUTPUT_FILE);
        final INDArray targetMeanBias = Nd4jIOUtils.readNDArrayFromFile(targetMeanBiasFile);

        final File targetUnexplainedVarianceFile = new File(modelPath, TARGET_UNEXPLAINED_VARIANCE_OUTPUT_FILE);
        final INDArray targetUnexplainedVariance = Nd4jIOUtils.readNDArrayFromFile(targetUnexplainedVarianceFile);

        final File principalLatentToTargetMapFile = new File(modelPath, PRINCIPAL_LATENT_TO_TARGET_MAP_OUTPUT_FILE);
        final INDArray principalLatentToTargetMap = Nd4jIOUtils.readNDArrayFromFile(principalLatentToTargetMapFile);

        return new CoverageModelParametersNDArray(targetList, targetMeanBias, targetUnexplainedVariance,
                principalLatentToTargetMap);
    }

    /**
     * Write to disk
     *
     * @param outputPath
     */
    public static void write(@Nonnull CoverageModelParametersNDArray model, @Nonnull final String outputPath) {
        /* create output directory if it doesn't exist */
        createOutputPath(outputPath);

        /* write targets list */
        final File targetListFile = new File(outputPath, TARGET_LIST_OUTPUT_FILE);
        TargetWriter.writeTargetsToFile(targetListFile, model.getTargetList());

        /* write target mean bias to file */
        final File targetMeanBiasFile = new File(outputPath, TARGET_MEAN_BIAS_OUTPUT_FILE);
        Nd4jIOUtils.writeNDArrayToFile(model.getTargetMeanBias(), targetMeanBiasFile);

        /* write target unexplained variance to file */
        final File targetUnexplainedVarianceFile = new File(outputPath, TARGET_UNEXPLAINED_VARIANCE_OUTPUT_FILE);
        Nd4jIOUtils.writeNDArrayToFile(model.getTargetUnexplainedVariance(), targetUnexplainedVarianceFile);

        /* writer principal map to file */
        final File principalLatentToTargetMapFile = new File(outputPath, PRINCIPAL_LATENT_TO_TARGET_MAP_OUTPUT_FILE);
        Nd4jIOUtils.writeNDArrayToFile(model.getPrincipalLatentToTargetMap(), principalLatentToTargetMapFile);
    }

    /**
     * This method "adapts" a model to a read count collection in the following sense:
     *
     *     - remove targets that are not included in the model from the read counts collection
     *     - remove targets that are in the read count collection from the model
     *     - rearrange model targets in the same order as read count collection targets
     *
     * The modifications are not done in-plane and the original input parameters remain intact.
     *
     * @param model a model
     * @param readCounts a read count collection
     * @return a pair of model and read count collection
     */
    public static ImmutablePair<CoverageModelParametersNDArray, ReadCountCollection> adaptModelToReadCountCollection(
            @Nonnull final CoverageModelParametersNDArray model, @Nonnull final ReadCountCollection readCounts,
            @Nonnull final Logger logger) {
        logger.info("Adapting model to read counts...");

        final List<Target> modelTargetList = model.getTargetList();
        final List<Target> readCountsTargetList = readCounts.targets();
        final Set<Target> mutualTargetList = Sets.intersection(new HashSet<>(modelTargetList),
                new HashSet<>(readCountsTargetList));
        final List<Target> finalTargetList = readCountsTargetList.stream()
                .filter(mutualTargetList::contains)
                .collect(Collectors.toList());
        final Set<Target> finalTargetsSet = new LinkedHashSet<>(finalTargetList);

        logger.info("Number of mutual targets: " + finalTargetList.size());
        if (finalTargetList.isEmpty()) {
            throw new UserException.BadInput("The interaction between model targets, and targets from read count" +
                    " collection is empty. Please check there the model is compatible with the given read count" +
                    " collection.");
        }

        if (modelTargetList.size() > finalTargetList.size()) {
            logger.info("The following targets dropped from the model: " + Sets.difference(new HashSet<>(modelTargetList),
                    finalTargetsSet).stream().map(Target::getName).collect(Collectors.joining(", ", "[", "]")));
        }

        if (readCountsTargetList.size() > finalTargetList.size()) {
            logger.info("The following targets dropped from read counts: " + Sets.difference(new HashSet<>(readCountsTargetList),
                    finalTargetsSet).stream().map(Target::getName).collect(Collectors.joining(", ", "[", "]")));
        }

        final ReadCountCollection subsetReadCounts = readCounts.subsetTargets(finalTargetsSet);

        /* fetch original model parameters */
        final INDArray originalModelTargetMeanBias = model.getTargetMeanBias();
        final INDArray originalModelTargetUnexplainedVariance = model.getTargetUnexplainedVariance();
        final INDArray originalModelPrincipalLatentToTargetMap = model.getPrincipalLatentToTargetMap();

        /* reorder */
        final int[] newTargetIndicesInOriginalModel = IntStream.range(0, modelTargetList.size())
                .filter(ti -> finalTargetsSet.contains(modelTargetList.get(ti)))
                .toArray();
        final INDArray newModelTargetMeanBias = Nd4j.create(new int[] {1, finalTargetList.size()});
        final INDArray newModelTargetUnexplainedVariance = Nd4j.create(new int[] {1, finalTargetList.size()});
        final INDArray newModelPrincipalLatentToTargetMap = Nd4j.create(new int[] {finalTargetList.size(),
                model.getNumLatents()});
        IntStream.range(0, finalTargetList.size())
                .forEach(ti -> {
                    newModelTargetMeanBias.put(0, ti,
                            originalModelTargetMeanBias.getDouble(0, newTargetIndicesInOriginalModel[ti]));
                    newModelTargetUnexplainedVariance.put(0, ti,
                            originalModelTargetUnexplainedVariance.getDouble(0, newTargetIndicesInOriginalModel[ti]));
                    newModelPrincipalLatentToTargetMap.get(NDArrayIndex.point(ti), NDArrayIndex.all())
                            .assign(originalModelPrincipalLatentToTargetMap.get(NDArrayIndex.point(newTargetIndicesInOriginalModel[ti]),
                                    NDArrayIndex.all()));
                });

        return ImmutablePair.of(new CoverageModelParametersNDArray(finalTargetList, newModelTargetMeanBias,
                newModelTargetUnexplainedVariance, newModelPrincipalLatentToTargetMap), subsetReadCounts);
    }

}
