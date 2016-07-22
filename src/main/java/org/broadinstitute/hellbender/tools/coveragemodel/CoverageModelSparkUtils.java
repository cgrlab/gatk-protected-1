package org.broadinstitute.hellbender.tools.coveragemodel;

import org.apache.commons.lang3.tuple.Pair;
import org.apache.commons.math3.util.FastMath;
import org.apache.spark.HashPartitioner;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.broadinstitute.hellbender.utils.param.ParamUtils;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;
import scala.Tuple2;

import javax.annotation.Nonnull;
import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

/**
 * @author Mehrtash Babadi &lt;mehrtash@broadinstitute.org&gt;
 */
public class CoverageModelSparkUtils {

    /**
     *
     * @param length
     * @param numBlocks
     * @param minBlockSize
     * @return
     */
    public static List<LinearSpaceBlock> createLinearSpaceBlocks(final int length, final int numBlocks,
                                                                 final int minBlockSize) {
        ParamUtils.isPositive(length, "The length of the linear space to be partitioned must be positive");
        ParamUtils.isPositive(numBlocks, "The number of blocks must be positive");
        ParamUtils.isPositive(minBlockSize, "Minimum block size must be positive");

        final int blockSize = FastMath.max(length/numBlocks, minBlockSize);
        final List<LinearSpaceBlock> blocks = new ArrayList<>();
        for (int begIndex = 0; begIndex < length; begIndex += blockSize) {
            blocks.add(new LinearSpaceBlock(begIndex, FastMath.min(begIndex + blockSize, length)));
        }
        /* the last block might be smaller than minBlockSize; we merge them */
        while (blocks.size() > numBlocks && blocks.size() > 1) {
            final int newBegIndex = blocks.get(blocks.size() - 2).getBegIndex();
            final int newEndIndex = blocks.get(blocks.size() - 1).getEndIndex();
            blocks.remove(blocks.size() - 1);
            blocks.remove(blocks.size() - 1);
            blocks.add(new LinearSpaceBlock(newBegIndex, newEndIndex));
        }
        return blocks;
    }

    /**
     *
     * Note: dup() is not needed TODO
     *
     * @param blocks
     * @param arr
     * @return
     */
    public static List<Tuple2<LinearSpaceBlock, INDArray>> chopINDArrayToBlocks(@Nonnull final List<LinearSpaceBlock> blocks,
                                                                                @Nonnull final INDArray arr) {
        return blocks.stream().map(block ->
                new Tuple2<>(block, arr.get(NDArrayIndex.interval(block.getBegIndex(), block.getEndIndex()))))
                .collect(Collectors.toList());
    }

    public static Map<LinearSpaceBlock, INDArray> mapINDArrayToBlocks(@Nonnull final List<LinearSpaceBlock> blocks,
                                                                      @Nonnull final INDArray arr) {
        return blocks.stream().map(block ->
                new Tuple2<>(block, arr.get(NDArrayIndex.interval(block.getBegIndex(), block.getEndIndex()))))
                .collect(Collectors.toMap(p -> p._1, p -> p._2));
    }

    /**
     * Assembles INDArray blocks in a {@code JavaPairRDD<LinearSpaceBlock, INDArray>} by collecting
     * them to a local list, sorting them based on their keys ({@link LinearSpaceBlock}), and
     * concatenating them along a given axis.
     *
     * @param blocksPairRDD an instance of {@code JavaPairRDD<LinearSpaceBlock, INDArray>}
     * @param axis axis to concat along
     * @return an instance of {@link INDArray}
     */
    public static INDArray assembleINDArrayBlocksFromRDD(@Nonnull JavaPairRDD<LinearSpaceBlock, INDArray> blocksPairRDD,
                                                         final int axis) {
        final List<INDArray> sortedBlocks = blocksPairRDD.collect().stream()
                .sorted((Lp, Rp) -> Lp._1.getBegIndex() - Rp._1.getBegIndex())
                .map(p -> p._2)
                .collect(Collectors.toList());
        return Nd4j.concat(axis, sortedBlocks.toArray(new INDArray[sortedBlocks.size()]));
    }

    /**
     * Assemble INDArray blocks in a {@code Collection<? extends Pair<LinearSpaceBlock, INDArray>}
     * by sorting them based on their keys ({@link LinearSpaceBlock}), and concatenating them along
     * a given axis.
     *
     * @param blocksCollection an instance of {@code Collection<? extends Pair<LinearSpaceBlock, INDArray>}
     * @param axis axis to concat along
     * @return an instance of {@link INDArray}
     */
    public static INDArray assembleINDArrayBlocksFromCollection(@Nonnull Collection<? extends Pair<LinearSpaceBlock, INDArray>> blocksCollection,
                                                                final int axis) {
        final List<INDArray> sortedBlocks = blocksCollection.stream()
                .sorted((Lp, Rp) -> Lp.getKey().getBegIndex() - Rp.getKey().getBegIndex())
                .map(Pair<LinearSpaceBlock, INDArray>::getValue)
                .collect(Collectors.toList());
        return Nd4j.concat(axis, sortedBlocks.toArray(new INDArray[sortedBlocks.size()]));
    }
    /**
     *
     * @param arr
     * @param blocks
     * @param ctx
     * @param persist
     * @return
     */
    public static JavaPairRDD<LinearSpaceBlock, INDArray> rddFromINDArray(@Nonnull final INDArray arr,
                                                                          @Nonnull final List<LinearSpaceBlock> blocks,
                                                                          @Nonnull final JavaSparkContext ctx,
                                                                          final boolean persist) {
        JavaPairRDD<LinearSpaceBlock, INDArray> rdd = ctx.parallelizePairs(
                CoverageModelSparkUtils.chopINDArrayToBlocks(blocks, arr), blocks.size())
                .partitionBy(new HashPartitioner(blocks.size()));
        if (persist) {
            rdd.cache();
        }
        return rdd;
    }
}
