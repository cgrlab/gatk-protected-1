package org.broadinstitute.hellbender.tools.coveragemodel;

import org.broadinstitute.hellbender.tools.exome.Target;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.buffer.util.DataTypeUtil;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.io.Serializable;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.stream.IntStream;

/**
 * @author Mehrtash Babadi &lt;mehrtash@broadinstitute.org&gt;
 */
public class CopyRatioPosteriorResults implements Serializable {

    private static final long serialVersionUID = 7573175667965873997L;

    /**
     *
     */
    public static final double MEAN_LOG_COPY_RATIO_ON_MISSING_TARGETS = 0.0;

    /**
     *
     */
    public static final double VAR_LOG_COPY_RATIO_ON_MISSING_TARGETS = 0.0;

    private final int[] activeTargetIndices;
    private final double[] logCopyRatioPosteriorMeans;
    private final double[] logCopyRatioPosteriorVariances;

    /**
     *
     * @param activeTargetIndices
     * @param logCopyRatioPosteriorMeans
     * @param logCopyRatioPosteriorVariances
     */
    public CopyRatioPosteriorResults(@Nonnull final int[] activeTargetIndices,
                                     @Nonnull final double[] logCopyRatioPosteriorMeans,
                                     @Nonnull final double[] logCopyRatioPosteriorVariances) {
        this.activeTargetIndices = activeTargetIndices.clone();
        this.logCopyRatioPosteriorMeans = logCopyRatioPosteriorMeans.clone();
        this.logCopyRatioPosteriorVariances = logCopyRatioPosteriorVariances.clone();
    }

    /**
     *
     * @return
     */
    public int[] getActiveTargetIndices() {
        return activeTargetIndices;
    }

    /**
     *
     * @param tb
     * @return
     */
    public double[] getLogCopyRatioPosteriorMeansOnTargetBlock(@Nonnull final LinearSpaceBlock tb) {
        return fillMissingValuesOnTargetBlock(logCopyRatioPosteriorMeans, tb, MEAN_LOG_COPY_RATIO_ON_MISSING_TARGETS);
    }

    /**
     *
     * @param tb
     * @return
     */
    public double[] getLogCopyRatioPosteriorVariancesOnTargetBlock(@Nonnull final LinearSpaceBlock tb) {
        return fillMissingValuesOnTargetBlock(logCopyRatioPosteriorVariances, tb, VAR_LOG_COPY_RATIO_ON_MISSING_TARGETS);
    }

    /**
     *
     * @param valuesOnActiveTargets
     * @param tb
     * @param missingTargetValuePlaceholder
     * @return
     */
    private double[] fillMissingValuesOnTargetBlock(@Nonnull final double[] valuesOnActiveTargets,
                                                    @Nonnull final LinearSpaceBlock tb,
                                                    final double missingTargetValuePlaceholder) {
        final int begTargetIndex = tb.getBegIndex();
        final int endTargetIndex = tb.getEndIndex();
        final double[] res = IntStream.range(0, tb.getNumTargets())
                .mapToDouble(n -> missingTargetValuePlaceholder).toArray();
        final int[] activeTargetIndicesInBlock = Arrays.stream(activeTargetIndices)
                .filter(ti -> ti >= begTargetIndex && ti < endTargetIndex)
                .toArray();
        if (activeTargetIndicesInBlock.length > 0) {
            final int offset;
            int counter = 0;
            while (activeTargetIndices[counter] < begTargetIndex) {
                counter++;
            }
            offset = counter;
            IntStream.range(0, activeTargetIndicesInBlock.length)
                    .forEach(idx -> res[activeTargetIndicesInBlock[idx] - begTargetIndex] =
                            valuesOnActiveTargets[offset + idx]);
        }
        return res;
    }

    /**
     *
     * @return
     */
    public double[] getLogCopyRatioPosteriorMeansOnActiveTargets() {
        return logCopyRatioPosteriorMeans;
    }

    /**
     *
     * @return
     */
    public double[] getLogCopyRatioPosteriorVariancesOnActiveTargets() {
        return logCopyRatioPosteriorVariances;
    }

}
