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

    /**
     *
     */
    public static final double VITERBI_COPY_RATIO_ON_MISSING_TARGETS = 0.0;

    private final int[] activeTargetIndices;
    private final double[] logCopyRatioPosteriorMeans;
    private final double[] logCopyRatioPosteriorVariances;
    private final double[] mostLikelyHiddenStateChain;
    private final int numActiveTargets;

    /**
     *
     * @param activeTargetIndices
     * @param logCopyRatioPosteriorMeans
     * @param logCopyRatioPosteriorVariances
     * @param mostLikelyHiddenStateChain
     */
    public CopyRatioPosteriorResults(@Nonnull final int[] activeTargetIndices,
                                     @Nonnull final double[] logCopyRatioPosteriorMeans,
                                     @Nonnull final double[] logCopyRatioPosteriorVariances,
                                     @Nullable final double[] mostLikelyHiddenStateChain) {
        this.activeTargetIndices = activeTargetIndices.clone();
        this.logCopyRatioPosteriorMeans = logCopyRatioPosteriorMeans.clone();
        this.logCopyRatioPosteriorVariances = logCopyRatioPosteriorVariances.clone();
        if (mostLikelyHiddenStateChain == null) {
            this.mostLikelyHiddenStateChain = null;
        } else {
            this.mostLikelyHiddenStateChain = mostLikelyHiddenStateChain.clone();
        }
        this.numActiveTargets = activeTargetIndices.length;
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
     * @param tb
     * @return
     */
    public double[] getMostLikelyHiddenStateChainOnTargetBlock(@Nonnull final LinearSpaceBlock tb) {
        if (hasMostLikelyHiddenStateChainData()) {
            return fillMissingValuesOnTargetBlock(mostLikelyHiddenStateChain, tb, VITERBI_COPY_RATIO_ON_MISSING_TARGETS);
        } else {
            return null;
        }
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

    /**
     *
     * @return
     */
    public double[] getMostLikelyHiddenStateChainOnActiveTargets() {
        if (mostLikelyHiddenStateChain != null) {
            return mostLikelyHiddenStateChain;
        } else {
            throw new IllegalStateException("The most likely chain of hidden states is not available.");
        }
    }

    /**
     *
     * @return
     */
    public boolean hasMostLikelyHiddenStateChainData() {
        return mostLikelyHiddenStateChain != null;
    }

}
