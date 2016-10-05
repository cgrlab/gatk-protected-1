package org.broadinstitute.hellbender.tools.coveragemodel;

import org.apache.commons.math3.util.FastMath;
import org.broadinstitute.hellbender.exceptions.UserException;
import org.broadinstitute.hellbender.tools.coveragemodel.interfaces.CopyRatioPosteriorCalculator;
import org.broadinstitute.hellbender.tools.exome.HashedListTargetCollection;
import org.broadinstitute.hellbender.tools.exome.Target;
import org.broadinstitute.hellbender.tools.exome.germlinehmm.CopyNumberTriState;
import org.broadinstitute.hellbender.tools.exome.germlinehmm.CopyNumberTriStateHiddenMarkovModel;
import org.broadinstitute.hellbender.tools.exome.germlinehmm.CopyNumberTriStateTransitionProbabilityCache;
import org.broadinstitute.hellbender.utils.hmm.ForwardBackwardAlgorithm;
import org.broadinstitute.hellbender.utils.hmm.ViterbiAlgorithm;

import javax.annotation.Nonnull;
import java.io.Serializable;
import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

/**
 * @author Mehrtash Babadi &lt;mehrtash@broadinstitute.org&gt;
 */
public class CoverageModelGermlineCopyNumberPosteriorCalculator implements
        CopyRatioPosteriorCalculator<CoverageModelCopyRatioEmissionData, CopyNumberTriState>, Serializable {

    private static final long serialVersionUID = -7155566973157829680L;

    private final CopyNumberTriStateHiddenMarkovModel<CoverageModelCopyRatioEmissionData> triStateHMM;

    private final static boolean CHECK_FOR_NANS = true;

    /**
     *
     * @param eventStartProbability
     * @param eventMeanSize
     */
    public CoverageModelGermlineCopyNumberPosteriorCalculator(final double eventStartProbability,
                                                              final double eventMeanSize) {
        final CoverageModelCopyRatioEmissionProbabilityCalculator emissionProbabilityCalculator =
                new CoverageModelCopyRatioEmissionProbabilityCalculator();
        triStateHMM = new CopyNumberTriStateHiddenMarkovModel<>(emissionProbabilityCalculator,
                eventStartProbability, eventMeanSize);
    }

    /**
     *
     * @param activeTargets list of targets corresponding to the provided list of emission data
     * @param emissionData list of emission probability calculation data
     * @return
     */
    @Override
    public CopyRatioPosteriorResults getCopyRatioPosteriorResults(
            @Nonnull final List<Target> activeTargets,
            @Nonnull final int[] activeTargetIndices,
            @Nonnull final List<CoverageModelCopyRatioEmissionData> emissionData) {
        verifyArgs(activeTargets, emissionData);

        /* run the forward-backward algorithm */
        final ForwardBackwardAlgorithm.Result<CoverageModelCopyRatioEmissionData, Target, CopyNumberTriState> result =
                ForwardBackwardAlgorithm.apply(emissionData, activeTargets, triStateHMM);

        final List<CopyNumberTriState> hiddenStates = triStateHMM.hiddenStates();
        final double[] hiddenStatesLogCopyRatios = hiddenStates.stream()
                .mapToDouble(s -> FastMath.log(s.copyRatio)).toArray();
        final double[] hiddenStatesLogCopyRatiosSquared = Arrays.stream(hiddenStatesLogCopyRatios)
                .map(d -> d * d).toArray();

        final List<double[]> hiddenStateProbabilities = IntStream.range(0, activeTargets.size())
                .mapToObj(ti -> triStateHMM.hiddenStates().stream()
                        .mapToDouble(s -> FastMath.exp(result.logProbability(ti, s)))
                        .toArray())
                .collect(Collectors.toList());

        if (CHECK_FOR_NANS) {
            final int[] badTargets = IntStream.range(0, activeTargets.size())
                    .filter(ti -> Arrays.stream(hiddenStateProbabilities.get(ti))
                            .anyMatch(p -> Double.isNaN(p) || Double.isInfinite(p))).toArray();
            if (badTargets.length > 0) {
                throw new RuntimeException("Some of the posterior probabilities are ill-defined; targets: " +
                    Arrays.stream(badTargets).mapToObj(String::valueOf).collect(Collectors.joining(", ", "[", "]")));
            }
        }

        final double[] logCopyRatioPosteriorMeans = hiddenStateProbabilities.stream()
                .mapToDouble(prob ->
                        prob[0] * hiddenStatesLogCopyRatios[0] +
                                prob[1] * hiddenStatesLogCopyRatios[1] +
                                prob[2] * hiddenStatesLogCopyRatios[2])
                .toArray();

        final double[] logCopyRatioPosteriorVariances = IntStream.range(0, activeTargets.size())
                .mapToDouble(ti -> {
                    final double[] prob = hiddenStateProbabilities.get(ti);
                    final double logCopyRatioSquaredExpectation =
                            prob[0] * hiddenStatesLogCopyRatiosSquared[0] +
                                    prob[1] * hiddenStatesLogCopyRatiosSquared[1] +
                                    prob[2] * hiddenStatesLogCopyRatiosSquared[2];
                    return logCopyRatioSquaredExpectation - logCopyRatioPosteriorMeans[ti] *
                            logCopyRatioPosteriorMeans[ti];
                }).toArray();

        return new CopyRatioPosteriorResults(activeTargetIndices, logCopyRatioPosteriorMeans, logCopyRatioPosteriorVariances);
    }

    @Override
    public CopyRatioHiddenMarkovModelResults<CoverageModelCopyRatioEmissionData,
            CopyNumberTriState> getCopyRatioHiddenMarkovModelResults(@Nonnull final List<Target> activeTargets,
                                                                     @Nonnull final List<CoverageModelCopyRatioEmissionData> emissionData) {
        final ForwardBackwardAlgorithm.Result<CoverageModelCopyRatioEmissionData, Target, CopyNumberTriState> fbResult =
                ForwardBackwardAlgorithm.apply(emissionData, activeTargets, triStateHMM);
        final List<CopyNumberTriState> viterbiResult = ViterbiAlgorithm.apply(emissionData, activeTargets, triStateHMM);
        return new CopyRatioHiddenMarkovModelResults<>(activeTargets, fbResult, viterbiResult);
    }


    /**
     * High level verification of input data
     *
     * @param activeTargets
     * @param data
     */
    private void verifyArgs(@Nonnull List<Target> activeTargets,
                            @Nonnull List<CoverageModelCopyRatioEmissionData> data) {
        if (activeTargets.size() < 2) {
            throw new UserException.BadInput("At least two active targets are required");
        }
        if (data.size() < 2) {
            throw new UserException.BadInput("At least two emission data entries are required");
        }
        if (activeTargets.size() != data.size()) {
            throw new UserException.BadInput(String.format("Number of active targets (%d) does not must match the length" +
                    " of emission data list (%d)", activeTargets.size(), data.size()));
        }
        if (activeTargets.stream().filter(t -> t == null).count() > 0) {
            throw new UserException.BadInput("Some of the active targets are null");
        }
        if (data.stream().filter(d -> d == null).count() > 0) {
            throw new UserException.BadInput("Some of the emission data entries are null");
        }

        activeTargets.stream()
                .collect(Collectors.groupingBy(Target::getContig)) /* group by contig */
                .values().stream() /* iterate through the target list of each contig */
                .forEach(targetList -> {
                    if (IntStream.range(0, targetList.size() - 1)
                            .filter(i -> targetList.get(i + 1).getStart() - targetList.get(i).getEnd() < 0).count() > 0) {
                        throw new UserException.BadInput("The list of active targets must be coordinate sorted and non-overlapping" +
                                " (except for their endpoints)");
                    }
                });
    }

    /**
     * Pre-compute transition matrices on a foreseeable list of subsequent targets
     * @param allTargets list of subsequent targets
     */
    @Override
    public void initializeCaches(@Nonnull final List<Target> allTargets) {
        final CopyNumberTriStateTransitionProbabilityCache cache = triStateHMM.getLogTransitionProbabilityCache();
        IntStream.range(0, allTargets.size() - 1)
                .map(i -> (int)CopyNumberTriStateHiddenMarkovModel.calculateDistance(allTargets.get(i), allTargets.get(i + 1)))
                .forEach(cache::cacheLogTransitionMatrix);
    }
}
