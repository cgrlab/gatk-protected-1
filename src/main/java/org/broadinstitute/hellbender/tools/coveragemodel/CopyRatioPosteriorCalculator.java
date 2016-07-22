package org.broadinstitute.hellbender.tools.coveragemodel;

import org.broadinstitute.hellbender.tools.exome.Target;

import javax.annotation.Nonnull;
import java.util.List;

/**
 * Classes that calculate posterior copy ratio expectations must implement this interface
 *
 * @param <D> emission data type
 *
 * @author Mehrtash Babadi &lt;mehrtash@broadinstitute.org&gt;
 */
public interface CopyRatioPosteriorCalculator<D> {

    /**
     * Calculates various posterior quantities for a given list of targets and emission data
     *
     * @param activeTargets list of targets corresponding to the provided list of emission data
     * @param activeTargetIndices indices of active targets in the full genomic-position-sorted list of targets
     * @param emissionData list of emission probability calculation data
     * @param calculateMostLikelyHiddenStates calculate most likely hidden states or not
     * @return posterior results
     */
    CopyRatioPosteriorResults getCopyRatioPosteriorResults(@Nonnull final List<Target> activeTargets,
                                                           @Nonnull final int[] activeTargetIndices,
                                                           @Nonnull final List<D> emissionData,
                                                           final boolean calculateMostLikelyHiddenStates);

    void initializeCaches(@Nonnull final List<Target> allTargets);
}