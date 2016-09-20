package org.broadinstitute.hellbender.tools.coveragemodel.interfaces;

import org.broadinstitute.hellbender.tools.coveragemodel.CopyRatioHiddenMarkovModelResults;
import org.broadinstitute.hellbender.tools.coveragemodel.CopyRatioPosteriorResults;
import org.broadinstitute.hellbender.tools.exome.Target;

import javax.annotation.Nonnull;
import java.util.List;

/**
 * Classes that calculate posterior copy ratio expectations must implement this interface
 *
 * @param <D> emission data type
 * @param <S> hidden state type
 *
 * @author Mehrtash Babadi &lt;mehrtash@broadinstitute.org&gt;
 */
public interface CopyRatioPosteriorCalculator<D, S> {

    /**
     * Calculates various posterior quantities for a given list of targets and emission data
     *
     * @param activeTargets list of targets corresponding to the provided list of emission data
     * @param activeTargetIndices indices of active targets in the full genomic-position-sorted list of targets
     * @param emissionData list of emission probability calculation data
     * @return posterior results
     */
    CopyRatioPosteriorResults getCopyRatioPosteriorResults(@Nonnull final List<Target> activeTargets,
                                                           @Nonnull final int[] activeTargetIndices,
                                                           @Nonnull final List<D> emissionData);

    CopyRatioHiddenMarkovModelResults<D, S> getCopyRatioHiddenMarkovModelResults(@Nonnull final List<Target> activeTargets,
                                                                                 @Nonnull final List<D> emissionData);

    void initializeCaches(@Nonnull final List<Target> allTargets);
}