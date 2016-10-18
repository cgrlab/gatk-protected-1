package org.broadinstitute.hellbender.tools.coveragemodel.math;

/**
 * @author Mehrtash Babadi &lt;mehrtash@broadinstitute.org&gt;
 */
public class UnivariateSolverDescription {

    private final double absoluteAccuracy, relativeAccuracy, functionValueAccuracy;

    public UnivariateSolverDescription(final double absoluteAccuracy, final double relativeAccuracy,
                                       final double functionValueAccuracy) {
        this.absoluteAccuracy = absoluteAccuracy;
        this.relativeAccuracy = relativeAccuracy;
        this.functionValueAccuracy = functionValueAccuracy;
    }

    public double getAbsoluteAccuracy() {
        return absoluteAccuracy;
    }

    public double getRelativeAccuracy() {
        return relativeAccuracy;
    }

    public double getFunctionValueAccuracy() {
        return functionValueAccuracy;
    }
}
