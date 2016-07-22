package org.broadinstitute.hellbender.tools.coveragemodel;

import org.broadinstitute.hellbender.utils.param.ParamUtils;

import java.io.Serializable;

/**
 *
 * @author Mehrtash Babadi &lt;mehrtash@broadinstitute.org&gt;
 */
public final class LinearSpaceBlock implements Serializable {

    private static final long serialVersionUID = 4571138983754570342L;

    /* begin index inclusive, end index not inclusive */
    private final int begIndex, endIndex, numTargets;

    public LinearSpaceBlock(final int begIndex, final int endIndex) {
        this.begIndex = ParamUtils.isPositiveOrZero(begIndex, "The begin index of a block must be non-negative.");
        this.endIndex = ParamUtils.inRange(endIndex, begIndex + 1, Integer.MAX_VALUE, "The block must at least" +
                " contain one element.");
        numTargets = endIndex - begIndex;
    }

    public int getBegIndex() { return begIndex; }

    public int getEndIndex() { return endIndex; }

    public int getNumTargets() { return numTargets; }

    @Override
    public boolean equals(Object o) {
        if (this == o) {
            return true;
        }
        if (!(o instanceof LinearSpaceBlock)) {
            return false;
        }

        final LinearSpaceBlock block = (LinearSpaceBlock) o;
        return (begIndex == block.begIndex) && (endIndex == block.endIndex);
    }

    /**
     * The best hash code is the begIndex!
     *
     * @return
     */
    @Override
    public int hashCode() {
//        int result = begIndex;
//        result = 31 * result + endIndex;
        return begIndex;
    }

    @Override
    public String toString() {
        return "[" + begIndex + ", " + endIndex + "]";
    }

    public LinearSpaceBlock clone() {
        return new LinearSpaceBlock(begIndex, endIndex);
    }
}
