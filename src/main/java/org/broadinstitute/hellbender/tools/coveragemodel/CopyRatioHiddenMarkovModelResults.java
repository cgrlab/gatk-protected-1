package org.broadinstitute.hellbender.tools.coveragemodel;

import org.broadinstitute.hellbender.tools.exome.Target;
import org.broadinstitute.hellbender.tools.exome.TargetCollection;
import org.broadinstitute.hellbender.utils.hmm.ForwardBackwardAlgorithm;

import java.io.Serializable;
import java.util.List;

/**
 * @author Mehrtash Babadi &lt;mehrtash@broadinstitute.org&gt;
 */
public class CopyRatioHiddenMarkovModelResults<D, S> implements Serializable {

    private static final long serialVersionUID = 1891158919985229044L;

    private final TargetCollection<Target> targetCollection;
    private final ForwardBackwardAlgorithm.Result<D, Target, S> fbResult;
    private final List<S> viterbiResult;

    public CopyRatioHiddenMarkovModelResults(final TargetCollection<Target> targetCollection,
                                             final ForwardBackwardAlgorithm.Result<D, Target, S> fbResult,
                                             final List<S> viterbiResult) {
        this.targetCollection = targetCollection;
        this.fbResult = fbResult;
        this.viterbiResult = viterbiResult;
    }

    public TargetCollection<Target> getTargetCollection() {
        return targetCollection;
    }

    public ForwardBackwardAlgorithm.Result<D, Target, S> getForwardBackwardResult() {
        return fbResult;
    }

    public List<S> getViterbiResult() {
        return viterbiResult;
    }

}
