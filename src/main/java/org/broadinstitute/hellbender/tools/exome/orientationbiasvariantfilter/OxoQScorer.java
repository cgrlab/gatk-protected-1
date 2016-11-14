package org.broadinstitute.hellbender.tools.exome.orientationbiasvariantfilter;

import org.apache.commons.lang3.tuple.Pair;
import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.RealMatrix;
import org.broadinstitute.hellbender.tools.picard.analysis.artifacts.SequencingArtifactMetrics;
import org.broadinstitute.hellbender.utils.Utils;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

//TODO: docs
public class OxoQScorer {

    // rows {PRO, CON} x cols {ref, alt}
    final static int PRO = 0;
    final static int CON = 1;
    final static int REF = 0;
    final static int ALT = 1;

    /**
     * Gets error rate collapsed over contexts.
     *
     * rows {PRO, CON} x cols {ref, alt}
     * TODO: Finish docs
     * TODO: What to do about more than one sample?
     * @param metrics
     * @return
     */
    public static Map<Pair<Character, Character>, RealMatrix> countOrientationBiasMetricsOverContext(final List<SequencingArtifactMetrics.PreAdapterDetailMetrics> metrics) {
        Utils.nonNull(metrics, "Input metrics cannot be null");

        // Artifact mode to a matrix
        final Map<Pair<Character, Character>, RealMatrix> result = new HashMap<>();

        // Collapse over context
        for (SequencingArtifactMetrics.PreAdapterDetailMetrics metric : metrics) {
            final Pair<Character, Character> key = Pair.of(metric.REF_BASE, metric.ALT_BASE);
            result.putIfAbsent(key, new Array2DRowRealMatrix(2, 2));
            result.get(key).addToEntry(OxoQScorer.PRO, OxoQScorer.ALT, metric.PRO_ALT_BASES);
            result.get(key).addToEntry(OxoQScorer.CON, OxoQScorer.ALT, metric.CON_ALT_BASES);
            result.get(key).addToEntry(OxoQScorer.PRO, OxoQScorer.REF, metric.PRO_REF_BASES);
            result.get(key).addToEntry(OxoQScorer.CON, OxoQScorer.REF, metric.CON_REF_BASES);
        }

        return result;
    }

    // TODO: docs
    public static Map<Pair<Character, Character>, Double> scoreOrientationBiasMetrics(final List<SequencingArtifactMetrics.PreAdapterDetailMetrics> metrics) {
        Utils.nonNull(metrics, "Input metrics cannot be null");

        // Artifact mode to a double
        final Map<Pair<Character, Character>, Double> result = new HashMap<>();

        final Map<Pair<Character, Character>, RealMatrix> counts = countOrientationBiasMetricsOverContext(metrics);

        for (Pair<Character, Character> artifactMode : counts.keySet()) {
            final RealMatrix count = counts.get(artifactMode);
            final double totalBases = count.getEntry(PRO, REF) + count.getEntry(PRO, ALT) +
                    count.getEntry(CON, REF) + count.getEntry(CON, ALT);
            final double score = -10 * Math.log10(Math.max(count.getEntry(PRO, ALT)/totalBases -
                    count.getEntry(CON, ALT)/totalBases, Math.pow(10, -10)));
            result.put(artifactMode, score);
        }

        return result;
    }

    // Do not allow instantiation of this class
    private OxoQScorer() {};
}
