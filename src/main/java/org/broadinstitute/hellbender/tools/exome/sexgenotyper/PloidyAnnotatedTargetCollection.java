package org.broadinstitute.hellbender.tools.exome.sexgenotyper;

import com.google.common.collect.Sets;
import htsjdk.samtools.util.Locatable;
import org.broadinstitute.hellbender.exceptions.UserException;
import org.broadinstitute.hellbender.tools.exome.Target;
import org.broadinstitute.hellbender.tools.exome.TargetCollection;
import org.broadinstitute.hellbender.utils.IndexRange;
import org.broadinstitute.hellbender.utils.SimpleInterval;

import javax.annotation.Nonnull;
import java.util.*;
import java.util.function.Function;
import java.util.stream.Collectors;

/**
 * A collection of {@link Target} instances along with helper methods for generating their genotype ploidy
 * annotations on-the-fly using provided contig ploidy annotations.
 *
 * @author Mehrtash Babadi &lt;mehrtash@broadinstitute.org&gt;
 */
public final class PloidyAnnotatedTargetCollection implements TargetCollection<Target> {

    /**
     * Map from targets to their ploidy annotations (based on target contigs)
     */
    private Map<Target, ContigPloidyAnnotation> targetToContigPloidyAnnotationMap;

    /**
     * List of autosomal targets
     */
    private List<Target> autosomalTargetList;

    /**
     * List of allosomal targets
     */
    private List<Target> allosomalTargetList;

    /**
     * List of all targets
     */
    private List<Target> fullTargetList;

    /**
     * Set of all targets
     */
    private Set<Target> fullTargetSet;


    /**
     * Public constructor.
     *
     * @param contigAnnotsList list of contig ploidy annotations.
     * @param targetList list of targets
     */
    public PloidyAnnotatedTargetCollection(@Nonnull final List<ContigPloidyAnnotation> contigAnnotsList,
                                           @Nonnull final List<Target> targetList) {
        performValidityChecks(targetList, contigAnnotsList);

        fullTargetList = Collections.unmodifiableList(targetList);
        fullTargetSet = Collections.unmodifiableSet(new HashSet<>(fullTargetList));

        /* map targets to ploidy annotations */
        final Map<String, ContigPloidyAnnotation> contigNameToContigPloidyAnnotationMap = contigAnnotsList.stream()
                .collect(Collectors.toMap(ContigPloidyAnnotation::getContigName, Function.identity()));
        targetToContigPloidyAnnotationMap = Collections.unmodifiableMap(
                fullTargetList.stream().collect(Collectors.toMap(Function.identity(),
                        target -> contigNameToContigPloidyAnnotationMap.get(target.getContig()))));

        /* autosomal and allosomal target lists */
        autosomalTargetList = Collections.unmodifiableList(fullTargetList.stream()
                .filter(target -> targetToContigPloidyAnnotationMap.get(target).getContigClass() == ContigClass.AUTOSOMAL)
                .collect(Collectors.toList()));

        allosomalTargetList = Collections.unmodifiableList(fullTargetList.stream()
                .filter(target -> targetToContigPloidyAnnotationMap.get(target).getContigClass() == ContigClass.ALLOSOMAL)
                .collect(Collectors.toList()));
    }

    /**
     * Returns an unmodifiable list of contained autosomal targets
     * @return unmodifiable list of contained autosomal targets
     */
    public List<Target> getAutosomalTargetList() {
        return Collections.unmodifiableList(autosomalTargetList);
    }

    /**
     * Returns an unmodifiable list of contained allosomal targets
     * @return unmodifiable list of contained allosomal targets
     */
    public List<Target> getAllosomalTargetList() {
        return Collections.unmodifiableList(allosomalTargetList);
    }

    /**
     * Retruns the ploidy of a target for a ploidy tag (= genotype identifer string)
     *
     * @param target target in question
     * @param ploidyTag ploidy tag (= genotype identifer string)
     * @return integer ploidy
     */
    public int getTargetPloidyByTag(@Nonnull final Target target, @Nonnull final String ploidyTag) {
        if (!fullTargetSet.contains(target)) {
            throw new IllegalArgumentException("Target \"" + target.getName() + "\" can not be found");
        }
        return targetToContigPloidyAnnotationMap.get(target).getPloidy(ploidyTag);
    }

    /**
     * Perform a number of checks on the arguments passed to the constructor:
     * <dl>
     *     <dt> Assert both lists are non-empty </dt>
     *     <dt> Assert targets have unique names </dt>
     *     <dt> Assert each contig is annotated only once </dt>
     *     <dt> Assert all targets have annotated contigs </dt>
     * </dl>
     *
     * @param targetList list of targets
     * @param contigAnnotsList list of contig ploidy annotations
     */
    private void performValidityChecks(final List<Target> targetList, final List<ContigPloidyAnnotation> contigAnnotsList) {
        /* assert the lists and non-empty */
        if (targetList.size() == 0) {
            throw new UserException.BadInput("Target list can not be empty");
        }
        if (contigAnnotsList.size() == 0) {
            throw new UserException.BadInput("Contig ploidy annotation list can not be empty");
        }

        /* assert targets have unique names */
        final Map<String, Long> targetNameCounts = targetList.stream()
                .map(Target::getName)
                .collect(Collectors.groupingBy(Function.identity(), Collectors.counting()));
        if (targetNameCounts.keySet().size() < targetList.size()) {
            throw new UserException.BadInput("Targets must have unique names. Non-unique target names: " +
                    targetNameCounts.keySet().stream()
                            .filter(name -> targetNameCounts.get(name) > 1)
                            .collect(Collectors.joining(", ")));
        }

        /* assert annotated contigs are unique */
        final Map<String, Long> contigAnnotsCounts = contigAnnotsList.stream()
                .map(ContigPloidyAnnotation::getContigName)
                .collect(Collectors.groupingBy(Function.identity(), Collectors.counting()));
        if (contigAnnotsCounts.keySet().size() < contigAnnotsList.size()) {
            throw new UserException.BadInput("Some contigs are multiply annotated: " +
                    contigAnnotsCounts.keySet().stream()
                            .filter(contig -> contigAnnotsCounts.get(contig) > 1) /* multiply annotated contigs */
                            .collect(Collectors.joining(", ")));
        }

        /* assert all contigs present in the target list are annotated */
        final Set<String> contigNamesFromTargets = targetList.stream()
                .map(Target::getContig).collect(Collectors.toSet());
        final Set<String> contigNamesFromAnnots = contigAnnotsList.stream()
                .map(ContigPloidyAnnotation::getContigName).collect(Collectors.toSet());
        final Set<String> missingContigs = Sets.difference(contigNamesFromTargets, contigNamesFromAnnots);
        if (missingContigs.size() > 0) {
            throw new UserException.BadInput("All contigs must be annotated. Annotations are missing for: " +
                    missingContigs.stream().collect(Collectors.joining(", ")));
        }

        /* assert all contigs have annotations for all ploidy classes */
        final Set<String> firstAnnotPloidyTagSet = contigAnnotsList.get(0).getGenotypesSet();
        if (contigAnnotsList.stream().filter(annot -> !annot.getGenotypesSet().equals(firstAnnotPloidyTagSet)).count() > 0) {
            throw new UserException.BadInput("Not all entries in the contig ploidy annotation list have the same " +
                    "ploidy tag set");
        }
    }

    @Override
    public int targetCount() {
        return fullTargetList.size();
    }

    @Override
    public Target target(int index) {
        return fullTargetList.get(index);
    }

    @Override
    public String name(Target target) {
        return target.getName();
    }

    @Override
    public SimpleInterval location(int index) {
        return location(target(index));
    }

    @Override
    public SimpleInterval location(Target target) {
        return new SimpleInterval(target.getContig(), target.getStart(), target.getEnd());
    }

    @Override
    public List<Target> targets() {
        return Collections.unmodifiableList(fullTargetList);
    }


    /**
     * Not implemented yet (currently is not required in the use case of this class)
     */
    @Override
    public int index(String name) {
        throw new UnsupportedOperationException();
    }

    /**
     * Not implemented yet (currently is not required in the use case of this class)
     */
    @Override
    public IndexRange indexRange(Locatable location) {
        throw new UnsupportedOperationException();
    }

}
