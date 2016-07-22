package org.broadinstitute.hellbender.tools.exome.sexgenotyper;

import com.google.cloud.dataflow.sdk.repackaged.com.google.common.collect.Sets;
import org.broadinstitute.hellbender.exceptions.UserException;
import org.broadinstitute.hellbender.utils.tsv.TableColumnCollection;

import javax.annotation.Nonnull;
import java.io.*;
import java.util.*;
import java.util.function.Function;
import java.util.stream.Collectors;

/**
 * A collection of {@link SexGenotypeData} along with I/O methods.
 *
 * @author Mehrtash Babadi &lt;mehrtash@broadinstitute.org&gt;
 */
public final class SexGenotypeDataCollection {
    private final List<SexGenotypeData> sexGenotypeDataList;
    private final Set<String> sampleNames;
    private final Map<String, SexGenotypeData> sampleNamesToSexGenotypeDataMap;

    /**
     * Public constructor (empty collection).
     */
    public SexGenotypeDataCollection() {
        sexGenotypeDataList = new ArrayList<>();
        sampleNames = new HashSet<>();
        sampleNamesToSexGenotypeDataMap = new HashMap<>();
    }

    /**
     * Public constructor from file.
     *
     * @param sexGenotypeDataFile sex genotype data file
     * @throws IOException if a read error occurs
     */
    public SexGenotypeDataCollection(@Nonnull final File sexGenotypeDataFile) throws IOException {
        this(new FileReader(sexGenotypeDataFile), sexGenotypeDataFile.getAbsolutePath());
    }

    /**
     * Public constructor from abtract reader.
     *
     * @param sexGenotypeDataSourceName a string identifer for the reader
     * @param sexGenotypeDataReader an instance of {@link Reader}
     * @throws IOException if a read error occurs
     */
    public SexGenotypeDataCollection(@Nonnull final Reader sexGenotypeDataReader,
                                     @Nonnull final String sexGenotypeDataSourceName) throws IOException {
        try (final SexGenotypeTableReader reader = new SexGenotypeTableReader(sexGenotypeDataSourceName, sexGenotypeDataReader)) {
            sexGenotypeDataList = reader.stream().collect(Collectors.toList());
        } catch (final IOException ex) {
            throw new UserException.CouldNotReadInputFile("Could not reader sex genotype data from " + sexGenotypeDataSourceName);
        }
        sampleNames = sexGenotypeDataList.stream().map(SexGenotypeData::getSampleName).collect(Collectors.toSet());
        if (sampleNames.size() < sexGenotypeDataList.size()) {
            throw new UserException.BadInput("Each sample must have a unique sex annotation");
        }
        sampleNamesToSexGenotypeDataMap = sexGenotypeDataList.stream()
                .collect(Collectors.toMap(SexGenotypeData::getSampleName, Function.identity()));
    }

    /**
     * Add an instance of {@link SexGenotypeData} to the collection
     *
     * @param sexGenotypeData an instance of {@link SexGenotypeData}
     */
    public void add(@Nonnull final SexGenotypeData sexGenotypeData) {
        sexGenotypeDataList.add(sexGenotypeData);
    }

    /**
     * Write the collection to a writer. If extended genotyping inference data is available, they will
     * be also written as optional columns.
     *
     * @param dataWriter an instance of {@link Writer}
     */
    public void write(@Nonnull final Writer dataWriter) {
        if (sexGenotypeDataList.size() == 0) {
            throw new IllegalStateException("The sex genotype data collection is empty");
        }
        /* check if extended genotyping information is available; first, check for nulls */
        boolean extended = true;
        if (sexGenotypeDataList.stream().filter(dat -> !dat.hasExtendedGenotypingInfo()).count() > 0) {
            extended = false;
        }
        Set<String> commonGenotypes = new HashSet<>();
        /* check if there is a non-empty intersection */
        if (extended) {
            /* pool all genotypes */
            commonGenotypes = new HashSet<>(sexGenotypeDataList.get(0).getSexGenotypesSet());
            for (final SexGenotypeData dat : sexGenotypeDataList) {
                commonGenotypes = Sets.intersection(commonGenotypes, dat.getSexGenotypesSet());
            }
            if (commonGenotypes.size() == 0) {
                extended = false;
            }
        }

        final TableColumnCollection columns;
        if (extended) {
            final List<String> columnNames = new ArrayList<>();
            columnNames.addAll(SexGenotypeTableColumn.MANDATORY_SEX_GENOTYPE_COLUMNS.names());
            columnNames.addAll(commonGenotypes);
            columns = new TableColumnCollection(columnNames);
        } else {
            columns = new TableColumnCollection(SexGenotypeTableColumn.MANDATORY_SEX_GENOTYPE_COLUMNS.names());
        }
        try (final SexGenotypeTableWriter writer = new SexGenotypeTableWriter(dataWriter, columns)) {
            writer.writeAllRecords(sexGenotypeDataList);
        } catch (final IOException e) {
            throw new UserException.CouldNotCreateOutputFile("Could not write sex genotype data", e);
        }
    }

    /**
     * Return the list of contained {@link SexGenotypeData} instances.
     *
     * @return a list of {@link SexGenotypeData}.
     */
    public List<SexGenotypeData> getSexGenotypeDataList() {
        return sexGenotypeDataList;
    }

    /**
     * Returns the {@link SexGenotypeData} for a given sample in the collection
     *
     * @param sampleName sample name string identifier
     * @return the corresponding {@link SexGenotypeData}
     */
    public SexGenotypeData getSampleSexGenotypeData(@Nonnull final String sampleName) {
        if (!sampleNames.contains(sampleName)) {
            throw new UserException.BadInput("Sample name \"" + sampleName + "\" is not in the collection");
        }
        return sampleNamesToSexGenotypeDataMap.get(sampleName);
    }

    /**
     * Equality comparison is based on the pairwise equality of the contained {@link SexGenotypeData}
     * entries.
     *
     * @param obj object to compare with
     * @return boolean
     */
    @Override
    public boolean equals(Object obj) {
        if (this == obj) {
            return true;
        }
        if (!(obj instanceof SexGenotypeDataCollection)) {
            return false;
        }

        final SexGenotypeDataCollection col = (SexGenotypeDataCollection) obj;
        return sexGenotypeDataList.equals(col.getSexGenotypeDataList());
    }

    @Override
    public int hashCode() {
        return sexGenotypeDataList.hashCode();
    }
}
