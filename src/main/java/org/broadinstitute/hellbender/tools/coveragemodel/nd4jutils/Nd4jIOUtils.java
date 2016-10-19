package org.broadinstitute.hellbender.tools.coveragemodel.nd4jutils;

import org.apache.commons.io.input.ReaderInputStream;
import org.apache.commons.io.output.WriterOutputStream;
import org.broadinstitute.hellbender.exceptions.UserException;
import org.broadinstitute.hellbender.utils.tsv.DataLine;
import org.broadinstitute.hellbender.utils.tsv.TableColumnCollection;
import org.broadinstitute.hellbender.utils.tsv.TableReader;
import org.broadinstitute.hellbender.utils.tsv.TableWriter;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.buffer.util.DataTypeUtil;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.shape.Shape;
import org.nd4j.linalg.factory.NDArrayFactory;
import org.nd4j.linalg.factory.Nd4j;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.io.*;
import java.nio.charset.Charset;
import java.util.Arrays;
import java.util.LinkedList;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

import static org.nd4j.linalg.factory.Nd4j.createArrayFromShapeBuffer;

/**
 * @author Mehrtash Babadi &lt;mehrtash@broadinstitute.org&gt;
 */
public class Nd4jIOUtils {

    /**
     * Write NDArray to a file.
     *
     * @param arr
     * @param outputFile
     */
    public static void writeNDArrayToBinaryDumpFile(@Nonnull final INDArray arr,
                                                    @Nonnull final File outputFile) {
        try (final DataOutputStream dos = new DataOutputStream(new FileOutputStream(outputFile))) {
            Nd4j.write(arr, dos);
        } catch (final IOException ex) {
            throw new UserException.CouldNotCreateOutputFile(outputFile, "Could not write the NDArray");
        }
    }


    /**
     * Read NDArray from a file.
     *
     * @param inputFile
     * @return
     */
    public static INDArray readNDArrayFromBinaryDumpFile(@Nonnull final File inputFile) {
        try (final DataInputStream dis = new DataInputStream(new FileInputStream(inputFile))) {
            return Nd4j.read(dis);
        } catch (final IOException ex) {
            throw new UserException.CouldNotReadInputFile(inputFile, "Could not read the input file");
        }
    }

    public static void writeNDArrayToTextFile(@Nonnull final INDArray arr,
                                              @Nonnull final File outputFile,
                                              @Nullable final List<String> rowNames,
                                              @Nullable final List<String> columnNames) {
        if (arr.rank() > 2) {
            throw new IllegalArgumentException("At the moment, only rank-1 and rank-2 NDArray objects can be saved");
        }
        final int rowDimension = arr.shape()[0];
        final int colDimension = arr.shape()[1];

        final TableColumnCollection columnNameCollection;
        if (columnNames == null) {
            final LinkedList<String> columnNameList = new LinkedList<>(IntStream.range(0, colDimension)
                    .mapToObj(colIndex -> String.format("COL_%d", colIndex))
                    .collect(Collectors.toList()));
            columnNameList.addFirst("ROW_NAME");
            columnNameCollection = new TableColumnCollection(columnNameList);
        } else {
            if (columnNames.size() != colDimension) {
                throw new IllegalArgumentException("The length of column name list does not match the column dimension" +
                        " of the provided NDArray");
            } else {
                final LinkedList<String> columnNameList = new LinkedList<>(columnNames);
                columnNameList.addFirst("ROW_NAME");
                columnNameCollection = new TableColumnCollection(columnNameList);
            }
        }

        final List<String> rowNamesList;
        if (rowNames == null) {
            rowNamesList = IntStream.range(0, rowDimension)
                    .mapToObj(colIndex -> String.format("ROW_%d", colIndex))
                    .collect(Collectors.toList());
        } else {
            if (rowNames.size() != rowDimension) {
                throw new IllegalArgumentException("The length of row name string array does not match the column dimension" +
                        " of the provided NDArray");
            } else {
                rowNamesList = rowNames;
            }
        }

        try (final TableWriter<DoubleVectorRow> arrayWriter = new TableWriter<DoubleVectorRow>(outputFile, columnNameCollection) {
            @Override
            protected void composeLine(DoubleVectorRow record, DataLine dataLine) {
                record.composeDataLine(dataLine);
            }}) {
            for (int ri = 0; ri < rowDimension; ri++) {
                arrayWriter.writeRecord(new DoubleVectorRow(rowNamesList.get(ri), arr.getRow(ri).dup().data().asDouble()));
            }
        } catch (final IOException ex) {
            throw new UserException.CouldNotCreateOutputFile(outputFile, "Could not write the array");
        }
    }


    public static INDArray readNDArrayFromTextFile(@Nonnull final File inputFile) {
        final List<INDArray> rows = new LinkedList<>();
        try (final TableReader<DoubleVectorRow> arrayReader = new TableReader<DoubleVectorRow>(inputFile) {
            @Override
            protected DoubleVectorRow createRecord(DataLine dataLine) {
                final int colDimension = columns().columnCount() - 1;
                final String[] dataLineToString = dataLine.toArray();
                if (dataLineToString.length != colDimension + 1) {
                    throw new UserException.BadInput("The input NDArray tsv file is malformed");
                } else {
                    final double[] rowData = Arrays.stream(dataLineToString, 1, colDimension + 1)
                            .mapToDouble(Double::new).toArray();
                    return new DoubleVectorRow(dataLineToString[0], rowData);
                }
            }
        }) {
            final int colDimension = arrayReader.columns().columnCount() - 1;
            arrayReader.iterator().forEachRemaining(rowData -> rows.add(Nd4j.create(rowData.data, new int[] {1, colDimension})));
        } catch (final IOException ex) {
            throw new UserException.CouldNotReadInputFile(inputFile, "Could not read NDArray tsv file");
        }
        return Nd4j.vstack(rows);
    }


    private static final class DoubleVectorRow {
        final String rowName;
        final double[] data;

        DoubleVectorRow(final String rowName, final double[] data) {
            this.rowName = rowName;
            this.data = data;
        }

        public void composeDataLine(final DataLine dataLine) {
            dataLine.append(rowName);
            dataLine.append(data);
        }
    }

}
