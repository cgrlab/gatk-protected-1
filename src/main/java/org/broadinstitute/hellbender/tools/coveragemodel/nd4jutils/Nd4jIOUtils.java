package org.broadinstitute.hellbender.tools.coveragemodel.nd4jutils;

import org.apache.commons.io.input.ReaderInputStream;
import org.apache.commons.io.output.WriterOutputStream;
import org.broadinstitute.hellbender.exceptions.UserException;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.buffer.util.DataTypeUtil;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.shape.Shape;
import org.nd4j.linalg.factory.Nd4j;

import javax.annotation.Nonnull;
import java.io.*;
import java.nio.charset.Charset;

import static org.nd4j.linalg.factory.Nd4j.createArrayFromShapeBuffer;

/**
 * @author Mehrtash Babadi &lt;mehrtash@broadinstitute.org&gt;
 */
public class Nd4jIOUtils {

    /**
     * Write NDArray to a writer.
     *
     * @param arr
     * @param writer
     * @param writerName
     */
    public static void writeNDArrayToWriter(@Nonnull final INDArray arr,
                                            @Nonnull final Writer writer,
                                            @Nonnull final String writerName) {
        try (final DataOutputStream dos = new DataOutputStream(new WriterOutputStream(writer))) {
            Nd4j.write(arr, dos);
        } catch (final IOException ex) {
            throw new UserException.CouldNotCreateOutputFile(writerName, "Could not write the NDArray", ex);
        }
    }

    /**
     * Write NDArray to a file.
     *
     * @param arr
     * @param outputFile
     */
    public static void writeNDArrayToFile(@Nonnull final INDArray arr,
                                          @Nonnull final File outputFile) {
        try (final DataOutputStream dos = new DataOutputStream(new FileOutputStream(outputFile))) {
            Nd4j.write(arr, dos);
        } catch (final IOException ex) {
            throw new UserException.CouldNotCreateOutputFile(outputFile, "Could not write the NDArray");
        }
    }

    /**
     * Read NDArray from a reader.
     *
     * @param reader
     * @param readerName
     * @return
     */
    public static INDArray readNDArrayFromReader(@Nonnull final Reader reader, @Nonnull final String readerName) {
        try (final DataInputStream dis = new DataInputStream(new ReaderInputStream(reader, "UTF-16"))) {
            return Nd4j.read(dis);
        } catch (final IOException ex) {
            throw new UserException.CouldNotReadInputFile(readerName, "Could not read INDArray");
        }
    }

    /**
     * Read NDArray from a file.
     *
     * @param inputFile
     * @return
     */
    public static INDArray readNDArrayFromFile(@Nonnull final File inputFile) {
        try (final DataInputStream dis = new DataInputStream(new FileInputStream(inputFile))) {
            return Nd4j.read(dis);
        } catch (final IOException ex) {
            throw new UserException.CouldNotReadInputFile(inputFile, "Could not read the input file");
        }
    }

}
