package org.broadinstitute.hellbender.tools.coveragemodel;

import org.apache.commons.lang3.tuple.ImmutablePair;
import org.apache.commons.math3.linear.*;
import org.apache.commons.math3.util.Precision;
import org.apache.logging.log4j.Logger;
import org.broadinstitute.hellbender.exceptions.UserException;
import org.broadinstitute.hellbender.utils.Utils;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.buffer.util.DataTypeUtil;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;

import javax.annotation.Nonnull;
import java.util.Arrays;

/**
 *
 * @author Mehrtash Babadi &lt;mehrtash@broadinstitute.org&gt;
 */
public class CoverageModelEMWorkspaceNDArrayUtils {

    static {
        Nd4j.create(1);
        DataTypeUtil.setDTypeForContext(DataBuffer.Type.DOUBLE);
    }

    /**
     * INDArray to Apache
     *
     * TODO this must be optimized
     *
     * @param matrix rank-2 INDArray
     * @return Apache matrix
     */
    public static RealMatrix convertINDArrayToApacheMatrix(@Nonnull final INDArray matrix) {
        if (matrix.rank() != 2) {
            throw new IllegalArgumentException("Input rank is not 2 (not matrix)");
        }
        final int[] shape = matrix.shape();
        final BlockRealMatrix out = new BlockRealMatrix(shape[0], shape[1]);
        for (int i=0; i<shape[0]; i++) {
            for (int j=0; j<shape[1]; j++) {
                out.setEntry(i, j, matrix.getDouble(i, j));
            }
        }
        return out;
    }

    /**
     * INDArray to Apache
     *
     * TODO this must be optimized
     *
     * @param vector rank-1 INDArray
     * @return Apache vector
     */
    public static RealVector convertINDArrayToApacheVector(@Nonnull final INDArray vector) {
        if (!vector.isVector()) {
            throw new IllegalArgumentException("Input INDArray is not a vector.");
        }
        final int length = vector.length();
        final RealVector out = new ArrayRealVector(length);
        for (int i=0; i<length; i++) {
            out.setEntry(i, vector.getDouble(i));
        }
        return out;
    }

    /**
     * Apache to INDArray
     *
     * TODO this must be optimized
     *
     * @param matrix Apache matrix
     * @return rank-2 INDArray
     */
    public static INDArray convertApacheMatrixToINDArray(@Nonnull final RealMatrix matrix) {
        final int[] shape = new int[]{matrix.getRowDimension(), matrix.getColumnDimension()};
        final INDArray out = Nd4j.create(shape);
        for (int i=0; i<shape[0]; i++) {
            for (int j=0; j<shape[1]; j++) {
                out.putScalar(new int[]{i,j}, matrix.getEntry(i,j));
            }
        }
        return out;
    }


    /**
     * Inverts a square INDArray matrix using apache commons (LU decomposition)
     * @param mat matrix to be inverted
     * @return inverted matrix
     */
    public static INDArray minv(@Nonnull final INDArray mat) {
        if (mat.isScalar()) {
            return Nd4j.onesLike(mat).divi(mat);
        }
        if (!mat.isSquare()) {
            throw new IllegalArgumentException("Invalid array: must be square matrix");
        }
        final RealMatrix rm = convertINDArrayToApacheMatrix(mat);
        final RealMatrix rmInverse = new LUDecomposition(rm).getSolver().getInverse();
        return convertApacheMatrixToINDArray(rmInverse);
    }

    /**
     * Solves a linear system using apache commons methods
     * @param mat
     * @param vec
     * @return
     */
    public static INDArray linsolve(@Nonnull final INDArray mat, @Nonnull final INDArray vec) {
        if (mat.isScalar()) {
            return vec.div(mat.getDouble(0));
        }
        if (!mat.isSquare()) {
            throw new IllegalArgumentException("invalid array: must be square matrix");
        }
        final RealVector sol = new LUDecomposition(convertINDArrayToApacheMatrix(mat))
                .getSolver().solve(convertINDArrayToApacheVector(vec));
        return Nd4j.create(sol.toArray(), vec.shape());
    }

    /**
     * Return the norm-infinity of {@code arr}
     * @param arr the array to be normed
     * @return norm-infinity
     */
    public static double getINDArrayNormInfinity(@Nonnull final INDArray arr) {
        return Transforms.abs(arr, true).maxNumber().doubleValue();
    }

    /**
     * Takes a square symmetric real matrix [M] and finds an orthogonal transformation [U] such that
     * [U] [M] [U]^T is diagonal, and diagonal entries are sorted in descending order.
     *
     * @param matrix a symmetric matrix
     * @param symmetrize enforce symmetry
     * @param logger a logger instance
     * @return [U]
     */
    public static ImmutablePair<double[], RealMatrix> getOrthogonalizerAndSorterTransformation(@Nonnull final RealMatrix matrix,
                                                                      final boolean symmetrize,
                                                                      @Nonnull final Logger logger) {
        if (matrix.getRowDimension() != matrix.getColumnDimension()) {
            throw new IllegalArgumentException("The input matrix must be square");
        }
        final RealMatrix finalMatrix;
        if (symmetrize) {
            final double symTol = 10 * matrix.getRowDimension() * matrix.getColumnDimension() * Precision.EPSILON;
            if (!MatrixUtils.isSymmetric(matrix, symTol)) {
                logger.info("The input matrix is not symmetric -- enforcing symmetrization");
                finalMatrix = matrix.add(matrix.transpose()).scalarMultiply(0.5);
            } else {
                finalMatrix = matrix;
            }
        } else {
            finalMatrix = matrix;
        }
        final EigenDecomposition decomposer = new EigenDecomposition(finalMatrix);
        final double[] eigs = decomposer.getRealEigenvalues();
        final RealMatrix VT = decomposer.getVT();
        return ImmutablePair.of(eigs, VT);
    }

    /**
     * Same as {@link #getOrthogonalizerAndSorterTransformation(RealMatrix, boolean, Logger)} but with
     * INDArray input/output.
     *
     * @param matrix a symmetric matrix
     * @param symmetrize enforce symmetry
     * @param logger a logger instance
     * @return an INDArray
     */
    public static ImmutablePair<INDArray, INDArray> getOrthogonalizerAndSorterTransformation(@Nonnull final INDArray matrix,
                                                                                             final boolean symmetrize,
                                                                                             @Nonnull final Logger logger) {
        final ImmutablePair<double[], RealMatrix> out = getOrthogonalizerAndSorterTransformation(
                convertINDArrayToApacheMatrix(matrix), symmetrize, logger);
        return ImmutablePair.of(Nd4j.create(out.left, new int[] {1, matrix.shape()[0]}),
                convertApacheMatrixToINDArray(out.right));
    }

    public static boolean hasBadValues(final INDArray arr) {
        return Arrays.stream(arr.data().asDouble()).anyMatch(d -> Double.isNaN(d) || Double.isInfinite(d));
    }

}
