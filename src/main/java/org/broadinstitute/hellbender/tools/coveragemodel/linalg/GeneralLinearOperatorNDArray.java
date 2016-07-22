package org.broadinstitute.hellbender.tools.coveragemodel.linalg;

import org.apache.commons.math3.exception.DimensionMismatchException;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.buffer.util.DataTypeUtil;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import javax.annotation.Nonnull;

/**
 * @author Mehrtash Babadi &lt;mehrtash@broadinstitute.org&gt;
 */
public class GeneralLinearOperatorNDArray extends GeneralLinearOperator<INDArray> {

    private final INDArray mat;

    public GeneralLinearOperatorNDArray(@Nonnull final INDArray mat) {
        /* Set Nd4j DType to Double in context */
        DataTypeUtil.setDTypeForContext(DataBuffer.Type.DOUBLE);
        Nd4j.create(1);

        if (mat.rank() != 2)
            throw new IllegalArgumentException("The provided INDArray must be rank-2.");
        this.mat = mat;
    }

    @Override
    public int getRowDimension() {
        return mat.shape()[0];
    }

    @Override
    public int getColumnDimension() {
        return mat.shape()[1];
    }

    @Override
    public INDArray operate(@Nonnull final INDArray x) throws DimensionMismatchException {
        return mat.mmul(x);
    }

    @Override
    public INDArray operateTranspose(@Nonnull final  INDArray x)
            throws DimensionMismatchException, UnsupportedOperationException {
        return mat.transpose().mmul(x);
    }

    @Override
    public boolean isTransposable() {
        return true;
    }
}
