package org.broadinstitute.hellbender.tools.coveragemodel.linalg;

import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.broadinstitute.hellbender.utils.test.BaseTest;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.buffer.util.DataTypeUtil;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.inverse.InvertMatrix;
import org.testng.annotations.Test;

import java.util.stream.IntStream;

/**
 * @author Mehrtash Babadi &lt;mehrtash@broadinstitute.org&gt;
 */
public class Nd4jUnitTest extends BaseTest {

//    @Test
//    public void testIndexing() {
//        INDArray arr = Nd4j.linspace(0, 8, 9).reshape(3, 3);
//        INDArray arr2 = Nd4j.create(3, 3);
//        INDArray arr3 = Nd4j.create(3, 3);
//        arr2.assign(arr);
//        arr3.assign(arr2);
//
//        arr.get(
//                NDArrayIndex.interval(1, 2),
//                NDArrayIndex.interval(0, 3)).muli(2);
//
//        arr2.addi(5);
//
//        System.out.println(arr);
//        System.out.println(arr2);
//        System.out.println(arr3);
//    }
//
//    @Test
//    public void testMul() {
//        int N = 100000;
//        INDArray arr = Nd4j.ones(N, 5);
//        INDArray diag = Nd4j.linspace(0, N, N).transpose();
//        INDArray arr2 = arr.mulColumnVector(diag);
//        System.out.println(arr2.get(NDArrayIndex.interval(0, 5), NDArrayIndex.interval(0, 5)));
//        System.out.println(arr2.shapeInfoToString());
//    }
//
//    @Test
//    public void testInv() {
//        int N = 5;
//        INDArray mat = Nd4j.diag(Nd4j.linspace(1, N, N));
//        System.out.println("Original matrix:");
//        System.out.println(mat);
//        System.out.println("Inverted matrix:");
//        System.out.println(InvertMatrix.invert(mat, true));
//    }
//
//    @Test
//    public void testCreate() {
//        int N = 5;
//        INDArray arr = Nd4j.create(N, N);
//        System.out.println(arr);
//    }
//

//    @Test
//    public void testDataBuffer() {
//        INDArray arr = Nd4j.linspace(0, 9, 10).reshape(2,5);
//        System.out.println(arr);
//        System.out.println(Nd4j.create(arr.data().asDouble(), new int[]{2,5}));
//    }

//    @Test
//    public void testVectorSlice() {
//        final INDArray arr = Nd4j.linspace(1, 5, 5);
//        System.out.println(arr.get(NDArrayIndex.interval(0, 2)));
//        System.out.println(arr.transpose().get(NDArrayIndex.interval(0, 2)));
//    }

}
