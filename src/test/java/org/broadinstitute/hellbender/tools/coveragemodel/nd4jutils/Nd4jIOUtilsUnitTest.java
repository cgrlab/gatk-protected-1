package org.broadinstitute.hellbender.tools.coveragemodel.nd4jutils;

import org.broadinstitute.hellbender.utils.test.BaseTest;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.testng.annotations.Test;

import java.io.File;

/**
 * @author Mehrtash Babadi &lt;mehrtash@broadinstitute.org&gt;
 */
public class Nd4jIOUtilsUnitTest extends BaseTest {

    @Test
    public void basicTest() {
        final INDArray arr = Nd4j.rand(1000, 10);
        final File outFile = createTempFile("Nd4j-IO-test", ".nd4j");
        Nd4jIOUtils.writeNDArrayToFile(arr, outFile);
        final INDArray arr2 = Nd4jIOUtils.readNDArrayFromFile(outFile); /* EOFException */
        /* TODO test equality */
    }
}
