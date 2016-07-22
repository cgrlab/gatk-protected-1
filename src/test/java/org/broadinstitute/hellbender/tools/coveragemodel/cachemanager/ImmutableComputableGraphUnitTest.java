package org.broadinstitute.hellbender.tools.coveragemodel.cachemanager;

import org.broadinstitute.hellbender.utils.test.BaseTest;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.testng.annotations.Test;

import java.util.Map;
import java.util.function.Function;

/**
 * @author Mehrtash Babadi &lt;mehrtash@broadinstitute.org&gt;
 */
public class ImmutableComputableGraphUnitTest extends BaseTest {

    public static Function<Map<String, ? extends Duplicable>, ? extends Duplicable> func_0 = col -> {
        final INDArray x = DuplicableNDArray.of(col.get("X"));
        final double y = DuplicableNumber.of(col.get("Y"));
        return new DuplicableNDArray(x.mul(y));
    };

    public static Function<Map<String, ? extends Duplicable>, ? extends Duplicable> func_1 = col -> {
        final INDArray xProdY = DuplicableNDArray.of(col.get("X_prod_Y"));
        final double y = DuplicableNumber.of(col.get("Y"));
        return new DuplicableNDArray(xProdY.add(y));
    };

    @Test
    public void testCyclicGraph() {
        throw new AssertionError();
    }

    @Test
    public void testOutdatedCaches() {
        throw new AssertionError();
    }

    @Test
    public void testUpToDateCaches() {
        throw new AssertionError();
    }

    @Test
    public void testComputeOnDemandNodes() {
        throw new AssertionError();
    }

    @Test
    public void testPrimitiveUpdating() {
        throw new AssertionError();
    }

    @Test
    public void testExternallyComputableUpdating() {
        throw new AssertionError();
    }

    @Test
    public void testCacheByTag() {
        throw new AssertionError();
    }

    @Test
    public void testCacheByNode() {
        throw new AssertionError();
    }

    @Test
    public void testCacheAutoUpdate() {
        throw new AssertionError();
    }

    @Test
    public void testUnchangedNodesSameReferenceAfterUpdate() {
        throw new AssertionError();
    }

    @Test
    public void test() {

        ImmutableComputableGraph col = ImmutableComputableGraph.builder()
                .addPrimitiveNode("X", new String[]{}, new DuplicableNDArray(null))
                .addPrimitiveNode("Y", new String[]{}, new DuplicableNumber<>(5.0))
                .addComputableNode("X_prod_Y", new String[]{}, new String[]{"X", "Y"}, func_0, true)
                .addComputableNode("Z", new String[]{"calc_Z"}, new String[]{"X_prod_Y", "Y"}, func_1, true)
                .build();

        System.out.println(col.graphToString());

        System.out.println();
        System.out.println();
        System.out.println();

        System.out.println(col.statusToString(true));

        System.out.println();
        System.out.println();
        System.out.println();

        System.out.println("Setting X to [1, 1]");
        col = col.setValue("X", new DuplicableNDArray(Nd4j.ones(2)));

        System.out.println();
        System.out.println();
        System.out.println();

        System.out.println(col.statusToString(true));

        System.out.println();
        System.out.println();
        System.out.println();

        System.out.println("Updating caches");
        col = col.updateAllCaches();

        System.out.println();
        System.out.println();
        System.out.println();

        System.out.println(col.statusToString(true));

        System.out.println();
        System.out.println();
        System.out.println();

        System.out.println("Setting X to [4, 4, 4, 4, 4, 4]");
        col = col.setValue("X", new DuplicableNDArray(Nd4j.ones(6).mul(4)));

        System.out.println();
        System.out.println();
        System.out.println();

        System.out.println(col.statusToString(true));

        System.out.println();
        System.out.println();
        System.out.println();

        System.out.println("Updating caches");
        col = col.updateAllCaches();

        System.out.println();
        System.out.println();
        System.out.println();

        System.out.println("Setting Y to 3");
        col = col.setValue("Y", new DuplicableNumber<>(3.0));

        System.out.println();
        System.out.println();
        System.out.println();

        System.out.println(col.statusToString(true));

        System.out.println("Updating caches");
        col = col.updateAllCaches();

        System.out.println();
        System.out.println();
        System.out.println();

        System.out.println(col.statusToString(true));

    }


}
