package org.broadinstitute.hellbender.tools.coveragemodel;

import com.google.common.collect.ImmutableMap;
import org.apache.commons.math3.util.FastMath;
import org.apache.spark.HashPartitioner;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.storage.StorageLevel;
import org.broadinstitute.hellbender.engine.spark.SparkContextFactory;
import org.broadinstitute.hellbender.utils.test.BaseTest;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.ojalgo.function.multiary.MultiaryFunction;
import org.testng.annotations.Test;
import scala.Tuple2;
import scala.collection.JavaConverters;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

/**
 * @author Mehrtash Babadi &lt;mehrtash@broadinstitute.org&gt;
 */
public class SparkTest extends BaseTest {

    public static final Map<String, String> nd4jSparkProperties = ImmutableMap.<String,String>builder()
            .put("spark.kryo.registrator", "org.broadinstitute.hellbender.tools.coveragemodel.nd4jutils.Nd4jRegistrator")
            .build();

    public class MyClass {
        public INDArray pers;
        public INDArray x;

        public MyClass(final int n, final double val) {
            x = Nd4j.ones(n, n).mul(val);
            pers = Nd4j.ones(100, 100);
        }

        public MyClass(final INDArray x, final INDArray pers) {
            this.x = x;
            this.pers = pers;
        }

        public MyClass add() {
            System.out.println("added");
            return new MyClass(x.add(1), pers);
        }

        public MyClass addInPlace() {
            System.out.println("added");
            x.addi(1);
            return this;
        }

    }

    @Test
    public void testBasicFunction() {
        final JavaSparkContext ctx = SparkContextFactory.getTestSparkContext(nd4jSparkProperties);

        int N = 200;

        List<Tuple2<LinearSpaceBlock, MyClass>> list1 = IntStream.range(0, N)
                .mapToObj(n -> new Tuple2<>(new LinearSpaceBlock(n, n+1), new MyClass(n+1, n+1))).collect(Collectors.toList());

        JavaPairRDD<LinearSpaceBlock, MyClass> rdd1 = ctx.parallelizePairs(list1, N ).partitionBy(new HashPartitioner(N));

        JavaPairRDD<LinearSpaceBlock, MyClass> rdd3 = ctx.parallelizePairs(list1, N);

        List<Tuple2<LinearSpaceBlock, Double>> list2 = IntStream.range(0, N)
                .mapToObj(n -> new Tuple2<>(new LinearSpaceBlock(n, n+1), (double)n + 1.5)).collect(Collectors.toList());

        List<LinearSpaceBlock> keys = IntStream.range(0, N).mapToObj(n -> new LinearSpaceBlock(n, n+1)).collect(Collectors.toList());

        JavaPairRDD<LinearSpaceBlock, Double> rdd2 = ctx.parallelizePairs(list2).partitionBy(new HashPartitioner(N));

        long t0 = System.nanoTime();
        rdd1.count();
        long t1 = System.nanoTime();
        rdd3.count();
        long t2 = System.nanoTime();

        System.out.println((t1-t0)/1000000);
        System.out.println((t2-t1)/1000000);

        rdd1.cache();
        rdd2.cache();

        HashPartitioner partitioner = new HashPartitioner(N);

        for (LinearSpaceBlock key : keys) {
            System.out.println(key + " => " + partitioner.getPartition(key));
        }

        System.out.println();

        for (int i=0; i<N; i++) {
            rdd1.collectPartitions(new int[]{i})[0].forEach(x -> System.out.println(x._1.getBegIndex()));
        }

        System.out.println();

        for (int i=0; i<N; i++) {
            rdd2.collectPartitions(new int[]{i})[0].forEach(x -> System.out.println(x._1.getBegIndex()));
        }

        System.out.println();

        for (int i=0; i<N; i++) {
            rdd1.join(rdd2).collectPartitions(new int[]{i})[0].forEach(x -> System.out.println(x._1.getBegIndex()));
        }




    }

}
