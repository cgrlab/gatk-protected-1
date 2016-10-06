package org.broadinstitute.hellbender.tools.coveragemodel.math;

import org.apache.commons.math3.analysis.solvers.BrentSolver;
import org.apache.commons.math3.exception.NoBracketingException;
import org.apache.commons.math3.exception.TooManyEvaluationsException;
import org.apache.commons.math3.util.FastMath;

import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.locks.Condition;
import java.util.concurrent.locks.Lock;
import java.util.concurrent.locks.ReentrantLock;
import java.util.function.Function;
import java.util.stream.Collectors;

/**
 * This class implements a synchronous Brent solver for solving multiple independent equations.
 * It is to be used in situations where function queries have a costly overhead, though, simultaneous
 * queries of multiple functions have the same overhead as single queries.
 *
 * Consider the tasking of solving N independent equations:
 *
 *      f_1(x_1) = 0,
 *      f_2(x_2) = 0,
 *      ...
 *      f_N(x_N) = 0
 *
 * One approach is to solve these equations sequentially. In certain situations, each function
 * evaluation may be cheap but could entail a costly overhead (e.g. if the functions are evaluated
 * in a distributed architecture). It is desirable to minimize this overhead by bundling as many
 * function calls as possible, and querying the function in "chunks".
 *
 * Consider te ideal situation where function evaluations are infinitely cheap, however, each query has
 * a considerable overhead time of \tau. Also, let us assume that the overhead of simultaneously
 * querying {f_1(x_1), ..., f_N(x_N)} is the same as that of a single query, i.e. f_i(x_i). If the
 * Brent solver requires k queries on average, the overhead cost of the sequential approach is
 * O(k N \tau). By making simultaneous queries, this class reduces the overhead to O(k \tau).
 *
 * This is achieved by instantiating N threads for the N Brent solvers, accumulating their queries
 * and suspending them until all threads announce their required query.
 *
 * TODO In the current implementation, we make a thread for each solver. This is OK if the number
 * TODO of equations is reasonably small (< 200). In the future, the class must take a max number
 * TODO of threads and limit concurrency.
 *
 * @author Mehrtash Babadi &lt;mehrtash@broadinstitute.org&gt;
 */
public final class SynchronizedBrentSolver {

    /**
     * Default value for the absolute accuracy of function evaluations
     */
    private static final double DEFAULT_FUNCTION_ACCURACY = 1e-15;

    /**
     * Stores queries from instantiated solvers
     */
    private final ConcurrentHashMap<Integer, Double> queries;

    /**
     * Stores function evaluations on {@link #queries}
     */
    private final ConcurrentHashMap<Integer, Double> results;

    /**
     * Number of queries before making a function call
     */
    private final int numberOfQueriesBeforeCalling;

    /**
     * The objective functions
     */
    private final Function<Map<Integer, Double>, Map<Integer, Double>> func;

    /**
     * A list of Brent solver jobs
     */
    private final List<BrentJobDescription> jobDescriptions;
    private final Set<Integer> jobIndices;

    private final Lock resultsLock = new ReentrantLock();
    private final Condition resultsAvailable = resultsLock.newCondition();
    private CountDownLatch solversCountDownLatch;

    /**
     * Public constructor
     *
     * @param func the objective function (must be able to evaluate multiple calls in one shot)
     * @param numberOfQueriesBeforeCalling Number of queries before making a function call (the default value is
     *                                     the number of equations)
     */
    public SynchronizedBrentSolver(final Function<Map<Integer, Double>, Map<Integer, Double>> func,
                                   final int numberOfQueriesBeforeCalling) {
        this.func = func;
        this.numberOfQueriesBeforeCalling = numberOfQueriesBeforeCalling;

        queries = new ConcurrentHashMap<>(numberOfQueriesBeforeCalling);
        results = new ConcurrentHashMap<>(numberOfQueriesBeforeCalling);
        jobDescriptions = new ArrayList<>();
        jobIndices = new HashSet<>();
    }

    /**
     * Add a Brent solver job
     *
     * @param index a unique index for the equation
     * @param min lower bound of the root
     * @param max upper bound of the root
     * @param x0 initial guess
     * @param absoluteAccuracy absolute accuracy
     * @param relativeAccuracy relative accuracy
     * @param functionValueAccuracy function value accuracy
     * @param maxEval maximum number of allowed evaluations
     */
    public void add(final int index, final double min, final double max, final double x0,
               final double absoluteAccuracy, final double relativeAccuracy,
               final double functionValueAccuracy, final int maxEval) {
        if (jobIndices.contains(index)) {
            throw new IllegalArgumentException("A job with index " + index + " already exists; job indices must" +
                    " be unique");
        }
        if (x0 <= min || x0 >= max) {
            throw new IllegalArgumentException(String.format("The initial guess \"%f\" for equation number \"%d\" is" +
                    " must lie inside the provided search bracket [%f, %f]", x0, index, min, max));
        }
        jobDescriptions.add(new BrentJobDescription(index, min, max, x0, absoluteAccuracy,
                relativeAccuracy, functionValueAccuracy, maxEval));
    }

    /**
     * Add a Brent solver job using the default function accuracy {@link #DEFAULT_FUNCTION_ACCURACY}
     *
     * @param index a unique index for the equation
     * @param min lower bound of the root
     * @param max upper bound of the root
     * @param x0 initial guess
     * @param absoluteAccuracy absolute accuracy
     * @param relativeAccuracy relative accuracy
     * @param maxEval maximum number of allowed evaluations
     */
    public void add(final int index, final double min, final double max, final double x0,
                    final double absoluteAccuracy, final double relativeAccuracy,
                    final int maxEval) {
        add(index, min, max, x0, absoluteAccuracy, relativeAccuracy, DEFAULT_FUNCTION_ACCURACY, maxEval);
    }

    /**
     * Solve the equations
     *
     * @return a map from equation indices to the summary of results
     * @throws InterruptedException if any of the solver threads are interrupted
     */
    public Map<Integer, BrentSolverSummary> solve() throws InterruptedException {
        if (jobDescriptions.isEmpty()) {
            return Collections.emptyMap();
        }
        final Map<Integer, BrentSolverWorker> solvers = new HashMap<>(jobDescriptions.size());
        solversCountDownLatch = new CountDownLatch(jobDescriptions.size());
        jobDescriptions.stream().forEach(job -> solvers.put(job.index, new BrentSolverWorker(job)));

        /* start solver threads */
        solvers.values().forEach(worker -> new Thread(worker).start());

        /* wait for all workers to finish */
        solversCountDownLatch.await();

        return solvers.entrySet()
                .stream()
                .collect(Collectors.toMap(Map.Entry::getKey, entry -> entry.getValue().getSummary()));
    }

    /**
     * Require an evaluation of equation {@param index} at {$param x}
     *
     * @param index equation index
     * @param x equation argument
     * @return evaluated function value
     * @throws InterruptedException if the waiting thread is interrupted
     */
    private double evaluate(final int index, final double x) throws InterruptedException {
        queries.put(index, x);
        resultsLock.lock();
        final double value;
        try {
            fetchResults();
            while (!results.containsKey(index)) {
                resultsAvailable.await();
            }
            value = results.get(index);
            results.remove(index);
        } finally {
            resultsLock.unlock();
        }
        return value;
    }

    /**
     * Check if enough queries are in. If so, make a call and signal the waiting threads
     */
    private void fetchResults() {
        resultsLock.lock();
        try {
            if (queries.size() >= FastMath.min(numberOfQueriesBeforeCalling, solversCountDownLatch.getCount())) {
                results.putAll(func.apply(queries));
                queries.clear();
                resultsAvailable.signalAll();
            }
        } finally {
            resultsLock.unlock();
        }
    }

    /**
     * This class stores the description of a {@link BrentSolver} job
     */
    private final class BrentJobDescription {
        final int index, maxEval;
        final double min, max, x0, absoluteAccuracy, relativeAccuracy, functionValueAccuracy;

        BrentJobDescription(final int index, final double min, final double max, final double x0,
                            final double absoluteAccuracy, final double relativeAccuracy,
                            final double functionValueAccuracy, final int maxEval) {
            this.index = index;
            this.min = min;
            this.max = max;
            this.x0 = x0;
            this.absoluteAccuracy = absoluteAccuracy;
            this.relativeAccuracy = relativeAccuracy;
            this.functionValueAccuracy = functionValueAccuracy;
            this.maxEval = maxEval;
        }
    }

    public enum BrentSolverStatus {
        /**
         * Solution could not be bracketed
         */
        NO_BRACKETING,

        /**
         * Too many function evaluations
         */
        TOO_MANY_EVALUATIONS,

        /**
         * The solver found the root successfully
         */
        SUCCESS,

        /**
         * The status is not determined yet
         */
        TBD
    }

    /**
     * Stores the summary of a {@link BrentSolver} job
     */
    public final class BrentSolverSummary {
        public final double x;
        public final int evaluations;
        public final BrentSolverStatus status;

        BrentSolverSummary(final double x, final int evaluations, final BrentSolverStatus status) {
            this.x = x;
            this.evaluations = evaluations;
            this.status = status;
        }
    }

    /**
     * A runnable version of {@link BrentSolver}
     */
    private final class BrentSolverWorker implements Runnable {
        final BrentSolver solver;
        final BrentJobDescription job;
        BrentSolverStatus status;
        double sol;

        BrentSolverWorker(final BrentJobDescription job) {
            solver = new BrentSolver(job.relativeAccuracy, job.absoluteAccuracy, job.functionValueAccuracy);
            this.job = job;
            status = BrentSolverStatus.TBD;
        }

        @Override
        public void run() {
            double sol;
            try {
                sol = solver.solve(job.maxEval, x -> {
                    final double value;
                    try {
                        value = evaluate(job.index, x);
                    } catch (final InterruptedException ex) {
                        throw new RuntimeException(String.format("Evaluation of equation (n=%d) was interrupted --" +
                                " can not continue", job.index));
                    }
                    return value;
                }, job.min, job.max, job.x0);
            } catch (final NoBracketingException ex) {
                status = BrentSolverStatus.NO_BRACKETING;
                sol = Double.NaN;
            } catch (final TooManyEvaluationsException ex) {
                status = BrentSolverStatus.TOO_MANY_EVALUATIONS;
                sol = Double.NaN;
            }
            if (status.equals(BrentSolverStatus.TBD)) {
                status = BrentSolverStatus.SUCCESS;
            }
            this.sol = sol;
            solversCountDownLatch.countDown();
            fetchResults();
        }

        BrentSolverSummary getSummary() {
            return new BrentSolverSummary(sol, solver.getEvaluations(), status);
        }
    }

}
