package org.broadinstitute.hellbender.tools.coveragemodel;

import org.apache.commons.math3.analysis.interpolation.BicubicInterpolatingFunction;
import org.apache.commons.math3.analysis.interpolation.BicubicInterpolator;
import org.apache.commons.math3.analysis.interpolation.PiecewiseBicubicSplineInterpolatingFunction;
import org.apache.commons.math3.analysis.interpolation.PiecewiseBicubicSplineInterpolator;
import org.apache.commons.math3.exception.NotFiniteNumberException;
import org.apache.commons.math3.special.Erf;
import org.apache.commons.math3.util.FastMath;
import org.apache.commons.math3.util.MathUtils;
import org.broadinstitute.hellbender.tools.coveragemodel.interfaces.TargetLikelihoodCalculator;
import org.broadinstitute.hellbender.tools.exome.Target;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.io.*;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Scanner;

/**
 * @author Mehrtash Babadi &lt;mehrtash@broadinstitute.org&gt;
 */
public class CoverageModelCopyRatioEmissionProbabilityCalculator implements
        TargetLikelihoodCalculator<CoverageModelCopyRatioEmissionData>, Serializable {

    private static final long serialVersionUID = -6985799468753075235L;

    private static final boolean CHECK_FOR_NANS = true;

    private static final String MU_TABLE_RESOURCE = "mu_table.tsv";
    private static final String PSI_TABLE_RESOURCE = "psi_table.tsv";
    private static final String LOG_NORM_TABLE_RESOURCE = "log_norm_table.tsv";

    private static final double MINIMUM_ALLOWED_PSI = 0.0;
    private static final double MAXIMUM_ALLOWED_PSI = 1.0;
    private static final double MINIMUM_ALLOWED_MU = -25.0;
    private static final double MAXIMUM_ALLOWED_MU = 10.0;

    private static final BicubicInterpolatingFunction logNormFactorSpline =
            getLogNormFactorInterpolatingFunction();

    private static final double LOG_2PI = FastMath.log(2 * FastMath.PI);

    /**
     * The relative error tolerance in normalization of the the emission probability function
     */
    private static double NORMALIZATION_ERROR_TOL = 1e-3;

    /**
     * The approximations are guaranteed to produce results at least with a following relative accuracy
     */
    public static double RELATIVE_ACCURACY = 1e-3;

    /**
     * The approximations are guaranteed to produce results at least with a following absolute accuracy
     */
    public static double ABSOLUTE_ACCURACY = 1e-3;

    /**
     * Calculate the log emission probability. The parameter {@param target} is not used since
     * {@param emissionData} contains the necessary information.
     *
     * @param emissionData an instance of {@link CoverageModelCopyRatioEmissionData}
     * @param copyRatio copy ratio for which the emission probability is calculated
     * @param target target on which the emission probability is calculated (this parameter is not used)
     * @return emission probability
     */
    @Override
    public double logLikelihood(@Nonnull CoverageModelCopyRatioEmissionData emissionData,
                                double copyRatio, @Nullable Target target) {
        final double mu = emissionData.getMu() + FastMath.log(copyRatio);
        final double psi = emissionData.getPsi();
        final double logProbabilityMass = getLogProbabilityMass(mu, psi);
        final double res = getUnnormalizedLogPDF(emissionData.getReadCount(), mu, psi)
                - logProbabilityMass;
        if (CHECK_FOR_NANS) {
            if (Double.isNaN(res) || Double.isInfinite(res)) {
                throw new RuntimeException("Something went wrong while calculating the likelihood; the" +
                        " emission data is: " + emissionData.toString() + " on target " + target);
            }
        }
        return res;
    }

    private static double getUnnormalizedLogPDF(final double readCount, final double mu, final double psi) {
        final double variance = psi + 1.0 / readCount;
        final double logReadCount = FastMath.log(readCount);
        return - logReadCount - 0.5 * (LOG_2PI + FastMath.log(variance) +
                FastMath.pow(logReadCount - mu, 2) / variance);
    }

    public static double getLogProbabilityMass(final double mu, final double psi) {
        if (mu > MAXIMUM_ALLOWED_MU) { /* the log norm factor is guaranteed to be smaller than 1e-4 */
            return 0.0;
        } else {
            final double muTrunc = FastMath.max(MINIMUM_ALLOWED_MU, mu);
            final double psiTrunc = FastMath.max(MINIMUM_ALLOWED_PSI, FastMath.min(MAXIMUM_ALLOWED_PSI, psi));
            return logNormFactorSpline.value(muTrunc, psiTrunc);
        }
    }

    private static double getUnnormalizedLogPDF__Deprecated(final double logn, final double mu, final double lambda) {
        return - 0.5 * lambda * FastMath.pow(logn - mu, 2);
    }

    /**
     * This functions uses a combination of direct summation and Euler-Maclaurin asymptotic series to
     * calculate the log probability mass,
     *
     *      log(\sum_{n=0}^{\infty} \exp{-0.5 * \Lambda * (\log(n) - \mu)^2})
     *
     * within the relative error tolerance {@link #NORMALIZATION_ERROR_TOL}
     *
     * @param mu
     * @param lambda
     * @return
     */
    public static double getLogProbabilityMass__Deprecated(final double mu, final double lambda) {
        final double nMode;
        final double logPDFMode;
        double relErr, logProbMass = 0;

        if (mu <= 0) {
            nMode = 1;
            logPDFMode = getUnnormalizedLogPDF__Deprecated(0, mu, lambda);
        } else { /* only an approximation is required */
            nMode = FastMath.floor(FastMath.exp(mu));
            logPDFMode = getUnnormalizedLogPDF__Deprecated(FastMath.log(nMode), mu, lambda);
        }

        /* some recurring sub-expressions */
        final double lm = lambda * mu;
        final double lmm = lm * mu;
        final double llmmm = lm * lmm;

        /* try truncated Euler-Maclaurin summation as a first estimate */
        final double normFactEulerMaclaurin_0 = FastMath.sqrt(FastMath.PI/(2*lambda)) *
                Erf.erfc(-(1 + lm)/FastMath.sqrt(2*lambda)) * FastMath.exp((1 + 2*lm)/(2*lambda));
        final double expFact = FastMath.exp(-lmm/2);
        try {
            MathUtils.checkFinite(normFactEulerMaclaurin_0);
            final double BernoulliCorrection_1 = + 2;
            final double BernoulliCorrection_2 = - lm / 12;
            final double BernoulliCorrection_3 = + lambda * (3 + 2*mu - 3*lm - 3*lmm + llmmm) / 720;
            final double normFactEulerMaclaurin_1 = normFactEulerMaclaurin_0 + expFact * BernoulliCorrection_1;
            final double normFactEulerMaclaurin_2 = normFactEulerMaclaurin_1 + expFact * BernoulliCorrection_2;
            final double normFactEulerMaclaurin_3 = normFactEulerMaclaurin_2 + expFact * BernoulliCorrection_3;
            logProbMass = FastMath.log(normFactEulerMaclaurin_3);
            MathUtils.checkFinite(logProbMass);
            double relErr_1 = FastMath.abs(normFactEulerMaclaurin_1 - normFactEulerMaclaurin_0) / normFactEulerMaclaurin_0;
            double relErr_2 = FastMath.abs(normFactEulerMaclaurin_2 - normFactEulerMaclaurin_1) / normFactEulerMaclaurin_1;
            double relErr_3 = FastMath.abs(normFactEulerMaclaurin_3 - normFactEulerMaclaurin_2) / normFactEulerMaclaurin_2;
            relErr = FastMath.max(FastMath.max(relErr_1, relErr_2), relErr_3);
        } catch (final NotFiniteNumberException ex) {
            relErr = 2 * NORMALIZATION_ERROR_TOL;
        }

        /* Euler-Maclaurin may fail either due to non-converging series or numerical over/underflow */
        if (relErr < NORMALIZATION_ERROR_TOL) {
            return logProbMass;
        } else {
            double prevSum = 1.0, sum = 1.0; /* old and new sums */
            double nRight = nMode; /* value of n for the right new term */
            double nLeft = nMode; /* value of n for the left new term */
            double decayRate = 2; /* series decay rate */
            double dsum, prevdsum = Double.MIN_VALUE; /* new and old added terms */
            while (relErr > NORMALIZATION_ERROR_TOL || decayRate > 1) {
                nRight += 1;
                nLeft -= 1;
                dsum = FastMath.exp(getUnnormalizedLogPDF__Deprecated(FastMath.log(nRight), mu, lambda) - logPDFMode);
                sum += dsum;
                if (nLeft > 0) {
                    sum += FastMath.exp(getUnnormalizedLogPDF__Deprecated(FastMath.log(nLeft), mu, lambda) - logPDFMode);
                }
                decayRate = dsum / prevdsum; /* estimation of series decay rate */
                /* geometric sum estimate of the remaining terms; gives an upper bound to the error*/
                relErr = 2 * (sum - prevSum) * decayRate / (prevSum * (1 - decayRate));
                prevSum = sum;
                prevdsum = dsum;
            }
            return FastMath.log(sum) + logPDFMode;
        }
    }

    /**
     * Loads a double array from a tab-separated file
     *
     * @return
     */
    private static double[] loadDoubleArrayTable(final InputStream inputStream) {
        final Scanner reader = new Scanner(inputStream);
        final List<Double> data = new ArrayList<>();
        while (reader.hasNextLine()) {
            data.add(Double.parseDouble(reader.nextLine()));
        }
        return data.stream().mapToDouble(d -> d).toArray();
    }

    /**
     * Loads a double 2D array from a tab-separated file
     *
     * @return
     */
    private static double[][] loadDouble2DArrayTable(final InputStream inputStream) {
        final Scanner reader = new Scanner(inputStream);
        final List<double[]> data = new ArrayList<>();
        while (reader.hasNextLine()) {
            data.add(Arrays.stream(reader.nextLine().split("\t"))
                    .mapToDouble(Double::parseDouble).toArray());
        }
        final int rows = data.size();
        final int cols = data.get(0).length;
        final double[][] data2DArray = new double[rows][cols];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                data2DArray[i][j] = data.get(i)[j];
            }
        }
        return data2DArray;
    }

    private static BicubicInterpolatingFunction getLogNormFactorInterpolatingFunction() {
        final Class<?> clazz = CoverageModelCopyRatioEmissionProbabilityCalculator.class;
        final double[] mu = loadDoubleArrayTable(clazz.getResourceAsStream(MU_TABLE_RESOURCE));
        final double[] psi = loadDoubleArrayTable(clazz.getResourceAsStream(PSI_TABLE_RESOURCE));
        final double[][] logNorm = loadDouble2DArrayTable(clazz.getResourceAsStream(LOG_NORM_TABLE_RESOURCE));
        final BicubicInterpolator interp = new BicubicInterpolator();
        return interp.interpolate(mu, psi, logNorm);
    }
}
