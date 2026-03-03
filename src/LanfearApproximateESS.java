package ccp.binomialess;

import ccp.algorithms.TreeDistances;
import ccp.model.WrappedBeastTree;
import org.apache.commons.math3.optim.PointValuePair;
import org.apache.commons.math3.optim.nonlinear.scalar.GoalType;
import org.apache.commons.math3.optim.nonlinear.scalar.ObjectiveFunction;
import org.apache.commons.math3.optim.nonlinear.scalar.noderiv.NelderMeadSimplex;
import org.apache.commons.math3.optim.nonlinear.scalar.noderiv.SimplexOptimizer;

import java.util.Arrays;
import java.util.List;
import java.util.stream.IntStream;

public class LanfearApproximateESS {

    public static final int MIN_SAMPLES_PER_LAG = 5;
    public static final double ALPHA = 0.05;
    public static final int MAX_ITERATIONS = 10000;
    public static final double REL = 1e-5;
    public static final double ABS = 1e-5;


    public static double calculateApproximateESS(final List<WrappedBeastTree> chain) {
        if (chain.size() <= 1) {
            return chain.size();
        }

        int n = chain.size();
        int maxLag = n - MIN_SAMPLES_PER_LAG - 1;
        maxLag = 200;

        double[] meanDistances = computeMeanDistances(chain, maxLag);
        // System.out.println(Arrays.toString(meanDistances));
        int[] lags = new int[maxLag];
        for (int i = 0; i < maxLag; i++) {
            lags[i] = i + 1;
        }

        double result = fitExponentialModel(meanDistances, lags);

        // Threshold calculation
        double adjustedThreshold = (1 - ALPHA) * result;

        // Find m
        double maxMean = Arrays.stream(meanDistances).max().orElse(meanDistances[meanDistances.length - 1]);
        int m;
        if (maxMean < adjustedThreshold) {
            m = meanDistances.length + 1;
        } else {
            // m = Arrays.stream(meanDistances).filter(x -> (x >= adjustedThreshold)).min().orElse(0);
            m = IntStream.range(0, meanDistances.length)
                    .filter(i -> meanDistances[i] >= adjustedThreshold)
                    .reduce((i1, i2) -> meanDistances[i1] < meanDistances[i2] ? i1 : i2)
                    .orElse(-1);
        }

        // Calculate D
        double D = maxMean;

        // Calculate S
        double S = 0.0;

        if (m > 1) {
            for (int k = 0; k < m - 1; k++) {
                double f = meanDistances[k];
                S += (n - k) * f;
            }
        }

        S += (n - m + 1) * (n - m) * D / 2;
        S = S / 2 / (n * n);

        // Calculate ESSc
        return 1 / (1 - 4 * S / D);
    }

    private static double fitExponentialModel(double[] meanDistances, int[] lags) {
        double maxValue = Arrays.stream(meanDistances).max().orElse(meanDistances[meanDistances.length - 1]);
        double[] initialGuess = {maxValue, 1};


        ObjectiveFunction objectiveFunction = new ObjectiveFunction(par -> {
            return evaluateFunction(par, lags, meanDistances, maxValue);
        });

        // int i = 0;
        // while (evaluateFunction(initialGuess, lags, meanDistances, maxValue) == Double.POSITIVE_INFINITY && i++ < 1000) {
        //     initialGuess[1] += 0.1;
        // }

        SimplexOptimizer optimizer = new SimplexOptimizer(REL, ABS);

        // Optimize
        PointValuePair result = optimizer.optimize(
                new NelderMeadSimplex(2),
                new org.apache.commons.math3.optim.MaxEval(MAX_ITERATIONS),
                GoalType.MINIMIZE,
                // new SimpleBounds(new double[]{1, 0}, new double[]{Double.MAX_VALUE, Double.MAX_VALUE}),
                new org.apache.commons.math3.optim.InitialGuess(initialGuess),
                objectiveFunction);

        return result.getPoint()[0];
    }

    private static double evaluateFunction(double[] par, int[] lags,
                                           double[] meanDistances, double maxValue) {
        // Extract parameters
        double par1 = par[0];
        double par2 = par[1];

        if (par1 < maxValue * 0.2) {
            return Double.MAX_VALUE;
        }
        if (par1 > 1.5 * maxValue) {
            return Double.MAX_VALUE;
        }
        if (par2 <= 1) {
            return Double.MAX_VALUE;
        }

        // Initialize the sum of squared differences
        double sumSqDiff = 0.0;

        // Compute the expected distances
        double[] expectedDistance = new double[meanDistances.length];
        for (int i = 0; i < meanDistances.length; i++) {
            expectedDistance[i] = par1 * (1 - Math.exp(-lags[i] / par2));
        }

        // Compute the sum of squared differences
        for (int i = 0; i < meanDistances.length; i++) {
            double diff = expectedDistance[i] - meanDistances[i];
            sumSqDiff += diff * diff;
        }

        // if (new Random().nextDouble() > 0.99) {
        //     System.out.println();
        // }

        return sumSqDiff;
    }

    public static double[] computeMeanDistances(final List<WrappedBeastTree> chain, int maxLag) {
        int n = chain.size();
        double[] meanDistances = new double[maxLag];
        for (int i = 0; i < n - 1; i++) {
            for (int j = i + 1; j < n; j++) {
                if (j - i <= maxLag) {
                    double rf = TreeDistances.robinsonsFouldDistance(chain.get(i), chain.get(j));
                    double rfrf = rf * rf;
                    meanDistances[j - i - 1] += rfrf;
                }
            }
        }

        for (int lag = 1; lag <= maxLag; lag++) {
            int ni = n - lag;
            meanDistances[lag - 1] /= ni;
        }

        return meanDistances;
    }

    // private static double getSequentialDistance(int thinning, List<WrappedBeastTree> chain) {
    //     double totalDistance = 0.0;
    //     int pairsCount = 0;
    //
    //     // Compare trees at thinning intervals
    //     for (int start = 0; start < chain.size() - thinning; start++) {
    //         int end = start + thinning;
    //         int rf = TreeDistances.robinsonsFouldDistance(chain.get(start), chain.get(end));
    //         totalDistance += rf * rf;
    //         pairsCount++;
    //     }
    //
    //     // Return average distance
    //     return pairsCount > 0 ? totalDistance / pairsCount : 0.0;
    // }

    // // Helper method to generate thinning intervals
    // private static int[] getThinnings(int maxThinning, int autocorrIntervals) {
    //     int[] thinnings = new int[autocorrIntervals];
    //     for (int i = 0; i < autocorrIntervals; i++) {
    //         thinnings[i] = (int) Math.round((double) i * maxThinning / autocorrIntervals);
    //     }
    //
    //     return thinnings;
    // }
}
