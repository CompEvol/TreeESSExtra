package ccp.binomialess;

import beast.base.evolution.tree.Tree;
import ccp.algorithms.TreeDistances;
import ccp.model.WrappedBeastTree;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * Method to estimate the tree ESS (effective sample size) of a chain of trees using the RF distance
 * based on the method by <a href="http://doi.org/10.1214/22-ba1339">Magee et al. 2024</a>,
 * which is an initial positive sequence estimator that uses estimates of covariance, variance, and mean
 * based on estimates of Frechet variance and mean on squared RF distances.
 *
 * @author Jonathan Klawitter (based on Magee et al. code)
 */
public class FrechetTreeESSEstimator {

    /** Only compute the correlation between x_{t} and x_{t+lag} if there are at least this many samples. */
    public static final int MIN_SAMPLES_PER_LAG = 5;

    /**
     * Estimates the ESS for the given chain of trees.
     * Unlike {@link #estimateFrechetTreeESS(List)} this method can be called with unwrapped tree,
     * which however results in them being wrapped, which can be slow for long chains.
     *
     * @param chain of trees for which ESS is estimated
     * @return estimated tree ESS
     */
    public static double estimateFrechetUnwrappedTreeESS(final List<Tree> chain) {
        if (chain.size() <= 1) {
            return chain.size();
        }

        List<WrappedBeastTree> wrappedChain = new ArrayList<>(chain.size());
        for (Tree tree : chain) {
            wrappedChain.add(new WrappedBeastTree(tree));
        }

        return estimateFrechetTreeESS(wrappedChain);
    }

    /**
     * Estimates the ESS for the given chain of trees.
     *
     * @param chain of trees for which ESS is estimated,
     *              wrapped ones for faster RF distance computation
     * @return estimated tree ESS
     */
    public static double estimateFrechetTreeESS(final List<WrappedBeastTree> chain) {
        if (chain.size() <= 1) {
            return chain.size();
        }

        double act = estimateFrechetTreeACT(chain);
        return chain.size() / act;
    }

    /**
     * Estimates the ACT for the given chain of trees
     * with the initial positive sequence estimator method
     * adapted for trees using squared RF distances.
     *
     * @param chain of tress for which ACT is estimated,
     *              wrapped ones for faster RF distance computation
     * @return estimated tree ACT
     */
    private static double estimateFrechetTreeACT(final List<WrappedBeastTree> chain) {
        int n = chain.size();
        int maxLag = n - MIN_SAMPLES_PER_LAG - 1; // ensure min number of samples per lag

        double[] autoCorrelation = new double[maxLag + 1];
        double[] rhos = new double[(maxLag + 1) / 2 + 1]; // only half because pairs are combined
        Arrays.fill(rhos, Double.NaN);
        double actSum = 0;

        /* The algorithm is an initial positive sequence estimator (as also used in Tracer).
         The difference is that variance and mean are computed differently,
         namely, we use approximations by Magee et al. for Frechet variance/mean

        > Frechet variance
        we want to compute the Frechet variance, but do not have the Frechet mean,
        so instead we approximate it with the sum of pairwise distances, for specific lag s:
              1/(n(n-1)) \sum_{i<j} d(x_i, x_j)^2
        see Eq (15) in Magee et al. supplementary material.

        > E[\Delta^2] mean squared distance for specific lag s:
              E[\Delta^2] = 1/(n-lag) \sum_{i=0}^{n-lag} d(x_i, x_{i+s})^2
        see Eq (16) in Magee et al. supplementary material.

        > Frechet covariance estimation
              0.5 (Var(X) + Var(Y) - E[\Delta^2]) / \sqrt( Var(X)Var(Y) )
        see Eq (18) in Magee et al. supplementary material.

        Magee et al. computed the whole distance matrix,
        but for very long chains, this matrix gets too large.
        However, we can compute the three full sums and then only subtract the lost values as we go.
        */
        double sumX = 0;
        double sumY = 0;
        double[] meanDistances = new double[maxLag + 1];
        double[] diffsX = new double[maxLag + 1];
        double[] diffsY = new double[maxLag + 1];
        for (int i = 0; i < n - 1; i++) {
            for (int j = i + 1; j < n; j++) {
                double rf = TreeDistances.robinsonsFouldDistance(chain.get(i), chain.get(j));
                double rfrf = rf * rf;
                sumX += rfrf;

                if (n - j <= maxLag) {
                    diffsX[n - j] += rfrf;
                }
                if (i < maxLag) {
                    diffsY[i + 1] += rfrf;
                }

                if (j - i <= maxLag) {
                    meanDistances[j - i] += rfrf;
                }
            }
        }
        sumY = sumX;

        for (int lag = 1; lag <= maxLag; lag++) {
            // X = \tau_{t} are all the trees from 0 to n-lag-1,
            //      so from sumX, we have to subtract from previous round distances to element n-lag
            sumX -= diffsX[lag];
            // Y = \tau_{t+lag} are all the trees from lag to n-1
            //      so from sumY, we have to subtract from previous round distances to element lag
            sumY -= diffsY[lag];
            // meanDistance[lag] = sum_{i = 0}^{n - lag} d(x_i, x_{i + lag})^2
            //      is already computed
            // and now we can weigh all three values correctly
            int ni = n - lag;
            double varianceX = sumX / (ni * (ni - 1));
            double varianceY = sumY / (ni * (ni - 1));
            meanDistances[lag] /= ni;

            // > covariance approximation
            double covariance = varianceX + varianceY - meanDistances[lag];
            // System.out.println("lag = " + lag + ", varX = " + varianceX +
            //         ", varY = " + varianceY + ", mean = " + meanDistances[lag] + ", cov = " + covariance);

            // If either of the variances are 0,
            // then one set of trees is all the same and the correlation is technically undefined.
            // Practically, this means there is very little variability in these trees or
            // the sampler is mixing poorly, and we can report this with a high correlation.
            if ((varianceX == 0) || (varianceY == 0)) { // exceptional case
                autoCorrelation[lag] = 1.0;
            } else { // normal case
                autoCorrelation[lag] = 0.5 * covariance / Math.sqrt(varianceX * varianceY);
            }

            // Combine past two rho estimates in joined rho and
            // check if we got a negative value, which is the stopping criteria.
            if (lag % 2 == 1) {
                int index = (lag + 1) / 2;
                if (lag == 1) {
                    // because autoCorrelation[0] = 1
                    rhos[index] = autoCorrelation[lag] + 1.0;
                } else {
                    rhos[index] = autoCorrelation[lag] + autoCorrelation[lag - 1];
                }

                // Same stopping criteria as used in Tracer,
                // namely only use terms with rho positive.
                if (rhos[index] <= 0) {
                    break;
                }

                // Smoothing as Magee et al. (which follow Vehtari et al.)
                if ((index > 1) && (rhos[index] > rhos[index - 1])) {
                    rhos[index] = rhos[index - 1];
                }

                actSum += rhos[index];
            }
        }

        double[] autoCorrelations = Arrays.stream(autoCorrelation).filter(x -> !Double.isNaN(x) && !(x == 0.0)).toArray();
        System.out.println(Arrays.toString(autoCorrelations));

        double act = -1 + 2 * actSum;

        // exception handling
        if (act < 0) {
            act = 1;
        }

        return act;
    }
}
