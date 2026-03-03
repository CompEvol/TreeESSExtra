package ccp.binomialess;

import org.apache.commons.math3.analysis.UnivariateFunction;
import org.apache.commons.math3.analysis.integration.MidPointIntegrator;
import org.apache.commons.math3.analysis.integration.RombergIntegrator;
import org.apache.commons.math3.analysis.integration.SimpsonIntegrator;
import org.apache.commons.math3.analysis.integration.TrapezoidIntegrator;
import org.apache.commons.math3.analysis.integration.UnivariateIntegrator;
import org.apache.commons.math3.optim.MaxEval;
import org.apache.commons.math3.optim.nonlinear.scalar.GoalType;
import org.apache.commons.math3.optim.univariate.BrentOptimizer;
import org.apache.commons.math3.optim.univariate.SearchInterval;
import org.apache.commons.math3.optim.univariate.UnivariateObjectiveFunction;
import org.apache.commons.math3.optim.univariate.UnivariatePointValuePair;

public class TargetDistribution {


    

    // Maximum number of iterations for numerical integration
    final int MAX_EVAL = 100000000;

    // How many steps between 0 and 1 when searching for quantiles
    final int ICDF_NSTEPS = 10000;


    // Numerical integration
    UnivariateIntegrator integrator = null;
    UnivariateIntegrator[] integrators = new UnivariateIntegrator[]{
            new MidPointIntegrator(),
            new SimpsonIntegrator(),
            new RombergIntegrator(),
            new TrapezoidIntegrator()
    };

    // The data
    int nObservations;
    int[] frequencies1;
    int[] frequencies2;
    int l1;
    int l2;
    
    double alpha = 1;
    double beta = 1;

    // Inference
    double minR = 0.0;
    double maxR = 1.0;
    JointBinomialDistribution dist;
    double normalisationZ;
    double logAlphaLikelihood = 0; // Maximum log-likelihood (used to ensure that integration is numerically stable)
    double logAlphaPosterior = 0; // Maximum posterior-prob (used to ensure that integration is numerically stable)

    public TargetDistribution(int[] frequencies1, int[] frequencies2, int l1, int l2) throws Exception {
        this.frequencies1 = frequencies1;
        this.frequencies2 = frequencies2;
        this.l1 = l1;
        this.l2 = l2;

        this.nObservations = Math.min(frequencies1.length, frequencies2.length); // Should be the same length
        this.dist = new JointBinomialDistribution(0.5);

    }


    /**
     * Get log-likelihood, conditional on r, a, and b
     *
     * @param r
     * @param a
     * @param b
     * @return
     */
    public double getLogLikelihood(double r, double a, double b) {


        if (r <= minR || r >= maxR || a <=0 || b <= 0) {
        	//System.out.println(r +  " " + a + " " + b + "  ninf");
            return Double.NEGATIVE_INFINITY;
        }

        dist.setR(r);
        dist.setA(a);
        dist.setB(b);

        // Iterate through observations, assuming independence
        double logP = 0;
        for (int i = 0; i < this.nObservations; i++) {
            int k1 = frequencies1[i];
            int k2 = frequencies2[i];
            logP += this.dist.getLogP(k1, k2, l1, l2);
            if (logP == Double.NEGATIVE_INFINITY) {
                //System.out.println(r + "  ninf");
                return logP;
            }
        }

        //System.out.println(r +  " " + a + " " + b + "  " + logP);
        return logP;

    }




    /**
     * As R approaches zero, the likelihood becomes flat.
     * Find the minimum value of R such that the likleihood is concave
     *
     * @return
     * @throws Exception
     */
    private double getMinR() throws Exception {

        double dr = 1e-6;
        double minR = dr;

        while (true) {
            int n1 = (int) Math.floor(minR * l1);
            int n2 = (int) Math.floor(minR * l2);

            if (n1 > 0 && n2 > 0) {
                break;
            }

            if (minR >= 1) {
                throw new Exception("Dev error: something went wrong finding the minimal value of R");
            }

            minR += dr;


        }

        // minR += dr;
        // minR *= 2;
        return minR;


    }
    
    
    /**
     * Returns double[] with three numbers -
     * first, the MLE or MAP of R
     * second, the log-likelihood or posterior of this estimate
     * third, the ESS = r*(l1+l2)
     *
     * @return
     * @throws Exception
     */
    public double[] getMaxLogP() throws Exception {
    	return null;
    }

    
    
    public void setAB(double a, double b) {
    	this.alpha = a;
    	this.beta = b;
    }
    

}




