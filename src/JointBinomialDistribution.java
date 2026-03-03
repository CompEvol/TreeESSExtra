package ccp.binomialess;

import org.apache.commons.math3.special.Beta;
import org.apache.commons.math3.util.CombinatoricsUtils;

public class JointBinomialDistribution {

    // Model parameters
    private double param_r;
    private double param_a;
    private double param_b;

    public JointBinomialDistribution(double r) {
        this.param_r = r;
        this.param_a = 1;
        this.param_b = 1;
    }

    public JointBinomialDistribution(double r, double a, double b) {
        this.param_r = r;
        this.param_a = a;
        this.param_b = b;
    }

    /**
     * Method to calculate the joint probability distribution, assuming that each clade support has a Beta(a,b) prior
     *
     * @param k1
     * @param k2
     * @param l1
     * @param l2
     * @return
     */
    public double getLogP(int k1, int k2, int l1, int l2) {


        // Calculate
        int n1 = (int) Math.floor(param_r * l1);
        int n2 = (int) Math.floor(param_r * l2);
        int x1 = (int) Math.floor(param_r * k1);
        int x2 = (int) Math.floor(param_r * k2);

        // Validate that k1>=0, k2>=0, and k1+k2 > 0
        if (k1 < 0 || k2 < 0 || k1 + k2 <= 0 || n1 == 0 || n2 == 0) {
        	
        	// Skip
        	return Double.NEGATIVE_INFINITY;
        }
        
        if (n1 == 0 || n2 == 0) {
//        	System.out.println("violation " + param_r + " , " + k1 + " " + l1 + " " + x1);
//        	System.out.println("l1=" + l1);
//        	System.out.println("l2=" + l2);
//        	System.out.println("x1=" + x1);
//        	System.out.println("x2=" + x2);
//        	System.out.println("k1=" + k1);
//        	System.out.println("k2=" + k2);
//        	System.out.println("n1=" + n1);
//        	System.out.println("n1=" + n1);
        	
            return Double.NEGATIVE_INFINITY;
        }


        // Combinatorial terms
        double binom1 = CombinatoricsUtils.binomialCoefficientLog(n1, x1);
        double binom2 = CombinatoricsUtils.binomialCoefficientLog(n2, x2);


        // Conditional on both k1 and k2 being non-zero
        double ascertainmentCorrection = Beta.logBeta(param_a, n1 + n2 + param_b);


        // Calculate log P
        double betaVal = Beta.logBeta(x1 + x2 + param_a, n1 + n2 - (x1 + x2) + param_b);
        double btm = Math.log(Math.exp(Beta.logBeta(param_a, param_b)) - Math.exp(ascertainmentCorrection));
        double logP = binom1 + binom2 + betaVal - btm;


        // System.out.println(binom1 + " " + binom2 + " " + betaVal + " " + btm + " " + ascertainmentCorrection);

        // Downweight according to how many (K1,K2) map to this (X1,X2) so that probability sums to 1 across all k1, k2
        double weight = Math.log(howManyK(x1, param_r) * howManyK(x2, param_r));

        if (logP - weight == Double.POSITIVE_INFINITY) {
            System.out.println(logP + ", " + weight + " pinf error " + param_r + " " + param_a + " " + param_b);
            //return Double.NEGATIVE_INFINITY;
        }

        return logP - weight;

    }

    /**
     * Count the number of integers k, such that floor(k*r)=x
     *
     * @param x
     * @param r
     * @return
     */
    public static int howManyK(int x, double r) {


        int minVal = (int) Math.ceil(1.0 * x / r);
        double maxVal = (1.0 * x + 1) / r;
        int diff = (int) (Math.floor(maxVal) - minVal);

        // Check if maxVal is an integer - this may cause numerical issues
        if (maxVal == Math.floor(maxVal)) {
            return diff;
        } else {
            return diff + 1;
        }

    }

    public void setR(double r) {
        this.param_r = r;
    }

    public double getR() {
        return this.param_r;
    }

    public void setA(double a) {
        this.param_a = a;
    }

    public double geta() {
        return this.param_a;
    }

    public void setB(double b) {
        this.param_b = b;
    }

    public double getB() {
        return this.param_b;
    }

}
