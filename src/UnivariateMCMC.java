package ccp.binomialess;

import org.apache.commons.math3.analysis.UnivariateFunction;
import org.apache.commons.math3.random.MersenneTwister;

import java.util.ArrayList;
import java.util.List;

public class UnivariateMCMC {
    MersenneTwister rng = new MersenneTwister();
    BactrianIntervalOperator operator;
    UnivariateFunction targetDistribution;
    double paramLower;
    double paramUpper;
    boolean logToScreen = false;
    int logEvery = 100;
    final int maxLogEvery = 50000;
    int burninEvery = 5; // Discard the first element every n logs

    public UnivariateMCMC(UnivariateFunction targetDistribution, double paramLower, double paramUpper) {
        this.targetDistribution = targetDistribution;
        this.paramLower = paramLower;
        this.paramUpper = paramUpper;
        this.operator = new BactrianIntervalOperator(paramLower, paramUpper, this.rng);
    }

    public void setLogEvery(int logEvery) {
        this.logEvery = logEvery;
    }
    

    public void setBurninEvery(int burnin) {
        this.burninEvery = burnin;
    }

    public void screenLogging(boolean logToScreen) {
        this.logToScreen = logToScreen;
    }

    /**
     * Runs MCMC until the ESS of the parameter exceeds the threshold or until time limit
     * Returns a List<Double> containing estimates
     *
     * @param minESS
     * @return
     */
    public List<Double> run(double minESS, double timeLimitSeconds) {
    	
    	
    	long startTime = System.currentTimeMillis();
        int maxListSize = (int) (minESS * 2);

        // Initial value
        double theta = (this.paramUpper - this.paramLower) / 2;
        double logP = this.targetDistribution.value(theta);

        // Estimates
        List<Double> estimates = new ArrayList<>();

        // Proposal variables
        double[] proposalArr = new double[2];
        double thetaPrime, logHR, logPPrime, logAlpha, alpha, u;

        // Repeat the MCMC loop until convergence
        long sampleNr = 0;
        int burninCycle = 0;
        while (true) {
        	
            // Make proposal
            this.operator.proposal(theta, proposalArr);
            thetaPrime = proposalArr[0];
            logHR = proposalArr[1];

            // Compute new logP
            logPPrime = this.targetDistribution.value(thetaPrime);

            // Acceptance probability
            logAlpha = logPPrime - logP + logHR;
            alpha = Math.min(1, Math.exp(logAlpha));

            u = rng.nextDouble();
            if (logHR != Double.NEGATIVE_INFINITY && u < alpha) {
                // Accept
                theta = thetaPrime;
                logP = logPPrime;
                this.operator.accept();
            } else {
                // Reject
                this.operator.reject();
            }

            if (logHR != Double.NEGATIVE_INFINITY) {
                this.operator.optimize(logAlpha);
            }

            // Sample value
            if (sampleNr % this.logEvery == 0) {

                if (logToScreen) {
                    double ess = ParameterUtils.calcESS(estimates);
                    System.out.println("Sample nr " + sampleNr + " theta = " + theta + ". MCMC ESS=" + ess + " / " + estimates.size());
                }
                estimates.add(theta);
                burninCycle++;

                // Thin the list to reduce computational overhead
                boolean timeToThin = estimates.size() > maxListSize;

                // From time to time randomly check that the ESS to sample size ratio is large
                // A list size over 200 becomes much slower to add entries to
                if (estimates.size() > 200 & this.rng.nextFloat() < 0.1) {
                    double ess = ParameterUtils.calcESS(estimates);
                    if (ess / estimates.size() < 0.5) {
                        timeToThin = true;
                    }

                }

                if (timeToThin && this.logEvery < maxLogEvery) {

                    int factor = 2;
                    this.logEvery = this.logEvery * factor;
                    estimates = thinList(estimates, maxListSize / factor);

                    if (logToScreen) {
                        double ess = ParameterUtils.calcESS(estimates);
                        System.out.println("Thinning list down to " + (maxListSize / factor) + ". MCMC ESS=" + ess);
                    }

                }

                // Remove front from list?
                if (burninCycle % this.burninEvery == 0) {
                    estimates.remove(0);
                    burninCycle = 0;
                }
            

            
	            // Check for convergence or timeout
                //System.out.println("time=" + timeElapsed + " " + startTime);\
                double timeElapsed = (System.currentTimeMillis() - startTime);
	            if (estimates.size() > minESS || timeElapsed > timeLimitSeconds*1000) {
	                double ess = ParameterUtils.calcESS(estimates);
	                if (ess > minESS || timeElapsed > timeLimitSeconds*1000) {
	                    if (logToScreen) {
	                        System.out.println("ESS = " + ess + " after " + sampleNr + " samples");
	                    }
	                    break;
	                }
	            }
            
            }

            sampleNr++;
        }

        return estimates;
    }

    // Downsample at regular intervals
    private static List<Double> thinList(List<Double> list, int newSize) {
        List<Double> sample = new ArrayList<Double>();
        int sampleEvery = Math.max(2, list.size() / newSize);

        // System.out.println("sample every " + sampleEvery + " ," + newSize + " ," + list.size());

        for (int i = 0; i < list.size(); i += sampleEvery) {
            sample.add(list.get(i));
        }


        return sample;
    }

}
