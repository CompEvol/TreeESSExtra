package ccp.binomialess;

import java.util.ArrayList;
import java.util.List;

import org.apache.commons.math3.analysis.MultivariateFunction;
import org.apache.commons.math3.analysis.UnivariateFunction;
import org.apache.commons.math3.random.MersenneTwister;

public class MultivariateMCMC {
	
    MersenneTwister rng = new MersenneTwister();
    List<BactrianIntervalOperator> operators;
    MultivariateFunction targetDistribution;
    boolean logToScreen = false;
    int logEvery = 1000;
    final int maxLogEvery = 50000;
    int burninEvery = 5; // Discard the first element every n logs

	public MultivariateMCMC(MultivariateFunction targetDistribution) {
		
		this.targetDistribution = targetDistribution;
        this.operators = new ArrayList<>();
        
        //for (int i = 0l i < dimension; )
        //this.operators.add(new BactrianIntervalOperator(paramLower, paramUpper, this.rng));
        
    }
	
	
	public void addParameter(double lower, double upper) {
		this.operators.add(new BactrianIntervalOperator(lower, upper, this.rng));
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
    public  List<List<Double>> run(double minESS, double timeLimitSeconds) {
    	
    	
    	long startTime = System.currentTimeMillis();
        int maxListSize = (int) (minESS * 2);
        int dimension = this.operators.size();

        // Initial values
        double theta[] = new double[dimension];
        double[] thetaPrime = new double[dimension];
        for (int i = 0; i < dimension; i ++) {
        	double lower = this.operators.get(i).lower;
        	double upper = this.operators.get(i).upper;
        	double x = (upper - lower) * this.rng.nextDouble() + lower;
        	theta[i] = x;
        	
        	//System.out.println(i + "=" + x);
        	
        }
        
        
        double logP = this.targetDistribution.value(theta);

        // Estimates
        List<List<Double>> estimates = new ArrayList<>();
        for (int i = 0; i < dimension; i ++) {
        	estimates.add(new ArrayList<>());
        }

        // Proposal variables
        double[] proposalArr = new double[2];
        double logHR, logPPrime, logAlpha, alpha, u;

        // Repeat the MCMC loop until convergence
        long sampleNr = 0;
        int burninCycle = 0;
        while (true) {
        	
        	// Select a proposal
        	int opNum = this.rng.nextInt(dimension);
        	BactrianIntervalOperator operator = this.operators.get(opNum);
        	
        	
            // Make proposal
        	operator.proposal(theta[opNum], proposalArr);
        	double paramValPrime = proposalArr[0];
        	thetaPrime[opNum] = paramValPrime;
            logHR = proposalArr[1];

            // Compute new logP
            logPPrime = this.targetDistribution.value(thetaPrime);

            // Acceptance probability
            logAlpha = logPPrime - logP + logHR;
            alpha = Math.min(1, Math.exp(logAlpha));
            
            
            //System.out.println("proposed " + paramValPrime + " from " + theta[opNum] + " for param " + opNum);

            u = rng.nextDouble();
            if (logHR != Double.NEGATIVE_INFINITY && u < alpha) {
            	
                // Accept
            	theta[opNum] = paramValPrime;
                logP = logPPrime;
                operator.accept();
                
            } else {
            	
                // Reject
                operator.reject();
            }

            if (logHR != Double.NEGATIVE_INFINITY) {
                operator.optimize(logAlpha);
            }

            // Sample value
            if (sampleNr % this.logEvery == 0) {

                if (logToScreen) {
                	double ess = ParameterUtils.calcESSOfTraces(estimates);
                	System.out.print("Sample nr " + sampleNr + ". MCMC minESS=" + ess + " / " + estimates.get(0).size());
                	for (int i = 0; i < dimension; i ++) {
                		System.out.print(" theta" + i + "=" + theta[i]);
    				}
                	System.out.println(". MCMC minESS=" + ess + " / " + estimates.get(0).size());
                	
                
                }
                
				for (int i = 0; i < dimension; i ++) {
					estimates.get(i).add(theta[i]);
				}
                
                burninCycle++;

                // Thin the list to reduce computational overhead
                boolean timeToThin = estimates.get(0).size() > maxListSize;

                // From time to time randomly check that the ESS to sample size ratio is large
                // A list size over 200 becomes much slower to add entries to
                if (estimates.get(0).size() > 200 & this.rng.nextFloat() < 0.1) {
                	double ess = ParameterUtils.calcESSOfTraces(estimates);
                    if (ess / estimates.size() < 0.5) {
                        timeToThin = true;
                    }

                }

                if (timeToThin && this.logEvery < maxLogEvery) {

                    int factor = 2;
                    this.logEvery = this.logEvery * factor;
                    for (int i = 0; i < dimension; i ++) {
                    	List<Double> thinned = thinList(estimates.get(i), maxListSize / factor);
                    	estimates.set(i, thinned);
                    }
                    

                    if (logToScreen) {
                    	double ess = ParameterUtils.calcESSOfTraces(estimates);
                        System.out.println("Thinning list down to " + (maxListSize / factor) + ". MCMC ESS=" + ess);
                    }

                }

                // Remove front from list?
                if (burninCycle % this.burninEvery == 0) {
                	for (int i = 0; i < dimension; i ++) {
                		 estimates.get(i).remove(0);
                    }
                    burninCycle = 0;
                }
            

            
	            // Check for convergence or timeout
                //System.out.println("time=" + timeElapsed + " " + startTime);\
                double timeElapsed = (System.currentTimeMillis() - startTime);
	            if (estimates.get(0).size() > minESS || timeElapsed > timeLimitSeconds*1000) {
	                double ess = ParameterUtils.calcESSOfTraces(estimates);
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
