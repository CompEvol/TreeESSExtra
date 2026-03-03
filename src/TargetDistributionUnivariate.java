package ccp.binomialess;

import org.apache.commons.math3.analysis.UnivariateFunction;
import org.apache.commons.math3.optim.MaxEval;
import org.apache.commons.math3.optim.nonlinear.scalar.GoalType;
import org.apache.commons.math3.optim.univariate.BrentOptimizer;
import org.apache.commons.math3.optim.univariate.SearchInterval;
import org.apache.commons.math3.optim.univariate.UnivariateObjectiveFunction;
import org.apache.commons.math3.optim.univariate.UnivariatePointValuePair;


/**
 * Optimises one parameter only - the ess
 */
public class TargetDistributionUnivariate extends TargetDistribution  implements UnivariateFunction {

	
	
	public TargetDistributionUnivariate(int[] frequencies1, int[] frequencies2, int l1, int l2) throws Exception {
		super(frequencies1, frequencies2, l1, l2);
		// TODO Auto-generated constructor stub
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
	@Override
    public double[] getMaxLogP() throws Exception {


        BrentOptimizer optimiser = new BrentOptimizer(1e-10, 1e-14);

        // Define the search interval (start, end, initial guess)
        SearchInterval searchInterval = new SearchInterval(this.minR, this.maxR, (this.maxR + this.minR) / 2);

        // Optimize to find the minimum
        UnivariatePointValuePair result = optimiser.optimize(
                new MaxEval(MAX_EVAL),
                new UnivariateObjectiveFunction(this),
                GoalType.MAXIMIZE,
                searchInterval
        );

        // Get the result
        double est = result.getPoint();
        double maxLogP = result.getValue();


        if (maxLogP == Double.POSITIVE_INFINITY) {
            throw new Exception("Positive infinity!");
        }

        double ESS = est * (this.l1 + this.l2);
        return new double[]{est, maxLogP, ESS};
    }
    

    // Called by integrator/optimiser.
    // Make sure to set univariateTask appropriately to return the right term
    @Override
    public double value(double r) {
    	double logP = getLogLikelihood(r, this.alpha, this.beta);
        return logP;
    }


	
	
}
