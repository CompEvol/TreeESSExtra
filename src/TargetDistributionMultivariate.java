package ccp.binomialess;

import org.apache.commons.math3.analysis.MultivariateFunction;
import org.apache.commons.math3.optim.InitialGuess;
import org.apache.commons.math3.optim.MaxEval;
import org.apache.commons.math3.optim.PointValuePair;
import org.apache.commons.math3.optim.SimpleBounds;
import org.apache.commons.math3.optim.nonlinear.scalar.GoalType;
import org.apache.commons.math3.optim.nonlinear.scalar.ObjectiveFunction;
import org.apache.commons.math3.optim.nonlinear.scalar.noderiv.BOBYQAOptimizer;
import org.apache.commons.math3.optim.nonlinear.scalar.noderiv.PowellOptimizer;
import org.apache.commons.math3.optim.nonlinear.scalar.noderiv.SimplexOptimizer;
import org.apache.commons.math3.optim.univariate.BrentOptimizer;
import org.apache.commons.math3.optim.univariate.SearchInterval;
import org.apache.commons.math3.optim.univariate.UnivariateObjectiveFunction;
import org.apache.commons.math3.optim.univariate.UnivariatePointValuePair;

public class TargetDistributionMultivariate extends TargetDistribution  implements MultivariateFunction {

	public TargetDistributionMultivariate(int[] frequencies1, int[] frequencies2, int l1, int l2) throws Exception {
		super(frequencies1, frequencies2, l1, l2);
	}
	
	

    /**
     * Returns double[] with five numbers -
     * first, the MLE or MAP of R, a, and b
     * second, the log-likelihood or posterior of this estimate
     * third, the ESS = r*(l1+l2)
     *
     * @return
     * @throws Exception
     */
	@Override
    public double[] getMaxLogP() throws Exception {


		// Or BOBYQAOptimizer or SimplexOptimizer or PowellOptimizer
		//SimplexOptimizer optimizer = new SimplexOptimizer(1e-6, 1e-6);
		//BOBYQAOptimizer optimizer = new BOBYQAOptimizer(5);
		PowellOptimizer optimizer = new PowellOptimizer(1e-8, 1e-8);


		// Lower  and upper bounds for (r, a, b)
		double[] lowerBounds = {0.0, 0.0, 0.0};  
        double[] upperBounds = {1.0, 10, 10}; 

        // Optimize to find the minimum
        PointValuePair result = optimizer.optimize(
                new MaxEval(10000),
                new InitialGuess(new double[]{0,0,0}),   // Initial guess for (r, a, b)
                new ObjectiveFunction(this),
                GoalType.MAXIMIZE
                //new SimpleBounds(lowerBounds, upperBounds) 
        );
        


        // Get the result
        double est[] = result.getPoint();
        double maxLogP = result.getValue();
        
        double rhat = est[0];
        double ahat = est[1];
        double bhat = est[2];
        
        // Transform
        //rhat = Math.exp(rhat) / (Math.exp(rhat) + 1);
		//ahat = Math.exp(ahat);
		//bhat = Math.exp(bhat);


        if (maxLogP == Double.POSITIVE_INFINITY) {
            throw new Exception("Positive infinity!");
        }

        double ESS = rhat * (this.l1 + this.l2);
        return new double[]{rhat, ahat, bhat, maxLogP, ESS};
    }
    
	

	@Override
	public double value(double[] params) {
		double r = params[0];
		double a = params[1];
		double b = params[2];
		
		
		// Transform out of real space
		//r = Math.exp(r) / (Math.exp(r) + 1);
		//a = Math.exp(a);
		//b = Math.exp(b);
		
		double logP = getLogLikelihood(r, a, b);
        return logP;
	}

}
