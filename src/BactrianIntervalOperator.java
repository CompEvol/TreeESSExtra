package ccp.binomialess;

import org.apache.commons.math3.random.MersenneTwister;

/**
 * Bactrian interval operator from BEAST 2
 * Makes a proposal on a univariate parameter bound between two values, and returns the hastings ratio
 * Optimises
 */
public class BactrianIntervalOperator {
    long nAccepts = 0;
    long nRejects = 0;

    MersenneTwister rng;
    double scaleFactor = 0.9;
    double lower, upper;

    public BactrianIntervalOperator(double lower, double upper, MersenneTwister rng) {
        this.lower = lower;
        this.upper = upper;
        this.rng = rng;
    }

    /**
     * value: initial value
     * Returns out:
     * index 0: the new value
     * index 1: the log hastings ratio
     *
     * @param out
     */
    public void proposal(double value, double[] out) {

        // Uniform operator with some probability
        if (this.rng.nextFloat() < 0.2) {
            double newV = rng.nextDouble() * (upper-lower) + lower;
            out[0] = newV;
            out[1] = 0;
            return;
        }

        double scale = getScaler(value, scaleFactor);

        // Transform value
        double y = (upper - value) / (value - lower);
        y *= scale;
        double newValue = (upper + lower * y) / (y + 1.0);

        if (newValue < lower || newValue > upper) {
        	 //out[0] = value;
             //out[1] = Double.NEGATIVE_INFINITY;
             //return;
            throw new RuntimeException("programmer error: new value proposed outside range");
        }

        // Ensure that the value is not sitting on the limit (due to numerical issues for example)
        if (newValue <= lower || newValue >= upper) {
            out[0] = value;
            out[1] = Double.NEGATIVE_INFINITY;
            return;
        }

        double logHR = Math.log(scale) + 2.0 * Math.log((newValue - lower) / (value - lower));
        out[0] = newValue;
        out[1] = logHR;
    }

    // Bactrian(0.95) distribution
    public double getScaler(double oldValue, double scaleFactor) {
        // Sample a bactrian random variable b
        double b;
        double m = 0.95;
        if (this.rng.nextBoolean()) {
            b = m + rng.nextGaussian() * Math.sqrt(1 - m * m);
        } else {
            b = -m + rng.nextGaussian() * Math.sqrt(1 - m * m);
        }


        // Convert to a scale factor
        double scale = 0;
        scale = scaleFactor * b;
        scale = Math.exp(scale);
        return scale;
    }

    public void optimize(double logAlpha) {
        double delta = calcDelta(logAlpha);
        double scaleFactorNew = this.scaleFactor;
        delta += Math.log(scaleFactorNew);
        scaleFactorNew = Math.exp(delta);
        this.scaleFactor = scaleFactorNew;
    }

    private double calcDelta(final double logAlpha) {
        final double target = 0.3;
        double count = nAccepts + nRejects + 1.0;
        final double deltaP = ((1.0 / count) * (Math.exp(Math.min(logAlpha, 0)) - target));

        if (deltaP > -Double.MAX_VALUE && deltaP < Double.MAX_VALUE) {
            return deltaP;
        }
        return 0;
    }

    public void accept() {
        this.nAccepts++;
    }

    public void reject() {
        this.nRejects++;
    }
}
