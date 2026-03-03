package ccp.binomialess;

import java.util.List;

public class ParameterUtils {

    final static int MAX_LAG = 2000;


    public static double calcESSOfTraces(List<List<Double>> traces) {

        double ess = Double.POSITIVE_INFINITY;
        for (int i = 0; i < traces.size(); i++) {
            double ess2 = ParameterUtils.calcESS(traces.get(i));
            if (ess2 < ess) ess = ess2;
        }
        return ess;

    }

    /**
     * return ESS time of a sample, batch version.
     * Can be used to calculate effective sample size
     * Code borrowed from BEAST 2 / Tracer
     *
     * @param trace: values from which the ACT is calculated
     */
    public static double calcESS(List<Double> trace) {
        return calcESS(trace.toArray(new Double[0]), 1);
    }

    public static double calcESS(Double[] trace, int sampleInterval) {
        return trace.length / (ACT(trace, sampleInterval) / sampleInterval);
    }

    public static double ACT(Double[] trace, int sampleInterval) {
        /** sum of trace, excluding burn-in **/
        double sum = 0.0;
        /** keep track of sums of trace(i)*trace(i_+ lag) for all lags, excluding burn-in  **/
        double[] squareLaggedSums = new double[MAX_LAG];
        double[] autoCorrelation = new double[MAX_LAG];
        for (int i = 0; i < trace.length; i++) {
            sum += trace[i];
            // calculate mean
            final double mean = sum / (i + 1);

            // calculate auto correlation for selected lag times
            // sum1 = \sum_{start ... totalSamples-lag-1} trace
            double sum1 = sum;
            // sum2 = \sum_{start+lag ... totalSamples-1} trace
            double sum2 = sum;
            for (int lagIndex = 0; lagIndex < Math.min(i + 1, MAX_LAG); lagIndex++) {
                squareLaggedSums[lagIndex] = squareLaggedSums[lagIndex] + trace[i - lagIndex] * trace[i];
                // The following line is the same approximation as in Tracer
                // (valid since mean *(samples - lag), sum1, and sum2 are approximately the same)
                // though a more accurate estimate would be
                // autoCorrelation[lag] = m_fSquareLaggedSums.get(lag) - sum1 * sum2
                autoCorrelation[lagIndex] = squareLaggedSums[lagIndex] - (sum1 + sum2) * mean + mean * mean * (i + 1 - lagIndex);
                autoCorrelation[lagIndex] /= (i + 1 - lagIndex);
                sum1 -= trace[i - lagIndex];
                sum2 -= trace[lagIndex];
            }
        }

        final int maxLag = Math.min(trace.length, MAX_LAG);
        double integralOfACFunctionTimes2 = 0.0;
        for (int lagIndex = 0; lagIndex < maxLag; lagIndex++) //{
            if (lagIndex == 0) //{
                integralOfACFunctionTimes2 = autoCorrelation[0];
            else if (lagIndex % 2 == 0)
                // fancy stopping criterion - see main comment in Tracer code of BEAST 1
                if (autoCorrelation[lagIndex - 1] + autoCorrelation[lagIndex] > 0) //{
                    integralOfACFunctionTimes2 += 2.0 * (autoCorrelation[lagIndex - 1] + autoCorrelation[lagIndex]);
                else
                    // stop
                    break;
        //}
        //}
        //}

        // auto correlation time
        return sampleInterval * integralOfACFunctionTimes2 / autoCorrelation[0];
    }

    public static double stdErrorOfMean(Double[] trace, int sampleInterval) {
        /** sum of trace, excluding burn-in **/
        double sum = 0.0;
        /** keep track of sums of trace(i)*trace(i_+ lag) for all lags, excluding burn-in  **/
        double[] squareLaggedSums = new double[MAX_LAG];
        double[] autoCorrelation = new double[MAX_LAG];
        for (int i = 0; i < trace.length; i++) {
            sum += trace[i];
            // calculate mean
            final double mean = sum / (i + 1);

            // calculate auto correlation for selected lag times
            // sum1 = \sum_{start ... totalSamples-lag-1} trace
            double sum1 = sum;
            // sum2 = \sum_{start+lag ... totalSamples-1} trace
            double sum2 = sum;
            for (int lagIndex = 0; lagIndex < Math.min(i + 1, MAX_LAG); lagIndex++) {
                squareLaggedSums[lagIndex] = squareLaggedSums[lagIndex] + trace[i - lagIndex] * trace[i];
                // The following line is the same approximation as in Tracer
                // (valid since mean *(samples - lag), sum1, and sum2 are approximately the same)
                // though a more accurate estimate would be
                // autoCorrelation[lag] = m_fSquareLaggedSums.get(lag) - sum1 * sum2
                autoCorrelation[lagIndex] = squareLaggedSums[lagIndex] - (sum1 + sum2) * mean + mean * mean * (i + 1 - lagIndex);
                autoCorrelation[lagIndex] /= (i + 1 - lagIndex);
                sum1 -= trace[i - lagIndex];
                sum2 -= trace[lagIndex];
            }
        }

        final int maxLag = Math.min(trace.length, MAX_LAG);
        double integralOfACFunctionTimes2 = 0.0;
        for (int lagIndex = 0; lagIndex < maxLag; lagIndex++) //{
            if (lagIndex == 0) //{
                integralOfACFunctionTimes2 = autoCorrelation[0];
            else if (lagIndex % 2 == 0)
                // fancy stopping criterion - see main comment in Tracer code of BEAST 1
                if (autoCorrelation[lagIndex - 1] + autoCorrelation[lagIndex] > 0) //{
                    integralOfACFunctionTimes2 += 2.0 * (autoCorrelation[lagIndex - 1] + autoCorrelation[lagIndex]);
                else
                    // stop
                    break;
        //}
        //}
        //}

        // auto correlation time
        return Math.sqrt(integralOfACFunctionTimes2 / trace.length);
    }

    public static TraceStatistics getTrace(List<Double> estimates) {
        double[] ests = new double[estimates.size()];
        for (int i = 0; i < ests.length; i++) {
            ests[i] = estimates.get(i);
        }
        TraceStatistics trace = new TraceStatistics(ests);
        return trace;
    }
}





