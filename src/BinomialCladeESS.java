package ccp.binomialess;

import java.util.List;

public class BinomialCladeESS {

    public static void main(String[] args) throws Exception {

        // Perform test
        // test1();
        // test2();
        // test3();
        test4();
    }

    /**
     * Return the MLE of the ESS under the binomial clade sampling method
     *
     * @param cladeFrequencies1
     * @param chainLength1
     * @param cladeFrequencies2
     * @param chainLength2
     * @return
     * @throws Exception
     */
    public static double getESSML(int[] cladeFrequencies1, int chainLength1, int[] cladeFrequencies2, int chainLength2) throws Exception {
        TargetDistributionUnivariate dist = new TargetDistributionUnivariate(cladeFrequencies1, cladeFrequencies2, chainLength1, chainLength2);

        dist.setAB(1, 1);

        double[] mle = dist.getMaxLogP();
        System.out.println("r = " + mle[0] + " with logP " + mle[1] + " and ESS " + mle[2]);

        return mle[2];
    }

//    public static double getESSML_Multivariate(int[] cladeFrequencies1, int chainLength1, int[] cladeFrequencies2, int chainLength2) throws Exception {
//    	TargetDistributionMultivariate dist = new TargetDistributionMultivariate(cladeFrequencies1, cladeFrequencies2, chainLength1, chainLength2);
//        
//        double[] mle = dist.getMaxLogP();
//        System.out.println("r=" + mle[0] + ", a=" + mle[1] + ", b=" + mle[2] + " with logP " + mle[3] + " and ESS " + mle[4]);
//
//        return mle[4];
//    }

    /**
     * Return the posterior mean and median of the ESS (plus the lower and upper 95% HPD)
     *
     * @param cladeFrequencies1
     * @param chainLength1
     * @param cladeFrequencies2
     * @param chainLength2
     * @return
     * @throws Exception
     */
    public static double[] getESSBayesian(int[] cladeFrequencies1, int chainLength1, int[] cladeFrequencies2, int chainLength2) throws Exception {
        TargetDistributionUnivariate dist = new TargetDistributionUnivariate(cladeFrequencies1, cladeFrequencies2, chainLength1, chainLength2);
        long startTime = System.currentTimeMillis();
        // System.out.println("Bay - ");
        // System.out.println(Arrays.toString(cladeFrequencies1));
        // System.out.println(chainLength1);
        // System.out.println(Arrays.toString(cladeFrequencies2));
        // System.out.println(chainLength2);
        UnivariateMCMC mcmc = new UnivariateMCMC(dist, 0, 1);
        // mcmc.screenLogging(true); // Comment out to avoid being spammed with print lines
        List<Double> estimates = mcmc.run(200, 60);
        // Multiply estimates by scale factor to get tree ESS
        double scale = chainLength1 + chainLength2;

        // Summarise MCMC chain
        TraceStatistics trace = ParameterUtils.getTrace(estimates);
        double mean = trace.getMean() * scale;
        double median = trace.getMedian() * scale;
        double lower = trace.getHpdLower() * scale;
        double upper = trace.getHpdUpper() * scale;
        long estimatedTime = System.currentTimeMillis() - startTime;
        // System.out.println("mean ESS = " + mean + ";  median " + median + "; and 95% hpd (" + lower + ", " + upper + ")");
        // System.out.println("total time: " + (estimatedTime / 1000) + " seconds");

        return new double[]{mean, median, lower, upper};
    }

    public static double[] getESSBayesianMultivariate(int[] cladeFrequencies1, int chainLength1, int[] cladeFrequencies2, int chainLength2, int timeLimitSeconds) throws Exception {
        TargetDistributionMultivariate dist = new TargetDistributionMultivariate(cladeFrequencies1, cladeFrequencies2, chainLength1, chainLength2);

        // System.out.println(Arrays.toString(cladeFrequencies1));
        // System.out.println(chainLength1);
        // System.out.println(Arrays.toString(cladeFrequencies2));
        // System.out.println(chainLength2);
        MultivariateMCMC mcmc = new MultivariateMCMC(dist);

        mcmc.addParameter(0, 1); // r
        mcmc.addParameter(0.2, 5); // a
        mcmc.addParameter(0.2, 5); // b

        // mcmc.screenLogging(true); // Comment out to avoid being spammed with print lines
        List<List<Double>> estimates = mcmc.run(200, timeLimitSeconds);

        // Multiply estimates by scale factor to get tree ESS
        double scale = chainLength1 + chainLength2;

        // Summarise MCMC chain
        TraceStatistics rTrace = ParameterUtils.getTrace(estimates.get(0));
        double mean = rTrace.getMean() * scale;
        double median = rTrace.getMedian() * scale;
        double lower = rTrace.getHpdLower() * scale;
        double upper = rTrace.getHpdUpper() * scale;
        // System.out.println("mean ESS = " + mean + ";  median " + median + "; and 95% hpd (" + lower + ", " + upper + ")");

        // Summarise a
        TraceStatistics aTrace = ParameterUtils.getTrace(estimates.get(1));
        // System.out.println("mean a = " + aTrace.getMean() + ";  median " + aTrace.getMedian() + "; and 95% hpd (" + aTrace.getHpdLower() + ", " + aTrace.getHpdUpper() + ")");

        // Summarise b
        // TraceStatistics bTrace = ParameterUtils.getTrace(estimates.get(2));
        // System.out.println("mean a = " + bTrace.getMean() + ";  median " + bTrace.getMedian() + "; and 95% hpd (" + bTrace.getHpdLower() + ", " + bTrace.getHpdUpper() + ")");

        return new double[]{mean, median, lower, upper};
    }


    // Test chains simulated under DS1 with ESS=200 in each chain. Chain lengths of 4174 and 4141
    public static void test1() throws Exception {
        int l1 = 4174;
        int l2 = 4141;

        int[] frequencies1 = new int[]{4174, 4158, 4174, 4174, 4174, 4174, 4174, 4174, 4174, 4174, 4174, 4174, 4164, 4165, 4174, 4174, 3754, 3963, 4129, 3913, 3461, 2928, 2683, 2789, 2244, 2434, 1701, 1638, 1232, 1206, 687, 336, 249, 414, 186, 85, 119, 153, 162, 132, 15, 94, 420, 47, 168, 0, 43, 7, 9, 23, 0, 0, 20, 0, 61, 9, 0, 10, 0, 66, 35, 30, 30, 22, 16, 16, 16, 13, 4, 3};
        int[] frequencies2 = new int[]{4141, 4141, 4141, 4141, 4141, 4141, 4141, 4141, 4141, 4141, 4141, 4141, 4134, 4131, 4130, 4123, 4074, 4065, 4055, 4017, 3482, 2833, 2691, 2631, 2521, 2257, 1845, 1474, 1307, 1163, 837, 331, 297, 247, 200, 128, 126, 115, 93, 92, 86, 71, 67, 58, 43, 39, 33, 25, 22, 22, 18, 17, 17, 16, 13, 10, 7, 7, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

        double ess = getESSML(frequencies1, l1, frequencies2, l2);
        System.out.println("Test1: MLE of ESS=" + ess);


        getESSBayesian(frequencies1, l1, frequencies2, l2);
        getESSBayesianMultivariate(frequencies1, l1, frequencies2, l2, 60);
    }

    /**
     * A tricky example that mixes poorly in MCMC
     * True ESS = 20, ACT = 20
     *
     * @throws Exception
     */
    public static void test2() throws Exception {
        int l1 = 413;
        int l2 = 399;

        int[] frequencies1 = new int[]{115, 0, 0, 413, 413, 157, 413, 413, 413, 413, 60, 298, 392, 413, 376, 99, 20, 413, 332, 413, 81, 21, 413, 413, 37, 387, 0, 413, 413, 282, 413, 413, 54, 21, 336, 413, 413, 36, 413, 413, 77, 356, 77, 413, 0, 413, 413, 413, 413, 413, 26, 413, 413, 413, 413, 157, 333};
        int[] frequencies2 = new int[]{102, 18, 21, 399, 399, 160, 399, 399, 399, 381, 23, 297, 361, 399, 360, 35, 0, 399, 319, 399, 80, 38, 399, 399, 0, 399, 18, 399, 399, 304, 399, 399, 55, 47, 380, 399, 399, 59, 399, 399, 40, 293, 0, 399, 19, 399, 399, 399, 399, 399, 0, 399, 399, 399, 399, 204, 376};

        double ess = getESSML(frequencies1, l1, frequencies2, l2);
        System.out.println("Test2: MLE of ESS=" + ess);

        getESSBayesian(frequencies1, l1, frequencies2, l2);
        getESSBayesianMultivariate(frequencies1, l1, frequencies2, l2, 60);
    }

    /**
     * Another tricky example that mixes poorly in MCMC
     * True ESS = 10, ACT = 20
     *
     * @throws Exception
     */
    public static void test3() throws Exception {
        int l1 = 210;
        int l2 = 194;

        int[] frequencies1 = new int[]{62, 0, 210, 210, 90, 210, 210, 210, 210, 21, 148, 210, 210, 193, 18, 210, 169, 210, 41, 0, 210, 210, 17, 210, 0, 210, 210, 131, 210, 210, 20, 168, 210, 210, 17, 210, 210, 59, 193, 42, 210, 210, 210, 210, 210, 210, 210, 210, 210, 210, 102, 189};
        int[] frequencies2 = new int[]{18, 18, 194, 194, 98, 194, 194, 194, 176, 23, 176, 177, 194, 176, 15, 194, 134, 194, 60, 17, 194, 194, 0, 194, 18, 194, 194, 160, 194, 194, 15, 194, 194, 194, 40, 194, 194, 19, 154, 0, 194, 194, 194, 194, 194, 194, 194, 194, 194, 194, 81, 171};

        double ess = getESSML(frequencies1, l1, frequencies2, l2);
        System.out.println("Test3: MLE of ESS=" + ess);

        getESSBayesian(frequencies1, l1, frequencies2, l2);
        getESSBayesianMultivariate(frequencies1, l1, frequencies2, l2, 60);
    }


    /**
     * ACT 1
     * True ESS is 1000
     *
     * @throws Exception
     */
    public static void test4() throws Exception {
        int l1 = 500;
        int l2 = 500;

        int[] frequencies1 = new int[]{245, 12, 1, 132, 23, 0, 500, 500, 500, 38, 1, 1, 3, 500, 1, 7, 0, 1, 3, 181, 44, 499, 1, 499, 494, 480, 489, 19, 7, 334, 14, 3, 19, 500, 68, 20, 22, 5, 2, 4, 20, 500, 298, 222, 499, 2, 191, 372, 5, 3, 500, 500, 500, 499, 3, 2, 4, 500, 30, 500, 459, 329, 1, 0, 0, 1, 277, 107, 3, 1};
        int[] frequencies2 = new int[]{257, 14, 1, 139, 35, 1, 500, 500, 500, 31, 0, 1, 4, 500, 2, 6, 1, 0, 4, 159, 33, 500, 0, 500, 491, 486, 484, 12, 13, 327, 14, 0, 20, 500, 68, 14, 16, 7, 2, 3, 14, 500, 290, 207, 500, 4, 196, 372, 7, 4, 500, 500, 500, 499, 5, 0, 1, 499, 23, 500, 465, 340, 0, 1, 1, 3, 309, 111, 4, 0};

        int[] frequencies3 = new int[]{245, 3, 0, 26, 119, 0, 500, 500, 499, 34, 4, 1, 500, 3, 0, 0, 7, 2, 43, 200, 498, 500, 488, 478, 484, 10, 13, 350, 12, 13, 2, 500, 72, 18, 14, 14, 1, 2, 22, 3, 500, 295, 239, 500, 1, 196, 370, 2, 5, 1, 500, 500, 498, 497, 8, 2, 1, 1, 500, 21, 500, 470, 325, 0, 0, 4, 274, 103, 4, 1, 2};
        int[] frequencies4 = new int[]{254, 9, 1, 133, 24, 1, 500, 500, 498, 36, 8, 3, 500, 3, 1, 2, 0, 1, 178, 42, 500, 500, 497, 486, 488, 11, 13, 346, 19, 10, 0, 500, 6, 55, 16, 14, 4, 1, 1, 1, 14, 500, 217, 284, 500, 1, 204, 394, 4, 1, 0, 500, 500, 499, 499, 0, 4, 1, 0, 500, 23, 500, 466, 323, 1, 3, 1, 293, 103, 0, 3, 0};


        double ess = getESSML(frequencies1, l1, frequencies2, l2);
        System.out.println("Test4: MLE of ESS=" + ess);
        getESSBayesian(frequencies1, l1, frequencies2, l2);
        getESSBayesianMultivariate(frequencies1, l1, frequencies2, l2, 60);
    }


}
