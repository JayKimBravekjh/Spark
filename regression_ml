from __future__ import print_function

from pyspark import SparkContext
from pyspark.sql import SQLContext
# $example on$
from pyspark.ml.regression import AFTSurvivalRegression
from pyspark.mllib.linalg import Vectors
# $example off$

if __name__ == "__main__":
    sc = SparkContext(appName="AFTSurvivalRegressionExample")
    sqlContext = SQLContext(sc)

    # $example on$
    training = sqlContext.createDataFrame([
        (1.218, 1.0, Vectors.dense(1.560, -0.605)),
        (2.949, 0.0, Vectors.dense(0.346, 2.158)),
        (3.627, 0.0, Vectors.dense(1.380, 0.231)),
        (0.273, 1.0, Vectors.dense(0.520, 1.151)),
        (4.199, 0.0, Vectors.dense(0.795, -0.226))], ["label", "censor", "features"])
    quantileProbabilities = [0.3, 0.6]
    aft = AFTSurvivalRegression(quantileProbabilities=quantileProbabilities,
                                quantilesCol="quantiles")

    model = aft.fit(training)

    # Print the coefficients, intercept and scale parameter for AFT survival regression
    print("Coefficients: " + str(model.coefficients))
    print("Intercept: " + str(model.intercept))
    print("Scale: " + str(model.scale))
    model.transform(training).show(truncate=False)
    # $example off$

    sc.stop()
Status API Training Shop Blog About Pricing
