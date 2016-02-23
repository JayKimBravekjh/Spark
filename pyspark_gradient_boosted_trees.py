

from __future__ import print_function

import sys

from pyspark import SparkContext
from pyspark.ml.classification import GBTClassifier
from pyspark.ml.feature import StringIndexer
from pyspark.ml.regression import GBTRegressor
from pyspark.mllib.evaluation import BinaryClassificationMetrics, RegressionMetrics
from pyspark.mllib.util import MLUtils
from pyspark.sql import Row, SQLContext

def testClassification(train, test):
    # Train a GradientBoostedTrees model.

    rf = GBTClassifier(maxIter=30, maxDepth=4, labelCol="indexedLabel")

    model = rf.fit(train)
    predictionAndLabels = model.transform(test).select("prediction", "indexedLabel") \
        .map(lambda x: (x.prediction, x.indexedLabel))

    metrics = BinaryClassificationMetrics(predictionAndLabels)
    print("AUC %.3f" % metrics.areaUnderROC)


def testRegression(train, test):
    # Train a GradientBoostedTrees model.

    rf = GBTRegressor(maxIter=30, maxDepth=4, labelCol="indexedLabel")

    model = rf.fit(train)
    predictionAndLabels = model.transform(test).select("prediction", "indexedLabel") \
        .map(lambda x: (x.prediction, x.indexedLabel))

    metrics = RegressionMetrics(predictionAndLabels)
    print("rmse %.3f" % metrics.rootMeanSquaredError)
    print("r2 %.3f" % metrics.r2)
    print("mae %.3f" % metrics.meanAbsoluteError)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        print("Usage: gradient_boosted_trees", file=sys.stderr)
        exit(1)
    sc = SparkContext(appName="Jay")
    sqlContext = SQLContext(sc)

    # Load and parse the data file into a dataframe.
    df = MLUtils.loadLibSVMFile(sc, "data/mllib/sample_libsvm_data.txt").toDF()

    # Map labels into an indexed column of labels in [0, numLabels)
    stringIndexer = StringIndexer(inputCol="label", outputCol="indexedLabel")
    si_model = stringIndexer.fit(df)
    td = si_model.transform(df)
    [train, test] = td.randomSplit([0.7, 0.3])
    testClassification(train, test)
    testRegression(train, test)
    sc.stop()
