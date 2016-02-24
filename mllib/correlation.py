from __future__ import print_function

import sys

from pyspark import SparkContext
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.stat import Statistics
from pyspark.mllib.util import MLUtils


if __name__ == "__main__":
    if len(sys.argv) not in [1, 2]:
        print("Usage: correlations (<file>)", file=sys.stderr)
        exit(-1)
    sc = SparkContext(appName="PythonCorrelations")
    if len(sys.argv) == 2:
        filepath = sys.argv[1]
    else:
        filepath = 'data/mllib/sample_linear_regression_data.txt'
    corrType = 'pearson'

    points = MLUtils.loadLibSVMFile(sc, filepath)\
        .map(lambda lp: LabeledPoint(lp.label, lp.features.toArray()))

    print()
    print('Summary of data file: ' + filepath)
    print('%d data points' % points.count())

    # Statistics (correlations)
    numFeatures = points.take(1)[0].features.size
    labelRDD = points.map(lambda lp: lp.label)
    for i in range(numFeatures):
        featureRDD = points.map(lambda lp: lp.features[i])
        corr = Statistics.corr(labelRDD, featureRDD, corrType)
        print('%d\t%g' % (i, corr))
    print()

    sc.stop()
