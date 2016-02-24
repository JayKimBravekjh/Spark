
from __future__ import print_function

import sys

from pyspark import SparkContext
from pyspark.mllib.random import RandomRDDs


if __name__ == "__main__":
    if len(sys.argv) not in [1, 2]:
        print("Usage: random_rdd_generation", file=sys.stderr)
        exit(-1)

    sc = SparkContext(appName="Jay")

    numExamples = 10000  # number of examples to generate
    fraction = 0.1  # fraction of data to sample

    # Example: RandomRDDs.normalRDD
    normalRDD = RandomRDDs.normalRDD(sc, numExamples)
    print('Generated RDD of %d examples sampled from the standard normal distribution'
          % normalRDD.count())
    print('  First 5 samples:')
    for sample in normalRDD.take(5):
        print('    ' + str(sample))
    print()

    # Example: RandomRDDs.normalVectorRDD
    normalVectorRDD = RandomRDDs.normalVectorRDD(sc, numRows=numExamples, numCols=2)
    print('Generated RDD of %d examples of length-2 vectors.' % normalVectorRDD.count())
    print('  First 5 samples:')
    for sample in normalVectorRDD.take(5):
        print('    ' + str(sample))
    print()

    sc.stop()
