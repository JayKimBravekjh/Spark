from __future__ import print_function

import random
import argparse
import numpy as np

from pyspark import SparkConf, SparkContext
from pyspark.mllib.clustering import GaussianMixture


def parseVector(line):
    return np.array([float(x) for x in line.split(' ')])


if __name__ == "__main__":
    """
    Parameters
    ----------
    :param inputFile:        Input file path which contains data points
    :param k:                Number of mixture components
    :param convergenceTol:   Convergence threshold. Default to 1e-3
    :param maxIterations:    Number of EM iterations to perform. Default to 100
    :param seed:             Random seed
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('inputFile', help='Input File')
    parser.add_argument('k', type=int, help='Number of clusters')
    parser.add_argument('--convergenceTol', default=1e-3, type=float, help='convergence threshold')
    parser.add_argument('--maxIterations', default=100, type=int, help='Number of iterations')
    parser.add_argument('--seed', default=random.getrandbits(19),
                        type=long, help='Random seed')
    args = parser.parse_args()

    conf = SparkConf().setAppName("GMM")
    sc = SparkContext(conf=conf)

    lines = sc.textFile(args.inputFile)
    data = lines.map(parseVector)
    model = GaussianMixture.train(data, args.k, args.convergenceTol,
                                  args.maxIterations, args.seed)
    for i in range(args.k):
        print(("weight = ", model.weights[i], "mu = ", model.gaussians[i].mu,
               "sigma = ", model.gaussians[i].sigma.toArray()))
    print(("Cluster labels (first 100): ", model.predict(data).take(100)))
    sc.stop()
