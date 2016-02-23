from __future__ import print_function

import sys
import re

import numpy as np
from pyspark import SparkContext
from pyspark.ml.clustering import KMeans, KMeansModel
from pyspark.mllib.linalg import VectorUDT, _convert_to_vector
from pyspark.sql import SQLContext
from pyspark.sql.types import Row, StructField, StructType


def parseVector(line):
    array = np.array([float(x) for x in line.split(' ')])
    return _convert_to_vector(array)


if __name__ == "__main__":

    FEATURES_COL = "features"

    if len(sys.argv) != 3:
        print("Usage: kmeans_example.py <file> <k>", file=sys.stderr)
        exit(-1)
    path = sys.argv[1]
    k = sys.argv[2]

    sc = SparkContext(appName="Jay")
    sqlContext = SQLContext(sc)

    lines = sc.textFile(path)
    data = lines.map(parseVector)
    row_rdd = data.map(lambda x: Row(x))
    schema = StructType([StructField(FEATURES_COL, VectorUDT(), False)])
    df = sqlContext.createDataFrame(row_rdd, schema)

    kmeans = KMeans().setK(2).setSeed(1).setFeaturesCol(FEATURES_COL)
    model = kmeans.fit(df)
    centers = model.clusterCenters()

    print("Cluster Centers: ")
    for center in centers:
        print(center)

    sc.stop()
