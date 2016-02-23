
from __future__ import print_function

from pyspark import SparkContext
from pyspark.sql import SQLContext
# $example on$
from pyspark.ml.feature import Word2Vec
# $example off$

if __name__ == "__main__":
    sc = SparkContext(appName="Jay")
    sqlContext = SQLContext(sc)

    # $example on$
    # Input data: Each row is a bag of words from a sentence or document.
    documentDF = sqlContext.createDataFrame([
        ("This is test".split(" "), ),
        ("real test".split(" "), ),
        ("real real test".split(" "), )
    ], ["text"])
    # Learn a mapping from words to Vectors.
    word2Vec = Word2Vec(vectorSize=3, minCount=0, inputCol="text", outputCol="result")
    model = word2Vec.fit(documentDF)
    result = model.transform(documentDF)
    for feature in result.select("result").take(3):
        print(feature)
    # $example off$

    sc.stop()
