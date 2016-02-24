
from __future__ import print_function

from pyspark import SparkContext
# $example on$
from pyspark.ml.feature import HashingTF, IDF, Tokenizer
# $example off$
from pyspark.sql import SQLContext

if __name__ == "__main__":
    sc = SparkContext(appName="Jay")
    sqlContext = SQLContext(sc)

    # $example on$
    sentenceData = sqlContext.createDataFrame([
        (0, "This is test"),
        (0, "test is test"),
        (1, "real test")
    ], ["label", "sentence"])
    tokenizer = Tokenizer(inputCol="sentence", outputCol="words")
    wordsData = tokenizer.transform(sentenceData)
    hashingTF = HashingTF(inputCol="words", outputCol="rawFeatures", numFeatures=20)
    featurizedData = hashingTF.transform(wordsData)
    idf = IDF(inputCol="rawFeatures", outputCol="features")
    idfModel = idf.fit(featurizedData)
    rescaledData = idfModel.transform(featurizedData)
    for features_label in rescaledData.select("features", "label").take(3):
        print(features_label)
    # $example off$

    sc.stop()
