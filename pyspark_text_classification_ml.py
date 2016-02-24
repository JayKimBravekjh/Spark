from __future__ import print_function

from pyspark import SparkContext
from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import HashingTF, Tokenizer
from pyspark.sql import Row, SQLContext


if __name__ == "__main__":
    sc = SparkContext(appName="SimpleTextClassificationPipeline")
    sqlContext = SQLContext(sc)

    # Prepare training documents, which are labeled.
    LabeledDocument = Row("id", "text", "label")
    training = sc.parallelize([(0, "a b c d e spark", 1.0),
                               (1, "b d", 0.0),
                               (2, "spark f g h", 1.0),
                               (3, "hadoop mapreduce", 0.0)]) \
        .map(lambda x: LabeledDocument(*x)).toDF()

    # Configure an ML pipeline, which consists of tree stages: tokenizer, hashingTF, and lr.
    tokenizer = Tokenizer(inputCol="text", outputCol="words")
    hashingTF = HashingTF(inputCol=tokenizer.getOutputCol(), outputCol="features")
    lr = LogisticRegression(maxIter=10, regParam=0.001)
    pipeline = Pipeline(stages=[tokenizer, hashingTF, lr])

    # Fit the pipeline to training documents.
    model = pipeline.fit(training)

    # Prepare test documents, which are unlabeled.
    Document = Row("id", "text")
    test = sc.parallelize([(4, "spark i j k"),
                           (5, "l m n"),
                           (6, "spark hadoop spark"),
                           (7, "apache hadoop")]) \
        .map(lambda x: Document(*x)).toDF()

    # Make predictions on test documents and print columns of interest.
    prediction = model.transform(test)
    selected = prediction.select("id", "text", "prediction")
    for row in selected.collect():
        print(row)

    sc.stop()
