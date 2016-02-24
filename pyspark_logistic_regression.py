
if __name__ == "__main__":

    if len(sys.argv) > 1:
        print("Usage: logistic_regression", file=sys.stderr)
        exit(-1)

    sc = SparkContext(appName="Jay")
    sqlContext = SQLContext(sc)

    # Load and parse the data file into a dataframe.
    df = MLUtils.loadLibSVMFile(sc, "data/mllib/sample_libsvm_data.txt").toDF()

    # Map labels into an indexed column of labels in [0, numLabels)
    stringIndexer = StringIndexer(inputCol="label", outputCol="indexedLabel")
    si_model = stringIndexer.fit(df)
    td = si_model.transform(df)
    [training, test] = td.randomSplit([0.7, 0.3])

    lr = LogisticRegression(maxIter=100, regParam=0.3).setLabelCol("indexedLabel")
    lr.setElasticNetParam(0.8)

    # Fit the model
    lrModel = lr.fit(training)

    predictionAndLabels = lrModel.transform(test).select("prediction", "indexedLabel") \
        .map(lambda x: (x.prediction, x.indexedLabel))

    metrics = MulticlassMetrics(predictionAndLabels)
    print("weighted f-measure %.3f" % metrics.weightedFMeasure())
    print("precision %s" % metrics.precision())
    print("recall %s" % metrics.recall())

    sc.stop()
