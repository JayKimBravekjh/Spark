

# $example on$
from pyspark.mllib.fpm import FPGrowth
# $example off$
from pyspark import SparkContext

if __name__ == "__main__":
    sc = SparkContext(appName="Jay")

    # $example on$
    data = sc.textFile("data/mllib/sample_fpgrowth.txt")
    transactions = data.map(lambda line: line.strip().split(' '))
    model = FPGrowth.train(transactions, minSupport=0.2, numPartitions=10)
    result = model.freqItemsets().collect()
    for fi in result:
        print(fi)
    # $example off$
