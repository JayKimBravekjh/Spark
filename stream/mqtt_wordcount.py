
"""
 A sample wordcount with MqttStream stream
 Usage: mqtt_wordcount.py <broker url> <topic>
 To run this in your local machine, you need to setup a MQTT broker and publisher first,
 Mosquitto is one of the open source MQTT Brokers, see
 http://mosquitto.org/
 Eclipse paho project provides number of clients and utilities for working with MQTT, see
 http://www.eclipse.org/paho/#getting-started
 and then run the example
    `$ bin/spark-submit --jars \
      external/mqtt-assembly/target/scala-*/spark-streaming-mqtt-assembly-*.jar \
      examples/src/main/python/streaming/mqtt_wordcount.py \
      tcp://localhost:1883 foo`
"""

import sys

from pyspark import SparkContext
from pyspark.streaming import StreamingContext
from pyspark.streaming.mqtt import MQTTUtils

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print >> sys.stderr, "Usage: mqtt_wordcount.py <broker url> <topic>"
        exit(-1)

    sc = SparkContext(appName="PythonStreamingMQTTWordCount")
    ssc = StreamingContext(sc, 1)

    brokerUrl = sys.argv[1]
    topic = sys.argv[2]

    lines = MQTTUtils.createStream(ssc, brokerUrl, topic)
    counts = lines.flatMap(lambda line: line.split(" ")) \
        .map(lambda word: (word, 1)) \
        .reduceByKey(lambda a, b: a+b)
    counts.pprint()

    ssc.start()
    ssc.awaitTermination()
