import glob
import os
import sys
import java
with java:
  from java.util import Properties
  from org.apache.flink.streaming.api.collector.selector import OutputSelector
  from org.apache.flink.api.common.serialization import SimpleStringSchema
  from org.apache.flink.streaming.connectors.kafka import FlinkKafkaConsumer09

exec_env = ExecutionEnvironment.get_execution_environment()
exec_env.set_parallelism(2)
t_config = TableConfig()
t_env = BatchTableEnvironment.create(exec_env, t_config)

props = Properties()
topic = "kafkaFlink"
config = {"bootstrap_servers": "localhost:9092",
          "group_id": "flink_test",
          "topics": [topic]}
props.setProperty("group_id", config['group_id'])
props.setProperty("zookeeper.connect", "localhost:2181")

consumer = FlinkKafkaConsumer09([config["topics"]], SimpleStringSchema(), props)

for message in consumer:
  print("topic: %s message=%s" % (message.topic, message.value))

print("end")
