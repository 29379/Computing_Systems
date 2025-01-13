from pyspark.sql import SparkSession
import findspark
import os
import sys

# os.environ['PYSPARK_PYTHON'] = sys.executable
# os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable

findspark.init()
spark = SparkSession.builder.appName("-").getOrCreate()
#example dataframe
df = spark.createDataFrame([("a", 1), ("b", 2)], ["letter", "number"])
df.show()