from pyspark.sql.functions import dayofmonth, month, hour
from pyspark.sql.session import SparkSession
from pyspark.sql.types import DoubleType, StructType, StructField, TimestampType, StringType
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.clustering import KMeans
import pyspark.sql.functions as func

# Initializing Spark Session
spark = SparkSession.builder.appName('Spark_Streaming_KMeans').getOrCreate()

# Setting up the schema
schema = StructType([StructField("dt", TimestampType(), True),
                     StructField("lat", DoubleType(), True),
                     StructField("lon", DoubleType(), True),
                     StructField("base", StringType(), True)]
                    )

# Reading the Data into spark
Dataset = spark.read.option("inferSchema", "false").schema(schema).csv("G:\\Data\\Input_Streaming_Spark.csv")

Dataset.show(5)
print("Raw Dataset")

# Defining feature array
assembler = VectorAssembler(inputCols=("lat", "lon"), outputCol='features')
Dataframe = assembler.transform(Dataset)

# Dataset with feature
Dataframe.show()
print("Dataset with feature")

# setting K means k = 20 and Max Iteration to 5
kmeans = KMeans().setK(20).setMaxIter(5)

# fitting out features into K means
model = kmeans.fit(Dataframe.select('features'))

# Save your model
# model.save("F:\\kMeans")

# Adding the prediction from K means to the Dataset
clusters = model.transform(Dataframe)

clusters.show()
print("K means predictions")

clusters.select(month("dt").alias("month"), dayofmonth("dt").alias("day"), hour("dt").alias(
    "hour"), "prediction").groupBy("month", "day", "hour", "prediction").agg(
    func.count("prediction").alias("count")).orderBy("day", "hour", "prediction").show()
print("Count Total")

clusters.select(hour("dt").alias("hour"), "prediction").groupBy("hour", "prediction").agg(
    func.count("prediction").alias("count")).orderBy(
    func.desc("count")).show()
print("Count Total ordered by count")

clusters.groupBy("prediction").count().show()
print("Counts in each cluster")
