import os
import numpy as np
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.tuning import ParamGridBuilder, TrainValidationSplit
from pyspark.ml.evaluation import BinaryClassificationEvaluator

dataset_name = '../datasets/appendicitis.csv'

spark = SparkSession.builder.appName("Appendicitis").getOrCreate()
df = spark.read.csv(dataset_name, header=True, inferSchema=True)
df = df.withColumnRenamed(df.columns[-1], "label")
for i in range(7):
    df = df.withColumnRenamed(df.columns[i], f"c{i+1}")
feature_columns = df.columns[:-1]
assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")
df = assembler.transform(df)

train_df, test_df = df.randomSplit([0.5, 0.5], seed=42)
print(f"Training data: ")
train_df.show()
print(f"Testing data: ")
test_df.show()

lr = LogisticRegression(featuresCol='features', labelCol='label')  
paramGrid = ParamGridBuilder() \
    .addGrid(lr.regParam, [0.1, 0.01]) \
    .build()
evaluator = BinaryClassificationEvaluator(labelCol='label') 
tvs = TrainValidationSplit(estimator=lr,
                           estimatorParamMaps=paramGrid,
                           evaluator=evaluator,
                           trainRatio=0.8)

tvsModel = tvs.fit(train_df)                                    # choosing best set of parameters

bestModel = tvsModel.bestModel
bestParams = {
    param[0].name: param[1] 
    for param in bestModel.extractParamMap().items() 
        if param[0].name in ['regParam']
}
print(f"\nBest parameters: {bestParams}\n")                     # showing best parameters

predictions = tvsModel.transform(test_df)                       # making predictions
predictions.select("features", "label", "probability", "prediction").show()    # showing predictions
print("Model evaluation: ", evaluator.evaluate(predictions))    # evaluating model
