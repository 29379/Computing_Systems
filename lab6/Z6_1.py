import os
import numpy as np
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import LogisticRegression, DecisionTreeClassifier
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
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
dt = DecisionTreeClassifier(featuresCol='features', labelCol='label')

paramGridLr = ParamGridBuilder() \
    .addGrid(lr.regParam, [0.1, 0.01, 0.0]) \
    .addGrid(lr.maxIter, [1, 5, 10]) \
    .build()
paramGridDt = ParamGridBuilder() \
    .addGrid(dt.maxDepth, [2, 3, 4]) \
    .addGrid(dt.maxBins, [2, 10, 20]) \
    .build()

evaluator = BinaryClassificationEvaluator(labelCol='label') 

print("\n- - - - - -  - - - - - - - -- - - - - - -- - - - - -- \nLogistic Regression: \n")
crossvalLr = CrossValidator(estimator=lr,
                          estimatorParamMaps=paramGridLr,
                          evaluator=evaluator,
                          numFolds=5) 

cvModelLr = crossvalLr.fit(train_df)

bestModelLr = cvModelLr.bestModel
bestParamsLr = {
    param[0].name: param[1] 
    for param in bestModelLr.extractParamMap().items() 
        if param[0].name in ['regParam', 'maxIter']
}
print(f"\nBest parameters: {bestParamsLr}\n")                     # showing best parameters

predictionsLr = cvModelLr.transform(test_df)                       # making predictions
predictionsLr.select("features", "label",  "probability", "prediction").show()    # showing predictions
print("Model evaluation: ", evaluator.evaluate(predictionsLr))    # evaluating model
avgMetricsLr = cvModelLr.avgMetrics
print(f"\nAverage metrics for Logistic Regression: {avgMetricsLr}")


print("\n- - - - - -  - - - - - - - -- - - - - - -- - - - - -- \nDecision Tree Classifier: \n")
crossvalDt = CrossValidator(estimator=dt,
                          estimatorParamMaps=paramGridDt,
                          evaluator=evaluator,
                          numFolds=5)

cvModelDt = crossvalDt.fit(train_df)

bestModelDt = cvModelDt.bestModel
bestParamsDt = {
    param[0].name: param[1] 
    for param in bestModelDt.extractParamMap().items() 
        if param[0].name in ['maxDepth', 'maxBins']
}
print(f"\nBest parameters: {bestParamsDt}\n")                     # showing best parameters

predictionsDt = cvModelDt.transform(test_df)                       # making predictions
predictionsDt.select("features", "label",  "probability", "prediction").show()    # showing predictions
print("Model evaluation: ", evaluator.evaluate(predictionsLr))    # evaluating model
avgMetricsDt = cvModelDt.avgMetrics
print(f"\nAverage metrics for Decision Tree: {avgMetricsLr}")

