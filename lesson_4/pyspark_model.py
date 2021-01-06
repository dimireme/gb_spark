# coding=utf-8
# ssh BD_274_ashadrin@89.208.223.141 -i ~/.ssh/id_rsa_gb_spark
# /spark2.4/bin/pyspark --driver-memory 512m --driver-cores 1 --master local[1]

from pyspark.ml import Transformer, Pipeline
from pyspark.ml.feature import RegexTokenizer, HashingTF, IDF
from pyspark.ml.classification import LogisticRegression, DecisionTreeClassifier, RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.tuning import TrainValidationSplit, ParamGridBuilder
from pyspark.sql.functions import udf, concat, col, lit
from pyspark.sql.types import IntegerType
from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("shadrin_spark").getOrCreate()

# ========================== DEFINE UDF ==========================
# rating => label.
def ratingToClass(rating):
    if rating < 8:
        return 1  # 'A'
    elif rating < 16:
        return 2  # 'B'
    elif rating < 32:
        return 3  # 'C'
    else:
        return 4  # 'D'


# label => rating range.
def classToRating(label):
    if (label == 'A'):
        return '1 - 7'
    elif (label == 'B'):
        return '8 - 15'
    elif (label == 'C'):
        return '16 - 31'
    elif (label == 'D'):
        return '32 - inf'
    else:
        return None


ratingToClassUdf = udf(ratingToClass)
classToRatingUdf = udf(classToRating)

# ========================== CUSTOM TRANSFORMER ==================
class ColumnSelector(Transformer):
    def __init__(self):
        super(ColumnSelector, self).__init__()
    def _transform(self, df):
        df = df \
            .withColumn('rating', col('rating').cast(IntegerType())) \
            .dropna(subset=('rating', 'first_5_sentences', 'last_5_sentences')) \
            .withColumn('rating_class', ratingToClassUdf(col("rating")).cast(IntegerType())) \
            .withColumn('sentences', concat(
                col('first_5_sentences'),
                lit(' '),
                col('last_5_sentences')
            )) \
            .select('link', 'rating', 'rating_class', 'sentences')
        return df


# ========================== READ DATA ==========================
train, test = spark.read.option("header", True) \
.csv("/user/admin/habr_data.csv") \
.randomSplit([.8, .2], seed=42)

print("There are " + str(train.count()) + " rows in the training set, and " + str(test.count()) + " in the test set")
# There are 8484 rows in the training set, and 2073 in the test set

# ========================= DEFINE PIPELINE =====================
column_selector = ColumnSelector()
regexTokenizer = RegexTokenizer(
    inputCol="sentences",
    outputCol="sentences_words",
    pattern="[^a-zа-яё]",
    gaps=True
).setMinTokenLength(3)

hashingTF = HashingTF(inputCol="sentences_words", outputCol="tf_words", numFeatures=pow(2, 18))
idf = IDF(inputCol="tf_words", outputCol="features")


# ================ CREATE PIPELINE ========================
def evaluateModel(method, metric_evaluator):
    pipeline = Pipeline(stages=[column_selector, regexTokenizer, hashingTF, idf, method])
    model = pipeline.fit(train)
    prediction_train = model.transform(train)
    prediction_test = model.transform(test)
    train_ml_metric = metric_evaluator.evaluate(prediction_train)
    test_ml_metric = metric_evaluator.evaluate(prediction_test)
    print("Train ML metric: " + str(train_ml_metric))
    print("Test  ML metric: " + str(test_ml_metric))


# ================= ML METRICS ==========================
evaluator = MulticlassClassificationEvaluator(predictionCol='prediction', labelCol='rating_class', metricName='f1')


# ================= LOGISTIC REGRESSION ======================
lr = LogisticRegression(
    maxIter=10,
    regParam=0.2,
    featuresCol='features',
    labelCol='rating_class',
    predictionCol='prediction'
)

evaluateModel(lr, evaluator)
# Train ML metric: 0.999292791276
# Test  ML metric: 0.335689172959


# ================= DECISION TREE ===================
dt = DecisionTreeClassifier(
    labelCol="rating_class",
    featuresCol="features",
    maxDepth=3
)

evaluateModel(dt, evaluator)
# Train ML metric: 0.239470480314
# Test  ML metric: 0.230254753346


# ================ RANDOM FOREST ====================
rf = RandomForestClassifier(
    labelCol="rating_class",
    featuresCol="features",
    maxDepth=2,
    numTrees=10
)

evaluateModel(rf, evaluator)
# Train ML metric: 0.147040516869
# Test  ML metric: 0.137823674985


# ============= LOGISTIC REGRESSION TUNING ==================
lr = LogisticRegression(
    maxIter=10,
    regParam=0.2,
    featuresCol='features',
    labelCol='rating_class',
    predictionCol='prediction'
)

pipeline = Pipeline(stages=[column_selector, regexTokenizer, hashingTF, idf, lr])

paramGrid = ParamGridBuilder()  \
    .addGrid(lr.regParam, [0.1, 0.3])\
    .addGrid(hashingTF.numFeatures, [pow(2, 10), pow(2, 14)]) \
    .build()

tvs = TrainValidationSplit(
    estimator=pipeline,
    estimatorParamMaps=paramGrid,
    evaluator=evaluator,
    trainRatio=0.8
)

tvsModel = tvs.fit(train)

prediction_train = tvsModel.transform(train)
prediction_test = tvsModel.transform(test)
train_ml_metric = evaluator.evaluate(prediction_train)
test_ml_metric = evaluator.evaluate(prediction_test)
print("Train ML metric: " + str(train_ml_metric))
print("Test  ML metric: " + str(test_ml_metric))

# Train ML metric: 0.498068036001
# Test  ML metric: 0.324369298692

prediction_test.select('rating', 'rating_class', 'prediction').show(20)

