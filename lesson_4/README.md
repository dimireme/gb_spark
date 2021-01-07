## Урок 4. Машинное обучение на pySpark на примере линейной регрессии

### Задание

1. Построить распределение статей в датасете по `rating` с `bin_size = 10`.
2. Написать функцию `ratingToClass(rating: Int): String`, которая определяет категорию статьи (A, B, C, D) на основе рейтинга. Границы для классов подобрать самостоятельно.
3. Добавить к датасету категориальную фичу `rating_class`. При добавлении колонки использовать `udf` из функции в предыдущем пункте.
4. Построить модель логистической регрессии (one vs all) для классификации статей по рассчитанным классам.
5. Получить `F1 score` для получившейся модели

### Решение

Первые пункты выполнил [в тетрадке в цепелине](http://185.241.193.174:9995/#/notebook/2FUFZ2PAD). Тут приведу скрины некоторых этапов. Остальной код запускал в консольном спарке (команда `/spark2.4/bin/pyspark --driver-memory 512m --driver-cores 1 --master local[1]`) на одной из нод кластера. Весь код есть в файле `pyspark_model.py` и [в тетрадке в цепелине](http://185.241.193.174:9995/#/notebook/2FUFZ2PAD). Далее некоторые фрагменты кода с комментариями.  

##### Распределение рейтинга в датасете

![Распределение рейтинга в датасете](/lesson_4/images/distribution_by_rating.png)

##### Разбиение рейтинга на классы

![Разбиение рейтинга на классы](/lesson_4/images/rating_to_classes.png)

По второму пункту оказалось, что модели классификации (логистическая регрессия, решающее дерево и случайный лес) принимают на вход в `labelCol` число вместо строки, поэтому классы выбрал `1`, `2`, `3` и `4`. 

После применения метода `ratingToClass` к исходному датафрейму, получил примерно одинаковое распределение статей по классам рейтинга. Далее решал задачу классификации на основании текста в первых и последних пяти предложениях. Эти два столбца склеил в столбец `sentences`.
 
Все первичные преобразования сделал в кастомном трансформере `ColumnSelector`. 

##### UDF для разбивки рейтинга на классы 

```python
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

ratingToClassUdf = udf(ratingToClass)
```

##### Трансформер для подготовки датасета

```python
from pyspark.ml import Transformer
from pyspark.sql.functions import concat, col, lit

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
```

##### Чтение датафрейма

```python
train, test = spark.read.option("header", True) \
.csv("/user/admin/habr_data.csv") \
.randomSplit([.8, .2], seed=42)

print("There are " + str(train.count()) + " rows in the training set, and " + str(test.count()) + " in the test set")
```

```text
There are 8484 rows in the training set, and 2073 in the test set
```

##### Элементы пайплайна 

```python
from pyspark.ml.feature import RegexTokenizer, HashingTF, IDF

column_selector = ColumnSelector()
regexTokenizer = RegexTokenizer(
    inputCol="sentences",
    outputCol="sentences_words",
    pattern="[^a-zа-яё]",
    gaps=True
).setMinTokenLength(3)

hashingTF = HashingTF(inputCol="sentences_words", outputCol="tf_words", numFeatures=pow(2, 18))
idf = IDF(inputCol="tf_words", outputCol="features")
```

##### Метод для запуска модели

```python
def evaluateModel(method, metric_evaluator):
    pipeline = Pipeline(stages=[column_selector, regexTokenizer, hashingTF, idf, method])
    model = pipeline.fit(train)
    
    prediction_train = model.transform(train)
    prediction_test = model.transform(test)
    
    train_ml_metric = metric_evaluator.evaluate(prediction_train)
    test_ml_metric = metric_evaluator.evaluate(prediction_test)
    
    print("Train ML metric: " + str(train_ml_metric))
    print("Test  ML metric: " + str(test_ml_metric))
```

##### Меитрика F1 score

```python
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

evaluator = MulticlassClassificationEvaluator(predictionCol='prediction', labelCol='rating_class', metricName='f1')
```

##### Модель логистической регрессии

```python
lr = LogisticRegression(
    maxIter=10,
    regParam=0.2,
    featuresCol='features',
    labelCol='rating_class',
    predictionCol='prediction'
)

evaluateModel(lr, evaluator)
```

```text
Train ML metric: 0.999292791276
Test  ML metric: 0.335689172959
```

##### Модель решающего дерева

```python
dt = DecisionTreeClassifier(
    labelCol="rating_class",
    featuresCol="features",
    maxDepth=3
)

evaluateModel(dt, evaluator)
```

```text
Train ML metric: 0.239470480314
Test  ML metric: 0.230254753346
```

##### Модель случайного леса

```python
rf = RandomForestClassifier(
    labelCol="rating_class",
    featuresCol="features",
    maxDepth=2,
    numTrees=10
)

evaluateModel(rf, evaluator)
```

```text
Train ML metric: 0.147040516869
Test  ML metric: 0.137823674985
```

##### Выводы

Наилучший результат дала модель логистической регрессии, `f1_score = 0.335`. Но эта модель сильно переобучилась. 

Наилучшие обобщающие свойства у модели решающего дерева. На тренировочной и тестовой выборках результат примерно одинаковый `f1_score = 0.23`.

##### Тюнинг модели логистической регрессии

Здесь переиспользовал ранее определённый эстиматор для подстчёта метрики f1 score.

```python
from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.tuning import TrainValidationSplit, ParamGridBuilder

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
```

```text
Train ML metric: 0.498068036001
Test  ML metric: 0.324369298692
```

```python
prediction_test.select('rating', 'rating_class', 'prediction').show(20)
```

    +------+------------+----------+                                                
    |rating|rating_class|prediction|
    +------+------------+----------+
    |     1|           1|       2.0|
    |     7|           1|       2.0|
    |    17|           3|       4.0|
    |    32|           4|       2.0|
    |    28|           3|       3.0|
    |     1|           1|       3.0|
    |    22|           3|       2.0|
    |    66|           4|       3.0|
    |     6|           1|       4.0|
    |     7|           1|       1.0|
    |    10|           2|       1.0|
    |    13|           2|       1.0|
    |     2|           1|       1.0|
    |    46|           4|       3.0|
    |    20|           3|       3.0|
    |    15|           2|       3.0|
    |    96|           4|       1.0|
    |    29|           3|       1.0|
    |    18|           3|       1.0|
    |    23|           3|       2.0|
    +------+------------+----------+

##### Выводы

При тюнинге модели в сетке параметров размер словаря существенно меньше чем в предыдущей реализации модели логистической регрессии (16к против 256к). Но результат примерно такой же (`f1_score = 0.32` на тестовой выборке). 

Так же видим, что мы избавились от переобучения модели на тренировочной выборке, так как использовали класс `TrainValidationSplit`, в котором тренировочная выборка делилась на тренировочную и валидационную.

