# Create UDF

%pyspark

df1 = spark.createDataFrame([(1, "andy", 20, "USA"), (2, "jeff", 23, "China"), (3, "james", 18, "USA")]) \
            .toDF("id", "name", "age", "country")

# Create udf create python lambda
from pyspark.sql.functions import udf
udf1 = udf(lambda e: e.upper())
df2 = df1.select(udf1(df1["name"]))
df2.show()

# UDF could also be used in filter, in this case the return type must be Boolean
# We can also use annotation to create udf
from pyspark.sql.types import *
@udf(returnType=BooleanType())
def udf2(e):
    if e >= 20:
        return True;
    else:
        return False

df3 = df1.filter(udf2(df1["age"]))
df3.show()

# UDF could also accept more than 1 argument.
udf3 = udf(lambda e1, e2: e1 + "_" + e2)
df4 = df1.select(udf3(df1["name"], df1["country"]).alias("name_country"))
df4.show()


# groupBy

%pyspark

df1 = spark.createDataFrame([(1, "andy", 20, "USA"), (2, "jeff", 23, "China"), (3, "james", 18, "USA")]) \
           .toDF("id", "name", "age", "country")

# You can call agg function after groupBy directly, such as count/min/max/avg/sum
df2 = df1.groupBy("country").count()
df2.show()

# Pass a Map if you want to do multiple aggregation
df3 = df1.groupBy("country").agg({"age": "avg", "id": "count"})
df3.show()

import pyspark.sql.functions as F
# Or you can pass a list of agg function
df4 = df1.groupBy("country").agg(F.avg(df1["age"]).alias("avg_age"), F.count(df1["id"]).alias("count"))
df4.show()

# You can not pass Map if you want to do multiple aggregation on the same column as the key of Map should be unique. So in this case
# you have to pass a list of agg functions
df5 = df1.groupBy("country").agg(F.avg(df1["age"]).alias("avg_age"), F.max(df1["age"]).alias("max_age"))
df5.show()


# Join on Single Field
%pyspark

df1 = spark.createDataFrame([(1, "andy", 20, 1), (2, "jeff", 23, 2), (3, "james", 18, 3)]).toDF("id", "name", "age", "c_id")
df1.show()

df2 = spark.createDataFrame([(1, "USA"), (2, "China")]).toDF("c_id", "c_name")
df2.show()

# You can just specify the key name if join on the same key
df3 = df1.join(df2, "c_id")
df3.show()

# Or you can specify the join condition expclitly in case the key is different between tables
df4 = df1.join(df2, df1["c_id"] == df2["c_id"])
df4.show()

# You can specify the join type afte the join condition, by default it is inner join
df5 = df1.join(df2, df1["c_id"] == df2["c_id"], "left_outer")
df5.show()



# Join on Multiple Fields
%pyspark

df1 = spark.createDataFrame([("andy", 20, 1, 1), ("jeff", 23, 1, 2), ("james", 12, 2, 2)]).toDF("name", "age", "key_1", "key_2")
df1.show()

df2 = spark.createDataFrame([(1, 1, "USA"), (2, 2, "China")]).toDF("key_1", "key_2", "country")
df2.show()

# Join on 2 fields: key_1, key_2

# You can pass a list of field name if the join field names are the same in both tables
df3 = df1.join(df2, ["key_1", "key_2"])
df3.show()

# Or you can specify the join condition expclitly in case when the join fields name is differetnt in the two tables
df4 = df1.join(df2, (df1["key_1"] == df2["key_1"]) & (df1["key_2"] == df2["key_2"]))
df4.show()


# Create large DF
%spark

spark.conf.set("spark.sql.autoBroadcastJoinThreshold", "-1")
// Generate some sample data for two data sets

var states = scala.collection.mutable.Map[Int, String]()
var items = scala.collection.mutable.Map[Int, String]()
val rnd = new scala.util.Random(42)

// Initialize states and items purchased

states += (0 -> "AZ", 1 -> "CO", 2-> "CA", 3-> "TX", 4 -> "NY", 5-> "MI")
items += (0 -> "SKU-0", 1 -> "SKU-1", 2-> "SKU-2", 3-> "SKU-3", 4 -> "SKU-4",
5-> "SKU-5")
// Create DataFrames
val usersDF = (0 to 100000)
    .map(id => (id, s"user_${id}", s"user_${id}@databricks.com", states(rnd.nextInt(5))))
    .toDF("uid", "login", "email", "user_state")


val ordersDF = (0 to 100000)
    .map(r => (r, r, rnd.nextInt(10000), 10 * r* 0.2d, states(rnd.nextInt(5)), items(rnd.nextInt(5))))
    .toDF("transaction_id", "quantity", "users_id", "amount", "state", "items")
    
    
// Do the join
val usersOrdersDF = ordersDF
    .join(broadcast(usersDF), $"users_id" === $"uid", "left")
    .select("users_id", "transaction_id")
    
// Show the joined results
usersOrdersDF.show(false)


# Save bucketing
import org.apache.spark.sql.functions._
import org.apache.spark.sql.SaveMode

// Save as managed tables by bucketing them in Parquet format
usersDF.orderBy(asc("uid"))
    .write.format("parquet")
    .bucketBy(8, "uid")
    .mode("overwrite")
    .saveAsTable("UsersTbl")

ordersDF.orderBy(asc("users_id"))
    .write.format("parquet")
    .bucketBy(8, "users_id")
    .mode("overwrite")
    .saveAsTable("OrdersTbl")

// Cache the tables
spark.sql("CACHE TABLE UsersTbl")
spark.sql("CACHE TABLE OrdersTbl")
// Read them back in

val usersBucketDF = spark.table("UsersTbl")
val ordersBucketDF = spark.table("OrdersTbl")

// Do the join and show the results
val joinUsersOrdersBucketDF = ordersBucketDF
    .join(usersBucketDF, $"users_id" === $"uid")


joinUsersOrdersBucketDF.show(false)