import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.{Column, DataFrame, SparkSession}
import org.apache.spark.sql.functions._


// titanic_train_df.createOrReplaceTempView("titanic_train")

// Create a Spark session
val spark = SparkSession.builder()
  .appName("TitanicEDA")
  .getOrCreate()


// Load the Titanic dataset
val titanic_train_df = spark.read
  .option("header", "true")
  .option("inferSchema", "true")
  .csv("C:/Users/DELL/Desktop/Scala/spark/train.csv")  

// Basic Data profiling
titanic_train_df.printSchema() 
titanic_train_df.show(5) // Show the first 5 rows
titanic_train_df.describe().show() // Summary statistics

//Data Cleansing and feature engineering
// Fill null values in the "Age" column with 0
val filledAgeDataFrame = titanic_train_df.na.fill(0, Seq("Age"))

// Fill null values in the "Cabin" column with "NA"
val filledCabinDataFrame = filledAgeDataFrame.na.fill("NA", Seq("Cabin"))

// Show the first few rows of the updated DataFrame
filledCabinDataFrame.show(5)

// Create a new feature "FamilySize" by combining "SibSp" and "Parch"
val titanicDataWithFamilySize = filledCabinDataFrame.withColumn("FamilySize", col("SibSp") + col("Parch"))
titanicDataWithFamilySize.show(5)


// Fill null values in the "Embarked" column with "NA" and expand other values
val EmbarkedDataFrame = titanicDataWithFamilySize.withColumn("Embarked",
  when(col("Embarked").isNull, "NA")
    .when(col("Embarked") === "C", "Cherbourg")
    .when(col("Embarked") === "Q", "Queenstown")
    .when(col("Embarked") === "S", "Southampton")
    .otherwise(col("Embarked"))
)

EmbarkedDataFrame.show(5)

val clean_titanic_train_df = EmbarkedDataFrame
clean_titanic_train_df.show(5)


// EDA
// Calculate the survival rate by passenger class
clean_titanic_train_df.groupBy("Pclass").agg(mean("Survived").alias("SurvivalRate")).show()

// Calculate the survival rate by gender
clean_titanic_train_df.groupBy("Sex").agg(mean("Survived").alias("SurvivalRate")).show()

// Calculate the average age of passengers by class
clean_titanic_train_df.groupBy("Pclass").agg(avg("Age").alias("AvgAge")).show()

// Calculate the average fare by class
clean_titanic_train_df.groupBy("Pclass").agg(avg("Fare").alias("AvgFare")).show()

// Calculate the count of passengers embarked from each port
clean_titanic_train_df.groupBy("Embarked").agg(count("PassengerId").alias("Count")).show()

// Calculate the survival rate based on the "Embarked" column
clean_titanic_train_df.groupBy("Embarked").agg(mean("Survived").alias("SurvivalRate")).show()


spark.stop()

