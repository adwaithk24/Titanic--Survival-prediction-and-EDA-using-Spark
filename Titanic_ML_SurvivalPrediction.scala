import org.apache.spark.sql.Row
import org.apache.spark.ml.linalg.DenseVector
import org.apache.spark.ml.classification.RandomForestClassificationModel
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.feature.{StringIndexer, VectorAssembler}
import org.apache.spark.ml.classification.RandomForestClassifier
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.sql.functions.col

val spark = SparkSession.builder()
  .appName("TitanicSurvivalPrediction")
  .getOrCreate()

// Loading the train, test and validation data
val trainData = spark.read
  .option("header", "true")
  .option("inferSchema", "true")
  .csv("C:/Users/DELL/Desktop/Scala/spark/train.csv")

val testData = spark.read
  .option("header", "true")
  .option("inferSchema", "true")
  .csv("C:/Users/DELL/Desktop/Scala/spark/test.csv")

val validationData = spark.read
  .option("header", "true")
  .option("inferSchema", "true")
  .csv("C:/Users/DELL/Desktop/Scala/spark/gender_submission.csv")

// Preprocess the data
val categoricalCols = Array("Pclass", "Sex", "Embarked")
val indexers = categoricalCols.map(col => new StringIndexer()
  .setInputCol(col)
  .setOutputCol(s"${col}_index")
  .setHandleInvalid("skip")
  .fit(trainData))

val assembler = new VectorAssembler()
  .setInputCols(categoricalCols.map(col => s"${col}_index"))
  .setOutputCol("features")

val pipeline = new Pipeline()
  .setStages(indexers :+ assembler)

val modelData = pipeline.fit(trainData).transform(trainData)
val testDataIndexed = pipeline.fit(trainData).transform(testData)

// Train a Random Forest classifier
val rf = new RandomForestClassifier()
  .setLabelCol("Survived")
  .setFeaturesCol("features")
  .setNumTrees(100) 

val model = rf.fit(modelData)

// Make predictions
val predictions = model.transform(testDataIndexed)

val finalPredictions = predictions.select("PassengerId", "prediction")
  .withColumnRenamed("prediction", "Survived")
  .withColumn("Survived", col("Survived").cast("double"))
finalPredictions.show(20)

// Calculate accuracy by matching with validationData
val correctPredictions = validationData.join(finalPredictions, Seq("PassengerId", "Survived"))

val accuracy = correctPredictions.count().toDouble / validationData.count()

// Output accuracy
println(s"Accuracy: ${accuracy * 100}%")

// Stop the Spark session
spark.stop()
