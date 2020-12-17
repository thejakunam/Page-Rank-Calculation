import org.apache.spark.{SparkConf, SparkContext}
import scala.collection.mutable.Map
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.feature.{HashingTF, Tokenizer}
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.sql.Row
import org.apache.spark.sql.functions._
import org.apache.spark.ml.feature.StopWordsRemover
import org.apache.spark.ml.feature.StringIndexer
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator

object PageRank {
  def main(args: Array[String]): Unit = {

    val sc = new SparkContext(new SparkConf().setAppName("PageRank"))

    val input = sc.textFile(args(0))
    val number_of_iterations = args(1)
    val output_directory = args(2)
    val initial_PR = 1.0


    val rdd_input = input.map(x => x.split(","))
    var airports = rdd_input.map(x => (x(1), x(4)))
    val firstRow = airports.first
    val airport_names = airports.filter(x => x != firstRow)
    val allairports = (airport_names.map { case (x, y) => x }.distinct() ++ airport_names.map { case (x, y) => y }.distinct()).distinct().collect()
    val out_links = airport_names.groupByKey().map(x => (x._1, x._2.size)).collect().map(x => (x._1, x._2)).toMap
    val rank = collection.mutable.Map() ++ allairports.map(x => (x, initial_PR)).toMap


    for (i <- 1 to number_of_iterations.toInt) {
      val out = collection.mutable.Map() ++ allairports.map(x => (x, 0.0)).toMap
      rank.keys.foreach((id) => rank(id) = rank(id) / out_links(id))
      for ((key, value) <- airport_names.collect()) {
        out.put(value, out(value) + rank(key))
      }
      val out1 = collection.mutable.Map() ++ out.map(x => (x._1, ((0.15 / allairports.size) + (1-0.15) * x._2)))
      out1.keys.foreach((id) => rank(id) = out1(id))
    }

    val result = rank.toSeq.sortBy(-_._2)
    sc.parallelize(result).saveAsTextFile(output_directory + "Output")
  }
}

object TopicModelling {
  def main(args : Array[String]) : Unit ={
    val sc = new SparkContext(new SparkConf().setAppName("PageRank"))

    al corpus: RDD[String] = sc.wholeTextFiles(args(0)).map(_._2)
    val stopWordSet = StopWordsRemover.loadDefaultStopWords("english").toSet
    val tokenized: RDD[Seq[String]] =
      corpus.map(_.toLowerCase.split("\\s")).map(_.filter(_.length > 3).filter(token => !stopWordSet.contains(token)).filter(_.forall(java.lang.Character.isLetter)))
    val termCounts: Array[(String, Long)] =
      tokenized.flatMap(_.map(_ -> 1L)).reduceByKey(_ + _).collect().sortBy(-_._2)
    val numStopwords = 20
    val vocabArray: Array[String] =
      termCounts.takeRight(termCounts.size - numStopwords).map(_._1)
    val vocab: Map[String, Int] = vocabArray.zipWithIndex.toMap

    val documents: RDD[(Long, Vector)] =
      tokenized.zipWithIndex.map { case (tokens, id) =>
        val counts = new mutable.HashMap[Int, Double]()
        tokens.foreach { term =>
          if (vocab.contains(term)) {
            val idx = vocab(term)
            counts(idx) = counts.getOrElse(idx, 0.0) + 1.0
          }
        }
        (id, Vectors.sparse(vocab.size, counts.toSeq))
      }

    val numTopics = 10
    val lda = new LDA().setK(numTopics).setMaxIterations(10)

    val ldaModel = lda.run(documents)

    val topicIndices = ldaModel.describeTopics(maxTermsPerTopic = 10)
    topicIndices.foreach { case (terms, termWeights) =>
      println("TOPIC:")
      terms.zip(termWeights).foreach { case (term, weight) =>
        println(s"${vocabArray(term.toInt)}\t$weight")
      }
      println()
    }

  }
}

object TweetAnalysis {
  def main(args : Array[String]) : Unit ={
    val sc = new SparkContext(new SparkConf().setAppName("PageRank"))

    val input = spark.read.option("header", "true"). option("inferSchema","true")
      . csv(args(0))
    val Array(training, test) = data.randomSplit(Array(0.7, 0.3))
    training.cache()

    val tokenizer = new Tokenizer().setInputCol("text").setOutputCol("set_of_words")

    val remover = new StopWordsRemover().setInputCol(tokenizer.getOutputCol).setOutputCol("filtered_text")

    val logistic_regression = new LogisticRegression().setMaxIter(10)

    val pipeline = new Pipeline()
      .setStages(Array(tokenizer, remover, logistic_regression))

    val cross_validation = new CrossValidator()
      .setEstimator(pipeline)
      .setEvaluator(new MulticlassClassificationEvaluator)
      .setNumFolds(2)
      .setParallelism(2)

    val cv_model = cross_validation.fit(training)
    val predictionAndLabels = cv_model.transform(test).select("label", "prediction").map{ case Row(l:Double,p:Double) => (p, l) }

  }
}