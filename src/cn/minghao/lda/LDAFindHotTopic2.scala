package cn.minghao.lda

import org.apache.log4j.{Level, Logger}
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.sql._
import org.apache.spark.ml.feature.Tokenizer
import org.apache.spark.ml.feature.CountVectorizer
import org.apache.spark.ml.clustering.LDA
import scala.collection.mutable.WrappedArray
import org.apache.spark.ml.feature.RegexTokenizer
import org.apache.lucene.analysis.Analyzer
import org.wltea.analyzer.lucene.IKAnalyzer
import org.apache.lucene.analysis.tokenattributes.CharTermAttribute
import org.apache.lucene.analysis.TokenStream
import java.io.StringReader

/**
  * 本程序为毕设spark lda聚类算法实现热点挖掘version 1.0
  * 工作计划：
  * 1、直接使用LDA算法进行聚类热点发现
  *	2、整理好数据集
  * 3、优化分词
  * 4、添加去停用词
  * 5、封装方法，使工程各方法分工合理
  * 6、搭建spark集群跑程序，搭建hadoop集群将数据从本地转移到hdfs中存储
  * 7、使用网页可视化展示热点
  * */
object LDAFindHotTopic2 {
  def main(args: Array[String]): Unit = {
    val startTime = System.nanoTime()
    //屏蔽日志
    Logger.getLogger("org.apache.spark").setLevel(Level.WARN)
    Logger.getLogger("org.eclipse.jetty.server").setLevel(Level.OFF)
    val spark = SparkSession
    .builder()
    .master("local[2]")  //设置运行方式是本地两个线程
    .appName("kmeans")    //应用的名字
    //设置数据仓库的位置，当时是为了解决一个错误而设置的，今后看能不能去掉
    .config("spark.sql.warehouse.dir", "file/:F:/spark-software/program/KMeans/spark-warehouse")  
    .getOrCreate()  //Gets an existing SparkSession or, if there is no existing one,
                    //creates a new one based on the options set in this builder.（官网对getOrCreate()方法的解释）
    //A wrapped version of this session in the form of a SQLContext, for backward compatibility.为的是向后兼容
    val sqlContext = spark.sqlContext  
    //(Scala-specific) Implicit methods available in Scala for converting common Scala objects into DataFrames.
    //scala中获取隐式的方法将普通的scala对象转化成DataFrames类型,不能将它移动上面的import位置，为什么？
    import sqlContext.implicits._
    //读取数据,数据格式:（数据1 数据2 ...），如：摘要,感知,基础的,物联网,技术,迅速发展 ,产业化
    val src = spark.read.textFile("F:/spark-software/program/LDA/data/test/*") 

    val srcDF = src.map { case (str: String) =>
      (str.toString())
    }.toDF("text")
    val trainingDF = srcDF
    
    //RegexTokenizer类指定匹配正则切分数据
    //设置输入text、输出words、正则
    val regextTokenizer = new RegexTokenizer().setInputCol("text").setOutputCol("words").setPattern(" ")
    //以,切分
    val wordsData = regextTokenizer.transform(trainingDF)

    var ldaInput = wordsData.select("words")
    //拟合词频向量
    val cvModel = new CountVectorizer().setInputCol("words").setOutputCol("features").fit(wordsData)
    val cvDF = cvModel.transform(wordsData)
    //无重复的词汇表
    val vocab = cvModel.vocabulary
    //初始化LDA模型
    var lda = new LDA().setK(10).setMaxIter(100)
    var ldaModel = lda.fit(cvDF)  //训练过程
    //返回主题-词
    val topics = ldaModel.describeTopics(maxTermsPerTopic=3)  //maxTermsPerTopic设置前多少个，即前n高的概率
    println("lda")  
    topics.show(false)  //false代表全部输出，不以省略的形式
    val indexes = topics.select("termIndices")  //返回高概率词索引
    indexes.rdd.foreach{
     case Row(wa: WrappedArray[Int]) =>
       println(vocab(wa(0))+"--"+vocab(wa(1))+"--"+vocab(wa(2)))
    }
    var ll  = ldaModel.logLikelihood(cvDF)
    var lp = ldaModel.logPerplexity(cvDF)
    println(s"The lower bound on the log likelihood of the entire corpus: $ll")
    println(s"The upper bound bound on perplexity: $lp")
   
   val endTime = System.nanoTime()
   val time = (endTime - startTime) / 1e9
   println("time: " + time)
  }
}
























































