package cn.minghao.test

import org.apache.spark.sql.SparkSession
import org.apache.log4j.Level
import org.apache.log4j.Logger

object test {
  def main(args: Array[String]): Unit = {
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
    //读取数据,数据格式:（id,数据1 数据2 ...），如：0,摘要 感知 基础的 物联网 技术 迅速发展 产业化
    val src = spark.sparkContext.textFile("F:/spark-software/program/LDA/data/data_txt/*")
    val distinct = src.distinct()
    //获取标题
    val coTitles = distinct.filter{str => str.startsWith("[")}.map{str => str.substring(4)}
    val titles = coTitles.distinct()
    //获取关键字
    val keywords = distinct.filter{str => str.startsWith("关键")}.map{str => str.substring(4)}	//注意此处不能传入“关键字”，否则结果为空，原因暂时不祥
    //获取摘要
    val abs = distinct.filter{str => str.startsWith("摘要")}.map{str => str.substring(3)}
    
    titles.repartition(1).saveAsTextFile("F:/spark-software/program/LDA/data/res/titles/")
    keywords.repartition(1).saveAsTextFile("F:/spark-software/program/LDA/data/res/keywords/")
    abs.repartition(1).saveAsTextFile("F:/spark-software/program/LDA/data/res/abs/")

  }
}

































































