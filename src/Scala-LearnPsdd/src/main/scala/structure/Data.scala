package structure

import java.io.File

import util.{BitSubSet, WeightedBitSubSet, Util}

import scala.collection.immutable.BitSet
import scala.collection.mutable
import scala.io.Source

/**
  * This class stores a dataset as a bitmap and allows bitset operations
  *
  * @param backend
  * @param weights
  * @param vars
  * @param elements
  */
class Data(backend: Array[Map[Int,Boolean]], weights: Array[Double], val vars: Array[Int], elements: BitSet) extends WeightedBitSubSet[Map[Int,Boolean]](backend, weights, elements){
  override def weightedIterator: Iterator[(Map[Int,Boolean],Double)] = elements.iterator.map { e => (backend(e),weights(e)) }

  override def empty = new Data(backend, weights, vars, BitSet.empty)

  override def filter(p: (Map[Int,Boolean]) => Boolean): Data = {
    new Data(backend,weights, vars, elements.filter { i => p(backend(i)) })
  }

  override def filterNot(p: (Map[Int,Boolean]) => Boolean): Data = {
    new Data(backend,weights, vars, elements.filterNot { i => p(backend(i)) })
  }

  override def intersect(that: BitSubSet[Map[Int,Boolean]]): Data = {
    require(this.backend eq that.backend)
    new Data(backend, weights, vars, this.elements intersect that.elements)
  }

  override def union(that: BitSubSet[Map[Int,Boolean]]): Data = {
    require(this.backend eq that.backend)
    new Data(backend, weights, vars, this.elements union that.elements)
  }


  override def diff(that: BitSubSet[Map[Int,Boolean]]): Data = {
    require(this.backend eq that.backend)
    val diffElements = this.elements diff that.elements
    new Data(backend, weights, vars, diffElements)
  }
  
  def checkOperation(that: BitSubSet[Map[Int,Boolean]]): Boolean = (this.backend eq that.backend)

  def copy() = new Data(backend.map(Map[Int,Boolean]()++_),weights.map(x=>x),vars.map(x=>x),BitSet()++elements)

  override def toString = "Data("+size+","+total+")"

  }

object PartialAssignment {

    def apply(backend: Array[Map[Int,Boolean]], weights: Array[Double], vars: Array[Int]): Data = new Data(backend, weights, vars, BitSet(backend.indices:_*))

    def readFromFile(file: File): Data = {
      var assignments = Seq[(String,Double)]()
      Source.fromFile(file).getLines().withFilter(_.nonEmpty).foreach{ line =>
        val split = line.split("\\|")
        // println("split: " + split)
        val (as,w) = if (split.size==1) (split.head,1.0) else (split(1),split.head.toDouble)
        // println("as: " + as)
        // println("w: " + w)
        assignments = (as,w) +: assignments
      }
      val (b,weights) = assignments.toArray.unzip
      var vars = mutable.SortedSet[Int]()
      val backend = b.map(line => line.split(",").map{
        case (v) => scala.math.abs(v.toInt) -> !v.contains("-")
        }.toMap)

      var counter = 0
      for (line <- b){
        val vars_in_line = mutable.SortedSet[Int]()
        for (elem <- line.split(",")){
          vars_in_line += scala.math.abs(elem.toInt)
        }
        if (counter == 0){
          vars = vars_in_line
        } else if (vars != vars_in_line) {
          println("Vars in one line are not equal to the variables of another line -- they must be all defined for the same...")
          throw new IllegalArgumentException
        }
        counter = counter + 1
      }
      println("[DATA] - variales found: " + vars)
      println("[DATA] - backend size: " + backend.size)
      // println("[DATA] - backend: " + backend)
      // // println("backend: " + backend)
      // for (map <- backend){
      //   println(map)
      // }

      Data(backend, weights, vars.toArray)
    }

    def readFromArray(samples: Array[Array[Int]],weights:Array[Double]): Data ={
      val backend = samples.map(Util.convertToAssignment(_))
      Data(backend, weights, (1 to backend.head.size).toArray)
    }

    def readDataAndWeights(assignments:Array[(Map[Int,Boolean],Double)]): Data = {
      val backend = assignments.map(_._1)
      val w = assignments.map(_._2)
      Data(backend,w, (1 to backend.head.size).toArray)
    }

  }


object Assignment {

    def apply(backend: Array[Map[Int,Boolean]], weights: Array[Double], vars: Array[Int]): Data = new Data(backend, weights, vars, BitSet(backend.indices:_*))

    def readFromFile(file: File): Data = {
      var assignments:Seq[(String, Double)] = Seq()
      Source.fromFile(file).getLines().withFilter(_.nonEmpty).foreach{ line =>
        val split = line.split("\\|")
        val (as,w) = if (split.size==1) (split.head,1.0) else (split(1),split.head.toDouble)
        assignments = assignments :+ (as,w)
      }
      val (b,weights) = assignments.toArray.unzip
      val backend = b.map(line => line.split(",").zipWithIndex.map{case (v,i) => i+1 -> !v.contains("0")}.toMap)

      Data(backend, weights, (1 to backend.head.size).toArray)
    }

    def readFromArray(samples: Array[Array[Int]],weights:Array[Double]): Data ={
      val backend = samples.map(Util.convertToAssignment(_))
      Data(backend, weights, (1 to backend.head.size).toArray)
    }

    def readDataAndWeights(assignments:Array[(Map[Int,Boolean],Double)]): Data = {
      val backend = assignments.map(_._1)
      val w = assignments.map(_._2)
      Data(backend,w, (1 to backend.head.size).toArray)
    }

  }


object Data {

  def apply(backend: Array[Map[Int,Boolean]], weights: Array[Double], vars: Array[Int]): Data = new Data(backend, weights, vars, BitSet(backend.indices:_*))

  def readFromFile(file: File): Data = {
    val assignments = mutable.Map[String,Double]()
    Source.fromFile(file).getLines().withFilter(_.nonEmpty).foreach{ line =>
      val split = line.split("\\|")
      val (as,w) = if (split.size==1) (split.head,1.0) else (split(1),split.head.toDouble)
      assignments.put(as,w+assignments.getOrElse(as,0.0))
    }
    val (b,weights) = assignments.toArray.unzip
    val backend = b.map(line => line.split(",").zipWithIndex.map{case (v,i) => i+1 -> !v.contains("0")}.toMap)

    Data(backend, weights, (1 to backend.head.size).toArray)
  }

  def readFromArray(samples: Array[Array[Int]],weights:Array[Double]): Data ={
    val backend = samples.map(Util.convertToAssignment(_))
    Data(backend, weights, (1 to backend.head.size).toArray)
  }

  def readDataAndWeights(assignments:Array[(Map[Int,Boolean],Double)]): Data = {
    val backend = assignments.map(_._1)
    val w = assignments.map(_._2)
    Data(backend,w, (1 to backend.head.size).toArray)
  }

}

/**
  * This class stores training validation an test data. Keeping the three together makes it easy to
  * apply the same operations on all of them in a PSDD
  * @param dataSets
  */
class DataSets(val dataSets: Array[Data]) {

  def this(train: Data, valid: Data, test: Data) = this(Array(train,valid,test))

  def empty = new DataSets(dataSets.map(_.empty))
  def filter(p: (Map[Int,Boolean]) => Boolean) = new DataSets(dataSets.map(_.filter(p)))
  def filterNot(p: (Map[Int,Boolean]) => Boolean) = new DataSets(dataSets.map(_.filterNot(p)))
  def intersect(other: DataSets) = new DataSets(dataSets.zip(other.dataSets).map{ case (a,b) =>a.intersect(b)})
  def union(other: DataSets) = new DataSets(dataSets.zip(other.dataSets).map{ case (a,b) =>a.union(b)})
  def diff(other: DataSets) = new DataSets(dataSets.zip(other.dataSets).map{ case (a,b) =>a.diff(b)})

  def isEmpty = dataSets.forall(_.isEmpty)

  def forall(p: Map[Int,Boolean]=>Boolean) = dataSets.forall(_.forall(p))
  def exists(p: Map[Int,Boolean]=>Boolean) = dataSets.exists(_.exists(p))

  def train = dataSets.head
  def valid = dataSets(1)
  def test = dataSets(2)

  override def equals(other: Any): Boolean = other match {
    case other: DataSets =>
      if (this.dataSets.length != other.dataSets.length) return false
      this.dataSets.zip(other.dataSets).forall{case (data1,data2) => data1==data2}
    case _ => false
  }

  def copy() = new DataSets(this.dataSets.map(_.copy()))

  override def toString: String = dataSets.mkString("[",",","]")
}
