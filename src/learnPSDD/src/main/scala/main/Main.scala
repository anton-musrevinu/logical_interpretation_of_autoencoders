package main

import algo.{SoftEM}

object Main {
  def main(args: Array[String]): Unit = {
    if (args.length < 6) {
      println("Please provide the name of the ensemble learner, the name of the dataset and the number of learners.")
    }else if (args.length < 7) {
      val learner = args(0) match {
        //args(1) - dataset path (without train/valid/test extension) (eg. train.data is added)
        //args(2) - vtree file path
        //args(3) - out dir path
        //args(4) - number of learniners
        //args(5) - iterationTimeForEM
        //args(6) - psdd file
        case "SoftEM" => new SoftEM(args(1), args(2), args(3), args(4).toInt, args(5).toInt, "")
      }
      learner.learn()
    } else {
        val learner = args(0) match {
        //args(1) - dataset path (without train/valid/test extension) (eg. train.data is added)
        //args(2) - vtree file path
        //args(3) - out dir path
        //args(4) - number of learniners
        //args(5) - iterationTimeForEM
        //args(6) - psdd file
        case "SoftEM" => new SoftEM(args(1), args(2), args(3), args(4).toInt, args(5).toInt, args(6))
      }
      learner.learn()
    }
  }
}
