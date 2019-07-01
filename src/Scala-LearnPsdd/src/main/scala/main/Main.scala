package main

import java.io.{PrintWriter, File}

import algo._
import operations._
import sdd.{SddManager, Vtree}
import structure._

import scala.util.Random
import scala.math._


object Main {


//  Different functionalities of the code
  val learnEnsemblePsdd = "learnEnsemblePsdd"
  val learnPsdd = "learnPsdd"
  val sdd2psdd = "sdd2psdd"
  val parameterLearning = "learnParams"
  val learnVtree = "learnVtree"
  val paramSearch = "paramSearch"
  val scratch = "scratch"
  val check = "check"
  val query = "query"

//  structure learning strategies
  val bottomUp = "BU"
  val topDown = "TD"
  val search = "search"
  val learnMethods = Array(bottomUp, topDown, search)


  val commands = Array(learnMethods.map(learnPsdd+" "+_).mkString(" ","|",""), sdd2psdd, parameterLearning, learnVtree, paramSearch, check, query)

//datastets
  val dataSets = Array("train","valid","test")


//  Vtrees
  val vtreeOptions = Map(
    "miMetis" -> "balanced vtree by top down selection of the split that minimizes the mutual information between the two parts, using metis.",
    "miBlossom" -> "balanced vtree by bottom up matching of the pairs that maximizes the average mutual information in a pair, using blossomV.",
    "balanced-ord" -> "balanced vtree using variable order",
    "rightLinear-ord" -> "right linear vtree using variable order",
    "leftLinea-ord" -> "left linear vtree using variable order",
    "balanced-rand" -> "balanced vtree using a random order",
    "rightLinear-rand" -> "right linear vtree using a random order",
    "leftLinear-rand" -> "left linear vtree a random order",
    "pairwiseWeights" -> "balanced vtree by top down selection of the split that minimizes the mutual information between the two parts, using exhaustive search.",
    "miGreedyBU" -> "balanced vtree by bottom up matching of the pairs that maximizes the average mutual information in a pair, using greedy selection."
  )

  val defaultVtree = "miBlossom"

// structure search operations
  val clonePrefix = "clone"
  val splitPrefix = "split"
  val split = (splitPrefix+"-(\\d+)").r
  val cloneOp = (clonePrefix+"-(\\d+)").r
  val operationTypes = Seq(clonePrefix+"-<k>", splitPrefix+"-<k>")
  val defaultOperationTypes = Seq(clonePrefix+"-3", splitPrefix+"-1")

// parameter calculators (differ in smoothing type)
  val no = "no"
  val mEstimatorPrefix = "m"
  val laplacePrefix = "l"
  val modelCountPrefix = "mc"
  val parameterCalculators = Map(
    "no" -> "No smoothing",
    mEstimatorPrefix+"-<m>" -> "m-estimator smoothing",
    laplacePrefix+"-<m>" -> "laplace smoothing, weighted with m",
    modelCountPrefix+"-<m>" -> "model count as pseudo count, weighted with m",
    modelCountPrefix+"-"+mEstimatorPrefix+"-<m>" -> "m-estimator, weighted with model count",
    modelCountPrefix+"-"+laplacePrefix+"-<m>" -> "laplace smoothing, weighted with model count and m"
  )
  val defaultParameterCalculator = new LaplaceParameterCalculator(1)
  val defaultParameterCalculatorString = "l-1"

  val float = "[-+]?\\d*\\.?\\d+(?:[eE][-+]?\\d+)?"
  val mEstimator = (mEstimatorPrefix+"-(" + float + ")").r
  val laplace = (laplacePrefix+"-(" + float + ")").r
  val mc = (modelCountPrefix+"-(" + float + ")").r
  val mcMEstimator = (modelCountPrefix+"-"+mEstimatorPrefix+"-(" + float + ")").r
  val mcLaplace = (modelCountPrefix+"-"+laplacePrefix+"-(" + float + ")").r

  implicit val paramCalcRead: scopt.Read[ParameterCalculator] = scopt.Read.reads{
    case `no` => ParameterCalculatorWithoutSmoothing
    case mEstimator(m) => new MEstimateParameterCalculator(m.toDouble)
    case laplace(m) => new LaplaceParameterCalculator(m.toDouble)
    case mc(m) => new ModelCountParameterCalculator(m.toDouble)
    case mcMEstimator(m) => new MCMEstimateParameterCalculator(m.toDouble)
    case mcLaplace(m) => new MCLaplaceParameterCalculator(m.toDouble)
  }

//  scorers
  val dllScorer = "dll"
  val dllPerSizeScorer = "dll/ds"
  val scorers = Array(dllScorer, dllPerSizeScorer)
  val defaultScorerString = dllPerSizeScorer
  val defaultScorer = DllPerDsizeScorer

  implicit val operationScorerRead: scopt.Read[OperationScorer] = scopt.Read.reads {
    case `dllScorer` => DllScorer
    case `dllPerSizeScorer` => DllPerDsizeScorer
  }


//  operation completion types
  val complete = "complete"
  val minimal = "min"
  val maxEdgesPrefix = "maxEdges"
  val maxDepthPrefix = "maxDepth"
  val operationCompletionTypes = Array(complete, minimal, maxDepthPrefix+"-<k>", maxEdgesPrefix+"-<k>")
  val defaultOperationCompletionType = MaxDepth(3)
  val defaultOperationCompletionTypeString = maxDepthPrefix+"-3"

  val maxEdges = (maxEdgesPrefix+"-(\\d+)").r
  val maxDepth = (maxDepthPrefix+"-(\\d+)").r

  implicit val completionTypeRead: scopt.Read[OperationCompletionType] = scopt.Read.reads {
    case `complete` => Complete
    case `minimal` => Minimal
    case maxEdges(k) => MaxEdges(k.toInt)
    case maxDepth(k) => MaxDepth(k.toInt)
  }


  //  save frequencies
  val allPrefix = "all"
  val bestPrefix = "best"
  val frequencyTypes = Array(allPrefix+"-<k>",bestPrefix+"-<k>")
  val defaultFrequency = Best(1)
  val defaultFrequencyString = bestPrefix+"-"+1

  val all = (allPrefix+"-(\\d+)").r
  val best = (bestPrefix+"-(\\d+)").r
  implicit val frequencyTypeRead: scopt.Read[SaveFrequency] = scopt.Read.reads {
    case all(k) => All(k.toInt)
    case best(k) => Best(k.toInt)
  }

  // print help
  def printHelp(): Unit = {
    println("LearnPsdd 1.0")
    println("Usage: PSDD "+commands.mkString("|")+" [options]")
  }

  /**
    * Configuration
    *
    * An object of this class stores the configuration for all the possible functionalities of this code
    */
  case class Config(
                     out: String = null,
                     train: File = null,
                     valid: File = null,
                     test: File = null,
                     vtree: File = null,
                     psdd: File = null,
                     query: File = null,
                     mode: String = null,
                     psdds: Seq[File] = null,
                     componentweights: Seq[Double] = null,
                     parameterCalculator: ParameterCalculator = defaultParameterCalculator,
                     vtreeMethod: String = defaultVtree,
                     debugLevel: Int = Int.MaxValue,
                     operationTypes: Seq[String] = defaultOperationTypes,
                     completionType: OperationCompletionType = defaultOperationCompletionType,
                     scorer: OperationScorer = defaultScorer,
                     maxIt: Int = Int.MaxValue,
                     structureChangeIt: Int = 1,
                     parameterLearningInt: Int = 3,
                     numComponentLearners: Int = 0,
                     keepSplitting: Boolean = false,
                     keepCloning: Boolean = false,
                     frequency: SaveFrequency = defaultFrequency,
                     parameterCalculators: Seq[ParameterCalculator] = null,
                     entropyOrder: Boolean=false,
                     fl_names: Seq[String] = null,
                     fl_nb_vars: Seq[Int]  = null,
                     fl_var_cat_dim: Seq[Int] = null,
                     fl_binary_encoded: Seq[Int] = null,
                     fl_encoded_start_idx: Seq[Int] = null,
                     fl_encoded_end_idx: Seq[Int] = null,
                     fl_to_query: Seq[String] = null,
                     data_bug: Boolean = false
                     ){
    val configString: String = Array(
      "trainData:\t"+  (if(train == null) "null" else train.getPath),
      "validData:\t"+  (if(valid == null) "null" else valid.getPath),
      "testData:\t"+  (if(test == null) "null" else test.getPath),
      "query:\t"+  (if(query == null) "null" else query.getPath),
      "vtreeFile:\t"+ (if(vtree == null) "null" else vtree.getPath),
      "psddFile:\t"+ (if(psdd== null) "null" else psdd.getPath),
      "out:\t"+ out
    ).mkString("\n")
  }

  /**
    * The learnEnsemblePsdd parser parses the options for learning an ensemble of psdds from data.
    */
  val learnEnsemblePsddParser = new scopt.OptionParser[Config](learnEnsemblePsdd+" softEM") {
    head("Learning the structure of an ensemble of psdds from data")

    opt[File]('p', "psdd").
      optional.
      valueName ("<file>").
      action{ (x,c) => c.copy(psdd = x)}.
      text("If no psdd is provided, the learning starts from a mixture of marginals.\n")

    opt[File]('v', "vtree").
      required().
      valueName ("<file>").
      action { (x, c) => c.copy(vtree = x) }

    opt[File]('d', "trainData").
      required().
      valueName ("<file>").
      action { (x, c) => c.copy(train = x) }

    opt[File]('b', "validData").
      optional().
      valueName ("<file>").
      action { (x, c) => c.copy(valid = x) }

    opt[File]('t', "testData").
      optional().
      valueName ("<file>").
      action { (x, c) => c.copy(test = x) }

    opt[String]('o', "out").
      required().
      valueName("<path>").
      action( (x, c) => c.copy(out = x) ).
      text("The folder for output\n")

    opt[Int]('c', "numComponentLearners").
      required().
      valueName("<numComponentLearners>").
      action((x, c) => c.copy(numComponentLearners = x)).
      text("The number of component learners to form the ensemble\n")

    opt[ParameterCalculator]('m',"smooth").
      optional().
      valueName("<smoothingType>").
      action((x,c) =>c.copy(parameterCalculator = x)).
      text (
        "default: " + defaultParameterCalculatorString + "\n" +
          parameterCalculators.map { case (k, v) => "\t * " + k + ": " + v }.mkString("\n")+"\n"
      )

    opt[OperationScorer]('s',"scorer").
      optional().
      valueName("<scorer>").
      action((x,c)=>c.copy(scorer = x)).
      text(
        "default: "+defaultScorerString+"\n" +
          scorers.map { case t => "\t * " + t }.mkString("\n")+"\n"
      )

    opt[Int]('e',"maxIt").
      optional().
      valueName("<maxIt>").
      action((x,c)=>c.copy(maxIt=x)).
      text(
        "this is the maximum number of ensemble learning iterations.\n"
      )


    opt[Int]("structureChangeIt").
      optional().
      valueName("<structureChangeIt>").
      action((x,c)=>c.copy(maxIt=x)).
      text(
        "this is the number of structure changes before a new round of parameter learning .\n"
      )

    opt[Int]("parameterLearningIt").
      optional().
      valueName("<parameterLearningIt>").
      action((x,c)=>c.copy(maxIt=x)).
      text(
        "this is the number of iterators for parameter learning before the structure of psdds changes again.\n"
      )


    checkConfig { c =>
      if (c.train==null) failure("train data is required")
      if (c.vtree==null) failure("vtree is required")
      if (c.out==null) failure("output path is required")
      if (c.numComponentLearners==0) failure("number of component learners to form the ensemble is required")
      success
    }

    help("help") text ("prints this usage text\n")

  }


  /**
    * The learnPSDD parser parses the options for learning a PSDD from data.
    * Furthermore, it provides information on the expected arguments
    */
  val learnPsddParser = new scopt.OptionParser[Config](learnPsdd +" "+ learnMethods.mkString("|")) {
    head("Learn the structure of a PSDD from data")

    opt[File]('p', "psdd").
      optional.
      valueName ("<file>").
      action{ (x,c) => c.copy(psdd = x)}.
      text("If no psdd is provided, the learning starts from a mixture of marginals.\n")

    opt[File]('v', "vtree").
      required().
      valueName ("<file>").
      action { (x, c) => c.copy(vtree = x) }

    opt[File]('d', "trainData").
      required().
      valueName ("<file>").
      action { (x, c) => c.copy(train = x) }

    opt[File]('b', "validData").
      optional().
      valueName ("<file>").
      action { (x, c) => c.copy(valid = x) }

    opt[File]('t', "testData").
      optional().
      valueName ("<file>").
      action { (x, c) => c.copy(test = x) }

    opt[String]('o', "out").
      required().
      valueName("<path>").
      action( (x, c) => c.copy(out = x) ).
      text("The folder for output\n")

    opt[ParameterCalculator]('m',"smooth").
      optional().
      valueName("<smoothingType>").
      action((x,c) =>c.copy(parameterCalculator = x)).
      text (
      "default: " + defaultParameterCalculatorString + "\n" +
        parameterCalculators.map { case (k, v) => "\t * " + k + ": " + v }.mkString("\n")+"\n"
    )

    opt[Seq[String]]('h', "opTypes").
      optional().
      valueName("<opType>,<opType>,...").
      action((x,c) => c.copy(operationTypes=x)).
      text(
        "default: " + defaultOperationTypes.mkString(",") + "\n" +
          "\toptions: "++ operationTypes.mkString(",")+"\n"+
          "\tIn split-k, k is the number of splits.\n"+
          "\tIn clone-k, k is the maximum number of parents to redirect to the clone.\n"
      )

    opt[OperationCompletionType]('c', "completion").
      optional().
      valueName("<completionType>").
      action((x,c)=>c.copy(completionType = x)).
      text(
        "default: "+defaultOperationCompletionTypeString+"\n" +
          operationCompletionTypes.map { case t => "\t * " + t }.mkString("\n")+"\n"
      )

    opt[OperationScorer]('s',"scorer").
      optional().
      valueName("<scorer>").
      action((x,c)=>c.copy(scorer = x)).
      text(
        "default: "+defaultScorerString+"\n" +
          scorers.map { case t => "\t * " + t }.mkString("\n")+"\n"
      )

    opt[Int]('e',"maxIt").
      optional().
      valueName("<maxIt>").
      action((x,c)=>c.copy(maxIt=x)).
      text(
        "For search, this is the maximum number of operations to be applied on the psdd.\n" +
          "\tFor bottom-up and top-down, at every level at most f*#vtreeNodesAtThisLevel operations will be applied.\n" +
          "\tdefault: maxInt\n"
      )

    opt[SaveFrequency]('f',"freq").
      optional().
      valueName("<freq>").
      action((x,c)=>c.copy(frequency = x)).
      text("method for saving psdds \n"+

        "\tdefault: "+defaultFrequencyString+"\n" +
        frequencyTypes.map { case t => "\t * " + t }.mkString("\n")+"\n"+
        "\tbest-k to only keep the best psdd on disk, all-k keeps all of the. A save attempt is made every k iterations\n"
      )


    opt[Int]('q',"debugLevel").
      optional().
      valueName("<level>").
      action((x,c)=>c.copy(debugLevel = x)).
      text("debug level\n")

    opt[Unit]("keepSplitting").
      optional().
      action((_,c)=>c.copy(keepSplitting = true)).
      text("Always for search (flag not needed then)\n")

    opt[Unit]("keepCloning").
      optional().
      action((_,c)=>c.copy(keepCloning = true)).
      text("Always for search (flag not needed then)\n")


    checkConfig { c =>
      if (c.train==null) failure("train data is required")
      if (c.vtree==null) failure("vtree is required")
      if (c.out==null) failure("output path is required")
      success
    }

    help("help") text ("prints this usage text\n")
  }

  /**
    * The SDD to PSDD parser parses the options for converting an SDD to a PSDD
    * and learning its parameters from data. (Choi et al, IJCAI 2015)
    * Furthermore, it provides information on the expected arguments
    */
  val sdd2psddParser = new scopt.OptionParser[Config](sdd2psdd) {
    override def showUsageOnError = true

    head("Learn a PSDD by doing parameter learning on an SDD")


    opt[File]('d', "trainData").
      required().
      valueName ("<file>").
      action { (x, c) => c.copy(train = x) }


    opt[File]('b', "validData").
      optional().
      valueName ("<file>").
      action { (x, c) => c.copy(valid = x) }

    opt[File]('t', "testData").
      optional().
      valueName ("<file>").
      action { (x, c) => c.copy(test = x) }


    opt[File]('v', "vtree").
      required().
      valueName ("<file>").
      action { (x, c) => c.copy(vtree = x) }

    opt[File]('s', "sdd").
      required().
      valueName ("<file>").
      action{ (x,c) => c.copy(psdd = x)}

    opt[String]('o', "out").
      required().
      valueName("<path>").
      action( (x, c) => c.copy(out = x) )

    opt[ParameterCalculator]('m',"smooth").
      optional().
      valueName("<smoothingType>").
      action((x,c) =>c.copy(parameterCalculator = x)).
      text (
      "default: " + defaultParameterCalculatorString + "\n" +
        parameterCalculators.map { case (k, v) => "\t * " + k + ": " + v }.mkString("\n"))


    checkConfig { c =>
      if (c.train==null) failure("train data is required")
      if (c.vtree==null) failure("vtree is required")
      if (c.psdd==null) failure("sdd is required")
      if (c.out==null) failure("output path is required")
      success
    }


    help("help") text ("prints this usage text\n")

  }

  /**
    * The Learn Parameters parser parses the options for learning the parameters for an existing psdd from data.
    * Furthermore, it provides information on the expected arguments
    */
  val learnParamsParser = new scopt.OptionParser[Config](parameterLearning) {
    override def showUsageOnError = true

    head("Learn the parameters of a PSDD")


    opt[File]('d', "trainData").
      required().
      valueName ("<file>").
      action { (x, c) => c.copy(train = x) }

    opt[File]('b', "validData").
      optional().
      valueName ("<file>").
      action { (x, c) => c.copy(valid = x) }

    opt[File]('t', "testData").
      optional().
      valueName ("<file>").
      action { (x, c) => c.copy(test = x) }


    opt[File]('v', "vtree").
      required().
      valueName ("<file>").
      action { (x, c) => c.copy(vtree = x) }

    opt[File]('p', "psdd").
      required().
      valueName ("<file>").
      action{ (x,c) => c.copy(psdd = x)}

    opt[String]('o', "out").
      required().
      valueName("<path>").
      action( (x, c) => c.copy(out = x) )

    opt[ParameterCalculator]('m',"smooth").
      optional().
      valueName("<smoothingType>").
      action((x,c) =>c.copy(parameterCalculator = x)).
      text (
      "default: " + defaultParameterCalculatorString + "\n" +
        parameterCalculators.map { case (k, v) => "\t * " + k + ": " + v }.mkString("\n"))


    checkConfig { c =>
      if (c.train==null) failure("train data is required")
      if (c.vtree==null) failure("vtree is required")
      if (c.psdd==null) failure("psdd is required")
      if (c.out==null) failure("output path is required")
      success
    }


    help("help") text ("prints this usage text\n")
  }


  /**
    * The Parameter Search parser parses the options for exploring different parameter learners for a given psdd  and data.
    * Furthermore, it provides information on the expected arguments
    */
  val paramSearchParser = new scopt.OptionParser[Config](paramSearch) {
    override def showUsageOnError = true

    head("Search for the best parameter calculator for a PSDD")


    opt[File]('d', "trainData").
      required().
      valueName ("<file>").
      action { (x, c) => c.copy(train = x) }

    opt[File]('b', "validData").
      optional().
      valueName ("<file>").
      action { (x, c) => c.copy(valid = x) }

    opt[File]('t', "testData").
      optional().
      valueName ("<file>").
      action { (x, c) => c.copy(test = x) }

    opt[File]('v', "vtree").
      required().
      valueName ("<file>").
      action { (x, c) => c.copy(vtree = x) }

    opt[File]('p', "psdd").
      required().
      valueName ("<file>").
      action{ (x,c) => c.copy(psdd = x)}

    opt[Seq[ParameterCalculator]]('m',"parameter calculators").
      optional().
      valueName("<paramCalc1>,<paramCalc2>,...").
      action((x,c) =>c.copy(parameterCalculators = x))


    checkConfig { c =>
      if (c.train==null) failure("train data is required")
      if (c.vtree==null) failure("vtree is required")
      if (c.psdd==null) failure("psdd is required")
      if (c.parameterCalculators==null) failure("parameterCalculators are required")
      success
    }

    help("help") text ("prints this usage text\n")

  }


  /**
    * The Vtree Learner parser parses the options for learning a vtree from data
    * Furthermore, it provides information on the expected arguments
    */
  val learnVtreeParser = new scopt.OptionParser[Config](learnVtree) {
    override def showUsageOnError = true

    head("learn/generate a vtree")


    opt[File]('d', "trainData").
      required().
      valueName ("<file>").
      action { (x, c) => c.copy(train = x) }

    opt[String]('v', "vtreeMethod") optional() valueName (vtreeOptions.keys.mkString("|")) action { (x, c) => c.copy(vtreeMethod = x) } validate { x =>
      if (vtreeOptions.contains(x)) success else failure(x + " is not a shallowValid vtree method.")} text (
      "default: " + defaultVtree + "\n" +
        vtreeOptions.map { case (k, v) => "\t * " + k + ": " + v }.mkString("\n"))

    opt[String]('o', "out").
      required().
      valueName("<path>").
      action( (x, c) => c.copy(out = x) )

    opt[Unit]('e',"entropyOrder").
      optional().
      action((_,c)=>c.copy(entropyOrder = true)).
      text("choose prime variables to have lower entropy\n")

    checkConfig { c =>
      if (c.train==null) failure("train data is required")
      success
    }


    help("help") text ("prints this usage text\n")
  }

  /**
    * The check parser parses the options for checking if a psdd is valild
    * Furthermore, it provides information on the expected arguments
    */
  val checkParser = new scopt.OptionParser[Config](check) {
    override def showUsageOnError = true

    head("Check if a PSDD is valid and calculate its likelihoods (in two ways)") //HERE

    opt[File]('d', "trainData").
      required().
      valueName ("<file>").
      action { (x, c) => c.copy(train = x) }

    opt[File]('b', "validData").
      optional().
      valueName ("<file>").
      action { (x, c) => c.copy(valid = x) }

    opt[File]('t', "testData").
      optional().
      valueName ("<file>").
      action { (x, c) => c.copy(test = x) }


    opt[File]('v', "vtree").
      required().
      valueName ("<file>").
      action { (x, c) => c.copy(vtree = x) }

    opt[File]('p', "psdd").
      required().
      valueName ("<file>").
      action{ (x,c) => c.copy(psdd = x)}

    checkConfig { c =>
      if (c.train==null) failure("train data is required")
      if (c.vtree==null) failure("vtree is required")
      if (c.psdd==null) failure("psdd is required")
      if (c.out==null) failure("output path is required")
      success
    }


    help("help") text ("prints this usage text\n")
  }

  val queryParser = new scopt.OptionParser[Config](query) {
    override def showUsageOnError = true

    head("Query an assembly of psdds") //HERE

    opt[String]('m', "mode").
      required().
      valueName ("<mode>").
      action { (x, c) => c.copy(mode = x) }

    opt[File]('d', "trainData").
      required().
      valueName ("<file>").
      action { (x, c) => c.copy(train = x) }

    opt[File]('b', "validData").
      required().
      valueName ("<file>").
      action { (x, c) => c.copy(valid = x) }

    opt[File]('q', "query").
      optional().
      valueName ("<file>").
      action { (x, c) => c.copy(query = x) }

    opt[File]('v', "vtree").
      required().
      valueName ("<file>").
      action { (x, c) => c.copy(vtree = x) }

    opt[String]('o', "out").
      optional().
      valueName("<path>").
      action( (x, c) => c.copy(out = x) )

    opt[Seq[Double]]('a', "componentweights").
      required().
      valueName("<double>,<double>,...").
      action( (x,c) => c.copy(componentweights = x) )

    opt[Seq[File]]('p', "psdds").
      required().
      valueName ("<file>,<file>,...").
      action( (x,c) => c.copy(psdds = x))


                     // fl_names: Seq[String] = null,
                     // fl_nb_vars: Seq[Int]  = null,
                     // fl_var_cat_dim: Seq[Int] = null,
                     // fl_binary_encoded: Seq[Int] = null,
                     // fl_encoded_start_idx: Seq[Int] = null,
                     // fl_encoded_end_idx: Seq[Int] = null,
                     // fl_to_query: Seq[String] = null,

    opt[Seq[String]]('y',"fl_names").
      required().
      valueName("<String>,<String>,...").
      action((x,c) => c.copy(fl_names = x)).
      text(
        "the fly info list of int: nb_vars, var_cat_dim, binary_encoded,encoded_start_idx,encoded_end_idx.\n"
      )

    opt[Seq[Int]]('x',"fl_nb_vars").
      required().
      valueName("<int>,<int>,...").
      action((x,c)=>c.copy(fl_nb_vars = x)).
      text(
        "the flx info list of int: nb_vars, var_cat_dim, binary_encoded,encoded_start_idx,encoded_end_idx.\n"
      )

    opt[Seq[Int]]('w',"fl_var_cat_dim").
      required().
      valueName("<int>,<int>,...").
      action((x,c)=>c.copy(fl_var_cat_dim = x)).
      text(
        "the flx info list of int: nb_vars, var_cat_dim, binary_encoded,encoded_start_idx,encoded_end_idx.\n"
      )

    opt[Seq[Int]]('z',"fl_binary_encoded").
      required().
      valueName("<int>,<int>,...").
      action((x,c)=>c.copy(fl_binary_encoded = x)).
      text(
        "the flx info list of int: nb_vars, var_cat_dim, binary_encoded,encoded_start_idx,encoded_end_idx.\n"
      )

    opt[Seq[Int]]('r',"fl_encoded_start_idx").
      required().
      valueName("<int>,<int>,...").
      action((x,c)=>c.copy(fl_encoded_start_idx = x)).
      text(
        "the flx info list of int: nb_vars, var_cat_dim, binary_encoded,encoded_start_idx,encoded_end_idx.\n"
      )

    opt[Seq[Int]]('s',"fl_encoded_end_idx").
      required().
      valueName("<int>,<int>,...").
      action((x,c)=>c.copy(fl_encoded_end_idx = x)).
      text(
        "the flx info list of int: nb_vars, var_cat_dim, binary_encoded,encoded_start_idx,encoded_end_idx.\n"
      )

    opt[Seq[String]]('u',"fl_to_query").
      required().
      valueName("<String>,<String>,...").
      action((x,c)=>c.copy(fl_to_query = x)).
      text(
        "the flx info list of int: nb_vars, var_cat_dim, binary_encoded,encoded_start_idx,encoded_end_idx.\n"
      )

    opt[Boolean]('g',"data_bug").
      required().
      valueName("<data_bug>").
      action((x,c)=>c.copy(data_bug=x)).
      text(
        "If data contains bug where two 00s are added between flx and fly.\n"
      )



    checkConfig { c =>
      if (c.vtree==null) failure("vtree is required")
      if (c.psdd==null) failure("psdd is required")
      if (c.out==null) failure("output path is required")
      if (c.mode=="classify" && c.query==null) failure("if mode == classify, query is required")
      if (c.mode=="analyse" && c.out==null) failure("if mode == analyse, out is required")
      success
    }


    help("help") text ("prints this usage text\n")
  }

    def main(args: Array[String]): Unit = {

      if (args.length==0 || args(0).contains("-h")) printHelp()
      else args(0) match{

        // easy way to check if assertions are on or off
        case "assertTest" =>
          try {
            assert(false)
            println("assertions are off")
          } catch {
            case e: AssertionError => println("assertions are on")
            case e: Throwable => throw e
          }


//          learn an ensemble of psdds from data
        case `learnEnsemblePsdd` =>
          if (args.length<2){
            println("provide an ensemble learning method. Options: softEM.")
          }else{
            if (args(1) != "softEM"){
              println("currently the only ensemble learning method that is supported is: softEM.")
            }

            else learnEnsemblePsddParser.parse(args.drop(2), Config()) match {
              case None =>
              case Some(config) =>

                val out = new File(config.out)
                Output.init(out)
                Debug.init(new File(out, "debug"), config.debugLevel)

                Output.addWriter("cmd")
                Output.writeln(args.mkString(" "), "cmd")
                Output.writeln("learnEnsemblePsdd", "cmd")
                Output.writeln(config.configString, "cmd")
                Output.closeWriter("cmd")


                Debug.writeln("data")
                val trainData = Data.readFromFile(config.train)
                val validData = if (config.valid == null) trainData.empty else Data.readFromFile(config.valid)
                val testData = if (config.test == null) trainData.empty else Data.readFromFile(config.test)
                val data = new DataSets(trainData, validData, testData)

                Debug.writeln("psdd manager")
                val sddMgr = new SddManager(Vtree.read(config.vtree.getPath))
                val vtree = VtreeNode.read(config.vtree)
                val psddMgr = new PsddManager(sddMgr, true)

                Debug.writeln("make/read psdd")
                val psdds = for (i<- 0 until config.numComponentLearners) yield {
                  if (config.psdd != null) psddMgr.readPsdd(config.psdd, vtree, data, config.parameterCalculator).asInstanceOf[PsddDecision]
                  else psddMgr.newPsdd(vtree, data, config.parameterCalculator)
                }

                println("Reading PSDD DONE")

                Debug.writeln("start learning...")

                val learner = new SoftEM(data, config.numComponentLearners, config.out, config.parameterCalculator, config.scorer, config.maxIt, config.structureChangeIt, config.parameterLearningInt)
                learner.learn(psdds, psddMgr)


                0 until config.numComponentLearners foreach { i => Output.savePsdds(psdds(i), "final_"+ "_l_"+i, asPsdd = true, asDot = true, asDot2 = true, asSdd = true, withVtree = true)}

                Output.writeln("==== DONE ====")
            }
          }


//          learn a psdd from data
        case `learnPsdd` =>
          if (args.length<2){
            println("provide a learnPsdd method. Options: "+learnMethods.mkString(", "))
          }
          else{
            val operationOrder = args(1)
            if (!learnMethods.contains(operationOrder)) {
              println("Error: incorrect learnPsdd method")
              printHelp()
            }

            else  learnPsddParser.parse(args.drop(2), Config()) match {
              case None =>
              case Some(config) =>

                val out = new File(config.out)
                Output.init(out)
                Debug.init(new File(out, "debug"), config.debugLevel)

                Output.addWriter("cmd")
                Output.writeln(args.mkString(" "), "cmd")
                Output.writeln("learnPsdd", "cmd")
                Output.writeln(config.configString, "cmd")
                Output.closeWriter("cmd")


                Debug.writeln("data")
                val trainData = Data.readFromFile(config.train)
                val validData = if (config.valid == null) trainData.empty else Data.readFromFile(config.valid)
                val testData = if (config.test == null) trainData.empty else Data.readFromFile(config.test)
                val data = new DataSets(trainData, validData, testData)

                Debug.writeln("psdd manager")
                val sddMgr = new SddManager(Vtree.read(config.vtree.getPath))
                val vtree = VtreeNode.read(config.vtree)
                val psddMgr = new PsddManager(sddMgr, true)

                Debug.writeln("make/read psdd")
                val psdd =
                  if (config.psdd!= null) psddMgr.readPsdd(config.psdd, vtree, data, config.parameterCalculator).asInstanceOf[PsddDecision]
                  else psddMgr.newPsdd(vtree, data, config.parameterCalculator)

                val operationFinders: Array[SpecificOperationQueue[_<:PsddOperation, _<:Object]] = config.operationTypes.map {
                  case cloneOp(k) => new CloneOperationQueue(new CloneOperationFinder(psddMgr, config.completionType, config.scorer, config.parameterCalculator, psdd, k.toInt))
                  case split(k) => new SplitOperationQueue(new SplitOperationFinder(psddMgr, config.completionType, config.scorer, config.parameterCalculator, psdd, k.toInt))
                }.toArray

                val learner = operationOrder match {
                  case `search` => new SearchLearner(operationFinders, config.parameterCalculator, config.maxIt, config.frequency)
                  case `bottomUp` => new BottomUpLearner(operationFinders, config.parameterCalculator, config.maxIt, config.keepSplitting, config.keepCloning, config.frequency)
                  case `topDown` => new TopDownLearner(operationFinders, config.parameterCalculator, config.maxIt, config.keepSplitting, config.keepCloning, config.frequency)
                }


                Debug.writeln("start learning...")
                learner.learn(psdd, psddMgr)

                Output.savePsdds(psdd, "final", asPsdd = true, asDot = true, asDot2 = true, asSdd = true, withVtree = true)
                Output.writeln("==== DONE ====")
            }
          }

//          Confert an sdd to a psdd and learn its parameters from data
        case `sdd2psdd` => sdd2psddParser.parse(args.drop(1), Config()) match {
          case None =>
          case Some(config) =>

            val pw = new PrintWriter(config.out+".cmd")
            pw.println(args.mkString(" "))
            pw.println("sdd2psdd")
            pw.println(config.configString)
            pw.flush()
            pw.close()

            val vtree = Vtree.read(config.vtree.getPath)
            val sddMgr = new SddManager(vtree)
            sddMgr.useAutoGcMin(false)

            val pVtree = VtreeNode.read(config.vtree)
            val psddMgr = new PsddManager(sddMgr, true)

            val trainData = Data.readFromFile(config.train)
            val validData = if (config.valid==null) trainData.empty else Data.readFromFile(config.valid)
            val testData = if (config.test==null) trainData.empty else Data.readFromFile(config.test)
            val data = new DataSets(trainData, validData, testData)

            val psdd = psddMgr.readPsddFromSdd(config.psdd, pVtree, data, config.parameterCalculator)

            PsddQueries.save(psdd, new File(config.out))

            println(dataSets.map(_+"Ll").mkString("\t"))
            println(dataSets.map(PsddQueries.logLikelihood(psdd,_)).mkString("\t"))
        }

//          learn the parameters of an existing psdd from data
        case `parameterLearning` => learnParamsParser.parse(args.drop(1), Config()) match {
          case None =>
          case Some(config) =>

            val pw = new PrintWriter(config.out+".cmd")
            pw.println(args.mkString(" "))
            pw.println("parameterLearning")
            pw.println(config.configString)
            pw.flush()
            pw.close()

            val sddMgr = new SddManager(Vtree.read(config.vtree.getPath))
            sddMgr.useAutoGcMin(false)

            val vtree = VtreeNode.read(config.vtree)
            val psddMgr = new PsddManager(sddMgr, true)

            val trainData = Data.readFromFile(config.train)
            val validData = if (config.valid==null) trainData.empty else Data.readFromFile(config.valid)
            val testData = if (config.test==null) trainData.empty else Data.readFromFile(config.test)
            val data = new DataSets(trainData, validData, testData)

            val psdd = psddMgr.readPsdd(config.psdd, vtree, data, config.parameterCalculator)
            PsddQueries.save(psdd, new File(config.out + ".psdd"))

            println(dataSets.map(_+"Ll").mkString("\t"))
            println(dataSets.map(PsddQueries.logLikelihood(psdd,_)).mkString("\t"))
        }

//          learn a vtree from data
        case `learnVtree` => learnVtreeParser.parse(args.drop(1), Config()) match {
          case None =>
          case Some(config) =>
            val pw = new PrintWriter(config.out+".cmd")
            pw.println(args.mkString(" "))
            pw.println("learnVtree")
            pw.println(config.configString)
            pw.flush()
            pw.close()

            val train = Data.readFromFile(config.train)
            val vtreeNode = config.vtreeMethod match {
              case "balancedBU" => VtreeNode.balancedBottomUp(train.vars.sorted)
              case "balanced-ord" => VtreeNode.balanced(train.vars.sorted)
              case "rightLinear-ord" => VtreeNode.rightLinear(train.vars.sorted)
              case "leftLinear-ord" => VtreeNode.leftLinear(train.vars.sorted)
              case "balanced-rand" => VtreeNode.balanced(Random.shuffle(train.vars.toSeq).toArray)
              case "rightLinear-rand" => VtreeNode.rightLinear(Random.shuffle(train.vars.toSeq).toArray)
              case "leftLinear-rand" => VtreeNode.leftLinear(Random.shuffle(train.vars.toSeq).toArray)
              case "pairwiseWeights" => VtreeLearner.learnExhaustiveTopDown(train)
              case "miGreedyBU" => VtreeLearner.learnGreedyBottomUp(train)
              case "miMetis" => VtreeLearner.learnMetisTopDown(train, 1, config.out, config.entropyOrder)
              case "miBlossom" => VtreeLearner.learnBlossomBottomUp(train, config.out, config.entropyOrder)
            }
            vtreeNode match{
              case vtree: VtreeInternal => 
                vtree.save(new File(config.out + ".vtree"))
                vtree.saveAsDot(new File(config.out + ".vtree.dot"))
              case _ => println("There were no variables, so no vtree could be learned")
            }
        }

//          test different parameter learners for a given psdd
        case `paramSearch` => paramSearchParser.parse(args.drop(1), Config()) match {
          case None =>
          case Some(config) =>

            val pw = new PrintWriter(config.out+".cmd")
            pw.println(args.mkString(" "))
            pw.println("parameterLearning")
            pw.println(config.configString)
            pw.flush()
            pw.close()

            val sddMgr = new SddManager(Vtree.read(config.vtree.getPath))
            sddMgr.useAutoGcMin(false)

            val vtree = VtreeNode.read(config.vtree)
            val psddMgr = new PsddManager(sddMgr, true)

            val trainData = Data.readFromFile(config.train)
            val validData = if (config.valid==null) trainData.empty else Data.readFromFile(config.valid)
            val testData = if (config.test==null) trainData.empty else Data.readFromFile(config.test)
            val data = new DataSets(trainData, validData, testData)

            println("params,trainLl,validLl,testLl")
            val psdd = psddMgr.readPsdd(config.psdd, vtree, data)
            val results = config.parameterCalculators.map { calc =>
              psddMgr.calculateParameters(psdd, calc, psdd)
              val res = (calc, PsddQueries.logLikelihood(psdd, "train") / data.train.total, PsddQueries.logLikelihood(psdd, "valid") / data.valid.total, PsddQueries.logLikelihood(psdd, "test") / data.test.total)
              println(res._1+","+res._2+","+res._3+","+res._4)
              res
            }

            println()
            val best = results.maxBy(_._3)
            println(best._1+","+best._2+","+best._3+","+best._4)
        }

//          check if the given psdd is valid (for debug purpose)
        case `check` =>checkParser.parse(args.drop(1), Config()) match {
          case None =>
          case Some(config) =>

            val sddMgr = new SddManager(Vtree.read(config.vtree.getPath))
            sddMgr.useAutoGcMin(false)


            print("Prepare psdd manager...")
            val vtree = VtreeNode.read(config.vtree)
            val psddMgr = new PsddManager(sddMgr, true)
            println(" done!")

            print("Read data...")
            val trainData = Data.readFromFile(config.train)
            val validData = if (config.valid == null) trainData.empty else Data.readFromFile(config.valid)
            val testData = if (config.test == null) trainData.empty else Data.readFromFile(config.test)
            val data = new DataSets(trainData, validData, testData)
            println(" done!")

            print("Read Psdd from "+config.psdd+"...")
            val psdd = psddMgr.readPsdd(config.psdd, vtree, data)
            println(" done!")

            println()

            print("Calculate train ll (fast)...")
            println("\t" + PsddQueries.logLikelihood(psdd, "train") / data.train.total)
            print("Calculate valid ll (fast)...")
            println("\t" + PsddQueries.logLikelihood(psdd, "valid") / data.valid.total)
            print("Calculate test ll (fast)... ")
            println("\t" + PsddQueries.logLikelihood(psdd, "test") / data.test.total)

            println()


            print("Calculate train ll (slow)...")
            println("\t" + data.train.weightedIterator.map { case (example, weight) => PsddQueries.logProb(psdd, example) * weight }.sum / data.train.total)
            print("Calculate valid ll (slow)...")
            println("\t" + data.valid.weightedIterator.map { case (example, weight) => PsddQueries.logProb(psdd, example) * weight }.sum / data.valid.total)
            print("Calculate test ll (slow)...")
            println("\t" + data.test.weightedIterator.map { case (example, weight) => PsddQueries.logProb(psdd, example) * weight }.sum / data.test.total)

            println()

            print("Check psdd validity...")
            val valid = PsddQueries.isValid(psdd)
            println(if (valid) "Valid!" else "Invalid!")

        }

        case `query` =>queryParser.parse(args.drop(1), Config()) match {
          case None =>
          case Some(config) =>

            val sddMgr = new SddManager(Vtree.read(config.vtree.getPath))
            sddMgr.useAutoGcMin(false)


            val pw = new PrintWriter(new File(config.out + "_" + config.mode + ".info"))

            print("Prepare psdd manager...")
            val vtree = VtreeNode.read(config.vtree)
            pw.write("reading vtree from: " + config.vtree + "\n")
            println("reading vtree from: " + config.vtree)
            val psddMgr = new PsddManager(sddMgr, true)
            println(" done!")

            print("Read data...")
            val trainData = Data.readFromFile(config.train)
            val validData = if (config.valid == null) trainData.empty else Data.readFromFile(config.valid)
            val testData = trainData.empty
            val data = new DataSets(trainData, validData, testData)
            println(" done!")

            println("Read Psdds from "+config.psdds+"...")
            val numComponents = config.psdds.size
            val psdds = Seq.tabulate(numComponents)(n => psddMgr.readPsdd(config.psdds(n), vtree, data))
            val componentweights = config.componentweights
            println("Psdd components given")
            for (i <- 0 to numComponents - 1){
              println("PSDD: " + i + " cw: " + componentweights(i) + "\n\tfile: " + config.psdds(i))
              pw.write("PSDD: " + i + " cw: " + componentweights(i) + "\n\tfile: " + config.psdds(i) + "\n")
            }
            println(" done!")

            println()

            println("getting fl information")
            val total_size = trainData.backend(0).size
            // nb_vars, var_cat_dim, binary_encoded,encoded_start_idx,encoded_end_idx
            val fls_names:Seq[String] = config.fl_names
            val fls_to_query = config.fl_to_query
            var fls_maps:Map[String,Seq[Int]] = Map()
            val nb_fls = fls_names.size
            val idx_nb_vars = 0
            val idx_var_cat_dim = 1
            val idx_binary_encoded = 2
            val idx_start_idx = 3
            val idx_end_idx = 4

            for (i <- 0 to nb_fls -1){
              var tmp_seq:Seq[Int] = Seq()
              tmp_seq = tmp_seq :+ config.fl_nb_vars(i)
              tmp_seq = tmp_seq :+ config.fl_var_cat_dim(i)
              tmp_seq = tmp_seq :+ config.fl_binary_encoded(i)
              tmp_seq = tmp_seq :+ config.fl_encoded_start_idx(i)
              tmp_seq = tmp_seq :+ config.fl_encoded_end_idx(i)
              fls_maps += (fls_names(i) -> tmp_seq)
            }

            var fl_info_str:String =("total_size          :  " + total_size)
            for (flx_name <- fls_names){
              fl_info_str += "\n" + flx_name + " -> nb_vars\t:" + fls_maps(flx_name)(idx_nb_vars)
              fl_info_str += "\n" + flx_name + " -> var_cat_dim\t:" + fls_maps(flx_name)(idx_var_cat_dim)
              fl_info_str += "\n" + flx_name + " -> binary_encoded\t:" + fls_maps(flx_name)(idx_binary_encoded)
              fl_info_str += "\n" + flx_name + " -> start_idx\t:" + fls_maps(flx_name)(idx_start_idx)
              fl_info_str += "\n" + flx_name + " -> end_idx\t:" + fls_maps(flx_name)(idx_end_idx)
            }
            println(fl_info_str)
            pw.write(fl_info_str + "\n")

            if(config.mode == "classify"){

              if (fls_to_query.length != 1){
                var out_str = "classify only supoorts one varibel to query but given where: " + fls_to_query
                println(out_str)
                pw.write(out_str + '\n')
                pw.close()
                return -1
              }

              var class_fl = fls_to_query(0)
              //fls_maps must contain fly

              var ymaps:Seq[Map[Int,Boolean]] = null
              if (fls_maps(class_fl)(idx_binary_encoded) == 1){
                ymaps = Seq.tabulate(fls_maps(class_fl)(idx_var_cat_dim))(x => int2map(x, fls_maps(class_fl)(idx_end_idx) - fls_maps(class_fl)(idx_start_idx), fls_maps("fly")(idx_start_idx)))
              }else {
                ymaps = Seq.tabulate(fls_maps(class_fl)(idx_var_cat_dim))(x => int2onehot(x, fls_maps(class_fl)(idx_end_idx) - fls_maps(class_fl)(idx_start_idx), fls_maps("fly")(idx_start_idx)))
              }
              println( "Calculated ymaps: " + ymaps)
              pw.write("Calculated ymaps: " + ymaps + '\n')

              var priors:Seq[BigDecimal] = Seq.tabulate(fls_maps(class_fl)(idx_var_cat_dim))(x =>
                Seq.tabulate(numComponents)(xx => PsddQueries.bigDecimalProb(psdds(xx), ymaps(x)) * componentweights(xx)).sum
                )
              println("Calculated Priors: " + priors)
              pw.write("Calculated Priors: " + priors + '\n')

              print("Read Assignment...")
              val assignment = Data.readFromFile(config.query)
              println(" done!")
              var accuracy:Seq[Int] = Seq()
              val nb_queries_total = (assignment.backend.length)
              val one_hundreth_of_total_queries = if (nb_queries_total > 100) (nb_queries_total/100).toInt else 1
              println("nb_queries_total: " + nb_queries_total)
              println("one_hundreth_of_total_queries: " + one_hundreth_of_total_queries)
              for ( i <- 0 to nb_queries_total -1) {
                var xmap:Map[Int,Boolean] = Map()
                var actual_label:Map[Int,Boolean] = Map()
                var actual_label_num:Int = -1
                assignment.backend(i).keys.foreach{j =>
                  if (j == 0){
                    println("INDEXING MISTAKE AT POS: adsfadsfasfd (assuming 1 - ..)")
                  }
                  if(fls_maps(class_fl)(idx_start_idx) < j && j <= fls_maps(class_fl)(idx_end_idx)){
                    actual_label += (j -> assignment.backend(i)(j))
                  } else {
                    xmap += (j -> assignment.backend(i)(j))
                  }
                }

                var highestProb:BigDecimal = 0.0
                var class_probabilities:Seq[BigDecimal] = Seq()
                var highestProbIdx = 0
                var correct_class_prob:BigDecimal = 0.0

                for (j <- 0 to fls_maps(class_fl)(idx_var_cat_dim) - 1){
                  var assignment_tmp = xmap ++ ymaps(j)
                  var result = Seq.tabulate(numComponents)(x => PsddQueries.bigDecimalProb(psdds(x), assignment_tmp) * componentweights(x)).sum
                  result = result/priors(j)
                  class_probabilities = class_probabilities :+ result

                  if (result > highestProb){
                    highestProb = result
                    highestProbIdx = j
                  }
                  if (ymaps(j) == actual_label){
                    correct_class_prob = result
                    actual_label_num = j
                  }
                }
                // var outputString = "For test point " + i + " the predicted label is: " + highestProbIdx + " actual_label: " + actual_label_num + " actual_label: " + actual_label
                // var wronglyclassified = ""
                // if(highestProbIdx != actual_label_num){
                //   // wronglyclassified = " -- Wrong - pcc: %.5f vs  ccc: %.5f".format(highestProb/class_probabilities.sum, correct_class_prob/class_probabilities.sum)
                //   // pw.write(outputString + wronglyclassified + "\n")
                // }
                // println(outputString + wronglyclassified)
                accuracy = accuracy :+ (if (highestProbIdx == actual_label_num) 1 else 0)
                if ((i) % one_hundreth_of_total_queries == 0 && i != 0){
                  var current_percent = BigDecimal((i/nb_queries_total.toDouble) * 100).setScale(0, BigDecimal.RoundingMode.CEILING)
                  var current_acc = accuracy.sum.toDouble/accuracy.size.toDouble

                  var outputString = "computed :" + current_percent + "% (" + i + ") \tof all queries current acc: " + current_acc
                  println(outputString)
                  pw.write(outputString + "\n")
                }

              }

              val testacc = accuracy.sum.toDouble/accuracy.size.toDouble
              val outputString2 = "\n\nThe accuracy over all queries is " + testacc + "\n"
              println(outputString2)
              pw.write(outputString2)
              pw.close
            }

            if(config.mode == "generative_query_bin"){

              print("Read Assignment...")
              val assignment = Assignment.readFromFile(config.query)
              var unassigned_vars:Set[Int] = Set()
              for(i <- fls_to_query){
                unassigned_vars = unassigned_vars ++ (fls_maps(i)(idx_start_idx) + 1 to fls_maps(i)(idx_end_idx)).toSet
              }
              val assigned_vars = (1 to total_size).toSet.diff(unassigned_vars)
              println(" done!\n\t -> unassigned_vars: " + unassigned_vars)

              val nb_queries_total = (assignment.backend.length)
              val one_hundreth_of_total_queries = if (nb_queries_total > 100) (nb_queries_total/100).toInt else 1
              println("nb_queries_total: " + nb_queries_total)
              println("one_hundreth_of_total_queries: " + one_hundreth_of_total_queries)

              val pw_samples = new PrintWriter(new File(config.out + "_bin.data"))

              val random = new Random

              var sumConfidence:BigDecimal = 0

              for ( i <- 0 to nb_queries_total -1) {
                var fl_sampled:Map[Int,Boolean] = Map()
                var fl_evidence:Map[Int,Boolean] = Map()
                assigned_vars.foreach{j =>
                  fl_evidence += (j -> assignment.backend(i)(j)) 
                }

                var unsassinged_stack = unassigned_vars.toList
                // var fl_sampled:Map[Int,Boolean] = Map()
                var fl_tmp_num:Map[Int,Boolean] = Map()
                var fl_tmp_div:Map[Int,Boolean] = Map()
                var new_var_idx = 0
                var new_var = 0

                for (j <- unassigned_vars){
                  // println(unsassinged_stack.length)
                  new_var_idx = random.nextInt(unsassinged_stack.length)
                  new_var = unsassinged_stack(new_var_idx)
                  unsassinged_stack = unsassinged_stack.dropRight(unsassinged_stack.length - new_var_idx) ++ unsassinged_stack.drop(new_var_idx + 1)

                  fl_tmp_num = (fl_evidence ++ fl_sampled) + (new_var -> true)
                  fl_tmp_div = (fl_evidence ++ fl_sampled)
                  // pr(FLx_j = true| fly + flx)
                  var prob_num:BigDecimal = Seq.tabulate(numComponents)(x => PsddQueries.bigDecimalProb(psdds(x), fl_tmp_num) * componentweights(x)).sum
                  var prob_div:BigDecimal = Seq.tabulate(numComponents)(x => PsddQueries.bigDecimalProb(psdds(x), fl_tmp_div) * componentweights(x)).sum
                  var prob_j = prob_num/prob_div
                  
                  var value_j = random.nextDouble() <= prob_j
                  fl_sampled += (new_var -> value_j)
                  // var tmpStr = "new_var_idx: %d, new_var: %d, prob_j: %.2f, value_j: %s, unsassinged_stack.length: %d\n".format(new_var_idx,new_var, prob_j, value_j.toString, unsassinged_stack.length)
                  // print(tmpStr)
                  // pw.write(tmpStr)
                }

                var fl_fully_assigned = (fl_evidence ++ fl_sampled)
                //Compute the probability of the fully assigmed fl conditional on the evidence given
                var prob_num:BigDecimal = Seq.tabulate(numComponents)(x => PsddQueries.bigDecimalProb(psdds(x), fl_fully_assigned) * componentweights(x)).sum
                var prob_div:BigDecimal = Seq.tabulate(numComponents)(x => PsddQueries.bigDecimalProb(psdds(x), fl_evidence) * componentweights(x)).sum
                var prob_fl_fully_assigned = prob_num/prob_div 
                sumConfidence = sumConfidence + prob_fl_fully_assigned

                for(j <- 1 to total_size - 1){
                  pw_samples.write("%d,".format(if (fl_fully_assigned(j)) 1 else 0))
                }
                pw_samples.write("%d".format(if (fl_fully_assigned(total_size)) 1 else 0))
                pw_samples.write(";" + prob_fl_fully_assigned + "\n")

                if ((i) % one_hundreth_of_total_queries == 0 && i != 0){
                  var current_percent = BigDecimal((i/nb_queries_total.toDouble) * 100).setScale(0, BigDecimal.RoundingMode.CEILING)
                  var current_confidence = sumConfidence/i.toDouble

                  var outputString = "computed :" + current_percent + "% (" + i + ") \tof all queries - ave confidence: " + current_confidence + "\n"
                  print(outputString)
                  pw.write(outputString)
                }

              }
              pw.write("Overall confidence in the genreated samples is: " + sumConfidence.toDouble/nb_queries_total.toDouble + "\n")
              pw.close
              pw_samples.close
            }

            if(config.mode == "generative_query_dis"){

              print("Read Assignment...")
              val assignment = Assignment.readFromFile(config.query)
              // val unassigned_vars = (1 to total_size).toSet.diff(assigned_vars)
              var unassigned_bin_vars:Set[Int] = Set()
              var nb_vars_to_query = 0
              for(i <- fls_to_query){
                unassigned_bin_vars = unassigned_bin_vars ++ (fls_maps(i)(idx_start_idx) + 1 to fls_maps(i)(idx_end_idx)).toSet
                nb_vars_to_query += fls_maps(i)(idx_nb_vars)
              }
              val assigned_bin_vars = (1 to total_size).toSet.diff(unassigned_bin_vars)
              println(" done!\n\t -> unassigned_bin_vars: " + unassigned_bin_vars)

              val nb_queries_total = (assignment.backend.length)
              val one_hundreth_of_total_queries = if (nb_queries_total > 100) (nb_queries_total/100).toInt else 1
              println("nb_queries_total: " + nb_queries_total)
              println("one_hundreth_of_total_queries: " + one_hundreth_of_total_queries)

              val pw_samples = new PrintWriter(new File(config.out + "_dis.data"))

              val random = new Random
              var sumConfidence:BigDecimal = 0
              
              for ( i <- 0 to nb_queries_total -1) {
                var fl_sampled:Map[Int,Boolean] = Map()
                var fl_evidence:Map[Int,Boolean] = Map()
                assigned_bin_vars.foreach{j =>
                  fl_evidence += (j -> assignment.backend(i)(j)) 
                }

                var unsassinged_stack = unassigned_bin_vars
                // var fl_sampled:Map[Int,Boolean] = Map()
                var fl_tmp_num:Map[Int,Boolean] = Map()
                var fl_tmp_div:Map[Int,Boolean] = Map()
                var new_var:Int = 0
                var unknownMaps:Seq[Map[Int,Boolean]] = null

                for (j <- 0 to nb_vars_to_query - 1){
                  //fruits.toVector(rnd.nextInt(fruits.size))
                  // new_var_idx = random.nextInt(unsassinged_stack.size)
                  new_var = unsassinged_stack.toVector(random.nextInt(unsassinged_stack.size))
                  var base_var = -1
                  
                  for(fl_part <- fls_to_query){
                    if (new_var > fls_maps(fl_part)(idx_start_idx) && new_var <= fls_maps(fl_part)(idx_end_idx)){
                      if (fls_maps(fl_part)(idx_binary_encoded) == 1){
                        var new_var_relative = new_var - fls_maps(fl_part)(idx_start_idx) - 1
                        var var_bin_length = (fls_maps(fl_part)(idx_end_idx) - fls_maps(fl_part)(idx_start_idx)) / fls_maps(fl_part)(idx_nb_vars)
                        base_var = (new_var_relative / var_bin_length) * var_bin_length
                        base_var = base_var + fls_maps(fl_part)(idx_start_idx) + 1
                        unsassinged_stack --= (base_var to (base_var + var_bin_length - 1 ))

                        unknownMaps = Seq.tabulate(fls_maps(fl_part)(idx_var_cat_dim))(x => 
                                  int2map(x, var_bin_length, base_var - 1))
                      } else { // onehot encoded variables
                        var new_var_relative = new_var - fls_maps(fl_part)(idx_start_idx) - 1
                        var var_bin_length = fls_maps(fl_part)(idx_var_cat_dim)
                        base_var = (new_var_relative / var_bin_length) * var_bin_length
                        base_var = base_var + fls_maps(fl_part)(idx_start_idx) + 1
                        unsassinged_stack --= (base_var to (base_var + var_bin_length - 1 ))

                        unknownMaps = Seq.tabulate(fls_maps(fl_part)(idx_var_cat_dim))(x => 
                          int2onehot(x, var_bin_length, base_var))
                      }
                    }
                  }
                  // if (j == nb_vars_to_query -1){
                  //   println("\n\n\n this should be empty now: " + unsassinged_stack + "\n\n\n")
                  // }

                  // println("unknownMaps: " + unknownMaps)
                  
                  var contender_probs:Seq[BigDecimal] = Seq()
                  for (contender <- unknownMaps){
                    fl_tmp_num = (fl_sampled ++ fl_evidence) ++ contender
                    fl_tmp_div = (fl_sampled ++ fl_evidence)

                    var prob_num:BigDecimal = Seq.tabulate(numComponents)(x => PsddQueries.bigDecimalProb(psdds(x), fl_tmp_num) * componentweights(x)).sum
                    var prob_div:BigDecimal = Seq.tabulate(numComponents)(x => PsddQueries.bigDecimalProb(psdds(x), fl_tmp_div) * componentweights(x)).sum

                    contender_probs = contender_probs :+ (prob_num/prob_div)
                  }

                  var contender_probs_norm:Seq[BigDecimal] = Seq()
                  for (contender_prob <- contender_probs){
                    contender_probs_norm = contender_probs_norm :+ contender_prob/(contender_probs.sum)
                  }
                  var value_j = draw_from_continuous(contender_probs)
                  // println("Drew: " + value_j + "  from dist: " + contender_probs_norm + "assigning: " + unknownMaps(value_j))
                  fl_sampled = fl_sampled ++ unknownMaps(value_j)
                  // var tmpStr = "new_var_idx: %d, new_var: %d, contender_probs_norm: %.2f, value_j: %s, unsassinged_stack.length: %d\n".format(new_var_idx,new_var, contender_probs_norm, value_j.toString, unsassinged_stack.length)
                  // print(tmpStr)
                  // pw.write(tmpStr)
                }

                var fl_fully_assigned = (fl_evidence ++ fl_sampled)
                //Compute the probability of the fully assigmed fl conditional on the evidence given
                var prob_num:BigDecimal = Seq.tabulate(numComponents)(x => PsddQueries.bigDecimalProb(psdds(x), fl_fully_assigned) * componentweights(x)).sum
                var prob_div:BigDecimal = Seq.tabulate(numComponents)(x => PsddQueries.bigDecimalProb(psdds(x), fl_evidence) * componentweights(x)).sum
                var prob_fl_fully_assigned = prob_num/prob_div 
                sumConfidence = sumConfidence + prob_fl_fully_assigned

                //Write resulting fl_assignment to file with corresponding probability
                for(j <- 1 to total_size - 1){
                  pw_samples.write("%d,".format(if (fl_fully_assigned(j)) 1 else 0))
                }
                pw_samples.write("%d".format(if (fl_fully_assigned(total_size)) 1 else 0))
                pw_samples.write(";" + prob_fl_fully_assigned + "\n")

                if ((i) % one_hundreth_of_total_queries == 0 && i != 0){
                  var current_percent = BigDecimal((i/nb_queries_total.toDouble) * 100).setScale(0, BigDecimal.RoundingMode.CEILING)
                  var current_confidence = sumConfidence.toDouble/i.toDouble

                  var outputString = "computed :" + current_percent + "% (" + i + ") \tof all queries - ave confidence: " + current_confidence + "\n"
                  print(outputString)
                  pw.write(outputString)
                }

              }

              pw.write("Overall confidence in the genreated samples is: " + sumConfidence.toDouble/nb_queries_total.toDouble + "\n")
              pw.close
              pw_samples.close
            }
            // if(config.mode == "analyse"){

            //   //top k infulencers
            //   val pw = new PrintWriter(new File(config.out))
            //   pw.write("flx_size: " + flx_size + "\n")
            //   pw.write("fly_size: " + fly_size + "\n")
            //   pw.write("fl_size:  " + (flx_size + fly_size) + "\n")


            //   //build fly
            //   for( i <- 1 to fly_size){
            //     var fly:Map[Int,Boolean] = Map()
            //     println("\n---------------- fly = " + i + " -------------------------\n")
            //     for( a <- flx_size + 1 to flx_size + fly_size){
            //       if(a - flx_size == i){
            //         fly += (a -> true)
            //       } else {
            //         fly += (a -> false)
            //       }
            //     }

            //     //build flx
            //     for ( j <- 1 to flx_size){
            //       var flx:Map[Int,Boolean] = Map()
            //       flx += (j -> true)

            //       //query psdds
            //       var fl = flx ++ fly
            //       var prob_num = Seq.tabulate(numComponents)(x => PsddQueries.bigDecimalProb(psdds(x), fl) * componentweights(x)).sum

            //       //compute pr(fly = i| flxj = True)
            //       var prob_div_fly = Seq.tabulate(numComponents)(x => PsddQueries.bigDecimalProb(psdds(x), flx) * componentweights(x)).sum
            //       var prob_fly = prob_num/prob_div_fly

            //       //compute pr(flxj = True| fly = i)
            //       var prob_div_flx = Seq.tabulate(numComponents)(x => PsddQueries.bigDecimalProb(psdds(x), fly) * componentweights(x)).sum
            //       var prob_flx = prob_num/prob_div_flx

            //       // println("pr(fly = " + i + " | flx" + j + " = T) = " + prob_fly)
            //       // println("pr(flx" + j + " = T | fly = " + i + " ) = " + prob_fly)
            //       println("flx =" + j + "; fly = " + i + ";\t pr(fly|flx) = " + prob_fly + ";\t pr(flx|fly) = " + prob_flx)
            //       pw.write("flx =" + j + "; fly = " + i + ";\t pr(fly|flx) = " + prob_fly + ";\t pr(flx|fly) = " + prob_flx + "\n")
            //     }
            //   }

            //   pw.close
            // }
            // //
            // // val assignment2 = Map(1 -> false, 2 -> false, 3 -> true)
            // // print("Calculate log probability for map...")
            // // println("\t" + PsddQueries.logProb(psdd, assignment2))
            // // print("Calculate log probability for map...")
            // // println("\t" + PsddQueries.bigDecimalProb(psdd, assignment2))

            // if(config.mode == "generate"){

            //   val num_batches_per_class = 1
            //   val batch_size = 100

            //   val pw = new PrintWriter(new File(config.out + "info.txt"))
            //   pw.write("flx_size: " + flx_size + "\n")
            //   pw.write("fly_size: " + fly_size + "\n")
            //   pw.write("fl_size:  " + (flx_size + fly_size) + "\n")
            //   pw.write("num_batches_per_class: " + num_batches_per_class + "\n")
            //   pw.write("batch_size: " + batch_size + "\n")

            //   val ymaps = Seq.tabulate(fly_cdim)(x => int2map(x, fly_size, flx_size))
            //   println("Calculated ymaps: " + ymaps)
            //   pw.write("Calculated ymaps: " + ymaps + '\n')

            //   //build fly
            //   var fly:Map[Int,Boolean] = ymaps(0)
            //   for( i <- 0 to fly_cdim - 1){
            //     val pw_file = new PrintWriter(new File(config.out + "samples_class_" + i + ".data"))
            //     fly = ymaps(i)
                
            //     //geneate for clas i
            //     val random = new Random
            //     var tmpStr:String  = "Drawing " + (num_batches_per_class * batch_size) + " samples from the Distribution conditioned on fly = " + i + " " + fly + '\n'
            //     print(tmpStr)
            //     pw.write(tmpStr)
            //     for(exp <- 0 to (num_batches_per_class * batch_size) - 1){
            //       print("#")
            //       var xvalues = List.range(1, flx_size + 1)
            //       var flx:Map[Int,Boolean] = Map()
            //       var j_idx = 0
            //       var j = 0
            //       for (var_count <- 1 to flx_size){
            //         // println(xvalues)
            //         j_idx = random.nextInt(xvalues.length)
            //         j = xvalues(j_idx)
            //         xvalues = xvalues.dropRight(xvalues.length - j_idx) ++ xvalues.drop(j_idx + 1)
            //         // println(xvalues)

            //         var fl_tmp = flx ++ fly
            //         fl_tmp += (j -> true)
            //         // pr(FLx_j = true| fly + flx)
            //         var prob_num:BigDecimal = Seq.tabulate(numComponents)(x => PsddQueries.bigDecimalProb(psdds(x), fl_tmp) * componentweights(x)).sum
            //         var prob_div_flx:BigDecimal = Seq.tabulate(numComponents)(x => PsddQueries.bigDecimalProb(psdds(x), fly ++ flx) * componentweights(x)).sum

            //         var prob_j = prob_num/prob_div_flx
            //         var value_j = random.nextDouble() <= prob_j
            //         flx += (j -> value_j)
            //         tmpStr = "it: %d, j_idx: %d, j: %d, prob_j: %.2f, value_j: %s, xvlaues.length: %d\n".format(var_count,j_idx,j, prob_j, value_j.toString, xvalues.length)
            //         // print(tmpStr)
            //         pw.write(tmpStr)

            //       }
            //       for(idx <- 1 to flx_size){
            //         pw_file.write("%d,".format(if (flx(idx)) 1 else 0))
            //       }
            //       for(idx <- 1 to fly_size - 1){
            //         pw_file.write("%d,".format(if (fly(idx + flx_size)) 1 else 0))
            //       }
            //       pw_file.write("%d\n".format(if (fly(fly_size + flx_size)) 1 else 0))
            //       print('\n')
            //     }
            //     print("\n")
            //     pw_file.close
            //   }
            //   pw.close
            // }


            // if(config.mode == "fl_samples"){

            //   val num_batches_per_class = 1
            //   val batch_size = 100

            //   val pw = new PrintWriter(new File(config.out + "info.txt"))
            //   pw.write("#flx_size: " + flx_size + "\n")
            //   pw.write("#fly_size: " + fly_size + "\n")
            //   pw.write("#fl_size:  " + (flx_size + fly_size) + "\n")
            //   pw.write("#num_batches_per_class: " + num_batches_per_class + "\n")
            //   pw.write("#batch_size: " + batch_size + "\n")

            //   val ymaps = Seq.tabulate(fly_cdim)(x => int2map(x, fly_size, flx_size))
            //   println("#Calculated ymaps: " + ymaps)
            //   pw.write("#Calculated ymaps: " + ymaps + '\n')

            //   //build fly
            //   for( i <- 1 to flx_nbvars){
            //     for (j <- 0 to fly_cdim - 1){
            //       var offset = flx_nbvars * flx_binVarSize + 1
            //       var flx_var:Map[Int,Boolean] = int2map(j, flx_binVarSize, offset)
            //       var tmpStr:String = "#\n#---------------- Variable = " + i + " category " + j + "short: sample_v_i_c_j -------------------------\n"
            //       println(tmpStr)
            //       pw.write(tmpStr)
                  

            //       //geneate for clas i
            //       val random = new Random
            //       tmpStr = "#Drawing " + (num_batches_per_class * batch_size) + " samples from the Distribution conditioned on fly = " + i + '\n'
            //       print(tmpStr)
            //       pw.write(tmpStr)
            //       for(exp <- 0 to (num_batches_per_class * batch_size) - 1){
            //         print("#")
            //         var xvalues = List.range(1, flx_size + 1)
            //         xvalues = xvalues.dropRight(xvalues.length - offset) ++ xvalues.drop(offset + flx_binVarSize)
            //         var flx:Map[Int,Boolean] = Map()
            //         var j_idx = 0
            //         var j = 0

            //         //Generate the rest of flx
            //         for (var_count <- 1 to flx_size - flx_binVarSize){
            //           println(xvalues)
            //           j_idx = random.nextInt(xvalues.length)
            //           j = xvalues(j_idx)
            //           xvalues = xvalues.dropRight(xvalues.length - j_idx) ++ xvalues.drop(j_idx + 1)
            //           println(xvalues)

            //           var fl_tmp = flx ++ flx_var
            //           fl_tmp += (j -> true)
            //           // pr(FLx_j = true| fly + flx)
            //           var prob_num:BigDecimal = Seq.tabulate(numComponents)(x => PsddQueries.bigDecimalProb(psdds(x), fl_tmp) * componentweights(x)).sum
            //           var prob_div_flx:BigDecimal = Seq.tabulate(numComponents)(x => PsddQueries.bigDecimalProb(psdds(x), flx_var ++ flx) * componentweights(x)).sum

            //           var prob_j = prob_num/prob_div_flx
            //           var value_j = random.nextDouble() <= prob_j
            //           flx += (j -> value_j)
            //           tmpStr = "it: %d, j_idx: %d, j: %d, prob_j: %.2f, value_j: %s, xvlaues.length: %d".format(var_count,j_idx,j, prob_j, value_j.toString, xvalues.length)
            //           println(tmpStr)
            //         }

            //         var highestProb:BigDecimal = 0.0
            //         var class_probabilities:Seq[BigDecimal] = Seq()
            //         var highestProbIdx = 0

            //         var priors:Seq[BigDecimal] = Seq.tabulate(fly_cdim)(x =>
            //           Seq.tabulate(numComponents)(xx => PsddQueries.bigDecimalProb(psdds(xx), ymaps(x)) * componentweights(xx)).sum
            //           )

            //         for (j <- 0 to fly_cdim -1){
            //           var assignment_tmp = flx ++ flx_var ++ ymaps(j)
            //           var result = Seq.tabulate(numComponents)(x => PsddQueries.bigDecimalProb(psdds(x), assignment_tmp) * componentweights(x)).sum
            //           result = result/priors(j)
            //           class_probabilities = class_probabilities :+ result

            //           if (result > highestProb){
            //             highestProb = result
            //             highestProbIdx = j
            //           }
            //         }
            //         tmpStr = "----- assigned label " + highestProbIdx + " to the sample drawn from " + i + " " + j
            //         println(tmpStr)
            //         pw.write(tmpStr)


            //         val pw_file = new PrintWriter(new File(config.out + "samples_v_" + i + "_c_" + j + ".data"))
            //         for(idx <- 1 to flx_size){
            //           pw_file.write("%d,".format(if (flx(idx)) 1 else 0))
            //         }
            //         for(idx <- 1 to fly_size){
            //           pw_file.write("%d,".format(if (ymaps(highestProbIdx)(idx + flx_size)) 1 else 0))
            //         }
            //         pw_file.write("\n")
            //       }
            //       print("\n")

            //     }
            //   }

            //   pw.close
            // }

        }

          // This is some scratch space that can be used to test stuff during implementation.
        case `scratch` =>
            println("scratch")
            // write scratch code here

      }

    }

  def int2map(i: Int, numPos: Int, strtIdx: Int): Map[Int,Boolean] = {
    val codeAsStr:String = int2bin(i, numPos)
    var resmap:Map[Int,Boolean] = Map()
    var idx:Int = -1
    var value:Boolean = false
    for( a <- 0 to numPos - 1){
      idx = a + strtIdx + 1
      value = codeAsStr(a) == '1'

      resmap += (idx -> value)
    }

    return resmap
  }

  def int2bin(i: Int, numPos: Int): String = {
    def nextPow2(i: Int, acc: Int): Int = if (i < acc) acc else nextPow2(i, 2 * acc)
    (nextPow2(i, math.pow(2,numPos).toInt)+i).toBinaryString.substring(1)
  }

  def int2onehot(i: Int, numPos: Int, strtIdx: Int): Map[Int,Boolean] = {
    var resmap:Map[Int,Boolean] = Map()
    var idx:Int = -1
    var value:Boolean = false
    for( a <- 0 to numPos - 1){
      idx = a + strtIdx + 1
      if (a == i){
        resmap += (idx -> true)
      } else {
        resmap += (idx -> false)
      }
    }
    return resmap
  }

  def draw_from_continuous(distribution: Seq[BigDecimal]): Int = {
    val random = new Random
    val u = random.nextDouble()
    var res:Int = -1
    var sum:BigDecimal = 0.0
    for (i <- 0 to distribution.size - 1){
      if (sum < u && u < distribution(i) + sum){
        res = i
      }
      sum += distribution(i)
    }
    return res
  }
}