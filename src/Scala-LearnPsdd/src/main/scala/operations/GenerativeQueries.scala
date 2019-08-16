package operations


import scala.util.Random
import scala.math._
import operations.PsddQueries
import structure._



object GenerativeQueries {

  def sample_discrete_variable_full(start_idx: Int, end_idx: Int, binary_encoded: Int,cat_dim: Int, assigned_vars: Map[Int, Boolean], psdds: Seq[PsddNode], componentweights: Seq[Double]):Map[Int, Boolean] = {

	var possible_assignments:Seq[Map[Int,Boolean]] = null
	if (binary_encoded == 1){
		possible_assignments = Seq.tabulate(cat_dim)(x => int2map(x, end_idx - start_idx, start_idx))
	}else {
		possible_assignments = Seq.tabulate(cat_dim)(x => int2onehot(x, end_idx - start_idx, start_idx))
	}
	return sample_discrete_variable(possible_assignments, assigned_vars, psdds, componentweights)
  }

  def sample_discrete_variable(possible_assignments: Seq[Map[Int, Boolean]], assigned_vars: Map[Int, Boolean], psdds: Seq[PsddNode], componentweights: Seq[Double]):Map[Int, Boolean] = {
	var highestProb: BigDecimal = -1
	var highestProbIdx = -1
	for (j <- 0 to (possible_assignments.size - 1)){
		var prob = PsddQueries.bigDecimalCoditionalProb(psdds, componentweights, assigned_vars, possible_assignments(j))
		if (prob > highestProb){
			highestProb = prob
			highestProbIdx = j
		}
	}
	return possible_assignments(highestProbIdx)
  }

  def sample_all_binary_variables(unsassinged_vars: Set[Int], fl_evidence:Map[Int,Boolean], psdds: Seq[PsddNode], componentweights: Seq[Double]): Map[Int, Boolean] = {
	var fl_sampled:Map[Int,Boolean] = Map()
	var new_var_idx = 0
	var new_var = 0
	var nb_vars_to_sample = unsassinged_vars.size
	var unsassinged_stack = unsassinged_vars.toList

	val random = new Random

	for (j <- 0 to (nb_vars_to_sample - 1)){
		// println(unsassinged_stack.length)
		new_var_idx = random.nextInt(unsassinged_stack.length)
		new_var = unsassinged_stack(new_var_idx)
		unsassinged_stack = unsassinged_stack.dropRight(unsassinged_stack.length - new_var_idx) ++ unsassinged_stack.drop(new_var_idx + 1)

		// Infer most probable value of the variable
		var new_assignment = sample_binary_variable(new_var, (fl_evidence ++ fl_sampled), psdds, componentweights)
		fl_sampled += new_assignment
	}
	return fl_sampled
  }

  def sample_binary_variable(new_var: Int, assigned_vars: Map[Int, Boolean], psdds: Seq[PsddNode], componentweights: Seq[Double]):(Int, Boolean) = {

  	var fl_tmp_num = assigned_vars + (new_var -> true)

	val random = new Random
	var prob_j = PsddQueries.bigDecimalCoditionalProb(psdds, componentweights, assigned_vars, fl_tmp_num)

	var value_j = random.nextDouble() <= prob_j
	return (new_var -> value_j)

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