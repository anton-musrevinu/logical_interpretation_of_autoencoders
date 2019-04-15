SMALL="small"
if [ $1 == $SMALL ]
then
	echo $SMALL
	TRAINDATA=~/code/data/exp_full_system_mnist/in/exp_full_system_mnist.train.data.small
	VALIDDATA=~/code/data/exp_full_system_mnist/in/exp_full_system_mnist.valid.data.small
	TESTDATA=~/code/data/exp_full_system_mnist/in/exp_full_system_mnist.test.data.small
else
	TRAINDATA=~/code/data/exp_full_system_mnist/in/exp_full_system_mnist.train.data
	VALIDDATA=~/code/data/exp_full_system_mnist/in/exp_full_system_mnist.valid.data
	TESTDATA=~/code/data/exp_full_system_mnist/in/exp_full_system_mnist.test.data
fi

java -jar ~/code/Scala-LearnPsdd/target/scala-2.11/psdd.jar "learnEnsemblePsdd" "softEM" \
			-v ~/code/data/exp_full_system_mnist/in/exp_full_system_mnist.vtree \
			-d $TRAINDATA \
			-b $VALIDDATA \
			-t $TESTDATA \
			-o ~/code/data/exp_full_system_mnist/out/ \
			-c 5

						# -p ~/code/data/exp_full_system_mnist/in/exp_full_system_mnist.psdd  \
