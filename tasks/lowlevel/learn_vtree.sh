EXEC="~/code/Scala-LearnPsdd/target/scala-2.11/psdd.jar"
TRAINDATA="~/code/Density-Estimation-Datasets/datasets/exp_full_system_mnist/exp_full_system_mnist.train.data"
OUT="~/code/learnPSDD/vtrees/exp_full_system"

echo java -jar ~/code/Scala-LearnPsdd/target/scala-2.11/psdd.jar learnVtree -d $TRAINDATA -o $OUT
java -jar ~/code/Scala-LearnPsdd/target/scala-2.11/psdd.jar -d $TRAINDATA -o $OUT