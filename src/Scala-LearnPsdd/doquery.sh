# java -jar ~/code/Scala-LearnPsdd/target/scala-2.11/psdd.jar query -v ./../data/it_9_l_4.vtree -p ./../data/it_9_l_4.psdd -d ./../data/exp_full_system.train.data -b ./../data/exp_full_system.valid.data -t ./../data/exp_full_system.test.data
DIR=./../learnPSDD/EnsembleExperiments/exp_full_system_mnist_exp4/models
MODE=classify
OUTFILE=$DIR/../classify.txt
VTREE=$DIR/it_50_l_0.vtree
PSDD0=$DIR/it_50_l_0.psdd
PSDD1=$DIR/it_50_l_1.psdd
PSDD2=$DIR/it_50_l_2.psdd
PSDD3=$DIR/it_50_l_3.psdd
PSDD4=$DIR/it_50_l_4.psdd
PSDD5=$DIR/it_50_l_5.psdd
PSDD6=$DIR/it_50_l_6.psdd
PSDD7=$DIR/it_50_l_7.psdd
PSDD8=$DIR/it_50_l_8.psdd
PSDD9=$DIR/it_50_l_9.psdd


java -jar ~/code/Scala-LearnPsdd/target/scala-2.11/psdd.jar query -v $VTREE\
                                                                  -m $MODE \
                                                                  -d ./../Density-Estimation-Datasets/datasets/exp_full_system_mnist/exp_full_system_mnist.train.data.small\
                                                                  -q ./../Density-Estimation-Datasets/datasets/exp_full_system_mnist/exp_full_system_mnist.test.data \
                                                                  -p $PSDD0,$PSDD1,$PSDD2,$PSDD3,$PSDD4,$PSDD5,$PSDD6,$PSDD7,$PSDD8,$PSDD9 \
                                                                  -o $OUTFILE\
                                                                  -a 0.09,0.12,0.11,0.10,0.10,0.06,0.10,0.13,0.05,0.15
