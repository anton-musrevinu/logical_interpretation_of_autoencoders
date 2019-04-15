NAME="exp_full_system"

echo scp -r s1455952@cdtcluster.inf.ed.ac.uk:./msc/tasks/$NAME/data_out/* ~/code/Density-Estimation-Datasets/datasets/$NAME/


java -jar ~/code/Scala-LearnPsdd/target/scala-2.11/psdd.jar learnVtree -d ~/code/Density-Estimation-Datasets/datasets/exp_full_system/exp_full_system.train.data -o ~/code/LearnPSDD/vtrees/exp_full_system