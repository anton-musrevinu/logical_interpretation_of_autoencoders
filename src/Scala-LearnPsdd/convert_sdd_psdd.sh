java -jar ~/code/Scala-LearnPsdd/target/scala-2.11/psdd.jar sdd2psdd \
			-s ~/code/data/exp_full_system_mnist/in/exp_full_system_mnist.sdd  \
			-v ~/code/data/exp_full_system_mnist/in/exp_full_system_mnist.vtree \
			-d ~/code/data/exp_full_system_mnist/in/exp_full_system_mnist.train.data \
			-b ~/code/data/exp_full_system_mnist/in/exp_full_system_mnist.valid.data \
			-t ~/code/data/exp_full_system_mnist/in/exp_full_system_mnist.test.data \
			-o ~/code/data/exp_full_system_mnist/in/exp_full_system_mnist.psdd