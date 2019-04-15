SDDFILE=/afs/inf.ed.ac.uk/user/s14/s1455952/code/data/exp_full_system_mnist/in/exp_full_system_mnist.sdd
VTREEFILE=/afs/inf.ed.ac.uk/user/s14/s1455952/code/data/exp_full_system_mnist/in/exp_full_system_mnist.vtree

python wmisdd.py --mode onehot --onehot_numvars 138 --onehot_xsize 128 --onehot_out_sdd $SDDFILE --onehot_out_vtree $VTREEFILE
