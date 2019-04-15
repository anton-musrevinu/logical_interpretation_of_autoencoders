
from wmisddsrc.manager.QueryManager import QueryManager
from wmisddsrc.elements.mylogger import Mylogger
import wmisddsrc.methods.ComputeWMI as ComputeWMI
import wmisddsrc.methods.OneHot as OneHot
from wmisddsrc.parsers.argextractor import get_args
import shutil
import os
import argparse
import datetime
import logging

def initLogging(name,console, file):

	l = logging.getLogger(name)
	l.setLevel(logging.INFO)
	formatter = logging.Formatter('%(asctime)s ; %(message)s')

	if file != None:
		filename = file + "{}_{}.log".format(name,datetime.datetime.now())
		fileHandler = logging.FileHandler(filename, mode='w')
		fileHandler.setFormatter(formatter)
		l.addHandler(fileHandler)

	if console:
		streamHandler = logging.StreamHandler()
		streamHandler.setFormatter(formatter)
		l.addHandler(streamHandler)
	#l.disabled = False

	myl = Mylogger(l,name, level = Mylogger.LEVEL_RESULTS)
	return myl


args = get_args() 

arg_mode_wmi = 'WMI'
arg_mode_onehot = 'onehot'
arg_logger_file = 'file'
arg_logger_console = 'console'
arg_logger_both = 'both'
arg_int_method_latte = 'latte'
arg_int_method_scipy = 'scipy'

query_name = args.name
tmpdir = args.tmpdir
logger_level = logging.INFO

if tmpdir == None:            
	tmpdir = os.path.abspath('./.tmpdir/')
	if os.path.exists(tmpdir):
		shutil.rmtree(tmpdir)
	os.mkdir(tmpdir)

if args.logger in [arg_logger_file,arg_logger_both]:
	if args.logger_file == None:
		raise argparse.ArgumentTypeError('looger_file has to be specified if looger is set to mode file')
	console = args.logger == arg_logger_both
elif args.logger == arg_logger_console:
	console = True
logger = initLogging(query_name, console, args.logger_file)
	

if args.mode == arg_mode_wmi:
	if args.integration_method == arg_int_method_latte:
		integration_method = ComputeWMI.DEF_COMUTE_LATTE
	elif args.integration_method == arg_int_method_scipy:
		integration_method = ComputeWMI.DEF_COMUTE_SCIPY
	else:
		raise argparse.ArgumentTypeError('integration method must be one: [latte,scipy]')

	queryManager = QueryManager(query_name, tmpdir, args.interror, logger)
	if args.kbfile == None:
		raise argparse.ArgumentTypeError('kbfile has to be specified')
	with open(args.kbfile,'r') as f:
		kbstrings = []
		for line in f:
			kbstrings.append(str(line))
		if len(kbstrings) > 1:
			raise argparse.ArgumentTypeError('kbfile must only have one line')
		kbstring = kbstrings[0]
	if args.wffile == None:
		wfstring = '1'
	else:
		wfstrings = []
		for line in f:
			wfstrings.append(str(line))
		if len(wfstrings) > 1:
			raise argparse.ArgumentTypeError('wffile must only have one line')
		wfstring = wfstrings[0]

	queryManager.do_wmi_base_parse(kbstring, wfstring, args.integration_method)
	result = queryManager.get_result()

if args.mode == arg_mode_onehot:
	if args.onehot_numvars == -1:
		raise argparse.ArgumentTypeError('number of variables has be provided for mode {}'.format(arg_mode_onehot))
	if args.onehot_fl_size == -1:
		raise argparse.ArgumentTypeError('number of xsize has be provided for mode {}'.format(arg_mode_onehot))

	onehot_kb = OneHot.create_all_contraints(args.onehot_numvars, args.onehot_fl_size, args.onehot_fl_categorical_dim)
	queryManager = QueryManager(query_name, tmpdir, args.interror, logger)
	queryManager.convert_to_sdd(onehot_kb, args.onehot_out_sdd, args.onehot_out_vtree,\
								 printModels = False, total_num_vars = args.onehot_numvars, precomputed_vtree = args.precomputed_vtree, cnf_dir = args.cnf_dir)


if not args.keeptmpdir:
	shutil.rmtree(tmpdir)










