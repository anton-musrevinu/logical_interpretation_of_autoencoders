import os
import platform
from subprocess import STDOUT, check_output, TimeoutExpired

LOWLEVEL_CMD = '../../../src/lowlevel/main.py'
BIN_DIR = os.path.join(os.path.abspath('./../../'),'./src/wmisdd/bin')
WMISDD_CMD = os.path.join(os.path.abspath('./../../'),'./src/wmisdd/wmisdd.py')

os.system('pwd')
FILES_DIR = os.path.join(os.path.abspath('.'), 'files')

def compile_cnf(cnf_file_name,test_name):
	cnfFile = os.path.join(FILES_DIR, cnf_file_name + '.cnf')
	test_file_name = cnf_file_name + '_' + test_name
	sddFile = os.path.join(FILES_DIR, test_file_name + '_cnf.sdd')
	vtreeFile = os.path.join(FILES_DIR, test_file_name + '_cnf.vtree')

	precomputed_vtree = False
	vtreeFlag = '-v' if precomputed_vtree else '-t right'
	vtreeSearchFlag = '-r 0' if precomputed_vtree else '-r 0'
	vtreeDotFile = vtreeFile + '.dot'
	sddDotFiel = sddFile + '.dot'
	if 'Linux' in platform.system():
		command = "{}/sdd-linux -c {} {} {} -R {} {} -m -V {} -S {}".format(BIN_DIR,cnfFile,\
			vtreeFlag, vtreeFile, sddFile, vtreeSearchFlag, vtreeDotFile, sddDotFiel)
	else:
		command = "{}/sdd-darwin -c {} {} -W {} -R {} -V {} -S {} {}".format(BIN_DIR, cnfFile,\
			vtreeFlag, vtreeFile, sddFile, vtreeDotFile, sddDotFiel,vtreeSearchFlag)

	print('\nexecuting command: {}'.format(command))
	output = check_output(command, stderr=STDOUT, timeout=60 * 60,shell = True)
	print('finished with output: {}\n'.format(str(output).replace('\\n','\n')))

	cmd_convert_to_pdf = 'dot -Tpdf {}.dot -o {}.pdf'.format(vtreeFile,vtreeFile)
	os.system(cmd_convert_to_pdf)

	cmd_convert_to_pdf = 'dot -Tpdf {}.dot -o {}.pdf'.format(sddFile,sddFile)
	os.system(cmd_convert_to_pdf)

	# cmd = 'python {} --mode WME --wme_in_sdd {} --wme_in_vtree {} --onehot_numvars {}'.format(WMISDD_CMD, sddFile, vtreeFile, int(cnf_file_name[-1]) * int(cnf_file_name[0]))
	# print('excuting: {}'.format(cmd))
	# os.system(cmd)

def compile_dnf(cnf_file_name):
	cnfFile = os.path.join(FILES_DIR, test_file_name + '.dnf')
	sddFile = os.path.join(FILES_DIR, test_file_name + '_dnf.sdd')
	vtreeFile = os.path.join(FILES_DIR, test_file_name + '_dnf.vtree')

	precomputed_vtree = False
	vtreeFlag = '-v' if precomputed_vtree else '-t balanced'
	vtreeSearchFlag = '-r 0' if precomputed_vtree else '-r 0'
	vtreeDotFile = vtreeFile + '.dot'
	sddDotFiel = sddFile + '.dot'
	if 'Linux' in platform.system():
		command = "{}/sdd-linux -d {} {} {} -R {} {} -m -V {} -S {}".format(BIN_DIR,cnfFile,\
			vtreeFlag, vtreeFile, sddFile, vtreeSearchFlag, vtreeDotFile, sddDotFiel)
	else:
		command = "{}/sdd-darwin -d {} {} -W {} -R {} -V {} -S {} {} ".format(BIN_DIR, cnfFile,\
			vtreeFlag, vtreeFile, sddFile, vtreeDotFile, sddDotFiel,vtreeSearchFlag)

	print('\nexecuting command: {}'.format(command))
	output = check_output(command, stderr=STDOUT, timeout=60 * 60,shell = True)
	print('finished with output: {}\n'.format(str(output).replace('\\n','\n')))

	cmd_convert_to_pdf = 'dot -Tpdf {}.dot -o {}.pdf'.format(vtreeFile,vtreeFile)
	os.system(cmd_convert_to_pdf)

	cmd_convert_to_pdf = 'dot -Tpdf {}.dot -o {}.pdf'.format(sddFile,sddFile)
	os.system(cmd_convert_to_pdf)

	cmd = 'python {} --mode WME --wme_in_sdd {} --wme_in_vtree {} --onehot_numvars {}'.format(WMISDD_CMD, sddFile, vtreeFile, int(cnf_file_name[-1]) * int(cnf_file_name[0]) )
	print('excuting: {}'.format(cmd))
	os.system(cmd)



# def write_vtree(num_vars, cat_dim, y_vec):


if __name__ == '__main__':
	test_file_name = '1_var_c4'
	test_name = 'anti_sym'

	compile_cnf(test_file_name, test_name)
	# compile_dnf(test_file_name)



