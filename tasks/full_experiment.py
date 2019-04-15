# This file contains:
# - functions for learning vtrees, SDDs, and PSDDs using exisiting resources
# - other helper functions


import os
import shutil
import re


# This function sets up LearnPSDD
def setup_learnpsdd(PWD):

    print("Setting up LearnPSDD...")

    # Add paths and assemble resource
    os.chdir(PWD+"/resources/LearnPSDD")
    os.environ['PATH'] += os.pathsep + 'lib'
    if 'LD_LIBRARY_PATH' not in os.environ:
	os.environ['LD_LIBRARY_PATH'] = os.pathsep + 'lib'
    else:
        os.environ['LD_LIBRARY_PATH'] += os.pathsep + 'lib'
    os.system("sbt assembly")
    os.chdir(PWD)

    # Copy library directory in order to run LearnPSDD from current directory
    if os.path.isdir(PWD+"/lib"):
        shutil.rmtree(PWD+"/lib")
    shutil.copytree(PWD+"/resources/LearnPSDD/lib", PWD+"/lib")

    # Create temporary folder to store LearnPSDD ouput
    if os.path.isdir(PWD+"/temp"):
        shutil.rmtree(PWD+"/temp")
    os.mkdir("temp")

    print("LearnPSDD set up successfully!")

    return


# This function sets the names of inputs and outputs according to our naming convention
def set_names(NAME):

    psdd = "models/"+NAME+"/"+NAME+".psdd"
    sdd = "models/"+NAME+"/"+NAME+".sdd"
    vtree = "models/"+NAME+"/"+NAME+".vtree"
    train = "data/"+NAME+"/"+NAME+"_train.data"
    valid = "data/"+NAME+"/"+NAME+"_valid.data"
    test = "data/"+NAME+"/"+NAME+"_test.data"

    return [psdd,sdd,vtree,train,valid,test]


# This function uses LearnPSDD to construct a PSDD from an SDD and data
def psdd_from_sdd(NAME,DATA,PWD):

    print("Learning PSDD from SDD and data file(s)...")

    setup_learnpsdd(PWD)
    [psdd,sdd,vtree,train,valid,test] = set_names(NAME)

    # Learn a PSDD from an SDD using the combination of data files available and Laplace smoothing
    if DATA[1]:
        if DATA[2]:
            os.system("java -jar resources/LearnPSDD/psdd.jar sdd2psdd -d "+train+" -b "+valid+" -t "+test+" -v "+vtree+" -s "+sdd+" -m l-1 -o temp/"+NAME+".psdd")
        else:
            os.system("java -jar resources/LearnPSDD/psdd.jar sdd2psdd -d "+train+" -b "+valid+" -v "+vtree+" -s "+sdd+" -m l-1 -o temp/"+NAME+".psdd")
    else:
        if DATA[2]:
            os.system("java -jar resources/LearnPSDD/psdd.jar sdd2psdd -d "+train+" -t "+test+" -v "+vtree+" -s "+sdd+" -m l-1 -o temp/"+NAME+".psdd")
        else:
            os.system("java -jar resources/LearnPSDD/psdd.jar sdd2psdd -d "+train+" -v "+vtree+" -s "+sdd+" -m l-1 -o temp/"+NAME+".psdd")

    # Copy PSDD to relevant models folder
    shutil.copyfile(PWD+"/temp/"+NAME+".psdd", os.path.join(PWD,psdd))

    # Remove copied library directory and temporary folder
    shutil.rmtree(PWD+"/lib")
    shutil.rmtree(PWD+"/temp")

    print("PSDD learnt from SDD and data file(s) successfully!")

    return


# This function learns an (unconstrained) PSDD from a vtree using LearnPSDD
def psdd_from_vtree(NAME,DATA,PWD):

    print("Learning PSDD from vtree and data file(s)...")

    setup_learnpsdd(PWD)
    [psdd,sdd,vtree,train,valid,test] = set_names(NAME)

    # Learn a PSDD from a vtree using the combination of data files available and Laplace smoothing
    if DATA[1]:
        if DATA[2]:
            os.system("java -jar resources/LearnPSDD/psdd.jar learnPsdd search -d "+train+" -b "+valid+" -t "+test+" -v "+vtree+" -m l-1 -o temp/"+NAME)
        else:
            os.system("java -jar resources/LearnPSDD/psdd.jar learnPsdd search -d "+train+" -b "+valid+" -v "+vtree+" -m l-1 -o temp/"+NAME)
    else:
        if DATA[2]:
            os.system("java -jar resources/LearnPSDD/psdd.jar learnPsdd search -d "+train+" -t "+test+" -v "+vtree+" -m l-1 -o temp/"+NAME)
        else:
            os.system("java -jar resources/LearnPSDD/psdd.jar learnPsdd search -d "+train+" -v "+vtree+" -m l-1 -o temp/"+NAME)

    # Copy PSDD to relevant models folder
    shutil.copyfile(PWD+"/temp/"+NAME+"/models/final.psdd", os.path.join(PWD,psdd))
    shutil.copyfile(PWD+"/temp/"+NAME+"/models/final.dot", os.path.join(PWD,"models/"+NAME,NAME+"_psdd.dot"))

    # Remove copied library directory and temporary folder
    shutil.rmtree(os.path.join(PWD, "lib"))
    shutil.rmtree(os.path.join(PWD, "temp"))

    print("PSDD learnt from vtree and data file(s) successfully!")

    return


# This function learns a vtree from data using LearnPSDD
def vtree_from_data(NAME,DATA,PWD):

    print("Learning vtree from training data...")

    setup_learnpsdd(PWD)
    [psdd,sdd,vtree,train,valid,test] = set_names(NAME)

    # Learn a PSDD from a vtree using the training data (by minimizing average pairwise mutual infomation between branches via Blossom)
    os.system("java -jar resources/LearnPSDD/psdd.jar learnVtree -d "+train+" -o temp/"+NAME)

    # Copy vtree to relevant models folder
    shutil.copyfile(PWD+"/temp/"+NAME+".vtree", os.path.join(PWD,vtree))

    # Remove copied library directory and temporary folder
    shutil.rmtree(os.path.join(PWD, "lib"))
    shutil.rmtree(os.path.join(PWD, "temp"))

    print("vtree learnt from training data successfully!")

    return


# This function uses the SDD package to construct an SDD from logical constraints
def sdd_from_constraints(NAME,PWD):

    print("Learning SDD from constraints...")

    create_sdd_manager(NAME)

    # Copy library and include directories in order to run constructor from the current directory
    if os.path.isdir(PWD+"/lib"):
        shutil.rmtree(PWD+"/lib")
    shutil.copytree(PWD+"/resources/sdd-2.0/lib", PWD+"/lib")
    if os.path.isdir(PWD+"/include"):
        shutil.rmtree(PWD+"/include")
    shutil.copytree(PWD+"/resources/sdd-2.0/include", PWD+"/include")

    # Run manager file to create SDD and vtree
    os.system("gcc -O2 -std=c99 models/"+NAME+"/"+NAME+"_sdd_manager.c -Iinclude -Llib -lsdd -lm -o models/"+NAME+"/"+NAME)
    os.system("./models/"+NAME+"/"+NAME)

    # Remove copied library and include directories
    shutil.rmtree(PWD+"/lib")
    shutil.rmtree(PWD+"/include")

    print("\nSDD learnt successfully from constraints!")

    return


# This function creates an SDD manager based on the logical constraints
def create_sdd_manager(NAME):

    print("Creating SDD manager...")

    # Open constraints file and save each constraint as an item in a list
    with open("models/"+NAME+"/"+NAME+".constraints") as f:
        constraints = f.readlines()

    # Save the number of variables separately
    num_vars = int(constraints[0])
    del constraints[0]

    f = open(os.path.join("models/"+NAME, NAME+"_sdd_manager.c"),"w")

    # Write beginning of SDD manager file
    f.write("""#include <stdio.h>
    #include <stdlib.h>
    #include "sddapi.h"
    // returns an SDD node representing ( node1 => node2 )
    SddNode* sdd_imply(SddNode* node1, SddNode* node2, SddManager* manager) {
    return sdd_disjoin(sdd_negate(node1,manager),node2,manager);
    }
    // returns an SDD node representing ( node1 <=> node2 )
    SddNode* sdd_equiv(SddNode* node1, SddNode* node2, SddManager* manager) {
    return sdd_conjoin(sdd_imply(node1,node2,manager),
    sdd_imply(node2,node1,manager),manager);
    }
    int main(int argc, char** argv) {
    ////////// SET UP VTREE AND MANAGER //////////
    Vtree* v = sdd_vtree_read("models/"""+NAME+"""/"""+NAME+""".vtree");
    SddManager* m = sdd_manager_new(v);
    // SddLiteral var_count = %d;
    // int auto_gc_and_minimize = 0;
    // SddManager* m = sdd_manager_create(var_count,auto_gc_and_minimize);
    // Vtree* v = sdd_manager_vtree(m);\n\n""" % num_vars)

    # Write list of variables
    variable_names = [chr(64 + i)+" = "+str(i) for i in range(1, num_vars + 1)]
    f.write("SddLiteral "+", ".join(variable_names)+";\n\n")

    # Write middle section, before constraints are entered
    f.write("""SddNode* delta = sdd_manager_true(m);
    SddNode* alpha;
    SddNode* beta;
    ////////// CONSTRUCT THEORY //////////\n\n""")

    # Parse constraints and convert them into the syntax of the SDD package
    for constraint in constraints:
        f.write("alpha = ")

        # Split string into characters and numbers
        chars_grouped = re.split("(\d+)", constraint)
        chars_split = [[char] if unicode(char, 'utf-8').isnumeric() else list(char) for char in chars_grouped]
        char_list = [char for sublist in chars_split for char in sublist]

	# Read in constraints character by character
        for char in char_list:
            if (char == " ") | (char == "\n"):
                continue
            elif (char == "(") | (char == ","):
                f.write(char)
            elif char == ")":
                f.write(",m"+char)
            elif char.isalpha():
                f.write("sdd_manager_literal("+char.upper()+",m)")
            elif unicode(char, 'utf-8').isnumeric():
                f.write("sdd_manager_literal("+chr(int(char)+64)+",m)")
            elif char == "!":
                f.write("sdd_negate")
            elif char == "|":
                f.write("sdd_disjoin")
            elif char == "&":
                f.write("sdd_conjoin")
            elif char == ">":
                f.write("sdd_imply")
            elif char == "=":
                f.write("sdd_equiv")
            else:
                print("Error: character "+char+" could not be parsed")
        f.write(";\n")
        f.write("delta = sdd_conjoin(delta,alpha,m);\n\n")

    # Write final section
    f.write("""////////// SAVE VTREE AND SDD //////////
    // SddNode* gamma = sdd_global_minimize_cardinality(delta, m);
    // sdd_manager_print(m);
	// SddModelCount g = sdd_model_count(gamma, m);
	SddModelCount d = sdd_model_count(delta, m);
	// printf("g: ");
	// printf("%llu",g);
	printf("Model count: ");
	printf("%llu",d);
    sdd_vtree_save(\"models/"""+NAME+"""/"""+NAME+""".vtree",v);
    sdd_vtree_save_as_dot(\"models/"""+NAME+"""/"""+NAME+"""_vtree.dot",v);
    sdd_save(\"models/"""+NAME+"""/"""+NAME+""".sdd",delta);
    sdd_save_as_dot(\"models/"""+NAME+"""/"""+NAME+"""_sdd.dot",delta);
    ////////// CLEAN UP //////////
    sdd_manager_free(m);
    return 0;
    }""")

    f.close()

    print("SDD manager created successfully!")

    return


# This function optionally re-fits a PSDD based on an existing one while preserving any logical constraints
def re_fit(NAME,DATA,PWD):

    print("Re-fitting the PSDD using the existing PSDD, vtree, and data file(s)...")

    setup_learnpsdd(PWD)
    [psdd,sdd,vtree,train,valid,test] = set_names(NAME)

    # Learn a PSDD from a vtree and PSDD using the combination of data files available and Laplace smoothing
    if DATA[1]:
        if DATA[2]:
            os.system("java -jar resources/LearnPSDD/psdd.jar learnPsdd search -d "+train+" -b "+valid+" -t "+test+" -v "+vtree+" -m l-1 -p "+psdd+" -o temp/"+NAME)
        else:
            os.system("java -jar resources/LearnPSDD/psdd.jar learnPsdd search -d "+train+" -b "+valid+" -v "+vtree+" -m l-1 -p "+psdd+" -o temp/"+NAME)
    else:
        if DATA[2]:
            os.system("java -jar resources/LearnPSDD/psdd.jar learnPsdd search -d "+train+" -t "+test+" -v "+vtree+" -m l-1 -p "+psdd+" -o temp/"+NAME)
        else:
            os.system("java -jar resources/LearnPSDD/psdd.jar learnPsdd search -d "+train+" -v "+vtree+" -m l-1 -p "+psdd+" -o temp/"+NAME)

    # Copy PSDD to relevant models folder
    shutil.copyfile(PWD+"/temp/"+NAME+"/models/final.psdd", os.path.join(PWD,psdd))
    shutil.copyfile(PWD+"/temp/"+NAME+"/models/final.dot", os.path.join(PWD,"models/"+NAME,NAME+"_psdd.dot"))

    # Remove copied library directory and temporary folder
    shutil.rmtree(PWD+"/lib")
    shutil.rmtree(PWD+"/temp")

    print("PSDD re-fitted successfully!")

    return

    © 2019 GitHub, Inc.
    Terms
    Privacy
    Security
    Status
    Help

    Contact GitHub
    Pricing
    API
    Training
    Blog
    About

