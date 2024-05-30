import sys
sys.path.append('/home/student7/LucaSchaufelberger/MasterThesis/ProjectScripts/ExternalRepositories/TheoDORE_3.0')
import theodore
from theodore.actions import analyze_tden
from colt.lazyimport import LazyImportCreator, LazyImporter
import packaging
import numpy as np
from theodore import lib_tden
from theodore import input_options
from theodore import lib_tden
from theodore import lib_exciton
from theodore import theo_header
#import cclib



_user_input = """
# Main input file
ifile = dens_ana.in :: existing_file, alias=f
# Print all keywords and their current values
keywords = false :: bool, alias=k
"""

input_manual = """
rtype='cclib'
rfile='mol0_opt.log'
print_OmFrag=False
Om_formula=0

"""

def theodore_workflow_S1_excdist(in_file,rfile):

    #theodore to localize T1
    theodore.actions.analyze_tden.AnalyzeTden.run = run_mod_S1exciton_size
    S1_excdist = theodore.actions.analyze_tden.AnalyzeTden.run(in_file,rfile)
    return S1_excdist

def run_mod_S1exciton_size(in_file,rfile):

    ioptions = input_options.tden_ana_options(in_file)

    ioptions['rfile']=rfile
    tdena = lib_tden.tden_ana(ioptions)

    tdena.read_dens()

    if 'at_lists' in ioptions or ioptions['eh_pop'] >= 1:
        tdena.compute_all_OmAt()

    #--------------------------------------------------------------------------#
    # Print-out
    #--------------------------------------------------------------------------#

    print("start modified --------------")



    for index,state in enumerate(tdena.state_list):
        print(tdena.state_list[index]['irrep'])
        if 'Sing' in tdena.state_list[index]['irrep']:
            S1_index = index
            print('found S1', index)
            break
    Om, OmAt = tdena.ret_Om_OmAt(tdena.state_list[S1_index])

    exca = lib_exciton.exciton_analysis()

    exca.get_distance_matrix(tdena.struc)
    RMSeh_S1 = exca.ret_RMSeh(Om, OmAt)


    return(RMSeh_S1)
