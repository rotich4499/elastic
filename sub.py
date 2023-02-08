from datetime import datetime, timedelta
from aiida.engine import run, submit
from aiida.orm import Code, Dict, Float, Str, StructureData
from aiida.plugins import DbImporterFactory
CodDbImporter = DbImporterFactory('cod')
from aiida.orm.nodes.data.upf import get_pseudos_from_structure 
from aiida_quantumespresso.workflows.pw.elastic_fit import Elasticwk

PwBandsWorkChain = WorkflowFactory('quantumespresso.pw.bands')
codename = 'qe-6.7@chpc'
code = Code.get_from_string(codename)

structure=load_node(1256)

params_scf = {
        'CONTROL': {
            'calculation': 'scf',
            'verbosity': 'high',
            'wf_collect': True
        },
        'SYSTEM': {
            'ecutwfc': 60.,
            'ecutrho': 600.,
            'occupations': 'smearing',
            'smearing': 'gaussian',
            'degauss': 0.01,
        },
        'ELECTRONS': {
            'mixing_mode': 'plain',
            'mixing_beta': 0.7,
            'conv_thr': 1.e-6,
            'diago_thr_init': 5.0e-6,
            'diago_full_acc': True
        },
    }
NUMMACH=2
options_res = {
                "custom_scheduler_commands": "#PBS -P MATS862\n#PBS -q normal\n#PBS -l mem={}gb".format(NUMMACH*118),
                "resources": { 'num_machines': NUMMACH , 'num_cores_per_machine':24},
                 ''
                "max_wallclock_seconds": 60*60*12 -1 ,  
                }
inputs_scf=  {'pw' : { 'parameters': params_scf , "metadata":{ "options" : options_res } },  
              'pseudo_family': 'SSSP/1.1/PBE/efficiency', 
              'kpoints_distance':0.5 }
overrides = Dict(dict=inputs_scf) 
            

pseudo_family = Str('SSSP/1.1/PBE/efficiency')

#builder= PwBandsWorkChain.get_builder_from_protocol(code=code,  structure=structure, protocol="fast",  overrides=overrides )
#from aiida.engine import  run_get_node
#res = submit(builder)
res = submit( Elasticwk, code=code, pseudo_family = Str('SSSP/1.1/PBE/efficiency'), structure=load_node(1256), base=overrides)
print(res)
