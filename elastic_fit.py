# -*- coding: utf-8 -*-
"""Elastic Constant  WorkChain."""
from aiida.engine import WorkChain, ToContext, calcfunction, if_, while_, BaseRestartWorkChain, process_handler, ProcessHandlerReport, ExitCode
from aiida.orm import Code, Dict, Float, Str, StructureData, ArrayData 
from aiida.plugins import CalculationFactory
from aiida_quantumespresso.workflows.protocols.utils import ProtocolMixin
from aiida.plugins import WorkflowFactory
import numpy as np
from numpy.distutils.misc_util import get_numpy_include_dirs
from .deform import deform
from .common_wf import generate_scf_input_params,delta_project_elast_fit
from scipy.optimize import curve_fit
PwBaseWorkChain = WorkflowFactory('quantumespresso.pw.base')

PwCalculation = CalculationFactory('quantumespresso.pw')
scale_facs = (0.00,0.02,0.018,0.016,0.014,0.012,0.014,0.012,0.010,0.008,0.006,0.004,0.002,-0.002,-0.004,-0.006,-0.008,-0.010,-0.012,-0.014,-0.016,-0.018,0.020)
labels = ['c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'c8', 'c9', 'c10', 'c11','c12','c13','c14','c15', 'c16', 'c17','c18', 'c19', 'c20' ]

@calcfunction
def func(x, a, b, c):
    return a*x*x + b*x + c

@calcfunction
def elastic_fit(**kwargs):
    """Store EOS data in Dict node."""
    elast = [(result.dict.volume, result.dict.energy, result.dict.energy_units)
            for label, result in kwargs.items()]
    #return Dict(dict={'elast':elast})
    vol = np.array([i[0] for i in elast])
    en = np.array([i[0] for i in elast])
    e1 = (en[:1]- en[0])/vol[0]
    popt, pconv = curve_fit(func, scale_facs[1:], e1)
    c33=popt[0] #c33 is in eV/ang^3
    c33_gpa = c33 * 160.217 # eV/ang^3
    #array = np.array(elast)
    #array.set_array('elast', np.array())
    elast = Dict(dict={'c33':Float(c33), 'c33_GPa':Float(c33_gpa)})
    return elast

class Elasticwk(ProtocolMixin,WorkChain):
    """WorkChain to compute Elastic constant using Quantum ESPRESSO."""

    @classmethod
    def define(cls, spec):
        """Specify inputs and outputs."""
        super(Elasticwk, cls).define(spec)
        spec.input('code', valid_type=Code)
        #spec.expose_inputs(PwBaseWorkChain, namespace='base', exclude=('kpoints',))
        spec.input('pseudo_family', valid_type=Str)
        spec.input('structure', valid_type=StructureData)
        spec.input('base', valid_type=Dict)
        spec.output('elast', valid_type=Dict)
        spec.outline(
            cls.run_energyvolume,
            cls.results,
        )

    def run_energyvolume(self):
        """Run calculations for equation of state."""
        # Create basic structure and attach it as an output
        structure = self.inputs.structure

        calculations = {}

        for label, factor in zip(labels, scale_facs):

            deform_structure = deform(structure, Float(factor))
            inputs = generate_scf_input_params(deform_structure, self.inputs.code,self.inputs.pseudo_family, self.inputs.base)

            self.report(
                'Running an SCF calculation for {} with scale factor {}'.
                format(structure.get_formula(), factor))
            future = self.submit(PwBaseWorkChain, **inputs)
            calculations[label] = future

        # Ask the workflow to continue when the results are ready and store them in the context
        return ToContext(**calculations)

    def results(self):
        """Process results."""
        inputs = {
            label: self.ctx[label].get_outgoing().get_node_by_label(
                'output_parameters')
            for label in labels
        }
        elast = elastic_fit(**inputs)
        #array = elast.get_array('elast')
        E0, V0 = delta_project_elastic_fit(array[:,0] , array[:,1])  # volumes and energies
        self.out('elast', elast)
        self.out('E0', E0)
        self.out('V0', V0)
