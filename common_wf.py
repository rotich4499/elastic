# -*- coding: utf-8 -*-
"""Helper functions."""
import numpy as np 
from aiida.plugins import CalculationFactory, DataFactory
from aiida_quantumespresso.utils.pseudopotential import validate_and_prepare_pseudos_inputs
from aiida.plugins import WorkflowFactory
from scipy.optimize import curve_fit

Dict = DataFactory('dict')
KpointsData = DataFactory('array.kpoints')
PwBaseWorkChain = WorkflowFactory('quantumespresso.pw.base')

def generate_scf_input_params(structure, code, pseudo_family, base):
    """Construct a builder for the `PwBaseWorkChain` class and populate its inputs.

    :return: `ProcessBuilder` instance for `PwBaseWorkChain` with preset inputs
    """
    parameters = {
        'CONTROL': {
            'calculation': 'scf',
            'tstress': True,  # Important that this stays to get stress
            'tprnfor': True,
        },
        'SYSTEM': {
            'ecutwfc': 60.,
            'ecutrho': 600.,
        },
        'ELECTRONS': {
            'conv_thr': 1.e-6,
        }
    }

    #kpoints = KpointsData()
    #kpoints.set_kpoints_mesh([21, 21, 2])

    builder = PwBaseWorkChain.get_builder_from_protocol(code, structure, 'fast', overrides=base.get_dict())
    #builder.code = code
    #builder.structure = structure
    #builder.kpoints = kpoints
    #builder.parameters = Dict(dict=parameters)
    #builder.pseudos = validate_and_prepare_pseudos_inputs(
    #    structure, pseudo_family=pseudo_family)
    builder.pw.metadata = base.get_dict()['pw']['metadata']
    #builder.metadata.description = "Equation of State"
    #builder.metadata.options = base.get_dict()['pw']['metadata']["options"]
    #builder.metadata.options.max_wallclock_seconds = 30 * 60

    return builder


def birch_murnaghan(V, E0, V0, B0, B01):
    """Compute energy by Birch Murnaghan formula."""
    r = (V0 / V)**(2. / 3.)
    return E0 + 9. / 16. * B0 * V0 * (r - 1.)**2 * (2. + (B01 - 4.) * (r - 1.))


def fit_birch_murnaghan_params(volumes_, energies_):
    """Fit Birch Murnaghan parameters."""
    from scipy.optimize import curve_fit

    volumes = np.array(volumes_)
    energies = np.array(energies_)
    params, covariance = curve_fit(
        birch_murnaghan,
        xdata=volumes,
        ydata=energies,
        p0=(
            energies.min(),  # E0
            volumes.mean(),  # V0
            0.1,  # B0
            3.,  # B01
        ),
        sigma=None)
    return params, covariance


def plot_eos(eos_pk):
    """
    Plots equation of state taking as input the pk of the ProcessCalculation
    printed at the beginning of the execution of run_eos_wf
    """
    import pylab as pl
    from aiida.orm import load_node
    eos_calc = load_node(eos_pk)

    data = []
    for V, E, units in eos_calc.outputs.result.dict.eos:
        data.append((V, E))

    data = np.array(data)
    params, _covariance = fit_birch_murnaghan_params(data[:, 0], data[:, 1])

    vmin = data[:, 0].min()
    vmax = data[:, 0].max()
    vrange = numpy.linspace(vmin, vmax, 300)

    pl.plot(data[:, 0], data[:, 1], 'o')
    pl.plot(vrange, birch_murnaghan(vrange, *params))

    pl.xlabel("Volume (ang^3)")
    # I take the last value in the list of units assuming units do not change
    pl.ylabel("Energy ({})".format(units))  # pylint: disable=undefined-loop-variable
    pl.show()


def delta_project_elast_fit(volumes, energies):  #pylint: disable=invalid-name
    """
    The fitting procedure implemented in this function
    was copied from the Delta Project Code.
    https://github.com/molmod/DeltaCodesDFT/blob/master/eosfit.py
    It is introduced to fully uniform the delta test procedure
    with the one performed with other codes, moreover it
    has the upside to not use scypi.
    """

    import numpy as np 

    #Does the fit always succeed?
    fitdata = np.polyfit(volumes**(-2. / 3.), energies, 3, full=True)
    ssr = fitdata[1]
    sst = np.sum((energies - np.average(energies))**2.)
    residuals0 = ssr / sst
    deriv0 = np.poly1d(fitdata[0])  #pylint: disable=invalid-name
    deriv1 = np.polyder(deriv0, 1)  #pylint: disable=invalid-name
    deriv2 = np.polyder(deriv1, 1)  #pylint: disable=invalid-name
    deriv3 = np.polyder(deriv2, 1)  #pylint: disable=invalid-name

    volume0 = 0
    x = 0
    for x in np.roots(deriv1):
        if x > 0 and deriv2(x) > 0:
            E0 = deriv0(x)  #pylint: disable=invalid-name
            volume0 = x**(-3. / 2.)
            break

    #Here something checking if the fit is good!
    #The choice of residuals0 > 0.01 it is not supported by a real scientific reason, just from experience.
    #Values ~ 0.1 are when fit random numbers, ~ 10^-5 appears for good fits. The check on the presence of
    #a minmum covers the situations when an almost linear dependence is fitted (very far from minimum)
    if volume0 == 0 or residuals0 > 0.01:
        return residuals0, volume0
    derivV2 = 4. / 9. * x**5. * deriv2(x)  #pylint: disable=invalid-name
    derivV3 = (-20. / 9. * x**(13. / 2.) * deriv2(x) - 8. / 27. * x**(15. / 2.) * deriv3(x))  #pylint: disable=invalid-name
    bulk_modulus0 = derivV2 / x**(3. / 2.)
    
    return E0, volume0, bulk_modulus0
