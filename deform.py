import numpy as np
def strain (e1,e2,e3,e4,e5,e6):
    e=np.array([[e1,e6/2,e5/2],
               [e6/2,e2,e4/2],
               [e5/2,e4/2,e3]])
    I=np.array([[1 , 0 , 0],
                [0 , 1 , 0],
                [0 , 0 , 1]])
    a = (e+I)
    return a

def deform(structure, scale):
    """Calculation function to rescale a structure

    :param structure: An AiiDA structure to rescale
    :param scale: The scale factor (for the lattice constant)
    :return: The rescaled structure
    """
    from aiida import orm

    ase = structure.get_ase()
    strm = strain (0,0,0.002,0,0,0)
    cellv = ase.get_cell()
    newcell = cellv.dot(strm)
    ase.set_cell(newcell, scale_atoms=True)
    return orm.StructureData(ase=ase)
