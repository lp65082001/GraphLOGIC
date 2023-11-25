from openmm.app import *
from openmm import *
from openmm.unit import *
import numpy as np
#from sys import stdout, exit, stderr

def min(pos,par):
    print("Energy minimize")
    psf = CharmmPsfFile(par)
    pdb = PDBFile(pos)
    params = CharmmParameterSet('./top_all22_prot.rtf', './par_all22_prot.prm')
    system = psf.createSystem(params, nonbondedMethod=NoCutoff,
                nonbondedCutoff=1*nanometer, constraints=HBonds)
    
    simulation = Simulation(psf.topology, system,VerletIntegrator(0.001))
    simulation.context.setPositions(pdb.positions)
    simulation.minimizeEnergy()


    simulation.reporters.append(PDBReporter('homo_minimize.pdb', 1))
    #simulation.reporters.append(StateDataReporter(stdout, 1, step=True,
    #    potentialEnergy=True, temperature=True))

    simulation.step(1)
    '''
    position = simulation.context.getState(getPositions=True).getPositions()
    
    em_pos = []
    for i in position:
        em_pos.append(i.x)
        em_pos.append(i.y)
        em_pos.append(i.z)
    '''
    print("Done")
    #return np.array(em_pos).reshape((-1,3)).astype('float')

min("./homo_initial.pdb","./homo_initial.psf")

