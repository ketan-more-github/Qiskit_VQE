# time PYTHONWARNINGS="ignore::DeprecationWarning" mpirun -np 20 python3 mpi.py

import qiskit_nature
from qiskit.algorithms.minimum_eigensolvers import NumPyMinimumEigensolver, VQE
from qiskit_nature.second_q.transformers import FreezeCoreTransformer
from qiskit_nature.second_q.formats.molecule_info import MoleculeInfo
from qiskit_nature.second_q.mappers import ParityMapper
from qiskit_nature.second_q.circuit.library import UCCSD, HartreeFock
from qiskit.algorithms.optimizers import SPSA, SLSQP, COBYLA
from qiskit_nature.second_q.drivers import PySCFDriver
import matplotlib.pyplot as plt
from qiskit.circuit.library import EfficientSU2
from mpi4py import MPI
import numpy as np
import time
import itertools
from qiskit_aer.primitives import Estimator
qiskit_nature.settings.use_pauli_sum_op = False
from qiskit_aer import Aer


def get_qubit_op(dist):
    
    molecule = MoleculeInfo(
        # Coordinates in Angstrom
        symbols=["Li", "H"],
        coords=([0.0, 0.0, 0.0], [dist, 0.0, 0.0]),
        multiplicity=1,  # = 2*spin + 1
        charge=0,
    )

    driver = PySCFDriver.from_molecule(molecule)

    # Get properties
    properties = driver.run()


    problem = FreezeCoreTransformer(
        freeze_core=True, remove_orbitals=[-3, -2]
    ).transform(properties)

    num_particles = problem.num_particles
    num_spatial_orbitals = problem.num_spatial_orbitals

    mapper = ParityMapper(num_particles=num_particles)
    qubit_op = mapper.map(problem.second_q_ops()[0])
    return qubit_op, num_particles, num_spatial_orbitals, problem, mapper


def exact_solver(qubit_op, problem):
    sol = NumPyMinimumEigensolver().compute_minimum_eigenvalue(qubit_op)
    result = problem.interpret(sol)
    return result



start_time = time.time()
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()



def compute_energy(dist):
    (qubit_op, num_particles, num_spatial_orbitals, problem, mapper) = get_qubit_op(dist)
    result = exact_solver(qubit_op, problem)
    exact_energy = result.total_energies[0].real
    init_state = HartreeFock(num_spatial_orbitals, num_particles, mapper)
    var_form = UCCSD(num_spatial_orbitals, num_particles, mapper, initial_state=init_state)
    vqe = VQE(noiseless_estimator , var_form, optimizer, initial_point=[0] * var_form.num_parameters)
    backend = Aer.get_backend('statevector_simulator')
    result = vqe.run(backend)
    vqe_calc = vqe.compute_minimum_eigenvalue(qubit_op)
    vqe_result = problem.interpret(vqe_calc).total_energies[0].real
    cpu_num = comm.Get_rank()
    return dist, exact_energy, vqe_result , cpu_num


distances = np.arange(0.5, 4.0, 0.2)
exact_energies = []
cpus = []
vqe_energies = []
optimizer = SLSQP(maxiter=10)
noiseless_estimator = Estimator(approximation=True)



distances_split = np.array_split(distances, size)


distances_local = comm.scatter(distances_split, root=0)


local_results = [compute_energy(dist) for dist in distances_local]


all_results = comm.gather(local_results, root=0)


if rank == 0:
    
    # all_results = [item for sublist in all_results for item in sublist]
    all_results = list(itertools.chain(*all_results))
   
    distances, exact_energies, vqe_energies , cpus = zip(*all_results)
    
 
    for dist, vqe_result, exact_energy , cpu_num in zip(distances, vqe_energies, exact_energies , cpus):
        print(f"Interatomic Distance: {np.round(dist, 2)}", 
              f"VQE Result: {vqe_result:.5f}",
              f"Exact Energy: {exact_energy:.5f}",
              f"CPU CORE: {cpu_num}"
              )
    
    print("All energies have been calculated")

    end_time = time.time()
    execution_time1 = end_time - start_time
    print(f"Execution time: {execution_time1} seconds")
