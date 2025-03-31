from qiskit import transpile
from qiskit import QuantumCircuit
from qiskit import QuantumRegister
from qiskit import ClassicalRegister
from qiskit.quantum_info.operators import Operator
from qiskit_aer import AerSimulator as Aer
from qiskit.visualization import plot_histogram as plt_hist
import matplotlib as plt
import numpy as np


# Define the Hadamard test circuit
def hadamard_test(theta, shots=1024):
    qc = QuantumCircuit(2, 1)
    qc.h(0)
    qc.x(0)
    qc.cp(2 * np.pi * theta, 0, 1)
    qc.h(0)
    
    qc.measure(0, 0)
    
    simulator = transpile(qc, backend=Aer)
    result = Aer.run(simulator, shots=shots).result()
    counts = result.get_counts()
    prob_0 = counts.get('0', 0) / shots
    prob_1 = counts.get('1', 0) / shots
    expectation = prob_0 - prob_1
    # Estimate theta from expectation
    estimated_theta = np.arccos(expectation) / (2 * np.pi)
    return estimated_theta


# True values of theta
theta_values = [0.5625, 0.1234]
# Number of shots for the experiment
shot_values = [10, 50, 100, 500, 1000, 5000, 10000, 50000]

# Store errors for each theta
errors = {theta: [] for theta in theta_values}

# Run Hadamard test for different shots and compute errors
for theta in theta_values:
    for shots in shot_values:
        estimated_theta = hadamard_test(theta, shots)
        error = abs(estimated_theta - theta)
        errors[theta].append(error)

# Plot error vs. number of shots
plt.figure(figsize=(8, 6))
for theta, err in errors.items():
    plt.loglog(shot_values, err, marker='o', label=f'True θ = {theta}')

# Plot theoretical bound (1/sqrt(shots))
sqrt_bound = [1 / np.sqrt(s) for s in shot_values]
plt.loglog(shot_values, sqrt_bound,
           linestyle='dashed', color='black', label='O(1/√shots)')

plt.xlabel("Number of Shots (log scale)")
plt.ylabel("Error (log scale)")
plt.title("Error vs. Number of Shots in Hadamard Test")
plt.legend()
plt.grid(True, which="both", linestyle="--")
plt.show()
