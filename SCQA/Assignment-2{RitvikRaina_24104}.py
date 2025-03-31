#!/usr/bin/env python
# coding: utf-8

# # Initialization

# In[50]:


from qiskit import transpile, circuit, QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.quantum_info import Operator, Statevector, DensityMatrix, Pauli
from qiskit_aer import AerSimulator
from qiskit_aer.primitives import Estimator as ES
from qiskit.visualization import plot_histogram as plt_hist
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.pyplot import figure
import qiskit.circuit.library as qlib
from qiskit_ibm_runtime import QiskitRuntimeService, EstimatorV2
from scipy.optimize import curve_fit
sv = Statevector.from_label
CCCZ = qlib.ZGate()


# # Problem 3(b): Quantum Fourier Transform

# In[2]:


def QFT_(n):
    qc = QuantumCircuit(n)
    for target in range(n):
        qc.h(target)
        for control in range(target+1, n):
            qc.cp(np.pi * (2**(target - control)), control, target)
        qc.barrier() #for better representation
    return qc

def IQFT_(n):
    qc = QuantumCircuit(n)
    for target in reversed(range(n)):
        for control in reversed(range(target+1, n)):
            qc.cp(np.pi * (2**(target - control)), control, target)
        qc.h(target)
        qc.barrier()
    return qc

n = 6 # set the number of qubits here


# In[3]:


QFT_(n).draw('mpl', fold = 50)


# In[4]:


IQFT_(n).draw('mpl', fold = 50)
#print("QFT circuit is:\n", QFT_(n), "\nInverse QFT is:\n", IQFT_(n))


# # Problem 4(a): Hadamard Test

# ##### Define the Hadamard test circuit

# In[5]:


def HadamardCircuit(theta):
    qc = QuantumCircuit(2, 1)
    
    # Apply Hadamard to the first qubit
    qc.h(0)
    # Applying X gate to convert |0> to |1> (eigenstate of the operator)
    qc.x(1)
    # Controlled-U gate (where U is the phase gate)
    qc.cp(2 * np.pi * theta, 0, 1)
    # Apply Hadamard again
    qc.h(0)
    # Measure the first qubit
    qc.measure(0, 0)

    return qc

def hadamard_test(theta, shots):
    # Simulate the circuit
    Aer = AerSimulator()
    simulator = transpile(HadamardCircuit(theta), backend=Aer)
    result = Aer.run(simulator, shots=shots).result()
    counts = result.get_counts()
    
    # Compute expectation value of the measurement
    prob_0 = counts.get('0', 0) / shots
    expectation = 2*prob_0 - 1
    
    # Estimate theta from expectation
    estimated_theta = np.arccos(expectation) / (2 * np.pi)
    # if theta< 0.5:
    #     estimated_theta = estimated_theta
    # elif theta>= 0.5:
    #     estimated_theta = 1 - estimated_theta
    return estimated_theta

HadamardCircuit(0.5625).draw("mpl")


# ##### Simulation and plotting

# In[ ]:


# True values of theta
theta_values = [0.5625, 0.1234]

# Number of shots for the experiment
shot_values = [2**12 ,2**14 ,2**16 ,2**18 ,2**20 ,2**22 ,2**24 ,2**26, 2**28]

# Store errors for each theta
errors = {theta: [] for theta in theta_values}

# Run Hadamard test for different shots and compute errors
for theta in theta_values:
    for shots in shot_values:
        estimated_theta = hadamard_test(theta, shots)
        error = abs(estimated_theta - theta)
        errors[theta].append(error)

# Plot error vs. number of shots
#plt.figure(figsize=(8, 6))
for theta, err in errors.items():
    plt.loglog(shot_values, err, marker='o', label=f'True θ = {theta}')

# Plot theoretical bound (1/sqrt(shots))
sqrt_bound = [1 / np.sqrt(s) for s in shot_values]
onebys = [1 / (s) for s in shot_values]
plt.loglog(shot_values, sqrt_bound, linestyle='dashed', color='black', label='O(1/√shots)')
plt.loglog(shot_values, onebys, linestyle='dashed', color='green', label='O(1/shots)')
plt.xlabel("Number of Shots (log scale)")
plt.ylabel("Error (log scale)")
plt.title("Error vs. Number of Shots in Hadamard Test")
plt.legend()
plt.grid(True, which="both", linestyle="--")
plt.show()


# ##### Explanation

# We observe that the error in the value of $\theta = 0.1234$ as number of shots increases, follows a linear trend and remains confined between $\frac{1}{shots}$ and $\frac{1}{\sqrt{shots}}$ and its closer to the expected scaling of $\frac{1}{\sqrt{shots}}$. 
# For the case of $\theta = 0.5625 > 0.5$ we see a very large error which is due to the symmetry of $cos(\theta)$ about $\pi$. This causes the probablities of getting the state $| 0 \rangle$ with equal probability for $\theta  = \pi \pm x$. We explore more on how to solve this in the <a id = # Explaination Section>next section</a>. <br>We have also given the plot for the value of $\theta = 1 - 0.5625 < 0.5$ ; $0.5625 > 0.5$  below for comparision.

# In[28]:


theta = 0.5625
errors = []
shot_values = [2**12 ,2**14 ,2**16 ,2**18 ,2**20 ,2**22 ,2**24]# ,2**26, 2**28]
for shots in shot_values:
    estimated_theta = hadamard_test(theta, shots)
    error = abs(estimated_theta - theta)
    errors.append(error)
plt.figure(figsize=(6,4))
plt.loglog(shot_values, errors, marker='o', label=f'True θ = {theta}')
plt.xlabel("Number of Shots (log scale)")
plt.ylabel("Error (log scale)")
plt.title("Error vs. Number of Shots in Hadamard Test")
plt.legend()
plt.grid(True, which="both", linestyle="--")
plt.show()


# In[ ]:


theta =1 - 0.5625
errors = []
shot_values = [2**12 ,2**14 ,2**16 ,2**18 ,2**20 ,2**22 ,2**24 ,2**26]#, 2**28]
for shots in shot_values:
    estimated_theta = hadamard_test(theta, shots)
    error = abs(estimated_theta - theta)
    errors.append(error)
plt.figure(figsize=(6,4))
plt.loglog(shot_values, errors, marker='o', label=f'True θ = {theta}')
plt.xlabel("Number of Shots (log scale)")
plt.ylabel("Error (log scale)")
plt.title("Error vs. Number of Shots in Hadamard Test")
plt.legend()
plt.grid(True, which="both", linestyle="--")
plt.show()


# # Problem 4(b): Kitaev Test

# ##### Defining the circuit and phase estimation algorithm

# In[7]:


def binary_to_decimal_float(binary_string): #Converts a string of binary numbers into a decimal number.
    integer_part, fractional_part = binary_string.split(".")
    decimal_value = 0
    for i, digit in enumerate(reversed(integer_part)):
        decimal_value += int(digit) * (2 ** i)
    for i, digit in enumerate(fractional_part):
        decimal_value += int(digit) * (2 ** -(i + 1))
    
    return decimal_value


def kitaev_test(theta, shots, im = False, d = 0): #Kitaev circuit with both Real and Imaginary Hadamard Tests
    qc = QuantumCircuit(2, 1)  
    qc.x(1)
    qc.h(0)
    if im:
        qc.s(0).inverse()
    qc.cp(2 * np.pi * theta * (2**d), 0, 1)
    qc.h(0)
    qc.measure(0, 0)
    
    # Run simulation
    simulator = AerSimulator()
    QC = transpile(qc, backend =  simulator)
    result = simulator.run(QC, shots= shots).result()
    counts = result.get_counts()
    p0 = counts.get('0' , 0)/shots

    return p0


def kitaev_estimate(theta, d, shots = shots): #Algorithm for interpreting the results
    cos_theta = 2 * kitaev_test(theta, im=False, d = d, shots = shots)   - 1 # Estimate cos(θ)
    sin_theta = 2 * kitaev_test(theta, im = True, d = d, shots = shots)  - 1 # Estimate sin(θ)

    # Use arctan2 to estimate θ
    theta_est = np.arctan2(sin_theta, cos_theta)
    if theta_est < 0:  # Ensure θ is in range [0, 2π]
        theta_est += 2 * np.pi
    return theta_est/(2*np.pi)


def findphase(theta, d_bit_precision, shots):
    c = "0."
    for coeff in range(d_bit_precision):
        c += (str(int(kitaev_estimate(theta, coeff, shots)>=0.5)))
    phase = binary_to_decimal_float(c)
    return phase
    


# ##### Explanation of the circuit and phase estimation <a name="Explaination Section"></a>

#  The Kitaev test has the same circuit as Hadamard test with one key difference, that the Unitary is raised to $2^j$ where, $j \in [0,d-1]$. 
# Each iteration of $j$ can provide us with the bit corresponding to the $j^{th}$ entry in the binary representaion of $\theta$.
# When the unitary is raised to $2^j$, if the estimated phase $(\theta)$ is larger than $ 0.5 * 2 \pi$ (refered henceforth as 0.5) then the bit corresponding to $j^{th}$ position is 1, else 0. This is since to get value $> 0.5$ in binary, the most significant bit should be 1.
# However to differentiate between $\phi > 0.5$ or $\phi < 0.5$ is tricky, since $cos(2\pi \phi)$ is symmetric about 0.5. 
# Thus we employ 'arctan2' function, which gives exact value of $\phi \in [0, 2\pi)$ based on the values of $cos(\phi)$ and $sin(\phi)$, which we can calculate by employing the **Real** and **Imaginary Hadamard** test.

# ##### Simulation of errors and plots

# In[16]:


# True values of theta
theta_values = [0.5625, 0.1234]
# Number of shots for the experiment
shot_values = [2**10, 2**12 ,2**14 ,2**16 ,2**18 ,2**20 ,2**22]# ,2**24 ,2**26, 2**28]
# Store errors for each theta
errors = {theta: [] for theta in theta_values}

# Run Kitaev test for different shots and compute errors
for theta in theta_values:
    for shots in shot_values:
        estimated_theta = findphase(theta, 10, shots)
        error = abs(estimated_theta - theta)
        errors[theta].append(error)

# Plot error vs. number of shots
for theta, err in errors.items():
    plt.loglog(shot_values, err, marker='o', label=f'True θ = {theta}')

# Plot theoretical bound (1/sqrt(shots))
sqrt_bound = [1 / np.sqrt(s) for s in shot_values]
onebys = [1 / (s) for s in shot_values]
plt.loglog(shot_values, sqrt_bound, linestyle='dashed', color='black', label='O(1/√shots)')
plt.loglog(shot_values, onebys, linestyle='dashed', color='green', label='O(1/shots)')
plt.xlabel("Number of Shots (log scale)")
plt.ylabel("Error (log scale)")
plt.title("Error vs. Number of Shots in Kitaev Test (8-Bit precision)")
plt.legend()
plt.grid(True, which="both", linestyle="--")
plt.show()


# # Problem 4(c): Quantum Phase Estimation

# ##### Defining circuits perliminaries

# In[9]:


def qft_dagger(qc, n):
    """n-qubit QFTdagger the first n qubits"""
    for qubit in range(n//2):
        qc.swap(qubit, n-qubit-1)
    for j in range(n):
        for m in range(j):
            qc.cp(-np.pi/float(2**(j-m)), m, j)
        qc.h(j)


# ##### QPE Circuit

# In[11]:


def qpe(theta, qubits): 
    d = qubits 
    n = d + 1
    qpe = QuantumCircuit(n, n)
    qpe.x(n-1)
    qpe.h(range(n-1))
    repetitions = 1
    for counting_qubit in range(n - 1):
        qpe.cp(2 * np.pi * theta * (2**counting_qubit), counting_qubit, n - 1); # This is CU^2^j
    qpe.barrier()
    # Apply inverse QFT
    qft_dagger(qpe, n - 1)
    # Measure
    qpe.barrier()
    qpe.measure(range(n-1), range(n - 1))

    return qpe


def qpe_est_theta(theta, d_bit_precision, shots):
    d = d_bit_precision
    Aer = AerSimulator()
    simulator = transpile(qpe(theta, d), backend=Aer)
    result = Aer.run(simulator, shots=shots).result()
    counts = result.get_counts()
    max_outcome = max(counts, key=counts.get)   
    measured_phase = binary_to_decimal_float("0" + "." + max_outcome[1:d]) 

    return measured_phase
    


qpe(1, 5).draw('mpl', fold = 50)


# ##### Simulation and Plotting

# In[57]:


ancillar_qubits =  5
d = ancillar_qubits
def shots_err():
    errors = []
    shots_list = [2**4, 2**6, 2**8, 2**10, 2**12, 2**14, 2**16]
    for shots in shots_list:    
        Aer = AerSimulator()
        simulator = transpile(qpe(theta, 9), backend=Aer)
        result = Aer.run(simulator, shots=shots).result()
        counts = result.get_counts()
        
        max_outcome = max(counts, key=counts.get)   
        measured_phase = binary_to_decimal_float("0" + "." + max_outcome[1:d]) 
        # Compute the absolute error (mod 1)
        error = abs(measured_phase - theta) % 1
        errors.append(error)

    return errors
    
def precision_err(theta, precision_list):
    errors_d = []
    
    for d in precision_list:
        
        Aer = AerSimulator()
        simulator = transpile(qpe(theta, d), backend=Aer)
        result = Aer.run(simulator, shots=2**15).result()
        counts = result.get_counts()
        
        max_outcome = max(counts, key=counts.get)   
        measured_phase = binary_to_decimal_float("0" + "." + max_outcome[1:d]) 
        # Compute the absolute error (mod 1)
        error = abs(measured_phase - theta) % 1
        errors_d.append(error)

    return errors_d

# True values of theta
theta_values = [0.5625, 0.1234]

# Number of shots for the experiment
shot_values = [2**4, 2**6, 2**8, 2**10, 2**12, 2**14, 2**16 ]

# Store errors for each theta
errors = {theta: [] for theta in theta_values}

# Run Hadamard test for different shots and compute errors
for theta in theta_values:
    for shots in shot_values:
        estimated_theta = est_theta(theta, d, shots)
        error = abs(estimated_theta - theta)
        errors[theta].append(error)

# Plot error vs. number of shots
#plt.figure(figsize=(8, 6))
for theta, err in errors.items():
    plt.loglog(shot_values, err, marker='o', label=f'True θ = {theta}')
# --- Plot Error vs Shots in log-log scale ---
plt.figure(figsize=(6,4))
sqrt_bound = [1 / np.sqrt(s) for s in shot_values]
onebys = [1 / (s) for s in shot_values]
plt.loglog(shot_values, sqrt_bound, linestyle='dashed', color='black', label='O(1/√shots)')
plt.loglog(shot_values, onebys, linestyle='dashed', color='green', label='O(1/shots)')
#plt.plot(shots_list, errors, 'o--', label='QPE Estimation Error')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Number of Shots (log scale)')
plt.ylabel('Error in Phase Estimation (log scale)')
plt.title(f'QPE Error vs. Shots (theta={theta}, {ancillar_qubits} ancillar qubits)')
plt.legend()
plt.tight_layout()
plt.show()


precision_list = [1,2,3,4,5,6,7,8,9,10,11,12]

plt.figure(figsize=(6,4))
for theta in theta_values:
    plt.plot(precision_list, precision_err(theta, precision_list), 'o--', label=f'True θ = {theta}')
plt.xscale('linear')
plt.yscale('log')
plt.xlabel('Number of ancillars')
plt.ylabel('Error in Phase Estimation (log scale)')
plt.title(f'QPE Error vs. Ancillars ( 2^{15} Shots)')
plt.legend()
plt.tight_layout()
plt.show()


# With more than 4 ancillars, we get exact value of 0.5625 since its has an exact binary representation. Thus, it will show 0 error at the output.

# # Problem 4(d)

# ##### Comparision of errors vs. number of shots 

# **QPE** converges to a low error estimate for a given $\theta$ based on the number of ancillars rather than number of shots.<br>
# **Kitaev Test** was inconclusive for scaling.<br>
# **Hadamard Test** gives $\frac{1}{\sqrt{shots}}$ scaling.

# ##### Faster convergence

# QPE and Kitaev are converging at the same rate.<br>
# Hadamard converges to low errors the slowest.

# ##### QPE vs. Hadamard (for same number of shots)

# QPE is faster than Hadamard for same number of shots as its precision is increased by ancillars which the Hadamard test, being on a single qubit, lacks.

# ##### Qubit's effect of QPE accuracy

# QPE gives better estimates for more number of qubits as seen from QPE Error vs Ancillar plot.

# ##### Ancillars vs. shots for QPE

# Increasing the number of ancillars did improve the accuracy but number of shots had no effect on it. This is because the estimate of $\theta$ in an ideal QPE is dependent on the number of ancillars and not shots. 

# # Problem 5(c): Grover Search

# #### Defining the Oracle

# In[11]:


num_qubit = 4
marked_state = ['001', '110'] #Input the marked states here
def oracle():
    qc = QuantumCircuit(num_qubit)
    #To change the marked state, change the string in the "ctrl_state" in the gates below.
    for state in range(len(marked_state)):
        qc.append(CCCZ.control(num_qubit - 1, ctrl_state = marked_state[state]), range(num_qubit))
    qc.name = "Oracle"
    
    return qc

oracle().draw('mpl')


# #### Defining the Diffusion Operator (Reflection about the initial state which was superposition of all basis states)

# In[12]:


def diffusion():
    qc = QuantumCircuit(num_qubit-1)
    qc.h(range(num_qubit-1))
    qc.x(range(num_qubit-1))
    qc.append(CCCZ.control(num_qubit - 2), range(num_qubit-1))
    qc.x(range(num_qubit-1))
    qc.h(range(num_qubit-1))
    qc.name = "Diffusion"
    
    return qc

diffusion().draw('mpl')


# #### Defining the Grover Operator

# In[ ]:


def GroverCircuit():
    qc = QuantumCircuit(num_qubit, num_qubit - 1)
    qc.x(num_qubit - 1)
    qc.h(range(num_qubit-1))
    for s in range(int(np.sqrt(num_qubit/2))): #We run this circuit for ~ (N/a)^0.5, where N = total states, a = marked states
        qc.append(oracle().to_gate(), range(num_qubit))
        qc.append(diffusion().to_gate(), range(num_qubit-1))
        qc.append(qlib.GlobalPhaseGate(-1))
    qc.barrier()
    qc.x(num_qubit-1)
    qc.name = 'Grover Circuit'
    qc.measure([0,1,2], [0,1,2])
    return qc

GroverCircuit().draw('mpl')


# #### Simulation

# In[14]:


Aer = AerSimulator()
simulator = transpile(GroverCircuit(), backend=Aer)
result = Aer.run(simulator, shots= 2**16).result()
counts = result.get_counts()
plt_hist(counts)


# # Run on Quantum computer

# In[39]:


QiskitRuntimeService().save_account(channel = "ibm_quantum",
             token = '52be9565dbffe2dc56695ecc5cd5ab3cc7a7889d3adec69e686c1cf9ae455c12bcbf637b832cc796050e836d3efcb3cc2c9ebfb3a505a28be781dd681f0b6c22', 
                                             overwrite=True)


# In[53]:


from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager as gppm
circuit = HadamardCircuit(0.1234)

Observables = [Pauli('ZI'), Pauli('IZ'), Pauli('ZZ')]
be = QiskitRuntimeService().backend('ibm_kyiv')
PassManager = gppm(optimization_level = 1, backend = be)
CircuitTranspiled = PassManager.run(circuit)
job = EstimatorV2(be).run([(CircuitTranspiled)])
print(job.job_id())

