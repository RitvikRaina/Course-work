from qiskit import transpile
from qiskit import QuantumCircuit as QC
from qiskit import QuantumRegister as QR
from qiskit import ClassicalRegister as CR
from qiskit.quantum_info.operators import Operator
from qiskit_aer import AerSimulator
from qiskit.visualization import plot_histogram as plt_hist
import matplotlib as plt
import numpy as np


#%%Hadamard Test
qr_sup = QR(2, "Superposition Register")
qr_num = QR(0, "L-bit number")
cr = CR(1, "Classical Register")
qc = QC(qr_sup, qr_num, cr)
qc.x(1)
qc.h(0)
U = Operator([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, complex(np.cos(2*np.pi*0.5), np.sin(2*np.pi*0.5))]
    ])

qc.unitary(U, [0, 1], label='U')
qc.h(0)
qc.measure(0,0)

qc.draw("mpl")

#%%Aer simulation with state vector representation or counts.
aer = AerSimulator()
qc_sim = transpile(qc, backend=aer)
result = aer.run(qc_sim, shots=200000).result()

m = 0
if m == 0:
    counts = result.get_counts()
    print(counts)


if m == 1:
    psi = result.get_statevector(qc_sim)
    psi.draw("latex")



#%%Rough Work

# U = QC(4)
# U.h(range(4))
# U.save_statevector()
# U.swap(2,3)
# U.swap(1,2)
# U.swap(0,1)
# U.x(range(4))
# U.save_statevector()

# U.draw("text")

# aer = AerSimulator()
# qc_sim = transpile(qc, backend=aer)
# result  = aer.run(qc_sim,shots = 4096).result()

# psi = result.get_statevector(qc_sim)
# psi.draw("latex")

# counts  = result.get_counts()
# print(counts)
# plt_hist(counts)



    
