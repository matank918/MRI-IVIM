"""
September 2020 by Oliver Gurney-Champion & Misha Kaandorp
oliver.gurney.champion@gmail.com / o.j.gurney-champion@amsterdamumc.nl
https://www.github.com/ochampion
Code is uploaded as part of our publication in MRM (Kaandorp et al. Improved unsupervised physics-informed deep learning for intravoxel-incoherent motion modeling and evaluation in pancreatic cancer patients. MRM 2021)
requirements:
numpy
torch
tqdm
matplotlib
scipy
joblib
"""

# import
import numpy as np
from config import hyperparams as hp_example
from utils import checkarg
from simulations import sim

dropout = [0.1, 0.2, 0.3, 0.4, 0.5]
arg = hp_example()
arg = checkarg(arg)

matNN = np.zeros([len(dropout), 3, 4])
stability = np.zeros([len(dropout), 3])


my_sim = sim(30, arg)
matlsq = my_sim.run_fit()

for i, drop in enumerate(dropout):
    # print('\n simulation at dropout of {drop}\n'.format(drop=drop))

    arg.net_pars.dropout = drop
    matNN[i, :, :], stability[i, :] = my_sim.run_NN()

