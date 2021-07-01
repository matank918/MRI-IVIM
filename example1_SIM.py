#!/usr/bin/env python3
# -*- coding: utf-8 -*-
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
# import simulations as sim
from config import hyperparams as hp_example_1
from simulations import sim

# Import parameters
arg = hp_example_1()

my_sim = sim(30, arg)
matNN, stability = my_sim.run_NN()
