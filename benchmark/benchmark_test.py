#!/usr/bin/env python

import sys
import numpy as np

from pytuq.utils.plotting import myrc

from benchmark import *
from autogen import *

myrc()

fcns = [
# From https://infinity77.net/global_optimization/test_functions.html
# 1D
    # SineSum(),
    # SineSum2(),
    # QuadxExp(),
    # LinxSin(),
    # SinexExp(),
    # SineLogSum(),
    # CosineSum(),
    # Sinex(),
    # CosineSum2(),

# N-D test functions, alphabetically
    # Ackley(),
    # Adjiman(),
    # Alpine01(),
    # Alpine02(),
    # AMGM(),
    # BartelsConn(),
    # Bird(),
    # Bohachevsky(),
    # Branin01(),
    # Branin02(),
    # Brent(),
    # Bukin02(),
    # Bukin04(),
    # Bukin6(),
    # CarromTable(),
    # Chichinadze(),
    # Cigar(),
    # Colville(),
    # CosineMixture(),
    # Damavandi(),
    # DeckkersAarts(),
    Dolan(),

# Many local minima (it's worth running these with higher sample points)
    # CrossInTray(),
    # DropWave(),
    # EggHolder(),
    # Griewank(),

# Trig
    # ChengSandu(),
    # Sine1d(),
    # Forrester(),
    # Friedman(),
    # GramacyLee(),
    # GramacyLee2(),
    # Higdon(),
    # Holsclaw(),
    # Lim(),
    # DampedCosine(),
]

for fcn in fcns:
    print(f"========== Function {fcn.name} ==========")
    print("Gradient check")
    x = np.random.rand(111, fcn.dim)
    assert(np.allclose(fcn.grad_(x, eps=1.e-8), fcn.grad(x), atol=1.e-5, rtol=1.e-3))

    print("Minimize")
    xmin = fcn.minimize()
    print(f"Minimum is at {xmin}")

    print(f"Domain is {fcn.domain}")

    print("Plotting 1d slice")
    fcn.plot_1d(ngr=555)

    if fcn.dim>1:
        print("Plotting 2d slice")
        fcn.plot_2d(ngr=55)

print(f"\nTotal number of functions: {len(fcns)}")