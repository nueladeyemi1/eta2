# -*- coding: utf-8 -*-
"""
Created on Mon Sep 24 20:05:04 2018

@author: sas37
"""

from R134a import *
from scipy.optimize import brentq

import matplotlib.pyplot as plt

from numpy import *

### an example of how to plot the a curve in python using matplotlib
## note that some of the values are strange because this code doesn't fully converge at
## some of the points
N = 15
Trange = linspace(-40,100,N)+273.15
p_ = zeros(N)
for i,T in enumerate(Trange):
    p_[i] = Psat(T)
plt.plot(Trange,p_)
plt.ylabel("Psat (kpa)")
plt.xlabel("T (K)")
plt.show()

#calculat the saturated vapour pressure at 273.15K
print(Psat(273.15))

#calculate the volume at 293K and 100 kPa
print(V_TP(293,100))


#calculate the enthapy at  T= 313.15 K and v= 2.54453 m^3/kg
print(H_TV(313.15,2.54453))

#calculat the entropy at  T= 313.15 K and v= 2.54453 m^3/kg, and check the
#state of the fluid. 
print(S_TV(313.15,2.54453))

#calculate the entropy at T = 368.15 k and P =3400 kPa
S1,state=(S_TP(368.15,3400))
print(S1)

#Can also calculate the enthalpy from the temperature and pressure
print(H_TP(368.15,3400))

#May have to use the function to search for unkown values.  For example use the
#entropy function and #the scipy solver "brentq" to search for a temperature
#which gives the same #entropy (which should be 368.15K), in the interval 
#350 to 390K
def thing(x):
    y= S_TP(x,3400)
    return y[0]-S1
T = brentq(thing,350,390)
print(T)
#Can also calculate the enthalpy from the temperature and pressure
H2 = Hsat_T(293.15)
print(H2)

#Inverting the psat function to give a temperature
SatPressure = 2500
def thing2(x):
    y = Psat(x)
    return y-SatPressure
SaturationT = brentq(thing2,273,370)
print(SaturationT)