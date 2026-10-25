# -*- coding: utf-8 -*-
"""
Created on Thu Sep 20 22:31:37 2018

@author: sas37
"""
import numpy as np
#    import eaa65_individual as eaa65
from   scipy.integrate import quad
from scipy.optimize import newton,fsolve,brentq
Vc=0.199334 / 102.03
Tc=101.08 + 273.15
Pc=4060.3

Tstandard=273.15

#standard internal energy (according to dow tables)
Hstandard=200
#standard internal energy (according to dow tables)
Sstandard=1
#integration intervals maximum
Maxintervals = 1000


def a(T):
    """
    
#The a parameters for the MBWR EOS
#of state. Taken from "Thermodynamic properties of HFC-134a" technical
#report T-134a-SI http://www2.dupont.com/Refrigerants/en_US/products/Suva/Suva134a.html
#INPUTS
#________
# T = temperature (K)
# a =  a parameters
    """
    #note that these paramters expect the volume to be in L/mol. A conversion
#is done at the end of the program to make them compatible with m3/kg
    
    R=0.08314471
# _a.m:14
    #parameters in the equation of state
    b=[0,-0.065455235227,5.8893751817,-137.61788409,22693.168845,-2926261.3296
       ,-0.0001192377619,-2.7214194543,1629.525368,729422.03182,-0.00011724519115,
       0.86864510013,-306.60168246,-0.025664047742,-2.4381835971,-316.03163961,
       0.34321651521,-0.010154368796,1.1734233787,- 0.027301766113,-663385.02898,
       - 64754799.101,-37295.219382,1261473589.9,- 647.4220007,123624.50399,
       - 1.5699196293,-518489.32204,-0.081396321392,30.325168842,0.00013399042297,
       - 0.15856192849,9.0679583743]

    T_=np.matrix([T,T ** 0.5,1.0,T ** - 1,T ** - 2,T ** - 3,T ** - 4]).transpose()

    a_=np.matrix(
               [[R,0,0,0,0,0,0],
                [b[1],b[2],b[3],b[4],b[5],0,0],
                [b[6],0,b[7],b[8],b[9],0,0],
                [b[10],0,b[11],b[12],0,0,0],
                [0,0,b[13],0,0,0,0],
                [0,0,0,b[14],b[15],0,0],
                [0,0,0,b[16],0,0,0],
                [0,0,0,b[17],b[18],0,0],
                [0,0,0,0,b[19],0,0],
                [0,0,0,0,b[20],b[21],0],
                [0,0,0,0,b[22],0,b[23]],
                [0,0,0,0,b[24],b[25],0],
                [0,0,0,0,b[26],0,b[27]],
                [0,0,0,0,b[28],b[29],0],
                [0,0,0,0,b[30],b[31],b[32]]
                ])
    
    #do the unit coversion
    a = a_*T_
    for n in range(1,10):
        a[n-1]=a[n-1] / (102.03) ** n
    
    for n in range(10,16):
        a[n-1]=a[n-1] / (102.03) ** (2*n - 17)

    
    return np.asarray(a, dtype=float).ravel()

    
def da_dT(T=None):
    """
#The a parameters for the MBWR EOS
#of state. Taken from "Thermodynamic properties of HFC-134a" technical
#report T-134a-SI http://www2.dupont.com/Refrigerants/en_US/products/Suva/Suva134a.html
#INPUTS
#________
# T = temperature (K)
# a =  a parameters
    
#note that these paramters expect the volume to be in L/mol. A conversion
#is done at the end of the program to make them compatible with m3/kg
    """
    R=0.08314471
    #parameters in the equation of state
    b=[0,- 0.065455235227,5.8893751817,- 137.61788409,22693.168845,- 2926261.3296
       ,- 0.0001192377619,- 2.7214194543,1629.525368,729422.03182,- 0.00011724519115,
       0.86864510013,- 306.60168246,- 0.025664047742,- 2.4381835971,- 316.03163961,
       0.34321651521,- 0.010154368796,1.1734233787,- 0.027301766113,- 663385.02898,
       - 64754799.101,- 37295.219382,1261473589.9,- 647.4220007,123624.50399,
       - 1.5699196293,- 518489.32204,- 0.081396321392,30.325168842,0.00013399042297,
       - 0.15856192849,9.0679583743]

    T_=np.matrix([1,0.5*T**-0.5,0,-1*T **-2,-2*T**-3,-3*T**-4,-4*T**-5]).transpose()
#            T    T^.5  1     1/T   T-2   T-3   T-4
    a_=np.matrix(
               [[R,0,0,0,0,0,0],
                [b[1],b[2],b[3],b[4],b[5],0,0],
                [b[6],0,b[7],b[8],b[9],0,0],
                [b[10],0,b[11],b[12],0,0,0],
                [0,0,b[13],0,0,0,0],
                [0,0,0,b[14],b[15],0,0],
                [0,0,0,b[16],0,0,0],
                [0,0,0,b[17],b[18],0,0],
                [0,0,0,0,b[19],0,0],
                [0,0,0,0,b[20],b[21],0],
                [0,0,0,0,b[22],0,b[23]],
                [0,0,0,0,b[24],b[25],0],
                [0,0,0,0,b[26],0,b[27]],
                [0,0,0,0,b[28],b[29],0],
                [0,0,0,0,b[30],b[31],b[32]]
                ])
    
    #do the unit coversion
    da_dt = a_*T_
    for n in range(1,10):
        da_dt[n-1]=da_dt[n-1] / (102.03) ** n
    
    for n in range(10,16):
        da_dt[n-1]=da_dt[n-1] / (102.03) ** (2*n - 17)

    
    return np.asarray(da_dt, dtype=float).ravel()


def P_TV(T, v):
    """
    Pressure from the MBWR equation of state.
    T = temperature (K)
    v = specific volume (m^3/kg)
    returns pressure in kPa
    """
    # P/100 = sum_{n=1}^{9} a_n/v^n
    #       + exp(-Vc^2/v^2) * sum_{n=10}^{15} a_n/v^(2n-17)
    aa = a(T)
    sum1 = np.zeros(15)

    for n in range(1, 10):
        sum1[n - 1] = aa[n - 1] / v ** n

    for n in range(10, 16):
        sum1[n - 1] = aa[n - 1] / v ** (2 * n - 17) * np.exp(-Vc ** 2 / v ** 2)

    return 100 * sum(sum1)


def dP_dT(T,v):
#calculate the rate dP/dT at constant volume. Taken from "Thermodynamic properties of HFC-134a" technical
#report T-134a-SI http://www2.dupont.com/Refrigerants/en_US/products/Suva/Suva134a.html
#INPUTS
#________
# v = specific volume (m^3/kg)
# T = temperature (K)
# P = pressure (KPa)

    #parameters in the equation of state
    
    #get the rate of change of a parameters
    da=da_dT(T)
    sum1=np.zeros(15)

    for n in range(1,10):
        sum1[n-1]=da[n-1] / v **n
    
    for n in range(10,16):
        sum1[n-1]=da[n-1] /( v ** (2*n - 17))*np.exp(- Vc ** 2 / v ** 2)

    
    dP_dT=100*(sum(sum1))

    return dP_dT
    
def dU_dV(T,v):
#calculate the rate dU/dV at constant T. Taken from "Thermodynamic properties of HFC-134a" technical
#report T-134a-SI http://www2.dupont.com/Refrigerants/en_US/products/Suva/Suva134a.html
#INPUTS
#________
# v = specific volume (m^3/kg)
# T = temperature (K)
#parameters in the equation of state
#get the rate of change of a parameters
    P    = P_TV(T,v)
    dU_dV= T*dP_dT(T,v) - P
    
    return dU_dV
    
def InternalEnergyChange(Ti,Tf,vi,vf):
    """
#calculate the rate dU/dV at constant T. Taken from "Thermodynamic properties of HFC-134a" technical
#report T-134a-SI http://www2.dupont.com/Refrigerants/en_US/products/Suva/Suva134a.html
#integrates the equation regardless of the two phase region
#USE:
#DU = _InternalEnergyChange(Ti,Tf,vi,vf)
#WHERE:
# DU = Change in internal energy between state i and state f (kJ/kg)
# Ti, Tf = Temperature of initial and final states (K)
# vi,vf  = specific volume of intial and final state (kg/m^3)
    """
    vinf=1000
    #parameters in the equation of state
    #integration between v1 and inf
    
    DU1=quad(lambda v:dU_dV(Ti,v),vi,vinf,limit=Maxintervals)
    #DU1=quadrature(lambda v:dU_dV(Ti,v),vi,vinf, vec_func=False,maxiter=Maxintervals)
    DU2=quad(IdealGasCv,Ti,Tf,limit=Maxintervals)
    #DU2=quadrature(IdealGasCv,Ti,Tf,vec_func=False,maxiter=Maxintervals)
    DU3=quad(lambda v:dU_dV(Tf,v),vinf,vf,limit=Maxintervals)
    #DU3=quadrature(lambda v:dU_dV(Tf,v),vinf,vf,vec_func=False,maxiter=Maxintervals)
    DU=DU1[0] + DU2[0] + DU3[0]

    
    return DU

def IdealGasCv(T):
    """
#returns the cv of ideal R134a
#Cv in J/mol/K
#T in K
    """
    
    cv=19.4006 + 0.258531*T - 0.000129665*T ** 2 - 8.314471

    #convert to kJ/Kg/K
    cv=cv / 1000 / 102.03*1000
# IdealGasCv.m:10
    return cv


def EntropyChange(Ti, Tf, vi, vf):
    """
    Change in specific entropy between two states, by integration of the
    equation of state.

    Ti, Tf = temperatures (K)
    vi, vf = specific volumes (m^3/kg)
    returns Δs in kJ/kg/K

    Path: (Ti, vi) -> (Ti, vinf) -> (Tf, vinf) -> (Tf, vf)
      (ds/dv)_T = (dP/dT)_v
      (ds/dT)_v = cv(T)/T   for the ideal-gas leg at vinf
    """
    vinf = 1000

    DS1 = quad(lambda v: dP_dT(Ti, v), vi, vinf, limit=Maxintervals)
    DS2 = quad(lambda T: IdealGasCv(T) / T, Ti, Tf, limit=Maxintervals)
    DS3 = quad(lambda v: dP_dT(Tf, v), vinf, vf, limit=Maxintervals)

    return DS1[0] + DS2[0] + DS3[0]


def EnthalpyChange(Ti, Tf, vi, vf):
    """
    Change in specific enthalpy between two states, by integration of the
    equation of state.

    Ti, Tf = temperatures (K)
    vi, vf = specific volumes (m^3/kg)
    returns Δh in kJ/kg

    h = u + P v, so Δh = Δu + Pf*vf - Pi*vi.
    Δu is integrated here (the provided InternalEnergyChange is not called):
      (du/dv)_T = T*(dP/dT)_v - P
      (du/dT)_v = cv(T)   for the ideal-gas leg at vinf
    """
    vinf = 1000

    DU1 = quad(lambda v: dU_dV(Ti, v), vi, vinf, limit=Maxintervals)
    DU2 = quad(IdealGasCv, Ti, Tf, limit=Maxintervals)
    DU3 = quad(lambda v: dU_dV(Tf, v), vinf, vf, limit=Maxintervals)
    DU = DU1[0] + DU2[0] + DU3[0]

    return DU + P_TV(Tf, vf) * vf - P_TV(Ti, vi) * vi



    
def H_TV(T,V):
    """
        #calculates the enthalpy from the Temperature and volume
        #Also handles the two phase region correctly, and works above and below critical
        #point. Might fail very close to critical temperature.
    
        #USE:
        # [H,{state,Hf,Hg}] = H_TV(T,V)
        # where:
        # V = specific volume(kJ/kg/K)
        # T = pressure (KPa)
        # H = specific enthalpy (kJ/kg)
        # state = fraction which is gas
        # Hf = saturation enthalpy of fluid (liquid) (kJ/kg)
        # Hg = saturation enthalpy of gas(i.e. vapour) (kJ/kg)
    """
    varargout={}


    #need to see if we are above critical point
    if T >= Tc:
        #above critical point
        H=EnthalpyChange(Tstandard,T,standardVf,V)
        state=1
        H=H + Hstandard

    else:
        #find out if we are in the two phase region
        vsatf=1 / SatDensityL(T)

        Pvap=Psat(T)

        vsatg=newton(lambda v:P_TV(T,v) - Pvap,8.314*T / Pvap / 1000 / 102*1000)

        if V < vsatf:
            #liquid
            H=EnthalpyChange(Tstandard,T,standardVf,V)
            state=0
            H=H + Hstandard

        else:
            if V > vsatg:
                #gas
                H=EnthalpyChange(Tstandard,T,standardVf,V)
                state=1
                H=H + Hstandard
            else:
                #two phase
                Hg=EnthalpyChange(Tstandard,T,standardVf,vsatg) + Hstandard
                Hf=EnthalpyChange(Tstandard,T,standardVf,vsatf) + Hstandard
                state=(V - vsatf) / (vsatg - vsatf)
                H=Hg*state + (1 - state)*Hf
                varargout['Hf']=Hf
# H_TV.m:62
                varargout['Hg']=Hg
# H_TV.m:63
        varargout['state']=state
# H_TV.m:66
    
    return H,varargout

def S_TV(T,V):
    """
#calculates the entropy from the Temperature and volume
#Also handles the two phase region correctly, and works above and below critical
#point. Might fail very close to critical temperature.
    
#USE:
# (S,{state,Sf,Sg}) = S_TV(T,V)
# where:
# V = specific volume(kJ/kg/K)
# T = pressure (KPa)
# S = specific entropy (kJ/kg/K)
# state = fraction which is gas
# Sf = saturation entropy of fluid (liquid) (kJ/kg/K)
# Sg = saturation entropy of gas(i.e. vapour) (kJ/kg/K)
    """

#work out if in two phase
    #need to identify if we are above the critical point.
    varargout={}
    
    if T >= Tc:
        #above critical point
        S=EntropyChange(Tstandard,T,standardVf,V)
        state=1
        S=S + Sstandard
    else:
        #below critical point, can compute two phase region.
        #find out if we are in the two phase region
        vsatf=1 / SatDensityL(T)
        Pvap=Psat(T)
        vsatg=newton(lambda v: P_TV(T,v) - Pvap,8.314*T / Pvap / 1000 / 102*1000)
        if V < vsatf:
            #liquid
            S=EntropyChange(Tstandard,T,standardVf,V)
            state=0
            S=S + Sstandard
        else:
            if V > vsatg:
                #gas
                S=EntropyChange(Tstandard,T,standardVf,V)
                state=1
                S=S + Sstandard
            else:
                #two phase
                Sg=EntropyChange(Tstandard,T,standardVf,vsatg) + Sstandard
                Sf=EntropyChange(Tstandard,T,standardVf,vsatf) + Sstandard
                state=(V - vsatf) / (vsatg - vsatf)
                S=Sg*state+ (1 - state)*Sf
                varargout['Sf']=Sf
                varargout['Sg']=Sg
    
    varargout['state']=state

    return S,varargout
    
def Psat(T):
    """
#Calculates the saturated vapour pressure (in kPa) - consistant with MBWR equation
#of state. Displays a warning and returns -1 if T is larger than the
#critical temperature.
#USE:
#Pvap= Psat(T)
#Where:
# T = temperature (K)
    """

    if T >= Tc:
        print('Warning, above critical temperature in Psat(T), Tc = 374.23K')
        Psat=- 1

    else:
        A=40.69889
        B=-2362.54
        C=-13.06883
        D=0.007616005
        E=0.2342564
        F=376.1111
        LogP=A + B / T + C*np.log10(T) + D*T + E*(F - T)/ T*np.log10(F - T)

        Psat=10 ** LogP

    
    return Psat

def SatDensityL(T):
    """
#Calculates the saturated density of liquid- consistent with MBWR equation of state
#USE:
#SatDensity= SatDensityL(T)
#WHERE:
# T = temperature (K)
# SatDensity = density of saturated liquid (kg/m^3)
    """

    Tr=T / Tc
    A=528.1464
    B=755.1834
    C=1028.676
    D=- 949.1172
    E=593.566
    SatDensity=A + (B*(1 - Tr) ** (1 / 3)) + (C*(1 - Tr) ** (2 / 3)) + (D*(1 - Tr)) + (E*(1 - Tr) ** (4 / 3))
    
    return SatDensity     
  
    
def V_TP(T,P):
    """
#Invert the equation of state for R134a
#of state. Taken from "Thermodynamic properties of HFC-134a" technical
#report T-134a-SI http://www2.dupont.com/Refrigerants/en_US/products/Suva/Suva134a.html
#USE:
#[Vf,Vg] = V_TP(T,P)
#WHERE:
# T = temperature (K)
# P = pressure (KPa)
# Vf = liquid volume (-1 if not in liquid or two phase region) (kg/m^3)
# Vg = liquid volume (-1 if not in gas or two phase region) (kg/m^3)
#supercritical fluid is treated as if gaseous
    """
#first work out if we are in the two phase region

#make sure we are not above the critical temperature
    if T>=Tc:
    #gas phase only
        Vg = newton(lambda v: P_TV(T,v)-P,8.314*T/P/1000/102*1000)
        Vf = -1;
    else:

        Pvap = Psat(T);
        rhosat = SatDensityL(T);
        vsat = 1/rhosat;
    
        if P==Pvap:
            #In two phase region
            #Vapour volume
            Vg = newton(lambda v:P_TV(T,v)-P,8.314*T/P/1000/102*1000);
            #liquid volume
            Vf = vsat
        elif P>Pvap: #liquid
            Vf = newton(lambda v:P_TV(T,v)-P,vsat);
            Vg = -1;
        elif  P<Pvap: #gas
            #in single phase region
            Vg = newton(lambda v:P_TV(T,v)-P,8.314*T/P/1000/102*1000);
            Vf = -1

    return Vf, Vg

def H_TP(T,P):
    """
#calculates the enthalpy from the Temperature and pressure
#In the two phase region should display a warning and return the liquid
#value
# #USE:
# [H] = H_TP(T,P)
# [H,state] = H_TP(T,P)
# where:
# P = Pressure (kPa)
# T = Temperature (K)
# H = specific enthalpy (kJ/kg)
# state = fraction which is gas
    """
    varargout={}
    vinf=1000

    #work out if in two phase 
    #calculate the pressure at the current conditions
    Vf,Vg=V_TP(T,P)
    if Vf == - 1:
        #all gas
        H=H_TV(T,Vg)
        state=1
    else:
        if Vg == - 1:
            #all liquid
            H=H_TV(T,Vf)
            state=0
        else:
            #two phase # value is indetermindate
            print('warning in the two phase region - use saturated enthalpy instead')
            H=H_TV(T,Vf)
            state=- 1
    
    varargout['state']=state

    return H,varargout

def H_TP(T,P):
    """
#calculates the enthalpy from the Temperature and pressure
#In the two phase region should display a warning and return the liquid
#value
# #USE:
# [H] = H_TP(T,P)
# [H,state] = H_TP(T,P)
# where:
# P = Pressure (kPa)
# T = Temperature (K)
# H = specific enthalpy (kJ/kg)
# state = fraction which is gas
    """
    varargout={}
    vinf=1000
# H_TP.m:14
    #work out if in two phase 
    #calculate the pressure at the current conditions
    Vf,Vg=V_TP(T,P)
    if Vf == - 1:
        #all gas
        H=H_TV(T,Vg)
        state=1
    else:
        if Vg == - 1:
            #all liquid
            H=H_TV(T,Vf)
            state=0
        else:
            #two phase # value is indetermindate
            print('warning in the two phase region - use saturated enthalpy instead')
            H=H_TV(T,Vf)
            state=- 1
    
    varargout['state']=state

    return H[0],varargout

def S_TP(T,P):
    """
#calculates the entropy from the Temperature and pressure
#In the two phase region should printlay a warning and return the liquid
#value
# #USE:
# [S] = S_TP(T,P)
# [S,state] = S_TP(T,P)
# where:
# P = Pressure (kPa)
# T = Temperature (K)
# S = specific entropy (kJ/kg)
# state = fraction which is gas
    """
    varargout={}
    vinf=1000

    #work out if in two phase 
    #calculate the pressure at the current conditions
    Vf,Vg=V_TP(T,P)
    if Vf == - 1:
        #all gas
        S=S_TV(T,Vg)
        state=1
    else:
        if Vg == - 1:
            #all liquid
            S=S_TV(T,Vf)
            state=0
        else:
            #two phase # value is indetermindate
            print('warning in the two phase region - use saturated enthalpy instead')
            S=S_TV(T,Vf)
            state=- 1
    
    varargout['state']=state

    return S[0],varargout

def Hsat_T(T):
#Calculates the saturated enthalpies. 
#Displays a warning if the temperature is above the critical points and returns
#entropies of -1.
#USE:
#Hf,Hg = Hsat_T(T)
#Where;
# T = Temperature(K)
# Hf = Enthalpy of the liquid phase (kJ/kg/K)
# Hg = Enthalpy of the gas phase (kJ/kg/K)

    if T > Tc:
        print('warning T is greater than critical temperature, unable to calculate the saturated enthalpy');
        Hf = -1
        Hg = -1
    else:
    
        #find out if we are in the two phase region
        vsatf = 1/SatDensityL(T)           
        Pvap = Psat(T)
    
        vsatg = newton(lambda v: P_TV(T,v) - Pvap,8.314*T / Pvap / 1000 / 102*1000)

        #calculate the saturation entropies 
        Hg =EnthalpyChange(Tstandard,T,standardVf,vsatg)+Hstandard;
        Hf =EnthalpyChange(Tstandard,T,standardVf,vsatf)+Hstandard;

    return Hf, Hg

def Ssat_T(T):
#Calculates the saturated entropies 
#Displays a warning if the temperature is above the critical points and returns
#entropies of -1.
#USE:
#sf,sg = Ssat_T(T)
#Where;
# T = Temperature(K)
# Sf= Entropy of the liquid phase (kJ/kg/K)
# Sg = Entropy of the gas phase (kJ/kg/K)

    if T > Tc:
        print('warning T is greater than critical temperature, unable to calculate the saturated enthalpy');
        Sf = -1
        Sg = -1
    else:
    
        #find out if we are in the two phase region
        vsatf = 1/SatDensityL(T)           
        Pvap = Psat(T)
    
        vsatg = newton(lambda v: P_TV(T,v) - Pvap,8.314*T / Pvap / 1000 / 102*1000)

        #calculate the saturation entropies 
        Sg =EntropyChange(Tstandard,T,standardVf,vsatg)+Sstandard;
        Sf =EntropyChange(Tstandard,T,standardVf,vsatf)+Sstandard;

    return Sf, Sg
    
#standard state 0C saturated liquid
standardVf=1 / SatDensityL(Tstandard)

    
if __name__ == '__main__':
    print("""EXAMPLE CALLS TO FUNCTIONS
--------------------------------------""")
    print("V_TP(390,3400)=",V_TP(390,3400))
    print("P_TV(298,0.01)= ", P_TV(298,0.01) )
    print("InternalEnergyChange(298,250,0.01,0.02)=", InternalEnergyChange(298,250,0.01,0.02))
    print("EntropyChange(298,250,0.01,0.02)=",EntropyChange(298,250,0.01,0.02) )
    print("EnthalpyChange(298,250,0.01,0.02)=",EnthalpyChange(298,250,0.01,0.02) )
    print("H_TV(313.15,2.54453)=",H_TV(313.15,2.54453))
    print("S_TV(313.15,2.54453)=",S_TV(313.15,2.54453))
    print("V_TP(T,P)=",V_TP(320,800))
    print("H_TP(320,800)=",H_TP(320,800))
    print("S_TP(320,800)=",S_TP(320,800))
    print("Hsat_T(300)=",Hsat_T(300))
    print("Ssat_T(300)=",Ssat_T(300))