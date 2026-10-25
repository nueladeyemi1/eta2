# Experimental heat-pump cycle for Spyder.
#
# 1. Open this file in Spyder. It must sit next to R134a.py, in /Users/admin.
# 2. Look at the working-directory box above the console. It should be
#    /Users/admin. If it is the MPhil folder, Spyder will import a different
#    R134a.py.
# 3. Press F5.
#
# The console prints one state at a time. The T-s and P-h figures open in
# the Plots pane and are also saved next to this file.
#
# To call a single function, type this in the console instead of pressing F5:
#
#     import R134a as r
#     print(r.Psat(17.5 + 273.15))

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq
import R134a as r


# --- declare the measurements -------------------------------------------
# R134a.py expects pressure in kPa and temperature in K.
P1_bar = 5.3
T1_C = 17.5
P2_bar = 21.5
T2_C = 81.9
T3_C = 63.6
T4_C = 20.0

P1 = P1_bar * 100.0
P2 = P2_bar * 100.0
T1 = T1_C + 273.15
T2 = T2_C + 273.15
T3 = T3_C + 273.15
T4_measured = T4_C + 273.15


def saturation_temperature(P):
    """Temperature (K) at which Psat(T) equals the given pressure (kPa)."""
    return brentq(lambda T: r.Psat(T) - P, 230.0, 370.0)


# Psat takes kelvin and returns kPa. Compare it with the measured pressure.
print("Psat at measured T1: {:.2f} kPa".format(float(r.Psat(T1))))
print("measured P1:         {:.2f} kPa".format(P1))

Tsat1 = saturation_temperature(P1)
Tsat2 = saturation_temperature(P2)
print("Tsat(P1) = {:.2f} C".format(Tsat1 - 273.15))
print("Tsat(P2) = {:.2f} C".format(Tsat2 - 273.15))


# --- state 1, evaporator exit -------------------------------------------
# H_TP(T, P) returns liquid when T is below Tsat(P). The reading 17.5 C is
# 0.04 K below Tsat(5.3 bar), so H_TP would give the liquid enthalpy.
# The compressor inlet is vapour. Use the saturated-vapour values instead.
# Hsat_T and Ssat_T each return (liquid, vapour).
hf1, hg1 = r.Hsat_T(Tsat1)
sf1, sg1 = r.Ssat_T(Tsat1)
hf1, hg1 = float(hf1), float(hg1)
sf1, sg1 = float(sf1), float(sg1)
h1, s1, T1_state = hg1, sg1, Tsat1
print("state 1  Hsat_T / Ssat_T   h = {:7.2f}   s = {:.4f}".format(h1, s1))


# --- state 2, compressor exit -------------------------------------------
# Superheated vapour: temperature and pressure are enough.
# The dictionary entry "state" is 1 for vapour, 0 for liquid, -1 inside
# the two-phase region.
h2, info2 = r.H_TP(T2, P2)
s2, _ = r.S_TP(T2, P2)
h2, s2 = float(h2), float(s2)
print("state 2  H_TP / S_TP       h = {:7.2f}   s = {:.4f}   state = {}".format(
    h2, s2, info2["state"]))


# --- state 3, condenser exit --------------------------------------------
# 63.6 C is below Tsat(P2), so the liquid has been cooled past boiling.
# Temperature and pressure fix the state. state == 0 means liquid.
h3, info3 = r.H_TP(T3, P2)
s3, _ = r.S_TP(T3, P2)
h3, s3 = float(h3), float(s3)
print("state 3  H_TP / S_TP       h = {:7.2f}   s = {:.4f}   state = {}".format(
    h3, s3, info3["state"]))

# Saturation values at the condenser pressure. State 3 is not on this
# curve; the numbers are only used to draw the condensation line.
hf2, hg2 = r.Hsat_T(Tsat2)
sf2, sg2 = r.Ssat_T(Tsat2)
hf2, hg2 = float(hf2), float(hg2)
sf2, sg2 = float(sf2), float(sg2)


# --- state 4, after the throttle ----------------------------------------
# The valve does not change the energy, so h4 = h3. That energy lies
# between liquid and vapour at P1, so the temperature is Tsat(P1).
# T4_C is the thermometer check. It is not passed to H_TP.
h4 = h3
T4 = Tsat1
x4 = (h4 - hf1) / (hg1 - hf1)
s4 = sf1 + x4 * (sg1 - sf1)
print("state 4  h4 = h3            h = {:7.2f}   s = {:.4f}   dryness = {:.3f}".format(
    h4, s4, x4))
print("T4 from the balance        {:.2f} C".format(T4 - 273.15))
print("T4 measured                {:.2f} C".format(T4_measured - 273.15))

w = h2 - h1
qh = h2 - h3
cop = qh / w
print("compressor work  {:.2f} kJ/kg".format(w))
print("heat delivered   {:.2f} kJ/kg".format(qh))
print("COP              {:.3f}".format(cop))


# --- background saturation curve ----------------------------------------
print("calculating the saturation curve for the plots...")
dome_T = np.linspace(250.0, 360.0, 8)
dome_sf, dome_sg, dome_hf, dome_hg, dome_P = [], [], [], [], []
for T in dome_T:
    sf_i, sg_i = r.Ssat_T(float(T))
    hf_i, hg_i = r.Hsat_T(float(T))
    dome_sf.append(float(sf_i))
    dome_sg.append(float(sg_i))
    dome_hf.append(float(hf_i))
    dome_hg.append(float(hg_i))
    dome_P.append(float(r.Psat(float(T))))

# Desuperheating leg of process 2-3, still at P2.
vapour_T, vapour_s, vapour_h = [], [], []
for T in np.linspace(Tsat2 + 0.5, T2, 5):
    vapour_s.append(float(r.S_TP(float(T), P2)[0]))
    vapour_h.append(float(r.H_TP(float(T), P2)[0]))
    vapour_T.append(float(T))
vapour_T = np.array(vapour_T)
vapour_s = np.array(vapour_s)
vapour_h = np.array(vapour_h)

# Subcooling leg of process 2-3, from the boiling temperature down to T3.
liquid_T, liquid_s, liquid_h = [], [], []
for T in np.linspace(Tsat2 - 0.4, T3, 4):
    liquid_s.append(float(r.S_TP(float(T), P2)[0]))
    liquid_h.append(float(r.H_TP(float(T), P2)[0]))
    liquid_T.append(float(T))
liquid_T = np.array(liquid_T)
liquid_s = np.array(liquid_s)
liquid_h = np.array(liquid_h)


# --- T-s -----------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7.2, 5.2))
ax.plot(dome_sf, dome_T, color="0.55", label="saturated liquid")
ax.plot(dome_sg, dome_T, color="0.55", label="saturated vapour")
ax.plot([s1, s2], [T1_state, T2], color="C0", lw=1.8, label="cycle")
ax.plot(np.r_[vapour_s[::-1], sg2, sf2, liquid_s],
        np.r_[vapour_T[::-1], Tsat2, Tsat2, liquid_T], color="C0", lw=1.8)
ax.plot([s3, s4], [T3, T4], color="C0", lw=1.8)
ax.plot([s4, s1], [T4, T1_state], color="C0", lw=1.8)
ax.plot([s1, s2, s3, s4], [T1_state, T2, T3, T4], "o", color="C0")
for name, si, Ti in zip(("1", "2", "3", "4"),
                        (s1, s2, s3, s4),
                        (T1_state, T2, T3, T4)):
    ax.annotate(name, (si, Ti), textcoords="offset points", xytext=(6, 4))
ax.set_xlabel("entropy (kJ/kg/K)")
ax.set_ylabel("temperature (K)")
ax.set_title("Experimental cycle, COP = {:.2f}".format(cop))
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig("/Users/admin/experimental_cycle_TS.png", dpi=150)


# --- P-h -----------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7.2, 5.2))
ax.plot(dome_hf, dome_P, color="0.55", label="saturation curve")
ax.plot(dome_hg, dome_P, color="0.55")
ax.plot([h1, h2], [P1, P2], color="C0", lw=1.8, label="cycle")
ax.plot(np.r_[vapour_h[::-1], hg2, hf2, liquid_h],
        np.full(len(vapour_h) + 2 + len(liquid_h), P2), color="C0", lw=1.8)
ax.plot([h3, h4], [P2, P1], color="C0", lw=1.8)
ax.plot([h4, h1], [P1, P1], color="C0", lw=1.8)
ax.plot([h1, h2, h3, h4], [P1, P2, P2, P1], "o", color="C0")
for name, hi, Pi in zip(("1", "2", "3", "4"),
                        (h1, h2, h3, h4),
                        (P1, P2, P2, P1)):
    ax.annotate(name, (hi, Pi), textcoords="offset points", xytext=(6, 4))
ax.set_xlabel("enthalpy (kJ/kg)")
ax.set_ylabel("pressure (kPa)")
ax.set_yscale("log")
ax.set_title("Experimental cycle, COP = {:.2f}".format(cop))
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig("/Users/admin/experimental_cycle_PH.png", dpi=150)
plt.show()
