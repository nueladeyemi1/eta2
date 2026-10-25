# Figure 2 model of the vapour-compression cycle, for Spyder.
#
# 1. Open this file in Spyder. It must sit next to R134a.py, in /Users/admin.
# 2. The working-directory box above the console must read /Users/admin.
# 3. Press F5.
#
# This is the model in the exercise caption, not the thermometer readings.
# The measured pressures choose the two black lines. The caption then
# places the four crosses:
#   evaporator outlet   3 K above boiling
#   compressor          isentropic (entropy stays the same)
#   condenser outlet    5 K below boiling
#   throttle            isenthalpic (enthalpy stays the same)
#
# R134a.py and the Figure 2 axis use kelvin. A difference of 3 K is the
# same size as 3 degrees Celsius. To convert:  T_K = T_C + 273.15.

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq
import R134a as r


# --- the two measured pressures, and the Figure 2 rules -----------------
P1_bar = 5.3
P2_bar = 21.5
SUPERHEAT = 3.0     # K above the evaporator boiling temperature
SUBCOOL = 5.0       # K below the condenser boiling temperature

P1 = P1_bar * 100.0    # kPa, which is what R134a.py expects
P2 = P2_bar * 100.0


def saturation_temperature(P):
    """Boiling temperature in kelvin at pressure P in kPa."""
    return brentq(lambda T: r.Psat(T) - P, 230.0, 370.0)


def show_T(name, T):
    print("{}  {:.2f} K   ({:.2f} C)".format(name, T, T - 273.15))


# Psat(T) needs kelvin and returns kPa. The search below inverts it:
# given the measured pressure, find the boiling temperature.
Tsat1 = saturation_temperature(P1)
Tsat2 = saturation_temperature(P2)
show_T("evaporator boiling T", Tsat1)
show_T("condenser boiling T ", Tsat2)


# --- state 1, evaporator outlet -----------------------------------------
# 3 K hotter than boiling, so the fluid is vapour. H_TP and S_TP apply.
T1 = Tsat1 + SUPERHEAT
h1, info1 = r.H_TP(T1, P1)
s1, _ = r.S_TP(T1, P1)
h1, s1 = float(h1), float(s1)
show_T("state 1", T1)
print("state 1  H_TP / S_TP   h = {:7.2f}   s = {:.4f}   state = {}".format(
    h1, s1, info1["state"]))


# --- state 2, compressor exit -------------------------------------------
# Isentropic: same entropy as state 1, at the condenser pressure.
# brentq searches for that temperature. Then H_TP gives the enthalpy.
T2 = brentq(lambda T: r.S_TP(T, P2)[0] - s1, Tsat2 + 0.5, Tsat2 + 40.0)
h2, info2 = r.H_TP(T2, P2)
s2, _ = r.S_TP(T2, P2)
h2, s2 = float(h2), float(s2)
show_T("state 2", T2)
print("state 2  H_TP / S_TP   h = {:7.2f}   s = {:.4f}   state = {}".format(
    h2, s2, info2["state"]))


# --- state 3, condenser outlet ------------------------------------------
# 5 K colder than boiling, so the fluid is liquid. H_TP and S_TP again.
T3 = Tsat2 - SUBCOOL
h3, info3 = r.H_TP(T3, P2)
s3, _ = r.S_TP(T3, P2)
h3, s3 = float(h3), float(s3)
show_T("state 3", T3)
print("state 3  H_TP / S_TP   h = {:7.2f}   s = {:.4f}   state = {}".format(
    h3, s3, info3["state"]))


# --- state 4, after the throttle ----------------------------------------
# Isenthalpic: h4 = h3. The mixture is at the evaporator boiling
# temperature. Hsat_T returns (liquid, vapour) and is used only for
# the dryness.
h4 = h3
T4 = Tsat1
hf1, hg1 = r.Hsat_T(Tsat1)
sf1, sg1 = r.Ssat_T(Tsat1)
hf1, hg1 = float(hf1), float(hg1)
sf1, sg1 = float(sf1), float(sg1)
x4 = (h4 - hf1) / (hg1 - hf1)
s4 = sf1 + x4 * (sg1 - sf1)
show_T("state 4", T4)
print("state 4  h4 = h3        h = {:7.2f}   s = {:.4f}   dryness = {:.3f}".format(
    h4, s4, x4))

# Saturation values at the condenser pressure, used to draw the
# horizontal condensation line. State 3 itself is 5 K below this.
hf2, hg2 = r.Hsat_T(Tsat2)
sf2, sg2 = r.Ssat_T(Tsat2)
hf2, hg2 = float(hf2), float(hg2)
sf2, sg2 = float(sf2), float(sg2)

w = h2 - h1
qh = h2 - h3
cop = qh / w
print("compressor work  {:.2f} kJ/kg".format(w))
print("heat delivered   {:.2f} kJ/kg".format(qh))
print("COP              {:.3f}".format(cop))


# --- grey saturation curve, as in Figure 2 ------------------------------
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

# Black isobar at P2: superheated vapour from T2 down to boiling,
# then liquid from boiling down to state 3.
vapour_T, vapour_s, vapour_h = [], [], []
for T in np.linspace(Tsat2 + 0.5, T2, 5):
    vapour_s.append(float(r.S_TP(float(T), P2)[0]))
    vapour_h.append(float(r.H_TP(float(T), P2)[0]))
    vapour_T.append(float(T))
vapour_T = np.array(vapour_T)
vapour_s = np.array(vapour_s)
vapour_h = np.array(vapour_h)

liquid_T, liquid_s, liquid_h = [], [], []
for T in np.linspace(Tsat2 - 0.4, T3, 4):
    liquid_s.append(float(r.S_TP(float(T), P2)[0]))
    liquid_h.append(float(r.H_TP(float(T), P2)[0]))
    liquid_T.append(float(T))
liquid_T = np.array(liquid_T)
liquid_s = np.array(liquid_s)
liquid_h = np.array(liquid_h)


# --- T-s: grey dome, black isobars, crosses at the four states ----------
fig, ax = plt.subplots(figsize=(7.2, 5.2))
ax.plot(dome_sf, dome_T, color="0.55", label="saturation line")
ax.plot(dome_sg, dome_T, color="0.55")
ax.plot([sf1, sg1], [Tsat1, Tsat1], color="black", lw=1, label="isobar")
ax.plot([sf2, sg2], [Tsat2, Tsat2], color="black", lw=1)
ax.plot(vapour_s, vapour_T, color="black", lw=1)
ax.plot(liquid_s, liquid_T, color="black", lw=1)
# 1-2 is vertical because the entropy is unchanged.
ax.plot([s1, s2], [T1, T2], color="C0", lw=1.8, label="model cycle")
ax.plot(np.r_[vapour_s[::-1], sg2, sf2, liquid_s],
        np.r_[vapour_T[::-1], Tsat2, Tsat2, liquid_T], color="C0", lw=1.8)
ax.plot([s3, s4], [T3, T4], color="C0", lw=1.8)
ax.plot([s4, sg1, s1], [Tsat1, Tsat1, T1], color="C0", lw=1.8)
ax.plot([s1, s2, s3, s4], [T1, T2, T3, T4], "x", color="C0", ms=8, mew=2)
for name, si, Ti in zip(("1", "2", "3", "4"),
                        (s1, s2, s3, s4),
                        (T1, T2, T3, T4)):
    ax.annotate(name, (si, Ti), textcoords="offset points", xytext=(6, 4))
ax.set_xlabel("entropy (kJ/kg/K)")
ax.set_ylabel("temperature (K)")
ax.set_title("Figure 2 model, COP = {:.2f}".format(cop))
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig("/Users/admin/model_cycle_TS.png", dpi=150)


# --- P-h -----------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7.2, 5.2))
ax.plot(dome_hf, dome_P, color="0.55", label="saturation line")
ax.plot(dome_hg, dome_P, color="0.55")
ax.plot([hf1, hg1], [P1, P1], color="black", lw=1, label="isobar")
ax.plot([hf2, hg2], [P2, P2], color="black", lw=1)
ax.plot(vapour_h, np.full_like(vapour_h, P2), color="black", lw=1)
ax.plot(liquid_h, np.full_like(liquid_h, P2), color="black", lw=1)
ax.plot([h1, h2], [P1, P2], color="C0", lw=1.8, label="model cycle")
ax.plot(np.r_[vapour_h[::-1], hg2, hf2, liquid_h],
        np.full(len(vapour_h) + 2 + len(liquid_h), P2), color="C0", lw=1.8)
ax.plot([h3, h4], [P2, P1], color="C0", lw=1.8)
ax.plot([h4, hg1, h1], [P1, P1, P1], color="C0", lw=1.8)
ax.plot([h1, h2, h3, h4], [P1, P2, P2, P1], "x", color="C0", ms=8, mew=2)
for name, hi, Pi in zip(("1", "2", "3", "4"),
                        (h1, h2, h3, h4),
                        (P1, P2, P2, P1)):
    ax.annotate(name, (hi, Pi), textcoords="offset points", xytext=(6, 4))
ax.set_xlabel("enthalpy (kJ/kg)")
ax.set_ylabel("pressure (kPa)")
ax.set_yscale("log")
ax.set_title("Figure 2 model, COP = {:.2f}".format(cop))
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig("/Users/admin/model_cycle_PH.png", dpi=150)
plt.show()
