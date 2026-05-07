# iom_app.py – IOM Upwind Sail‑Trim / VMG Optimizer
# Full single‑file drop‑in for Streamlit Community Cloud

import streamlit as st
import numpy as np

# ------------------------------------------------------------------
# CONSTANTS (simplified IOM‑scale model)
# ------------------------------------------------------------------
RHO_AIR = 1.225
RHO_WATER = 1025.0
G = 9.81

DISPLACEMENT = 4.0      # kg
SAIL_HEIGHT = 1.5        # m
SAIL_AREA_MAIN = 0.22    # m²
SAIL_AREA_JIB  = 0.10    # m²
KEEL_AREA = 0.015        # m²
KEEL_AR = 4.0
HYDRO_EFF = 0.9
HEEL_STIFFNESS = 0.07    # m per rad

# ------------------------------------------------------------------
# BASIC MODELS
# ------------------------------------------------------------------
def sail_forces(AWA, AWS, sheet, twist, camber, area):
    """Lift/drag for a single sail surface."""
    z = np.linspace(0, SAIL_HEIGHT, 8)
    c = area / SAIL_HEIGHT * np.ones_like(z)
    twist_prof = twist * (z / SAIL_HEIGHT)
    alpha = np.radians(AWA - sheet) - np.radians(twist_prof)

    # approximate 2D sail coefficients
    CL = 1.3 * alpha * (1 - 4.0 * (camber - 0.1) ** 2)
    CD = 0.01 + 0.02 * CL ** 2

    q = 0.5 * RHO_AIR * AWS ** 2
    L = np.sum(q * CL * c * np.gradient(z))
    D = np.sum(q * CD * c * np.gradient(z))

    F_drive = L * np.sin(np.radians(AWA)) - D * np.cos(np.radians(AWA))
    F_side  = L * np.cos(np.radians(AWA)) + D * np.sin(np.radians(AWA))
    M_heel  = F_side * (SAIL_HEIGHT * 0.4)
    return F_drive, F_side, M_heel


def hydro_forces(Vb, F_side):
    """Simple hydrodynamic resistance + side‑force balance."""
    CL_h = F_side / max(0.5 * RHO_WATER * KEEL_AREA * Vb ** 2, 1e-6)
    CD_h = 0.01 + CL_h ** 2 / (np.pi * HYDRO_EFF * KEEL_AR)
    R = 0.5 * RHO_WATER * CD_h * KEEL_AREA * Vb ** 2 + 0.40 * Vb ** 2
    return R


def boat_equilibrium(TWA, TWS,
                     main_sheet, main_twist, main_camber,
                     jib_sheet,  jib_twist,  jib_camber):
    """Iterate boat speed for force equilibrium, return (Vb, heel)."""
    Vb = 1.0
    for _ in range(80):
        AWA = np.degrees(np.arctan2(TWS * np.sin(np.radians(TWA)),
                                    TWS * np.cos(np.radians(TWA)) - Vb))
        AWS = np.hypot(TWS * np.sin(np.radians(TWA)),
                       TWS * np.cos(np.radians(TWA)) - Vb)

        # slot effect: jib sees slightly higher AWA
        Fm = sail_forces(AWA, AWS, main_sheet, main_twist, main_camber, SAIL_AREA_MAIN)
        Fj = sail_forces(AWA + 4, AWS, jib_sheet, jib_twist, jib_camber, SAIL_AREA_JIB)

        F_drive = Fm[0] + Fj[0]
        F_side  = Fm[1] + Fj[1]
        M_heel  = Fm[2] + Fj[2]
        R_hydro = hydro_forces(Vb, F_side)

        err = F_drive - R_hydro
        Vb += 0.05 * err / (abs(R_hydro) + 1e-6)
        if abs(err) < 0.02:
            break

    # stable heel angle
    ratio = M_heel / max(DISPLACEMENT * G * HEEL_STIFFNESS, 1e-6)
    ratio = max(-1.0, min(1.0, ratio))
    heel = np.degrees(np.arcsin(ratio))
    return max(Vb, 0), heel



## ------------------------------------------------------------------
# AUTOMATIC TRIM OPTIMIZER
# ------------------------------------------------------------------
from scipy.optimize import differential_evolution

def optimise_trim_for_vmg(TWA, TWS):
    """Search parameter space for max VMG at given conditions."""

    # bounds: (main_sheet, main_twist, main_camber, jib_sheet, jib_twist, jib_camber)
    bounds = [
        (5, 25), (0, 10), (0.05, 0.20),
        (5, 25), (0, 10), (0.05, 0.20)
    ]

    def objective(x):
        ms, mt, mc, js, jt, jc = x
        Vb, _ = boat_equilibrium(TWA, TWS, ms, mt, mc, js, jt, jc)
        VMG = Vb * np.cos(np.radians(TWA))
        return -VMG

    result = differential_evolution(objective, bounds, maxiter=60, popsize=10, tol=1e-3)
    ms, mt, mc, js, jt, jc = result.x
    Vb, heel = boat_equilibrium(TWA, TWS, ms, mt, mc, js, jt, jc)
    VMG = Vb * np.cos(np.radians(TWA))
    return {
        "main_sheet": ms, "main_twist": mt, "main_camber": mc,
        "jib_sheet": js,  "jib_twist": jt,  "jib_camber": jc,
        "Vb": Vb, "heel": heel, "VMG": VMG
    }
# ------------------------------------------------------------------
# STREAMLIT INTERFACE
# ------------------------------------------------------------------
st.set_page_config(page_title="IOM Upwind VMG Optimizer", layout="centered")
st.title("⛵ IOM Sail Trim – Close‑Hauled VMG Model")

st.sidebar.header("Wind")
TWS = st.sidebar.slider("True Wind Speed (m/s)", min_value=1.0, max_value=8.0, value=4.0, step=0.1)
TWA = st.sidebar.slider("True Wind Angle (° from bow)", min_value=25, max_value=90, value=40, step=1)

st.sidebar.header("Mainsail Trim")
main_sheet = st.sidebar.slider("Main Sheet Angle (°)", min_value=5.0, max_value=25.0, value=15.0, step=0.5)
main_twist = st.sidebar.slider("Main Twist (° foot→head)", 0.0, 10.0, 5.0, 0.5)
main_camber = st.sidebar.slider("Main Camber fraction", 0.05, 0.2, 0.10, 0.005)

st.sidebar.header("Jib Trim")
jib_sheet = st.sidebar.slider("Jib Sheet Angle (°)", 5.0, 25.0, 12.0, 0.5)
jib_twist = st.sidebar.slider("Jib Twist (° foot→head)", 0.0, 10.0, 4.0, 0.5)
jib_camber = st.sidebar.slider("Jib Camber fraction", 0.05, 0.2, 0.10, 0.005)

# ------------------------------------------------------------------
# CALCULATION
# ------------------------------------------------------------------
Vb, heel = boat_equilibrium(TWA, TWS,
                            main_sheet, main_twist, main_camber,
                            jib_sheet,  jib_twist,  jib_camber)

VMG = Vb * np.cos(np.radians(TWA))

st.subheader("Upwind Performance Estimate")
st.metric("Boat Speed", f"{Vb:.2f} m/s")
st.metric("Heel Angle", f"{heel:.1f}°")
st.metric("VMG (TWA {TWA}°)", f"{VMG:.2f} m/s")

# quick VMG tip
if TWA < 35:
    st.info("TWA below 35° = likely pinching; VMG decreases.")
elif TWA > 50:
    st.info("TWA above 50° = sailing too free for best VMG upwind.")
# ------------------------------------------------------------------
# AUTOMATIC OPTIMISATION (find best VMG)
# ------------------------------------------------------------------
st.subheader("Automatic Optimisation")
if st.button("Optimise Trim for Max VMG"):
    with st.spinner("Computing... this may take 5‑10 seconds"):
        opt = optimise_trim_for_vmg(TWA, TWS)

    st.success(f"Best VMG for TWS {TWS:.1f} m/s at TWA {TWA}°")
    st.write(f"Main Sheet = {opt['main_sheet']:.1f}°")
    st.write(f"Main Twist = {opt['main_twist']:.1f}°")
    st.write(f"Main Camber = {opt['main_camber']:.3f}")
    st.write(f"Jib Sheet = {opt['jib_sheet']:.1f}°")
    st.write(f"Jib Twist = {opt['jib_twist']:.1f}°")
    st.write(f"Jib Camber = {opt['jib_camber']:.3f}")
    st.metric("Optimised Boat Speed", f"{opt['Vb']:.2f} m/s")
    st.metric("Heel Angle", f"{opt['heel']:.1f}°")
    st.metric("VMG", f"{opt['VMG']:.2f} m/s")
    
# ------------------------------------------------------------------
# POLAR PLOT
# ------------------------------------------------------------------
if st.button("Generate Upwind Polar"):
    angles = np.arange(25, 91, 2)
    vmgs = []
    for ang in angles:
        Vb_a, _ = boat_equilibrium(ang, TWS,
                                   main_sheet, main_twist, main_camber,
                                   jib_sheet,  jib_twist,  jib_camber)
        vmgs.append(Vb_a * np.cos(np.radians(ang)))

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
    ax.plot(np.radians(angles), vmgs, color='navy')
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)
    ax.set_title(f"VMG Polar – TWS {TWS:.1f} m/s")
    st.pyplot(fig)

st.caption("Prototype model for IOM trim sensitivity – close‑hauled VMG experiment.")
