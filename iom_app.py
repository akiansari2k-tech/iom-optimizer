# iom_app.py – International One Metre (IOM) Sail Trim Optimizer
# Streamlit app to optimise VMG by adjusting sheet, twist, and camber

import streamlit as st
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# ------------------------------------------------------------------
# Constants
# ------------------------------------------------------------------
RHO_AIR = 1.225
RHO_WATER = 1025.0
G = 9.81

DISPLACEMENT = 4.0      # kg
SAIL_HEIGHT = 1.5       # m
SAIL_AREA = 0.32        # m²
KEEL_AREA = 0.015       # m²
KEEL_AR = 4.0
HYDRO_EFF = 0.9
HEEL_STIFFNESS = 0.07   # m lever arm per rad (approx small‑angle stiffness)

# ------------------------------------------------------------------
# Aerodynamic model
# ------------------------------------------------------------------
def sail_forces(AWA, AWS, sheet, twist, camber):
    """Return drive, side, and heeling‑moment from sails."""
    z = np.linspace(0, SAIL_HEIGHT, 8)
    c = SAIL_AREA / SAIL_HEIGHT * np.ones_like(z)
    twist_prof = twist * (z / SAIL_HEIGHT)
    alpha = np.radians(AWA - sheet) - np.radians(twist_prof)

    CL = 6.0 * alpha * (1 - 4.0 * (camber - 0.1) ** 2)
    CD = 0.01 + 0.02 * CL ** 2

    q = 0.5 * RHO_AIR * AWS ** 2
    L = np.sum(q * CL * c * np.gradient(z))
    D = np.sum(q * CD * c * np.gradient(z))

    F_drive = L * np.sin(np.radians(AWA)) - D * np.cos(np.radians(AWA))
    F_side = L * np.cos(np.radians(AWA)) + D * np.sin(np.radians(AWA))
    M_heel = F_side * (SAIL_HEIGHT * 0.4)
    return F_drive, F_side, M_heel

# ------------------------------------------------------------------
# Hydrodynamic model
# ------------------------------------------------------------------
def hydro_forces(Vb, F_side):
    CL_hydro = F_side / max(0.5 * RHO_WATER * KEEL_AREA * Vb ** 2, 1e-6)
    CD_hydro = 0.01 + CL_hydro ** 2 / (np.pi * HYDRO_EFF * KEEL_AR)
    R_hull = 0.5 * RHO_WATER * CD_hydro * KEEL_AREA * Vb ** 2 + 0.1 * Vb ** 2
    return R_hull

# ------------------------------------------------------------------
# Force & moment balance
# ------------------------------------------------------------------
def boat_equilibrium(TWA, TWS, sheet, twist, camber):
    Vb = 1.0
    for _ in range(60):
        AWA = np.degrees(np.arctan2(
            TWS * np.sin(np.radians(TWA)),
            TWS * np.cos(np.radians(TWA)) - Vb
        ))
        AWS = np.hypot(TWS * np.sin(np.radians(TWA)),
                       TWS * np.cos(np.radians(TWA)) - Vb)

        F_drive, F_side, M_heel = sail_forces(AWA, AWS, sheet, twist, camber)
        R_hydro = hydro_forces(Vb, F_side)

        err = F_drive - R_hydro
        Vb += 0.1 * err / (abs(R_hydro) + 1e-6)
        if abs(err) < 0.01:
            break

   ratio = M_heel / max(DISPLACEMENT * G * HEEL_STIFFNESS, 1e-6)
ratio = max(-1.0, min(1.0, ratio))  # clamp between -1 and 1
heel = np.degrees(np.arcsin(ratio))

    return max(Vb, 0), heel

# ------------------------------------------------------------------
# VMG optimisation
# ------------------------------------------------------------------
def negative_VMG(trim, TWA, TWS):
    sheet, twist, camber = trim
    Vb, _ = boat_equilibrium(TWA, TWS, sheet, twist, camber)
    return -Vb * np.cos(np.radians(TWA))

def optimize_trim(TWA, TWS):
    bounds = [(5, 25), (0, 10), (0.05, 0.2)]
    res = minimize(negative_VMG, x0=[15, 5, 0.1],
                   args=(TWA, TWS), bounds=bounds,
                   method="L-BFGS-B")
    sheet, twist, camber = res.x
    Vb, heel = boat_equilibrium(TWA, TWS, sheet, twist, camber)
    VMG = Vb * np.cos(np.radians(TWA))
    return sheet, twist, camber, Vb, heel, VMG

# ------------------------------------------------------------------
# Streamlit interface
# ------------------------------------------------------------------
st.set_page_config(page_title="IOM Sail Trim Optimizer", layout="centered")
st.title("⛵ IOM Sail Trim Optimizer")

st.sidebar.header("Wind Conditions")
TWS = st.sidebar.slider("True Wind Speed (m/s)", min_value=1.0, max_value=6.0, value=4.0, step=0.1)
TWA = st.sidebar.slider("True Wind Angle (°)", min_value=30, max_value=180, value=45, step=1) 
st.sidebar.header("Trim Controls (manual test)")
sheet = st.sidebar.slider("Sheet angle (°)", min_value=5.0, max_value=25.0, value=15.0, step=0.5)
twist = st.sidebar.slider("Twist (° foot→head)", min_value=0.0, max_value=10.0, value=5.0, step=0.5)
camber = st.sidebar.slider("Camber fraction", min_value=0.05, max_value=0.2, value=0.1, step=0.005)

# Manual prediction
st.subheader("Manual Trim Prediction")
Vb, heel = boat_equilibrium(TWA, TWS, sheet, twist, camber)
VMG = Vb * np.cos(np.radians(TWA))
st.metric("Boat Speed", f"{Vb:.2f} m/s")
st.metric("Heel Angle", f"{heel:.1f}°")
st.metric("VMG", f"{VMG:.2f} m/s")

# Optimiser
st.subheader("Automatic Optimisation (Max VMG)")
if st.button("Run Optimiser"):
    s_opt, t_opt, c_opt, Vb_opt, heel_opt, VMG_opt = optimize_trim(TWA, TWS)
    st.success(f"Optimal trim for {TWS:.1f} m/s @ {TWA}° TWA")
    st.write(f"- Sheet = {s_opt:.1f}°")
    st.write(f"- Twist = {t_opt:.1f}°")
    st.write(f"- Camber = {c_opt:.3f}")
    st.write(f"Speed = {Vb_opt:.2f} m/s Heel = {heel_opt:.1f}° VMG = {VMG_opt:.2f} m/s")

# Polar plot
st.subheader("Generate Polar Diagram")
if st.button("Create Polar Plot"):
    TWAs = np.arange(30, 181, 5)
    speeds = []
    for angle in TWAs:
        _, _, _, Vb_ang, _, _ = optimize_trim(angle, TWS)
        speeds.append(Vb_ang)

    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
    ax.plot(np.radians(TWAs), speeds, color='navy')
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)
    ax.set_title(f"IOM Polar – TWS {TWS:.1f} m/s")
    st.pyplot(fig)
  
