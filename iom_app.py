# iom_app.py – IOM Upwind Sail-Trim / VMG Optimizer
# Empirical model calibrated to real IOM racing data
# All trim inputs in mm (sheet length, twist, camber depth)

import streamlit as st
import numpy as np
from scipy.optimize import differential_evolution
import matplotlib.pyplot as plt

# NumPy 2.0 compatibility
if not hasattr(np, "trapezoid"):
    np.trapezoid = np.trapz

# ------------------------------------------------------------------
# IOM BOAT PARAMETERS
# ------------------------------------------------------------------
RHO_AIR   = 1.225
RHO_WATER = 1025.0
G         = 9.81

DISPLACEMENT    = 4.0       # kg
LWL             = 1.0       # m waterline length
SAIL_AREA_TOTAL = 0.32      # m² (main 0.22 + jib 0.10)
SAIL_AREA_MAIN  = 0.22
SAIL_AREA_JIB   = 0.10
GM              = 0.12      # metacentric height (m)
COE_HEIGHT      = 0.55      # centre of effort height above waterline (m)

MAIN_BOOM_RADIUS = 215.0    # mm
JIB_BOOM_RADIUS  = 230.0    # mm

MAIN_CHORD_MM = (SAIL_AREA_MAIN / 1.5) * 1000.0   # ≈ 147 mm
JIB_CHORD_MM  = (SAIL_AREA_JIB  / 1.5) * 1000.0   # ≈  67 mm


# ------------------------------------------------------------------
# SAIL DRIVE MODEL (works directly in mm)
# ------------------------------------------------------------------
def sail_drive_coefficient(AWA_deg, sheet_mm, twist_mm, camber_mm,
                           chord_mm, boom_radius_mm, area_fraction,
                           is_main=True):
    """
    Empirical drive coefficient for one sail.
    All trim inputs in mm — no angle conversion needed.

    Sheet mm: controls leech tension. Main optimum ~10 mm, jib ~50 mm.
    Camber mm: depth of sail curvature — main source of lift.
    Twist mm: how much the leech opens at the head.
    """
    # --- Camber-based lift ---
    # Camber fraction drives the "built-in" angle of attack
    camber_frac = camber_mm / chord_mm
    camber_alpha = 35.0 * camber_frac       # e.g. 20/147 = 0.136 → 4.8°

    # AWA contribution to effective alpha
    alpha_eff = camber_alpha + (AWA_deg - 15.0) * 0.4

    # Efficiency peak
    alpha_opt = 10.0 + 15.0 * camber_frac
    sigma_alpha = 8.0 + 15.0 * camber_frac
    efficiency = np.exp(-0.5 * ((alpha_eff - alpha_opt) / sigma_alpha) ** 2)

    # --- Sheet length / leech tension (in mm directly) ---
    if is_main:
        sheet_opt = 10.0      # mm — your father's preferred setting
        sheet_sigma = 8.0     # mm — tight peak: 5-20 mm is good
    else:
        sheet_opt = 50.0      # mm — jib needs more room in the slot
        sheet_sigma = 18.0    # mm — broader: 30-70 mm is workable

    sheet_offset = sheet_mm - sheet_opt

    # Asymmetric: easing beyond optimum is worse than overtightening
    if sheet_offset > 0:
        # Eased: power spills off leech, drag increases
        sheet_penalty = np.exp(-0.5 * (sheet_offset / sheet_sigma) ** 2)
    else:
        # Tight: slightly reduces flow but less harmful
        tight_sigma = sheet_sigma * 1.8
        sheet_penalty = np.exp(-0.5 * (sheet_offset / tight_sigma) ** 2)

    sheet_penalty = max(sheet_penalty, 0.10)

    # --- Twist (in mm) ---
    # Convert twist mm to approximate degrees for the aero calculation
    twist_deg = np.degrees(np.arcsin(np.clip(twist_mm / boom_radius_mm, -1, 1)))
    twist_opt = 5.0 + 0.1 * AWA_deg
    twist_penalty = 1.0 - 0.25 * ((twist_deg - twist_opt) / 8.0) ** 2
    twist_penalty = np.clip(twist_penalty, 0.3, 1.0)

    # --- Combined drive coefficient ---
    AWA_rad = np.radians(AWA_deg)
    Cd = 1.6 * np.sin(AWA_rad) * efficiency * sheet_penalty * twist_penalty * area_fraction

    # Very low effective alpha kills drive
    if alpha_eff < 1.0:
        Cd *= max(0.0, alpha_eff)

    return max(Cd, 0.0)


# ------------------------------------------------------------------
# TOTAL AERO FORCE
# ------------------------------------------------------------------
def total_aero_force(TWA_deg, TWS, Vb,
                     main_sheet_mm, main_twist_mm, main_camber_mm,
                     jib_sheet_mm, jib_twist_mm, jib_camber_mm):
    """Total driving force and heeling force in Newtons."""
    twa_rad = np.radians(TWA_deg)
    Vax = TWS * np.cos(twa_rad) - Vb
    Vay = TWS * np.sin(twa_rad)
    AWS = np.hypot(Vax, Vay)
    AWA_deg = np.degrees(np.arctan2(Vay, Vax))

    if AWS < 0.1 or AWA_deg < 1.0:
        return 0.0, 0.0

    q = 0.5 * RHO_AIR * AWS ** 2

    main_frac = SAIL_AREA_MAIN / SAIL_AREA_TOTAL
    jib_frac  = SAIL_AREA_JIB  / SAIL_AREA_TOTAL

    Cd_main = sail_drive_coefficient(
        AWA_deg, main_sheet_mm, main_twist_mm, main_camber_mm,
        MAIN_CHORD_MM, MAIN_BOOM_RADIUS, main_frac, is_main=True
    )
    Cd_jib = sail_drive_coefficient(
        AWA_deg + 3.0, jib_sheet_mm, jib_twist_mm, jib_camber_mm,
        JIB_CHORD_MM, JIB_BOOM_RADIUS, jib_frac, is_main=False
    )

    Cd_total = Cd_main + Cd_jib
    F_drive = q * SAIL_AREA_TOTAL * Cd_total

    AWA_rad = np.radians(AWA_deg)
    heel_ratio = np.cos(AWA_rad) / max(np.sin(AWA_rad), 0.1)
    F_heel = F_drive * heel_ratio * 0.8

    return F_drive, F_heel


# ------------------------------------------------------------------
# HULL RESISTANCE
# ------------------------------------------------------------------
def hull_resistance(Vb):
    """Speed-dependent hull resistance for an IOM."""
    if Vb < 1e-4:
        return 0.0

    Re = max(Vb * LWL / 1.0e-6, 1000)
    Cf = 0.075 / (np.log10(Re) - 2.0) ** 2

    Fn = Vb / np.sqrt(G * LWL)
    Cr = 0.006 * (1.0 + 8.0 * Fn ** 3)

    wetted_area = 0.14
    q = 0.5 * RHO_WATER * Vb ** 2

    return q * (Cf + Cr) * wetted_area


# ------------------------------------------------------------------
# EQUILIBRIUM SOLVER
# ------------------------------------------------------------------
def boat_equilibrium(TWA_deg, TWS,
                     main_sheet_mm, main_twist_mm, main_camber_mm,
                     jib_sheet_mm, jib_twist_mm, jib_camber_mm):
    """Find equilibrium boat speed by relaxation."""
    Vb = 0.5

    for iteration in range(150):
        F_drive, F_heel = total_aero_force(
            TWA_deg, TWS, Vb,
            main_sheet_mm, main_twist_mm, main_camber_mm,
            jib_sheet_mm, jib_twist_mm, jib_camber_mm
        )

        M_heel = F_heel * COE_HEIGHT
        sin_heel = np.clip(M_heel / (DISPLACEMENT * G * GM), -0.99, 0.99)
        heel_deg = np.degrees(np.arcsin(sin_heel))

        cos_heel = np.cos(np.radians(heel_deg))
        F_drive_eff = F_drive * cos_heel

        R = hull_resistance(Vb)
        net = F_drive_eff - R

        if Vb > 0.1:
            step = 0.015 * net / max(R, 0.01)
        else:
            step = 0.005 * np.sign(net)

        Vb = max(Vb + step, 0.001)

        if abs(net) < 0.0005 and iteration > 10:
            break

    if Vb < 0.02:
        return 0.0, 0.0

    return Vb, heel_deg


# ------------------------------------------------------------------
# OPTIMIZER
# ------------------------------------------------------------------
def optimise_trim_for_vmg(TWS):
    """Find TWA + trim (all in mm) that maximises upwind VMG."""
    bounds = [
        (28, 50),                         # TWA degrees
        (0, 80),  (15, 60), (10, 40),    # main: sheet mm, twist mm, camber mm
        (15, 100), (15, 60), (10, 40)    # jib:  sheet mm, twist mm, camber mm
    ]

    def objective(x):
        twa = x[0]
        Vb, _ = boat_equilibrium(twa, TWS, x[1], x[2], x[3], x[4], x[5], x[6])
        return -Vb * np.cos(np.radians(twa))

    result = differential_evolution(
        objective, bounds,
        maxiter=60, popsize=14, tol=1e-3,
        seed=42, polish=True, workers=1
    )

    twa = result.x[0]
    ms, mt, mc = result.x[1], result.x[2], result.x[3]
    js, jt, jc = result.x[4], result.x[5], result.x[6]

    Vb, heel = boat_equilibrium(twa, TWS, ms, mt, mc, js, jt, jc)
    vmg = Vb * np.cos(np.radians(twa))

    return {
        "TWA": twa,
        "main_sheet_mm": ms, "main_twist_mm": mt, "main_camber_mm": mc,
        "jib_sheet_mm":  js, "jib_twist_mm":  jt, "jib_camber_mm":  jc,
        "Vb": Vb, "heel": heel, "VMG": vmg
    }


# ------------------------------------------------------------------
# STREAMLIT UI
# ------------------------------------------------------------------
st.set_page_config(page_title="IOM Upwind VMG Optimizer", layout="centered")
st.title("⛵ IOM Sail Trim – Close-Hauled VMG Model")

# --- Wind ---
st.sidebar.header("Wind")
TWS = st.sidebar.slider("True Wind Speed (m/s)", 1.0, 8.0, 4.0, 0.1)
TWA = st.sidebar.slider("True Wind Angle (° from bow)", 25, 90, 38, 1)

# --- Mainsail Trim (mm) ---
st.sidebar.header("Mainsail Trim")
main_sheet_mm  = st.sidebar.slider("Main Sheet Length (mm)", 0, 80, 10, 1)
main_twist_mm  = st.sidebar.slider("Main Twist (mm)", 10, 60, 30, 1)
main_camber_mm = st.sidebar.slider("Main Camber Depth (mm)", 5, 40, 20, 1)

# --- Jib Trim (mm) ---
st.sidebar.header("Jib Trim")
jib_sheet_mm  = st.sidebar.slider("Jib Sheet Length (mm)", 0, 100, 50, 1)
jib_twist_mm  = st.sidebar.slider("Jib Twist (mm)", 10, 60, 35, 1)
jib_camber_mm = st.sidebar.slider("Jib Camber Depth (mm)", 5, 40, 25, 1)

# ------------------------------------------------------------------
# CURRENT-SETTING RESULTS
# ------------------------------------------------------------------
Vb, heel = boat_equilibrium(TWA, TWS,
                            main_sheet_mm, main_twist_mm, main_camber_mm,
                            jib_sheet_mm, jib_twist_mm, jib_camber_mm)
VMG = Vb * np.cos(np.radians(TWA))

st.subheader("Upwind Performance Estimate")
col1, col2, col3 = st.columns(3)
col1.metric("Boat Speed", f"{Vb:.2f} m/s")
col2.metric("Heel Angle", f"{heel:.1f}°")
col3.metric(f"VMG @ TWA {TWA}°", f"{VMG:.2f} m/s")

if TWA < 33:
    st.info("TWA below 33° – pinching hard; VMG likely dropping.")
elif TWA > 50:
    st.info("TWA above 50° – sailing too free for best upwind VMG.")

# ------------------------------------------------------------------
# AUTOMATIC OPTIMISATION
# ------------------------------------------------------------------
st.subheader("Automatic Optimisation (Best Upwind Angle + Trim)")
if st.button("Optimise for Max VMG"):
    with st.spinner("Searching best angle and trim..."):
        opt = optimise_trim_for_vmg(TWS)

    st.success(f"Best upwind VMG for TWS {TWS:.1f} m/s")
    st.write(f"**Optimum TWA = {opt['TWA']:.1f}°**")

    cA, cB = st.columns(2)
    with cA:
        st.markdown("**Mainsail**")
        st.write(f"Sheet:  {opt['main_sheet_mm']:.0f} mm")
        st.write(f"Twist:  {opt['main_twist_mm']:.0f} mm")
        st.write(f"Camber: {opt['main_camber_mm']:.0f} mm")
    with cB:
        st.markdown("**Jib**")
        st.write(f"Sheet:  {opt['jib_sheet_mm']:.0f} mm")
        st.write(f"Twist:  {opt['jib_twist_mm']:.0f} mm")
        st.write(f"Camber: {opt['jib_camber_mm']:.0f} mm")

    c1, c2, c3 = st.columns(3)
    c1.metric("Boat Speed", f"{opt['Vb']:.2f} m/s")
    c2.metric("Heel Angle", f"{opt['heel']:.1f}°")
    c3.metric("VMG",        f"{opt['VMG']:.2f} m/s")

# ------------------------------------------------------------------
# POLAR PLOT
# ------------------------------------------------------------------
if st.button("Generate Upwind Polar"):
    angles = np.arange(25, 91, 2)
    speeds, vmgs = [], []
    for ang in angles:
        Vb_a, _ = boat_equilibrium(ang, TWS,
                                   main_sheet_mm, main_twist_mm, main_camber_mm,
                                   jib_sheet_mm, jib_twist_mm, jib_camber_mm)
        speeds.append(Vb_a)
        vmgs.append(Vb_a * np.cos(np.radians(ang)))

    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(5, 5))
    ax.plot(np.radians(angles), speeds, color='navy', label='Boat speed')
    ax.plot(np.radians(angles), vmgs, color='crimson', label='VMG')
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)
    ax.set_thetamin(0)
    ax.set_thetamax(90)
    ax.set_title(f"Upwind Polar – TWS {TWS:.1f} m/s")
    ax.legend(loc='lower right', fontsize=8)
    st.pyplot(fig)

st.caption("IOM upwind VMG model – all trim in mm, empirical model calibrated to racing data.")
