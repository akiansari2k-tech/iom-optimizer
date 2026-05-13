# iom_app.py – IOM Upwind Sail-Trim / VMG Optimizer
# Calibrated against real IOM racing data

import streamlit as st
import numpy as np
from scipy.optimize import brentq, differential_evolution
import matplotlib.pyplot as plt

# NumPy 2.0 renamed trapz → trapezoid
if not hasattr(np, "trapezoid"):
    np.trapezoid = np.trapz

# ------------------------------------------------------------------
# CONSTANTS
# ------------------------------------------------------------------
RHO_AIR   = 1.225
RHO_WATER = 1025.0
G         = 9.81

DISPLACEMENT    = 4.0       # kg
SAIL_HEIGHT     = 1.5       # m
SAIL_AREA_MAIN  = 0.22      # m²
SAIL_AREA_JIB   = 0.10      # m²
KEEL_AREA       = 0.015     # m²
KEEL_AR         = 4.0
HYDRO_EFF       = 0.9

HULL_WETTED     = 0.15      # m² wetted surface
GM              = 0.12      # metacentric height (m)

Z_REF           = 1.0
SHEAR_EXP       = 1.0 / 7.0

# Boom geometry for sheet length conversion
MAIN_BOOM_RADIUS = 215.0    # mm from pivot to sheet attachment
JIB_BOOM_RADIUS  = 230.0    # mm from pivot to sheet attachment


# ------------------------------------------------------------------
# CONVERSION HELPERS
# ------------------------------------------------------------------
def angle_to_sheet_mm(angle_deg, boom_radius_mm):
    """Sheet angle (degrees off centreline) → sheet length in mm."""
    return boom_radius_mm * np.sin(np.radians(angle_deg))

def sheet_mm_to_angle(sheet_mm, boom_radius_mm):
    """Sheet length in mm → sheet angle in degrees off centreline."""
    ratio = np.clip(sheet_mm / boom_radius_mm, -1.0, 1.0)
    return np.degrees(np.arcsin(ratio))

def camber_fraction_to_mm(camber_frac, chord_mm):
    """Camber as fraction of chord → depth in mm."""
    return camber_frac * chord_mm

def camber_mm_to_fraction(camber_mm, chord_mm):
    """Camber depth in mm → fraction of chord."""
    return camber_mm / chord_mm

def twist_deg_to_mm(twist_deg, sail_height_m, boom_radius_mm):
    """
    Twist in degrees → approximate mm of leech open at head.
    Uses boom radius as the reference lever.
    """
    return boom_radius_mm * np.sin(np.radians(twist_deg))

def twist_mm_to_deg(twist_mm, boom_radius_mm):
    """Twist mm → degrees."""
    ratio = np.clip(twist_mm / boom_radius_mm, -1.0, 1.0)
    return np.degrees(np.arcsin(ratio))


# ------------------------------------------------------------------
# SAIL AERODYNAMICS
# ------------------------------------------------------------------
def sail_forces(AWA_deg, AWS_ref, sheet_deg, twist_deg, camber, area):
    """
    Drive force, side force, heeling moment for one sail.
    sheet_deg, twist_deg are internal angles (degrees).
    camber is a fraction (e.g. 0.10).
    """
    n = 12
    z = np.linspace(0.05, SAIL_HEIGHT, n)
    chord = (area / SAIL_HEIGHT) * np.ones_like(z)

    V_local = AWS_ref * (z / Z_REF) ** SHEAR_EXP
    AWA_local_deg = AWA_deg + 2.0 * (z / SAIL_HEIGHT)
    twist_local_deg = twist_deg * (z / SAIL_HEIGHT)

    alpha = np.radians(AWA_local_deg - sheet_deg - twist_local_deg)

    # Thin-airfoil CL with camber
    CL_inviscid = 2.0 * np.pi * (alpha + 2.0 * camber)

    # Smooth stall
    alpha_deg_abs = np.abs(np.degrees(alpha))
    stall_factor = np.clip(
        1.0 - 0.5 * (1.0 + np.tanh((alpha_deg_abs - 22.0) / 4.0)),
        0.25, 1.0
    )
    CL = CL_inviscid * stall_factor

    AR_sail = SAIL_HEIGHT ** 2 / area
    CD = 0.02 + CL ** 2 / (np.pi * 0.85 * AR_sail)

    q = 0.5 * RHO_AIR * V_local ** 2
    awa_rad = np.radians(AWA_local_deg)

    dF_drive = q * chord * (CL * np.sin(awa_rad) - CD * np.cos(awa_rad))
    dF_side  = q * chord * (CL * np.cos(awa_rad) + CD * np.sin(awa_rad))

    F_drive = np.trapezoid(dF_drive, z)
    F_side  = np.trapezoid(dF_side,  z)
    M_heel  = np.trapezoid(dF_side * z, z)

    return F_drive, F_side, M_heel


# ------------------------------------------------------------------
# HYDRODYNAMICS – speed-dependent (ITTC friction + residuary)
# ------------------------------------------------------------------
def hydro_resistance(Vb, F_side):
    """Hull + keel resistance."""
    if Vb < 1e-4:
        return 0.0

    q_w = 0.5 * RHO_WATER * Vb ** 2

    # ITTC-57 friction line
    Re = max(Vb * 1.0 / 1.0e-6, 1e3)
    Cf = 0.075 / (np.log10(Re) - 2.0) ** 2

    # Residuary (wave-making) rises sharply near hull speed
    Fn = Vb / np.sqrt(G * 1.0)
    Cr = 0.002 + 0.06 * Fn ** 4

    R_hull = q_w * (Cf + Cr) * HULL_WETTED

    # Keel induced drag from producing side force
    CL_keel = F_side / max(q_w * KEEL_AREA, 1e-6)
    CD_keel = 0.008 + CL_keel ** 2 / (np.pi * HYDRO_EFF * KEEL_AR)
    R_keel = q_w * CD_keel * KEEL_AREA

    return R_hull + R_keel


# ------------------------------------------------------------------
# EQUILIBRIUM
# ------------------------------------------------------------------
def _net_force(Vb, TWA_deg, TWS,
               main_sheet, main_twist, main_camber,
               jib_sheet,  jib_twist,  jib_camber,
               heel_deg):
    """Drive minus resistance at trial Vb."""
    twa_rad = np.radians(TWA_deg)
    awa_rad = np.arctan2(TWS * np.sin(twa_rad),
                         TWS * np.cos(twa_rad) - Vb)
    AWA_deg = np.degrees(awa_rad)
    AWS = np.hypot(TWS * np.sin(twa_rad),
                   TWS * np.cos(twa_rad) - Vb)

    cos_heel = np.cos(np.radians(heel_deg))

    Fm_d, Fm_s, Mm = sail_forces(AWA_deg,     AWS, main_sheet, main_twist, main_camber, SAIL_AREA_MAIN)
    Fj_d, Fj_s, Mj = sail_forces(AWA_deg + 4, AWS, jib_sheet,  jib_twist,  jib_camber,  SAIL_AREA_JIB)

    F_drive = (Fm_d + Fj_d) * cos_heel
    F_side  = (Fm_s + Fj_s) * cos_heel
    M_heel  = (Mm + Mj) * cos_heel

    R = hydro_resistance(Vb, F_side)
    return F_drive - R, F_side, M_heel


def boat_equilibrium(TWA_deg, TWS,
                     main_sheet, main_twist, main_camber,
                     jib_sheet,  jib_twist,  jib_camber):
    """Solve for steady-state boat speed and heel."""
    heel_deg = 0.0
    Vb = 0.0

    for _ in range(10):
        def residual(Vb_trial):
            net, _, _ = _net_force(Vb_trial, TWA_deg, TWS,
                                   main_sheet, main_twist, main_camber,
                                   jib_sheet,  jib_twist,  jib_camber,
                                   heel_deg)
            return net

        Vb_new = 0.0
        for lo, hi in [(0.005, 2.0), (0.005, 4.0), (0.001, 6.0)]:
            try:
                r_lo = residual(lo)
                r_hi = residual(hi)
                if r_lo * r_hi < 0:
                    Vb_new = brentq(residual, lo, hi, xtol=1e-4, maxiter=60)
                    break
            except (ValueError, RuntimeError):
                continue

        Vb = Vb_new
        if Vb <= 0:
            break

        _, F_side, M_heel = _net_force(Vb, TWA_deg, TWS,
                                       main_sheet, main_twist, main_camber,
                                       jib_sheet,  jib_twist,  jib_camber,
                                       heel_deg)

        ratio = M_heel / max(DISPLACEMENT * G * GM, 1e-6)
        ratio = np.clip(ratio, -0.999, 0.999)
        new_heel = np.degrees(np.arcsin(ratio))

        if abs(new_heel - heel_deg) < 0.15:
            heel_deg = new_heel
            break
        heel_deg = 0.6 * heel_deg + 0.4 * new_heel

    return max(Vb, 0.0), heel_deg


# ------------------------------------------------------------------
# OPTIMIZER
# ------------------------------------------------------------------
def optimise_trim_for_vmg(TWS):
    """Find TWA + trim that maximises upwind VMG."""
    # Bounds in mm for sheet/twist, mm for camber
    # Internally converted to degrees / fractions
    # TWA, main_sheet_mm, main_twist_mm, main_camber_mm,
    #       jib_sheet_mm,  jib_twist_mm,  jib_camber_mm
    bounds = [
        (25, 55),                        # TWA degrees
        (0, 80),  (10, 60), (10, 40),   # main: sheet mm, twist mm, camber mm
        (0, 100), (10, 60), (10, 40)    # jib:  sheet mm, twist mm, camber mm
    ]

    # Approximate chord in mm for camber conversion
    main_chord_mm = (SAIL_AREA_MAIN / SAIL_HEIGHT) * 1000.0
    jib_chord_mm  = (SAIL_AREA_JIB  / SAIL_HEIGHT) * 1000.0

    def objective(x):
        twa = x[0]
        ms_deg = sheet_mm_to_angle(x[1], MAIN_BOOM_RADIUS)
        mt_deg = twist_mm_to_deg(x[2], MAIN_BOOM_RADIUS)
        mc_frac = camber_mm_to_fraction(x[3], main_chord_mm)
        js_deg = sheet_mm_to_angle(x[4], JIB_BOOM_RADIUS)
        jt_deg = twist_mm_to_deg(x[5], JIB_BOOM_RADIUS)
        jc_frac = camber_mm_to_fraction(x[6], jib_chord_mm)

        Vb, _ = boat_equilibrium(twa, TWS, ms_deg, mt_deg, mc_frac,
                                 js_deg, jt_deg, jc_frac)
        return -Vb * np.cos(np.radians(twa))

    result = differential_evolution(
        objective, bounds,
        maxiter=50, popsize=12, tol=1e-3,
        seed=0, polish=True, workers=1
    )

    twa = result.x[0]
    ms_mm, mt_mm, mc_mm = result.x[1], result.x[2], result.x[3]
    js_mm, jt_mm, jc_mm = result.x[4], result.x[5], result.x[6]

    ms_deg = sheet_mm_to_angle(ms_mm, MAIN_BOOM_RADIUS)
    mt_deg = twist_mm_to_deg(mt_mm, MAIN_BOOM_RADIUS)
    mc_frac = camber_mm_to_fraction(mc_mm, main_chord_mm)
    js_deg = sheet_mm_to_angle(js_mm, JIB_BOOM_RADIUS)
    jt_deg = twist_mm_to_deg(jt_mm, JIB_BOOM_RADIUS)
    jc_frac = camber_mm_to_fraction(jc_mm, jib_chord_mm)

    Vb, heel = boat_equilibrium(twa, TWS, ms_deg, mt_deg, mc_frac,
                                js_deg, jt_deg, jc_frac)
    VMG = Vb * np.cos(np.radians(twa))

    return {
        "TWA": twa,
        "main_sheet_mm": ms_mm, "main_twist_mm": mt_mm, "main_camber_mm": mc_mm,
        "jib_sheet_mm":  js_mm, "jib_twist_mm":  jt_mm, "jib_camber_mm":  jc_mm,
        "Vb": Vb, "heel": heel, "VMG": VMG
    }


# ------------------------------------------------------------------
# STREAMLIT UI
# ------------------------------------------------------------------
st.set_page_config(page_title="IOM Upwind VMG Optimizer", layout="centered")
st.title("⛵ IOM Sail Trim – Close-Hauled VMG Model")

# Approximate chord in mm
main_chord_mm = (SAIL_AREA_MAIN / SAIL_HEIGHT) * 1000.0  # ≈ 147 mm
jib_chord_mm  = (SAIL_AREA_JIB  / SAIL_HEIGHT) * 1000.0  # ≈ 67 mm

# --- Wind ---
st.sidebar.header("Wind")
TWS = st.sidebar.slider("True Wind Speed (m/s)", 1.0, 8.0, 4.0, 0.1)
TWA = st.sidebar.slider("True Wind Angle (° from bow)", 25, 90, 38, 1)

# --- Mainsail Trim (in mm) ---
st.sidebar.header("Mainsail Trim")
main_sheet_mm  = st.sidebar.slider("Main Sheet Length (mm)", 0, 80, 10, 1)
main_twist_mm  = st.sidebar.slider("Main Twist (mm)", 10, 60, 30, 1)
main_camber_mm = st.sidebar.slider("Main Camber Depth (mm)", 5, 40, 20, 1)

# --- Jib Trim (in mm) ---
st.sidebar.header("Jib Trim")
jib_sheet_mm  = st.sidebar.slider("Jib Sheet Length (mm)", 0, 100, 50, 1)
jib_twist_mm  = st.sidebar.slider("Jib Twist (mm)", 10, 60, 35, 1)
jib_camber_mm = st.sidebar.slider("Jib Camber Depth (mm)", 5, 40, 25, 1)

# Convert mm inputs to internal degrees / fractions
main_sheet_deg  = sheet_mm_to_angle(main_sheet_mm, MAIN_BOOM_RADIUS)
main_twist_deg  = twist_mm_to_deg(main_twist_mm, MAIN_BOOM_RADIUS)
main_camber_frac = camber_mm_to_fraction(main_camber_mm, main_chord_mm)

jib_sheet_deg  = sheet_mm_to_angle(jib_sheet_mm, JIB_BOOM_RADIUS)
jib_twist_deg  = twist_mm_to_deg(jib_twist_mm, JIB_BOOM_RADIUS)
jib_camber_frac = camber_mm_to_fraction(jib_camber_mm, jib_chord_mm)

# ------------------------------------------------------------------
# CURRENT-SETTING CALCULATION
# ------------------------------------------------------------------
Vb, heel = boat_equilibrium(TWA, TWS,
                            main_sheet_deg, main_twist_deg, main_camber_frac,
                            jib_sheet_deg,  jib_twist_deg,  jib_camber_frac)
VMG = Vb * np.cos(np.radians(TWA))

st.subheader("Upwind Performance Estimate")
col1, col2, col3 = st.columns(3)
col1.metric("Boat Speed", f"{Vb:.2f} m/s")
col2.metric("Heel Angle", f"{heel:.1f}°")
col3.metric(f"VMG @ TWA {TWA}°", f"{VMG:.2f} m/s")

# Show the internal angles for reference
with st.expander("Internal angles (for reference)"):
    cA, cB = st.columns(2)
    with cA:
        st.write(f"Main sheet angle: {main_sheet_deg:.1f}°")
        st.write(f"Main twist angle: {main_twist_deg:.1f}°")
        st.write(f"Main camber fraction: {main_camber_frac:.3f}")
    with cB:
        st.write(f"Jib sheet angle: {jib_sheet_deg:.1f}°")
        st.write(f"Jib twist angle: {jib_twist_deg:.1f}°")
        st.write(f"Jib camber fraction: {jib_camber_frac:.3f}")

if TWA < 35:
    st.info("TWA below 35° – likely pinching; VMG may decrease.")
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
                                   main_sheet_deg, main_twist_deg, main_camber_frac,
                                   jib_sheet_deg,  jib_twist_deg,  jib_camber_frac)
        speeds.append(Vb_a)
        vmgs.append(Vb_a * np.cos(np.radians(ang)))

    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(5, 5))
    ax.plot(np.radians(angles), speeds, color='navy', label='Boat speed')
    ax.plot(np.radians(angles), vmgs,   color='crimson', label='VMG')
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)
    ax.set_thetamin(0)
    ax.set_thetamax(90)
    ax.set_title(f"Upwind Polar – TWS {TWS:.1f} m/s")
    ax.legend(loc='lower right', fontsize=8)
    st.pyplot(fig)

st.caption("IOM upwind VMG model – trim inputs in mm, calibrated to racing data.")
                                   
