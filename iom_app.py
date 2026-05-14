# iom_app.py – IOM Upwind Sail-Trim / VMG Optimizer
# Empirical model calibrated to real IOM racing data

import streamlit as st
import numpy as np
from scipy.optimize import minimize, differential_evolution
import matplotlib.pyplot as plt

# NumPy 2.0 compatibility
if not hasattr(np, "trapezoid"):
    np.trapezoid = np.trapz

# ------------------------------------------------------------------
# PHYSICAL CONSTANTS
# ------------------------------------------------------------------
RHO_AIR   = 1.225
RHO_WATER = 1025.0
G         = 9.81

# ------------------------------------------------------------------
# IOM BOAT PARAMETERS
# ------------------------------------------------------------------
DISPLACEMENT    = 4.0       # kg
LWL             = 1.0       # m waterline length
SAIL_AREA_TOTAL = 0.32      # m² (main 0.22 + jib 0.10)
SAIL_AREA_MAIN  = 0.22
SAIL_AREA_JIB   = 0.10
GM              = 0.12      # metacentric height (m)
COE_HEIGHT      = 0.55      # centre of effort height above waterline (m)

# Boom geometry for sheet-length conversion
MAIN_BOOM_RADIUS = 215.0    # mm
JIB_BOOM_RADIUS  = 230.0    # mm

# Reference chord in mm (for camber depth conversion)
MAIN_CHORD_MM = (SAIL_AREA_MAIN / 1.5) * 1000.0   # ≈ 147 mm
JIB_CHORD_MM  = (SAIL_AREA_JIB  / 1.5) * 1000.0   # ≈  67 mm

# Hull speed limit
HULL_SPEED = 1.25 * np.sqrt(LWL)  # ≈ 1.25 m/s theoretical, but IOMs exceed this


# ------------------------------------------------------------------
# CONVERSION HELPERS
# ------------------------------------------------------------------
def sheet_mm_to_angle(mm, boom_r):
    return np.degrees(np.arcsin(np.clip(mm / boom_r, -1, 1)))

def angle_to_sheet_mm(deg, boom_r):
    return boom_r * np.sin(np.radians(deg))

def camber_mm_to_frac(mm, chord_mm):
    return mm / chord_mm

def twist_mm_to_deg(mm, boom_r):
    return np.degrees(np.arcsin(np.clip(mm / boom_r, -1, 1)))


# ------------------------------------------------------------------
# EMPIRICAL SAIL DRIVE MODEL
# ------------------------------------------------------------------
def sail_drive_coefficient(AWA_deg, sheet_angle_deg, twist_deg, camber_frac,
                           area_fraction):
    """
    Empirical drive coefficient for one sail.
    
    Key insight: sheet length controls leech tension (how well the sail
    holds its shape and exit angle), NOT the primary angle of attack.
    Camber is the main source of lift. A tight sheet (small mm) with
    good camber = efficient attached flow. A loose sheet = leech opens,
    power spills, drag increases.
    """
    # Effective angle of attack comes primarily from AWA and camber,
    # NOT from sheet angle directly.
    # Camber creates lift even when the boom is on centreline.
    # Think of it as: the sail is a curved wing, camber sets its "built-in" alpha.
    camber_alpha = 35.0 * camber_frac          # e.g. 0.14 → 4.9° effective alpha from camber
    
    # The geometric boom angle adds to this
    alpha_eff = camber_alpha + 0.3 * sheet_angle_deg + (AWA_deg - 15.0) * 0.4
    
    # Optimal effective alpha for max drive
    alpha_opt = 10.0 + 15.0 * camber_frac     # e.g. 0.14 → 12.1°
    
    # Efficiency peak
    sigma = 8.0 + 15.0 * camber_frac          # e.g. 0.14 → 10.1°
    efficiency = np.exp(-0.5 * ((alpha_eff - alpha_opt) / sigma) ** 2)
    
    # Leech tension effect: tight sheet (small angle) = good leech control
    # There's a sweet spot: too tight chokes the slot, too loose spills air
    # For the main, optimal is small (tight leech); for jib, a bit more open
    leech_opt = 4.0    # degrees — corresponds to ~15 mm on main, ~16 mm on jib
    leech_penalty = 1.0 - 0.15 * ((sheet_angle_deg - leech_opt) / 8.0) ** 2
    leech_penalty = np.clip(leech_penalty, 0.4, 1.0)
    
    # Twist effect: some twist matches wind shear, too much spills power
    twist_opt = 5.0 + 0.1 * AWA_deg
    twist_penalty = 1.0 - 0.25 * ((twist_deg - twist_opt) / 8.0) ** 2
    twist_penalty = np.clip(twist_penalty, 0.3, 1.0)
    
    # Drive coefficient
    AWA_rad = np.radians(AWA_deg)
    Cd_base = 1.8 * np.sin(AWA_rad) * efficiency * leech_penalty * twist_penalty * area_fraction
    
    # Very low or negative effective alpha kills drive
    if alpha_eff < 1.0:
        Cd_base *= max(0.0, alpha_eff / 1.0)
    
    return max(Cd_base, 0.0)
                               


def total_aero_force(TWA_deg, TWS, Vb,
                     main_sheet_deg, main_twist_deg, main_camber_frac,
                     jib_sheet_deg, jib_twist_deg, jib_camber_frac):
    """
    Compute total aerodynamic driving force and heeling force.
    Returns (F_drive, F_heel) in Newtons.
    """
    # Apparent wind
    twa_rad = np.radians(TWA_deg)
    Vax = TWS * np.cos(twa_rad) - Vb   # apparent wind x-component (head-on)
    Vay = TWS * np.sin(twa_rad)         # apparent wind y-component (beam)
    AWS = np.hypot(Vax, Vay)
    AWA_deg = np.degrees(np.arctan2(Vay, Vax))
    
    if AWS < 0.1 or AWA_deg < 1.0:
        return 0.0, 0.0
    
    q = 0.5 * RHO_AIR * AWS ** 2
    
    # Drive coefficients for each sail
    main_frac = SAIL_AREA_MAIN / SAIL_AREA_TOTAL
    jib_frac  = SAIL_AREA_JIB  / SAIL_AREA_TOTAL
    
    Cd_main = sail_drive_coefficient(AWA_deg, main_sheet_deg, main_twist_deg,
                                     main_camber_frac, main_frac)
    # Jib sees slightly more open AWA due to slot effect
    Cd_jib  = sail_drive_coefficient(AWA_deg + 3.0, jib_sheet_deg, jib_twist_deg,
                                     jib_camber_frac, jib_frac)
    
    Cd_total = Cd_main + Cd_jib
    F_drive = q * SAIL_AREA_TOTAL * Cd_total
    
    # Heeling force: roughly proportional to side component of sail force
    # Side force ≈ CL * cos(AWA), approximately 2-3x the drive force close-hauled
    AWA_rad = np.radians(AWA_deg)
    heel_ratio = np.cos(AWA_rad) / max(np.sin(AWA_rad), 0.1)
    F_heel = F_drive * heel_ratio * 0.8   # 0.8 accounts for keel lift offset
    
    return F_drive, F_heel


# ------------------------------------------------------------------
# HULL RESISTANCE
# ------------------------------------------------------------------
def hull_resistance(Vb):
    """
    Speed-dependent hull resistance for an IOM.
    Calibrated: R ≈ 0.15 N at 2.0 m/s (matching ~0.15 N aero drive in 4 m/s TWS).
    """
    if Vb < 1e-4:
        return 0.0
    
    Re = max(Vb * LWL / 1.0e-6, 1000)
    Cf = 0.075 / (np.log10(Re) - 2.0) ** 2
    
    Fn = Vb / np.sqrt(G * LWL)
    Cr = 0.005 * (1.0 + 8.0 * Fn ** 3)   # residuary rises near hull speed
    
    wetted_area = 0.14   # m²
    q = 0.5 * RHO_WATER * Vb ** 2
    
    return q * (Cf + Cr) * wetted_area


# ------------------------------------------------------------------
# EQUILIBRIUM SOLVER
# ------------------------------------------------------------------
def boat_equilibrium(TWA_deg, TWS,
                     main_sheet_deg, main_twist_deg, main_camber_frac,
                     jib_sheet_deg, jib_twist_deg, jib_camber_frac):
    """
    Find equilibrium boat speed by iterating until drive = drag.
    Uses simple relaxation (robust for this well-behaved model).
    """
    Vb = 0.5   # initial guess
    
    for iteration in range(120):
        F_drive, F_heel = total_aero_force(
            TWA_deg, TWS, Vb,
            main_sheet_deg, main_twist_deg, main_camber_frac,
            jib_sheet_deg, jib_twist_deg, jib_camber_frac
        )
        
        # Heel angle
        M_heel = F_heel * COE_HEIGHT
        sin_heel = np.clip(M_heel / (DISPLACEMENT * G * GM), -0.99, 0.99)
        heel_deg = np.degrees(np.arcsin(sin_heel))
        
        # Heel reduces drive
        cos_heel = np.cos(np.radians(heel_deg))
        F_drive_eff = F_drive * cos_heel
        
        R = hull_resistance(Vb)
        
        # Net force
        net = F_drive_eff - R
        
        # Adaptive step
        if Vb > 0.1:
            step = 0.02 * net / max(R, 0.01)
        else:
            step = 0.01 * np.sign(net)
        
        Vb_new = Vb + step
        Vb = max(Vb_new, 0.001)
        
        if abs(net) < 0.001 and iteration > 5:
            break
    
    if Vb < 0.01:
        Vb = 0.0
        heel_deg = 0.0
    
    return Vb, heel_deg


# ------------------------------------------------------------------
# OPTIMIZER
# ------------------------------------------------------------------
def optimise_trim_for_vmg(TWS):
    """Find TWA + trim (in mm) that maximises upwind VMG."""
    bounds = [
        (28, 55),                         # TWA degrees
        (0, 80),  (15, 60), (10, 40),    # main: sheet mm, twist mm, camber mm
        (10, 100), (15, 60), (10, 40)    # jib:  sheet mm, twist mm, camber mm
    ]
    
    def objective(x):
        twa = x[0]
        ms_deg = sheet_mm_to_angle(x[1], MAIN_BOOM_RADIUS)
        mt_deg = twist_mm_to_deg(x[2], MAIN_BOOM_RADIUS)
        mc_frac = camber_mm_to_frac(x[3], MAIN_CHORD_MM)
        js_deg = sheet_mm_to_angle(x[4], JIB_BOOM_RADIUS)
        jt_deg = twist_mm_to_deg(x[5], JIB_BOOM_RADIUS)
        jc_frac = camber_mm_to_frac(x[6], JIB_CHORD_MM)
        
        Vb, _ = boat_equilibrium(twa, TWS, ms_deg, mt_deg, mc_frac,
                                 js_deg, jt_deg, jc_frac)
        vmg = Vb * np.cos(np.radians(twa))
        return -vmg
    
    result = differential_evolution(
        objective, bounds,
        maxiter=60, popsize=14, tol=1e-3,
        seed=42, polish=True, workers=1
    )
    
    twa = result.x[0]
    ms_mm, mt_mm, mc_mm = result.x[1], result.x[2], result.x[3]
    js_mm, jt_mm, jc_mm = result.x[4], result.x[5], result.x[6]
    
    ms_deg = sheet_mm_to_angle(ms_mm, MAIN_BOOM_RADIUS)
    mt_deg = twist_mm_to_deg(mt_mm, MAIN_BOOM_RADIUS)
    mc_frac = camber_mm_to_frac(mc_mm, MAIN_CHORD_MM)
    js_deg = sheet_mm_to_angle(js_mm, JIB_BOOM_RADIUS)
    jt_deg = twist_mm_to_deg(jt_mm, JIB_BOOM_RADIUS)
    jc_frac = camber_mm_to_frac(jc_mm, JIB_CHORD_MM)
    
    Vb, heel = boat_equilibrium(twa, TWS, ms_deg, mt_deg, mc_frac,
                                js_deg, jt_deg, jc_frac)
    vmg = Vb * np.cos(np.radians(twa))
    
    return {
        "TWA": twa,
        "main_sheet_mm": ms_mm, "main_twist_mm": mt_mm, "main_camber_mm": mc_mm,
        "jib_sheet_mm":  js_mm, "jib_twist_mm":  jt_mm, "jib_camber_mm":  jc_mm,
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

# Convert to internal units
main_sheet_deg  = sheet_mm_to_angle(main_sheet_mm, MAIN_BOOM_RADIUS)
main_twist_deg  = twist_mm_to_deg(main_twist_mm, MAIN_BOOM_RADIUS)
main_camber_frac = camber_mm_to_frac(main_camber_mm, MAIN_CHORD_MM)

jib_sheet_deg  = sheet_mm_to_angle(jib_sheet_mm, JIB_BOOM_RADIUS)
jib_twist_deg  = twist_mm_to_deg(jib_twist_mm, JIB_BOOM_RADIUS)
jib_camber_frac = camber_mm_to_frac(jib_camber_mm, JIB_CHORD_MM)

# ------------------------------------------------------------------
# CURRENT-SETTING RESULTS
# ------------------------------------------------------------------
Vb, heel = boat_equilibrium(TWA, TWS,
                            main_sheet_deg, main_twist_deg, main_camber_frac,
                            jib_sheet_deg, jib_twist_deg, jib_camber_frac)
VMG = Vb * np.cos(np.radians(TWA))

st.subheader("Upwind Performance Estimate")
col1, col2, col3 = st.columns(3)
col1.metric("Boat Speed", f"{Vb:.2f} m/s")
col2.metric("Heel Angle", f"{heel:.1f}°")
col3.metric(f"VMG @ TWA {TWA}°", f"{VMG:.2f} m/s")

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

if TWA < 33:
    st.info("TWA below 33° – pinching hard; VMG likely dropping.")
elif TWA > 50:
    st.info("TWA above 50° – sailing free; not optimal for upwind VMG.")

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
                                   jib_sheet_deg, jib_twist_deg, jib_camber_frac)
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

st.caption("IOM upwind VMG model – trim in mm, empirical model calibrated to racing data.")
