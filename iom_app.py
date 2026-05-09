# iom_app.py – IOM Upwind Sail-Trim / VMG Optimizer
# Streamlit Community Cloud single-file version

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution

# ------------------------------------------------------------------
# CONSTANTS - simplified IOM scale model
# ------------------------------------------------------------------
RHO_AIR = 1.225
RHO_WATER = 1025.0
G = 9.81

DISPLACEMENT = 4.0        # kg
SAIL_HEIGHT = 1.5         # m
SAIL_AREA_MAIN = 0.22     # m²
SAIL_AREA_JIB = 0.10      # m²
KEEL_AREA = 0.015         # m²
KEEL_AR = 4.0
HYDRO_EFF = 0.9
HEEL_STIFFNESS = 0.07     # m per rad


# ------------------------------------------------------------------
# SAIL FORCE MODEL
# ------------------------------------------------------------------
def sail_forces(AWA, AWS, sheet, twist, camber, area):
    """
    Simplified lift/drag model for one sail.
    AWA, sheet and twist are in degrees.
    AWS is in m/s.
    """

    if AWS <= 0:
        return 0.0, 0.0, 0.0

    z = np.linspace(0.05, SAIL_HEIGHT, 12)
    dz = np.gradient(z)
    chord = area / SAIL_HEIGHT

    # Wind shear
    z_ref = 1.0
    shear_exp = 1 / 7
    V_profile = AWS * (z / z_ref) ** shear_exp

    # Apparent wind opens slightly toward head
    AWA_profile = AWA + 2.0 * (z / SAIL_HEIGHT)

    # Twist opens sail progressively from foot to head
    twist_profile = twist * (z / SAIL_HEIGHT)

    # Local angle of attack
    alpha = np.radians(AWA_profile - sheet - twist_profile)

    # Simple cambered sail lift coefficient
    camber_factor = max(0.4, 1 - 4.0 * (camber - 0.10) ** 2)
    CL = 1.05 * alpha * camber_factor

    # Stall penalty
    stall_angle = np.radians(18)
    CL = np.where(np.abs(alpha) > stall_angle, CL * 0.5, CL)

    # Drag coefficient
    CD = 0.01 + 0.02 * CL ** 2

    q = 0.5 * RHO_AIR * V_profile ** 2

    L = np.sum(q * CL * chord * dz)
    D = np.sum(q * CD * chord * dz)

    awa_rad = np.radians(AWA)

    F_drive = L * np.sin(awa_rad) - D * np.cos(awa_rad)
    F_side = L * np.cos(awa_rad) + D * np.sin(awa_rad)
    M_heel = F_side * (SAIL_HEIGHT * 0.4)

    return F_drive, F_side, M_heel


# ------------------------------------------------------------------
# HYDRODYNAMIC MODEL
# ------------------------------------------------------------------
def hydro_forces(Vb, F_side):
    """
    Simplified hydrodynamic resistance and keel side-force drag.
    """

    Vb = max(Vb, 0.05)

    dynamic_pressure = 0.5 * RHO_WATER * KEEL_AREA * Vb ** 2
    CL_h = F_side / max(dynamic_pressure, 1e-6)

    induced_drag = CL_h ** 2 / (np.pi * HYDRO_EFF * KEEL_AR)
    CD_h = 0.01 + induced_drag

    keel_drag = 0.5 * RHO_WATER * CD_h * KEEL_AREA * Vb ** 2

    # Extra hull drag approximation
    hull_drag = 0.50 * Vb ** 2

    return keel_drag + hull_drag


# ------------------------------------------------------------------
# BOAT EQUILIBRIUM MODEL
# ------------------------------------------------------------------
def boat_equilibrium(
    TWA,
    TWS,
    main_sheet,
    main_twist,
    main_camber,
    jib_sheet,
    jib_twist,
    jib_camber,
):
    """
    Iterate boat speed until drive force roughly equals resistance.
    Returns boat speed and heel angle.
    """

    Vb = 0.8

    for _ in range(100):
        twa_rad = np.radians(TWA)

        apparent_x = TWS * np.cos(twa_rad) - Vb
        apparent_y = TWS * np.sin(twa_rad)

        AWA = np.degrees(np.arctan2(apparent_y, apparent_x))
        AWS = np.hypot(apparent_y, apparent_x)

        # Prevent odd negative/behind apparent wind behaviour
        AWA = max(1.0, min(120.0, AWA))

        # Jib sees slightly more open apparent wind due to slot effect
        Fm = sail_forces(
            AWA,
            AWS,
            main_sheet,
            main_twist,
            main_camber,
            SAIL_AREA_MAIN,
        )

        Fj = sail_forces(
            AWA + 4.0,
            AWS,
            jib_sheet,
            jib_twist,
            jib_camber,
            SAIL_AREA_JIB,
        )

        F_drive = Fm[0] + Fj[0]
        F_side = Fm[1] + Fj[1]
        M_heel = Fm[2] + Fj[2]

        R_hydro = hydro_forces(Vb, F_side)

        err = F_drive - R_hydro

        # Stable update
        Vb += 0.04 * err / max(abs(R_hydro), 1.0)
        Vb = max(0.0, min(Vb, 3.0))

        if abs(err) < 0.02:
            break

    righting_moment = DISPLACEMENT * G * HEEL_STIFFNESS
    ratio = M_heel / max(righting_moment, 1e-6)
    ratio = np.clip(ratio, -1.0, 1.0)

    heel = np.degrees(np.arcsin(ratio))

    return max(Vb, 0.0), heel


# ------------------------------------------------------------------
# OPTIMISER
# ------------------------------------------------------------------
def optimise_trim_for_vmg(TWS):
    """
    Finds best TWA and trim combination for upwind VMG.
    """

    bounds = [
        (30.0, 55.0),     # TWA
        (5.0, 25.0),      # main sheet
        (0.0, 10.0),      # main twist
        (0.05, 0.20),     # main camber
        (5.0, 25.0),      # jib sheet
        (0.0, 10.0),      # jib twist
        (0.05, 0.20),     # jib camber
    ]

    def objective(x):
        TWA, ms, mt, mc, js, jt, jc = x

        Vb, heel = boat_equilibrium(
            TWA,
            TWS,
            ms,
            mt,
            mc,
            js,
            jt,
            jc,
        )

        VMG = Vb * np.cos(np.radians(TWA))

        # Penalise excessive heel
        heel_penalty = max(0.0, abs(heel) - 25.0) * 0.02

        # Penalise dead/unstable results
        if Vb <= 0.01:
            return 999.0

        return -(VMG - heel_penalty)

    result = differential_evolution(
        objective,
        bounds,
        maxiter=40,
        popsize=10,
        tol=0.01,
        polish=True,
        workers=1,
    )

    TWA, ms, mt, mc, js, jt, jc = result.x

    Vb, heel = boat_equilibrium(
        TWA,
        TWS,
        ms,
        mt,
        mc,
        js,
        jt,
        jc,
    )

    VMG = Vb * np.cos(np.radians(TWA))

    return {
        "TWA": TWA,
        "main_sheet": ms,
        "main_twist": mt,
        "main_camber": mc,
        "jib_sheet": js,
        "jib_twist": jt,
        "jib_camber": jc,
        "Vb": Vb,
        "heel": heel,
        "VMG": VMG,
    }


# ------------------------------------------------------------------
# STREAMLIT INTERFACE
# ------------------------------------------------------------------
st.set_page_config(
    page_title="IOM Upwind VMG Optimizer",
    layout="centered",
)

st.title("⛵ IOM Sail Trim – Close-Hauled VMG Model")

st.sidebar.header("Wind")

TWS = st.sidebar.slider(
    "True Wind Speed (m/s)",
    min_value=1.0,
    max_value=8.0,
    value=4.0,
    step=0.1,
)

TWA = st.sidebar.slider(
    "True Wind Angle (° from bow)",
    min_value=25,
    max_value=90,
    value=40,
    step=1,
)

st.sidebar.header("Mainsail Trim")

main_sheet = st.sidebar.slider(
    "Main Sheet Angle (°)",
    min_value=5.0,
    max_value=25.0,
    value=15.0,
    step=0.5,
)

main_twist = st.sidebar.slider(
    "Main Twist (° foot to head)",
    min_value=0.0,
    max_value=10.0,
    value=5.0,
    step=0.5,
)

main_camber = st.sidebar.slider(
    "Main Camber Fraction",
    min_value=0.05,
    max_value=0.20,
    value=0.10,
    step=0.005,
)

st.sidebar.header("Jib Trim")

jib_sheet = st.sidebar.slider(
    "Jib Sheet Angle (°)",
    min_value=5.0,
    max_value=25.0,
    value=12.0,
    step=0.5,
)

jib_twist = st.sidebar.slider(
    "Jib Twist (° foot to head)",
    min_value=0.0,
    max_value=10.0,
    value=4.0,
    step=0.5,
)

jib_camber = st.sidebar.slider(
    "Jib Camber Fraction",
    min_value=0.05,
    max_value=0.20,
    value=0.10,
    step=0.005,
)


# ------------------------------------------------------------------
# CURRENT TRIM CALCULATION
# ------------------------------------------------------------------
Vb, heel = boat_equilibrium(
    TWA,
    TWS,
    main_sheet,
    main_twist,
    main_camber,
    jib_sheet,
    jib_twist,
    jib_camber,
)

VMG = Vb * np.cos(np.radians(TWA))

st.subheader("Current Trim Performance Estimate")

col1, col2, col3 = st.columns(3)

col1.metric("Boat Speed", f"{Vb:.2f} m/s")
col2.metric("Heel Angle", f"{heel:.1f}°")
col3.metric("VMG", f"{VMG:.2f} m/s")

if TWA < 35:
    st.info("TWA below 35° is likely pinching. VMG may reduce.")
elif TWA > 50:
    st.info("TWA above 50° is probably too free for best upwind VMG.")


# ------------------------------------------------------------------
# AUTOMATIC OPTIMISATION
# ------------------------------------------------------------------
st.subheader("Automatic Optimisation")

if st.button("Optimise for Max Upwind VMG"):
    with st.spinner("Searching best angle and trim..."):
        opt = optimise_trim_for_vmg(TWS)

    st.success(f"Best result for TWS {TWS:.1f} m/s")

    col1, col2, col3 = st.columns(3)
    col1.metric("Optimum TWA", f"{opt['TWA']:.1f}°")
    col2.metric("Boat Speed", f"{opt['Vb']:.2f} m/s")
    col3.metric("VMG", f"{opt['VMG']:.2f} m/s")

    st.write("### Recommended Trim")
    st.write(f"Main Sheet: **{opt['main_sheet']:.1f}°**")
    st.write(f"Main Twist: **{opt['main_twist']:.1f}°**")
    st.write(f"Main Camber: **{opt['main_camber']:.3f}**")
    st.write(f"Jib Sheet: **{opt['jib_sheet']:.1f}°**")
    st.write(f"Jib Twist: **{opt['jib_twist']:.1f}°**")
    st.write(f"Jib Camber: **{opt['jib_camber']:.3f}**")
    st.write(f"Heel Angle: **{opt['heel']:.1f}°**")


# ------------------------------------------------------------------
# POLAR PLOT
# ------------------------------------------------------------------
st.subheader("Upwind Polar")

if st.button("Generate Upwind Polar"):
    angles = np.arange(25, 91, 2)
    vmgs = []
    speeds = []

    for ang in angles:
        Vb_a, _ = boat_equilibrium(
            ang,
            TWS,
            main_sheet,
            main_twist,
            main_camber,
            jib_sheet,
            jib_twist,
            jib_camber,
        )

        speeds.append(Vb_a)
        vmgs.append(Vb_a * np.cos(np.radians(ang)))

    best_idx = int(np.argmax(vmgs))
    best_angle = angles[best_idx]
    best_vmg = vmgs[best_idx]

    fig, ax = plt.subplots(subplot_kw={"projection": "polar"})
    ax.plot(np.radians(angles), vmgs)
    ax.scatter(np.radians(best_angle), best_vmg)

    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_title(f"Upwind VMG Polar – TWS {TWS:.1f} m/s")

    st.pyplot(fig)

    st.write(
        f"Best VMG on this polar: **{best_vmg:.2f} m/s** "
        f"at **{best_angle}° TWA**."
    )


st.caption(
    "Prototype model for IOM trim sensitivity. "
    "Useful for experimentation, not yet a calibrated race prediction model."
    )
