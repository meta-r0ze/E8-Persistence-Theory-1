#!python3

import math
import argparse
import sys

# ==========================================
# 0. EXTERNAL REFERENCE VALUES (Experimental Truth)
# ==========================================
# Sources: CODATA 2022 & PDG 2024

REFS = {
    "me_mev":       0.51099895000,
    "me_mev_print": 0.51099,
    "alpha_inv":  137.035999177,
    "alpha_inv_err": 0.000000085,

    "alpha_s":    0.1179,
    "alpha_s_err": 0.0009,

    "sin2_w":     0.22290, # On-Shell
    "sin2_w_err": 0.00010,

    "vev":        246.22,
    "gf":         1.1663787e-5,
    "mh":         125.25,
    "lambda":     0.129,

    "G_coupling": 1.752e-45,
    "Mp":         1.22091e19
}

PI = math.pi

# ==========================================
# 1. HELPER FUNCTIONS
# ==========================================

def to_latex_sci(num, precision=4):
    """Converts a float to LaTeX scientific notation."""
    if num == 0: return "0"
    exponent = int(math.floor(math.log10(abs(num))))
    mantissa = num / (10**exponent)
    if exponent == 0:
        return f"{num:.{precision}f}"
    return f"{mantissa:.{precision}f} \\times 10^{{{exponent}}}"

def print_section(title, latex_mode=False):
    if latex_mode: return
    print(f"\n{'#'*70}")
    print(f"  {title}")
    print(f"{'#'*70}\n")

def print_derivation(name, tag, formula_sym, latex_sym, formula_num, result,
                     latex_mode=False, ref_key=None, unit="", context="observed value",
                     formula_step=None):

    # --- LATEX OUTPUT MODE ---
    if latex_mode:
        if formula_step is not None:
            val_step_str = f"{result:.9f}" if abs(formula_step) > 0.001 and abs(formula_step) < 1000 else to_latex_sci(formula_step, 5)
            print(f"%<*{tag}StepVal>{val_step_str}%</{tag}StepVal>")

        # 1. Value Tag
        val_str = f"{result:.9f}" if abs(result) > 0.001 and abs(result) < 1000 else to_latex_sci(result, 5)
        print(f"%<*{tag}Val>{val_str}%</{tag}Val>")

        # 2. Formula Tag
        print(f"%<*{tag}Eq>{latex_sym}%</{tag}Eq>")

        # 3. Accuracy/Diff Tags (if ref exists)
        if ref_key and ref_key in REFS:
            target = REFS[ref_key]
            diff = result - target

            if target != 0:
                match_pct = (1 - abs(diff/target)) * 100
            else:
                match_pct = 0

            # Experimental Value
            print(f"%<*{tag}ExperimentalValue>{to_latex_sci(target, 5)}%</{tag}ExperimentalValue>")

            # Accuracy Percentage Tag
            print(f"%<*{tag}Acc>{match_pct:.4f}\\%%</{tag}Acc>")

            # Full Accuracy Sentence Tag
            acc_text = f"The geometric prediction captures {match_pct:.4f}\\% of the {context}."
            print(f"%<*{tag}AccText>{acc_text}%</{tag}AccText>")

            # Diff Tag
            print(f"%<*{tag}Diff>{to_latex_sci(diff, 3)}%</{tag}Diff>")

        print("") # Spacer in tex file
        return

    # --- HUMAN READABLE MODE ---
    print(f"--- {name} ---")
    print(f"Formula:  {formula_sym}")
    print(f"Filled:   {formula_num}")
    print(f"Calculated: {result:.12g} {unit}")

    # LaTeX Snippet hint
    latex_str = to_latex_sci(result, 5) if abs(result) < 0.001 or abs(result) > 1000 else f"{result:.5f}"
    print(f"LaTeX Copy: \\mathbf{{{latex_str}}} {unit}")

    if ref_key and ref_key in REFS:
        target = REFS[ref_key]
        diff = result - target

        if target != 0:
            match_pct = (1 - abs(diff/target)) * 100
        else:
            match_pct = 0

        err_key = f"{ref_key}_err"
        sigma_msg = ""
        if err_key in REFS:
            sigma = abs(diff / REFS[err_key])
            sigma_msg = f" ({sigma:.1f}σ)"

        print(f"Target:     {target:.12g} {unit}")
        print(f"Accuracy:   {match_pct:.6f}%{sigma_msg}")

    print("")

def main():
    parser = argparse.ArgumentParser(description="Calculate E8 Persistence Constants")
    parser.add_argument('--latex', action='store_true', help='Output in catchfilebetweentags format')
    args = parser.parse_args()

    LATEX_MODE = args.latex
    
    # ==========================================
    # 2. Experimental
    # ==========================================
    if LATEX_MODE:
        # Output basic experimental values as tags too if needed
        print(f"%<*MeMeV>{REFS['me_mev']}%</MeMeVPrint>")
        print(f"%<*MeMeVPrint>{REFS['me_mev_print']}%</MeMeVPrint>")
        print("")

    # ==========================================
    # 2. SYSTEM I: INVARIANTS
    # ==========================================
    print_section("SYSTEM I: THE INVARIANT SUBSTRATE", LATEX_MODE)

    D     = 4
    DELTA = 43
    SIGMA = 5
    NU    = 16
    CHI   = 2

    if not LATEX_MODE:
        print(f"Invariants: S = {{ D={D}, Delta={DELTA}, Sigma={SIGMA}, Nu={NU}, Chi={CHI} }}")

    # Derived Capacities
    H_SYS = NU + SIGMA + CHI
    H_FULL = H_SYS + (2 * D)
    N = 2 * NU

    if LATEX_MODE:
        # Output basic invariants as tags too if needed
        print(f"%<*InvHSys>{H_SYS}%</InvHSys>")
        print(f"%<*InvHFull>{H_FULL}%</InvHFull>")
        print(f"%<*InvN>{N}%</InvN>")
    elif not LATEX_MODE:
        print(f"Capacities: H_sys={H_SYS}, H_full={H_FULL}, N={N}")

    # ==========================================
    # 3. SYSTEM II: THE VACUUM IMPEDANCE
    # ==========================================
    print_section("SYSTEM II: THE GEOMETRIC IMPEDANCE (Table II Audit)", LATEX_MODE)

    # 1. Energy Vessel (Circumference)
    comp_DE = PI * DELTA

    # 2. Info Model (Boundary)
    comp_DI = CHI

    # 3. Protocol (Alignment)
    comp_MI = -1 / ((D * DELTA) - SIGMA)

    # 4. Governor (Vacuum Pressure)
    comp_G = -(CHI / DELTA)

    # 5. Temporal Tax (Entropy)
    comp_T = (1 / pow(N, 3)) * (CHI / SIGMA) * (1 - (SIGMA / (D * DELTA)))

    # 6. Persistence Margin (Resolution)
    comp_PM = 1 / (H_FULL * (SIGMA + 1) * pow(DELTA, 2))

    # Summation
    ALPHA_INV_GEO = comp_DE + comp_DI + comp_MI + comp_G + comp_T + comp_PM
    ALPHA_GEO = 1.0 / ALPHA_INV_GEO

    # Table breakdown for human mode
    if not LATEX_MODE:
        print(f"{'COMPONENT':<25} | {'FORMULA':<25} | {'VALUE':<15}")
        print("-" * 70)
        print(f"{'Energy Vessel':<25} | {'π * Δ':<25} | {comp_DE:+.8f}")
        print(f"{'Info Model':<25} | {'χ':<25} | {comp_DI:+.8f}")
        print(f"{'Protocol':<25} | {'-1/(DΔ - σ)':<25} | {comp_MI:+.8f}")
        print(f"{'Governor':<25} | {'-χ/Δ':<25} | {comp_G:+.8f}")
        print(f"{'Temporal Tax':<25} | {'Eq 16a':<25} | {comp_T:+.8e}")
        print(f"{'Persistence Margin':<25} | {'Eq 16b':<25} | {comp_PM:+.8e}")
        print("-" * 70)
        print(f"{'TOTAL IMPEDANCE':<25} | {'SUM':<25} | {ALPHA_INV_GEO:.9f}")
        print("-" * 70)
        print("")
    else:
        # Export components for Table II generation
        print(f"%<*CompDE>{comp_DE:.5f}%</CompDE>")
        print(f"%<*CompDI>{comp_DI:.5f}%</CompDI>")
        print(f"%<*CompMI>{comp_MI:.5f}%</CompMI>")
        print(f"%<*CompG>{comp_G:.5f}%</CompG>")
        print(f"%<*CompT>{to_latex_sci(comp_T)}%</CompT>")
        print(f"%<*CompPM>{to_latex_sci(comp_PM)}%</CompPM>")
        print("")

    print_derivation(
        name="Fine Structure Constant Inverse",
        tag="AlphaInv",
        formula_sym="Sum(Components)",
        latex_sym=r"\pi\Delta + \chi - \frac{1}{D\Delta - \sigma} - \frac{\chi}{\Delta} + T + PM",
        formula_num="See Table",
        result=ALPHA_INV_GEO,
        latex_mode=LATEX_MODE,
        ref_key="alpha_inv"
    )

    # ==========================================
    # 4. SYSTEM III: EFFECTIVE FIELD LIMITS
    # ==========================================
    print_section("SYSTEM III: THE EFFECTIVE FIELD LIMITS", LATEX_MODE)

    # --- Strong Coupling ---
    numerator_s = NU + (1.0 / D)
    ALPHA_S_GEO = numerator_s / ALPHA_INV_GEO

    print_derivation(
        name="Strong Coupling (α_s)",
        tag="AlphaS",
        formula_sym="(ν + 1/D) / α⁻¹",
        latex_sym=r"\frac{\nu + 1/D}{\alpha^{-1}}",
        formula_num=f"({NU} + 0.25) / {ALPHA_INV_GEO:.4f}",
        result=ALPHA_S_GEO,
        latex_mode=LATEX_MODE,
        ref_key="alpha_s",
        context="PDG World Average"
    )

    # --- Weak Mixing Angle ---
    denom_weak = (D * DELTA) + NU + SIGMA
    SIN2_THETA_W_GEO = DELTA / denom_weak

    print_derivation(
        name="Weak Mixing Angle (sin²θ_W)",
        tag="WeakAngle",
        formula_sym="Δ / (DΔ + ν + σ)",
        latex_sym=r"\frac{\Delta}{D\Delta + \nu + \sigma}",
        formula_num=f"{DELTA} / {denom_weak}",
        result=SIN2_THETA_W_GEO,
        latex_mode=LATEX_MODE,
        ref_key="sin2_w",
        context="On-Shell definition"
    )

    # --- Higgs VEV ---
    I_S = (D * DELTA) + NU

    # 1. Tree Level (Bare Geometric Floor)
    V_MEV_BARE = ((CHI * pow(DELTA, 2)) - I_S) * ALPHA_INV_GEO * REFS["me_mev"]

    # 2. Radiative Correction (Manifold Polarization)
    # Correction = 1 + (alpha / D)
    POLARIZATION = 1.0 + (ALPHA_GEO / D)

    # 3. Physical VEV
    V_MEV_PHYS = V_MEV_BARE * POLARIZATION
    V_GEV_PHYS = V_MEV_PHYS / 1000.0

    print_derivation(
        name="Higgs VEV (v)",
        tag="HiggsVEV",
        formula_sym="v_tree * (1 + α/D)",
        latex_sym=r"v_{geo} \left( 1 + \frac{\alpha}{D} \right)",
        formula_num=f"{V_MEV_BARE/1000.0:.2f} * {POLARIZATION:.6f}",
        result=V_GEV_PHYS,
        latex_mode=LATEX_MODE,
        ref_key="vev",
        unit="GeV",
        context="electroweak scale",
        formula_step=V_MEV_BARE
    )

    # --- Fermi Constant ---
    GF_GEO = 1.0 / (math.sqrt(CHI) * pow(V_GEV_PHYS, 2))

    print_derivation(
        name="Fermi Constant (G_F)",
        tag="FermiConst",
        formula_sym="1 / (√χ * v_phys²)",
        latex_sym=r"\frac{1}{\sqrt{\chi} v_{phys}^2}",
        formula_num=f"1 / (√{CHI} * {V_GEV_PHYS:.2f}²)",
        result=GF_GEO,
        latex_mode=LATEX_MODE,
        ref_key="gf",
        unit="GeV^-2",
        context="experimental value"
    )

    # --- Higgs Parameters ---
    LAMBDA_GEO = (SIGMA - CHI) / H_SYS
    MH_GEO = math.sqrt(2 * LAMBDA_GEO) * V_GEV_PHYS

    print_derivation(
        name="Higgs Self-Coupling (λ)",
        tag="HiggsLambda",
        formula_sym="(σ - χ) / H_sys",
        latex_sym=r"\frac{\sigma - \chi}{H_{sys}}",
        formula_num=f"({SIGMA} - {CHI}) / {H_SYS}",
        result=LAMBDA_GEO,
        latex_mode=LATEX_MODE,
        ref_key="lambda",
        context="experimental central value"
    )

    print_derivation(
        name="Higgs Mass (m_H)",
        tag="HiggsMass",
        formula_sym="√(2λ) * v",
        latex_sym=r"\sqrt{2\lambda} v",
        formula_num=f"√(2 * {LAMBDA_GEO:.4f}) * {V_GEV_PHYS:.2f}",
        result=MH_GEO,
        latex_mode=LATEX_MODE,
        ref_key="mh",
        unit="GeV",
        context="observed mass"
    )

    # --- Electron Yukawa ---
    # Compare calculated PM against Standard Model expectation (sqrt(2)*me/v)
    YE_SM_EXPECTED = (math.sqrt(2) * (REFS["me_mev"]/1000.0)) / REFS["vev"]

    # We cheat slightly on the "ref_key" here to force the comparison logic to work with a calculated target
    REFS["ye_sm"] = YE_SM_EXPECTED

    print_derivation(
        name="Electron Yukawa (y_e)",
        tag="ElectronYukawa",
        formula_sym="PM (Persistence Margin)",
        latex_sym=r"\text{PM}_{geo}",
        formula_num=f"{comp_PM:.4e}",
        result=comp_PM,
        latex_mode=LATEX_MODE,
        ref_key="ye_sm",
        context="Standard Model expectation"
    )

    # ==========================================
    # 5. GRAVITY & PLANCK MASS
    # ==========================================
    print_section("GRAVITY & HIERARCHY", LATEX_MODE)

    # --- Residual Capacity ---
    B_RES = NU - (CHI / (SIGMA - CHI)) - ALPHA_GEO

    print_derivation(
        name="Residual Capacity (B_res)",
        tag="ResidualCap",
        formula_sym="ν - χ/(σ-χ) - α",
        latex_sym=r"\nu - \frac{\chi}{\sigma-\chi} - \alpha",
        formula_num=f"{NU} - {CHI/3:.4f} - {ALPHA_GEO:.4e}",
        result=B_RES,
        latex_mode=LATEX_MODE
    )

    # --- Gravitational Coupling ---
    EXP_G = DELTA / 2.0
    ALPHA_G_GEO = B_RES * pow(ALPHA_GEO, EXP_G)

    print_derivation(
        name="Gravitational Coupling (α_G)",
        tag="GravCoupling",
        formula_sym="B_res * α^(Δ/2)",
        latex_sym=r"B_{res} \alpha^{\Delta/2}",
        formula_num=f"{B_RES:.4f} * α^{EXP_G}",
        result=ALPHA_G_GEO,
        latex_mode=LATEX_MODE,
        ref_key="G_coupling",
        context="dimensionless coupling"
    )

    # --- Planck Mass ---
    MP_MEV_GEO = REFS["me_mev"] / math.sqrt(ALPHA_G_GEO)
    MP_GEV_GEO = MP_MEV_GEO / 1000.0

    print_derivation(
        name="Planck Mass (M_P)",
        tag="PlanckMass",
        formula_sym="m_e / √α_G",
        latex_sym=r"\frac{m_e}{\sqrt{\alpha_G}}",
        formula_num=f"m_e / √{ALPHA_G_GEO:.4e}",
        result=MP_GEV_GEO,
        latex_mode=LATEX_MODE,
        ref_key="Mp",
        unit="GeV",
        context="hierarchy scale"
    )

if __name__ == "__main__":
    main()
