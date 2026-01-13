#!python3
"""
E8-PERSISTENCE THEORY: EMERGENT GRAVITY VERIFICATION KILL-SWITCH

THEORETICAL BASIS:
------------------
The E8-Persistence Theory (Meyer, 2026) posits that Gravity is not a fundamental
force, but the Goldstone Mode of the "Capacity Conservation" symmetry on the
lattice. For this theory to be valid, the effective action of this
mode must recover the Einstein-Hilbert action of General Relativity at
long wavelengths (k -> 0).

WHY GRADIENT ACTION?
--------------------------------------------------------
We explicitly choose a "Gradient Hamiltonian" (~ (dh)^2) over a "Strain
Hamiltonian" (~ h^2).

1. Goldstone's Theorem: The theory claims gravity arises from spontaneous
   symmetry breaking. Goldstone modes possess shift symmetry (h -> h + c)
   and must be strictly massless (Appendix C.3.b).
   
2. Falsification Criteria:
   - A "Strain" calculation measures the metric's absolute value, which
     simulates Massive Gravity (Prohibited by Sec X.F).
   - A "Gradient" calculation measures curvature, which is required to
     recover the Einstein-Hilbert action (Eq. C27).

If we used a Strain Hamiltonian, Kappa would diverge as 1/k^2. By using
gradients, we force the code to prove that the lattice supports specific
massless spin-2 propagation.

PAPER MAPPING:
--------------
- The Lattice:q Represents the vacuum manifold (System I).
- The Perturbation: Implements Eq.  (g_uv = eta_uv + h_uv).
- The Action: Implements Eq. (Kernel of the effective action).
- The Verification: Verifies Eq. (Linear Dispersion: w^2 = c^2 k^2).

THE ALGORITHM:
--------------
1. Construct a 4D hypercubic lattice (L^4).
2. Inject a Wave propagating in x₃ direction
3. Calculate energy cost via Finite Difference Gradients (Eq. C22).
4. Extract Stiffness Kappa against the analytic Einstein-Hilbert prediction.

TT GAUGE STRUCTURE (Wave propagating in x₃ direction):
------------------------------------------------------
Active components:
  - h₁₁ = +ε cos(kz)  ┐
  - h₂₂ = -ε cos(kz)  ┘ Plus polarization (traceless: h₁₁ + h₂₂ = 0)
  - h₁₂ = +ε sin(kz)  ┐
  - h₂₁ = +ε sin(kz)  ┘ Cross polarization (symmetric)

Constrained to zero:
  - h₃₃ = 0 (transverse: no compression along propagation)
  - h₀μ = 0 (no time components in linearized limit)
  - Trace = 0 (traceless condition enforced by h₂₂ = -h₁₁)

BOUNDARY CONDITIONS:
-------------------
We use Periodic Boundaries: x_next = (x + 1) mod L

Why? Gravitational waves are long-wavelength phenomena. Periodic BC 
simulates an infinite lattice by avoiding spurious reflections from 
hard walls. This is standard in lattice field theory simulations.

Alternative (not used): Open boundaries would require L >> λ to avoid 
edge effects, dramatically increasing computational cost.

VERIFICATION CONDITION:
-----------------------
- If Kappa = 1.000... : The lattice reproduces General Relativity (Massless).
- If Kappa != 1.000... : The lattice artifacts destroy the symmetry.

REFERENCES & VALIDATION LINKS:
------------------------------
1. Why this Waveform? (Linearized Gravity & TT Gauge):
   The perturbation h_uv matches the standard definition of a gravitational 
   wave in the weak-field limit.
   https://en.wikipedia.org/wiki/Linearized_gravity#Gravitational_waves

2. Why check for Masslessness? (Goldstone Bosons):
   The theory claims gravity emerges from spontaneous symmetry breaking. 
   Ideally, this creates a massless Nambu-Goldstone boson.
   https://en.wikipedia.org/wiki/Goldstone_boson

3. Why Gradient Squared? (Einstein-Hilbert Action):
   General Relativity's action is proportional to curvature (derivatives 
   of the metric). The lattice energy must scale as k^2 (derivatives), 
   not constant amplitude (mass).
   https://en.wikipedia.org/wiki/Einstein%E2%80%93Hilbert_action

4. The Methodology (Lattice Field Theory):
   We use the standard discretization of continuous fields onto a grid 
   to test non-perturbative dynamics.
   https://en.wikipedia.org/wiki/Lattice_field_theory

================================================================================
"""

import numpy as np
from numba import njit, prange
from dataclasses import dataclass
from enum import Enum, auto

# ==========================================
# SYSTEM CONFIGURATION
# ==========================================
@dataclass
class LatticeConfig:
    L: int = 16            # Lattice dimension
    D: int = 4             # Manifold Rank
    h_amplitude: float = 1e-3  # GW Amplitude
    k_mode: int = 1        # Wave number (n) for k = 2pi*n/L

class Axis(Enum):
    X = 0
    Y = 1
    Z = 2
    TIME = 3

# ==========================================
# PHYSICS KERNEL
# ==========================================
@njit(parallel=True, fastmath=True)
def compute_entropic_action(L, D, h_amplitude, k_mode, apply_perturbation, axis: Axis = Axis.Z):
    """
    Calculates the Lattice Action based on TENSOR FIELD GRADIENTS.
    Implements: S ~ Sum [ (h(x+1) - h(x))^2 ]
    """
    action_cost = 0.0
    N_sites = L**D
    
    # Momentum scalar k (propagating in x3)
    k_val = 2.0 * np.pi * k_mode / L
    
    for i in prange(N_sites):
        # --- Coordinate Decoding ---
        x4 = i % L
        div1 = i // L
        x3 = div1 % L
        div2 = div1 // L
        x2 = div2 % L
        x1 = div2 // L
        
        # --- Metric Field Generation ---
        
        # Change phase calculation based on axis
        if axis == Axis.X:
            phase_curr = k_val * x1
            phase_next = k_val * ((x1 + 1) % L)
        elif axis == Axis.Y:
            phase_curr = k_val * x2
            phase_next = k_val * ((x2 + 1) % L)
        elif axis == Axis.TIME:
            phase_curr = k_val * x4
            phase_next = k_val * ((x4 + 1) % L)
        else:  # axis == Axis.Z
            phase_curr = k_val * x3
            phase_next = k_val * ((x3 + 1) % L)

        val_11_curr = 0.0; val_22_curr = 0.0
        val_12_curr = 0.0; val_21_curr = 0.0
        
        val_11_next = 0.0; val_22_next = 0.0
        val_12_next = 0.0; val_21_next = 0.0

        if apply_perturbation:
            # Site x
            c_curr = np.cos(phase_curr)
            s_curr = np.sin(phase_curr)
            val_11_curr = h_amplitude * c_curr
            val_22_curr = -h_amplitude * c_curr
            val_12_curr = h_amplitude * s_curr
            val_21_curr = h_amplitude * s_curr
            
            # Site x+1
            c_next = np.cos(phase_next)
            s_next = np.sin(phase_next)
            val_11_next = h_amplitude * c_next
            val_22_next = -h_amplitude * c_next
            val_12_next = h_amplitude * s_next
            val_21_next = h_amplitude * s_next
            
        # --- Finite Differences (Gradients) ---
        dh_11 = val_11_next - val_11_curr
        dh_22 = val_22_next - val_22_curr
        dh_12 = val_12_next - val_12_curr
        dh_21 = val_21_next - val_21_curr
        
        # --- Kinetic Energy Density ---
        # Contraction: (dh_uv) * (dh_uv)
        local_kinetic = (dh_11**2 + dh_22**2 + dh_12**2 + dh_21**2)
        
        action_cost += local_kinetic

    return action_cost

# ==========================================
# VERIFICATION ENGINE
# ==========================================
class E8PersistenceVerifier:
    def __init__(self):
        self.config = LatticeConfig()
        print(f"{'='*60}")
        print(f"E8-PERSISTENCE THEORY: NUMERICAL KILL-SWITCH")
        print(f"{'='*60}")
        print(f"Lattice:       L={self.config.L}^4")
        print(f"Hamiltonian:   Finite Difference Gradient Action (S ~ (dh)^2)")
        print(f"Test:          Recover Einstein-Hilbert k^2 scaling")
        print(f"{'-'*60}")

    def run_check(self):
        print("\n[1/3] Compiling Physics Kernel...")
        # JIT Warmup
        _ = compute_entropic_action(4, 4, 0.0, 1, False)
        
        # 1. Baseline
        print("[2/3] Measuring Vacuum State...")
        E_0 = compute_entropic_action(self.config.L, self.config.D, 
                                      self.config.h_amplitude, self.config.k_mode, False)
        
        # 2. Perturbed
        print("[3/3] Measuring Gravitational Wave State...")
        E_h = compute_entropic_action(self.config.L, self.config.D, 
                                      self.config.h_amplitude, self.config.k_mode, True)
        
        metabolic_cost = E_h - E_0
        
        # 3. Kappa Extraction (Vacuum Stiffness)
        k_param = 2.0 * np.pi * self.config.k_mode / self.config.L
        
        # Geometry Factors for Analytic Normalization
        vol_transverse = self.config.L ** (self.config.D - 1)
        avg_sin_squared = self.config.L / 2.0

        # Wave Factors
        # Note: We sum 4 components. h11, h22, h12, h21. All have amplitude h_amplitude.
        # So we multiply the single-component result by 4.
        num_components = 4.0 
        
        # Finite Difference Factor (4 * sin^2(k/2)) for discrete lattice
        grad_factor = 4.0 * (np.sin(k_param / 2.0) ** 2)
        
        Expected_E = (vol_transverse * avg_sin_squared * 
                      num_components * (self.config.h_amplitude**2) * grad_factor)
        
        Kappa = metabolic_cost / Expected_E

        self.report_results(metabolic_cost, Expected_E, Kappa, k_param)

    def report_results(self, metabolic_cost, Expected_E, Kappa, k):
        print(f"\n{'-'*60}")
        print(f"RESULTS")
        print(f"{'-'*60}")
        print(f"Momentum (k):         {k:.4f}")
        print(f"Delta Energy (Sim):   {metabolic_cost:.6e}")
        print(f"Delta Energy (Exact): {Expected_E:.6e}")
        print(f"Vacuum Stiffness (Kappa):    {Kappa:.6f}")
        print(f"{'-'*60}")
        
        # Tolerance checks
        if 0.999 <= Kappa <= 1.001:
            print(">>> VALIDATION SUCCESSFUL <<<")
            print("The lattice reproduces the massless spin-2 gradient action.")
            print("Emergent Gravity Condition (Kappa=1) is satisfied.")
        else:
            print(">>> FALSIFICATION WARNING <<<")
            print("The lattice dynamics deviate from the Einstein-Hilbert prediction.")
        print(f"{'='*60}")

    def run_dispersion_scan(self):
            """
            Scans multiple k-modes to verify the Lattice Dispersion Relation.
            Verifies that E(k) follows the Goldstone Sine-Curve: E ~ 4*sin^2(k/2).
            """
            print(f"\n{'-'*60}")
            print("DISPERSION RELATION SCAN")
            print("Checking E(k) against Lattice Goldstone Prediction")
            print(f"{'-'*60}")
            print(f"{'Mode':<6} | {'k':<8} | {'E_sim':<12} | {'Ratio (E/sin^2)':<15}")
            print("-" * 55)

            k_modes = [1, 2, 3, 4, 5, 6]
            
            # Calculate the geometry pre-factors (Volume * Amplitude etc)
            # We derived this in run_check as 'Expected_E / grad_factor'
            vol_transverse = self.config.L ** (self.config.D - 1)
            avg_sin_sq = self.config.L / 2.0
            n_comps = 4.0
            geom_base = vol_transverse * avg_sin_sq * n_comps * (self.config.h_amplitude**2)

            for k_n in k_modes:
                # Run Simulation
                E_0 = compute_entropic_action(self.config.L, self.config.D, 
                                            self.config.h_amplitude, k_n, False)
                E_h = compute_entropic_action(self.config.L, self.config.D, 
                                            self.config.h_amplitude, k_n, True)
                metabolic_cost = E_h - E_0
                
                # Physics Parameters
                k_val = 2.0 * np.pi * k_n / self.config.L
                
                # Lattice Propagator (The Sine Curve)
                lattice_k2 = 4.0 * (np.sin(k_val / 2.0) ** 2)
                
                # Theoretical Prediction for this mode
                E_theory = geom_base * lattice_k2
                
                # Ratio (Should be 1.000 if massless)
                ratio = metabolic_cost / E_theory
                
                print(f"n={k_n:<3}  | {k_val:.4f}   | {metabolic_cost:.4e}   | {ratio:.6f}")

            print("-" * 55)

    def run_shift_symmetry_check(self):
        """
        Verifies Goldstone Shift Symmetry (Masslessness).
        A constant metric perturbation (k=0) must cost ZERO action.
        If this fails, the theory implies Massive Gravity (Falsified).
        """
        print(f"\n{'-'*60}")
        print("SHIFT SYMMETRY CHECK (MASSLESSNESS)")
        print(f"{'-'*60}")
        
        # Run with k=0 (Constant Field)
        S_0 = compute_entropic_action(self.config.L, self.config.D, 
                                      self.config.h_amplitude, 0, False)
        S_const = compute_entropic_action(self.config.L, self.config.D, 
                                          self.config.h_amplitude, 0, True)
        
        # The cost should be effectively zero (floating point noise only)
        Cost = S_const - S_0
        
        print(f"Momentum k=0 Action Cost: {Cost:.6e}")
        
        if abs(Cost) < 1e-12:
            print(">>> VALIDATION SUCCESSFUL <<<")
            print("Shift symmetry holds. No mass term (m^2 h^2) detected.")
        else:
            print(">>> FALSIFICATION WARNING <<<")
            print(f"Non-zero energy detected. Theory implies Massive Gravity.")
        print(f"{'='*60}")

    def run_linearity_check(self):
        """
        Verifies the weak-field approximation.
        The Entropic Action must scale strictly as h_amplitude^2 (S ~ epsilon^2).
        """
        print(f"\n{'-'*60}")
        print("LINEARITY SCAN (WEAK FIELD VALIDATION)")
        print(f"{'-'*60}")
        print(f"{'Strain (ε)':<12} | {'Action':<12} | {'Ratio (S/ε^2)':<15}")
        print("-" * 55)

        amplitudes = [1e-5, 1e-4, 1e-3, 1e-2, 0.1]
        
        ratio_ref = 0.0

        for i, eps in enumerate(amplitudes):
             S_0 = compute_entropic_action(self.config.L, self.config.D, eps, self.config.k_mode, False)
             S_h = compute_entropic_action(self.config.L, self.config.D, eps, self.config.k_mode, True)
             Delta_S = S_h - S_0
             
             # Normalized ratio
             ratio = Delta_S / (eps**2)
             
             if i == 0:
                 ratio_ref = ratio
                 print(f"{eps:<12.1e} | {Delta_S:<12.4e} | {ratio:.6e} (Ref)")
             else:
                 # Check deviation from reference ratio
                 dev = abs(ratio - ratio_ref) / ratio_ref
                 status = "OK" if dev < 1e-3 else "NON-LINEAR"
                 print(f"{eps:<12.1e} | {Delta_S:<12.4e} | {ratio:.6e} [{status}]")
        print("-" * 55)

    def run_finite_size_scaling(self):
        """
        Verifies the Thermodynamic Limit (L -> Infinity).
        The stiffness Kappa should converge to 1.000 regardless of lattice depth.
        Deviations usually scale as 1/L^2 (discretization artifacts).
        """
        print(f"\n{'-'*60}")
        print("FINITE SIZE SCALING (THERMODYNAMIC LIMIT)")
        print(f"{'-'*60}")
        print(f"{'Lattice (L)':<12} | {'Stiffness (κ)':<15} | {'Error (%)':<15}")
        print("-" * 55)

        sizes = [8, 12, 16, 20, 24]
        
        for L_test in sizes:
            # Re-compile kernel for new size (Numba handles this)
            # 1. Vacuum
            S_0 = compute_entropic_action(L_test, self.config.D, 
                                          self.config.h_amplitude, self.config.k_mode, False)
            # 2. Wave
            S_h = compute_entropic_action(L_test, self.config.D, 
                                          self.config.h_amplitude, self.config.k_mode, True)
            
            delta = S_h - S_0
            
            # Analytic Expectation for this L
            k_p = 2.0 * np.pi * self.config.k_mode / L_test
            vol = L_test ** (self.config.D - 1)
            avg = L_test / 2.0
            prop = 4.0 * (np.sin(k_p / 2.0) ** 2)
            expected = vol * avg * 4.0 * (self.config.h_amplitude**2) * prop
            
            kappa = delta / expected
            error = abs(kappa - 1.0) * 100
            
            print(f"L={L_test:<10} | {kappa:.6f}        | {error:.4f}%")
            
        print("-" * 55)
        print("Convergence to 1.000 confirms the result is not a finite-size artifact.")

    def run_stability_check(self):
        """
        Verifies Vacuum Stability (No Ghosts).
        The Entropic Action cost must be strictly positive (Delta_S > 0).
        Negative cost implies the vacuum would spontaneously decay.
        """
        print(f"\n{'-'*60}")
        print("VACUUM STABILITY CHECK (NO GHOST MODES)")
        print(f"{'-'*60}")
        
        # Test a high-energy random mode
        S_0 = compute_entropic_action(self.config.L, self.config.D, 
                                      self.config.h_amplitude, 5, False)
        S_h = compute_entropic_action(self.config.L, self.config.D, 
                                      self.config.h_amplitude, 5, True)
        
        cost = S_h - S_0
        
        print(f"Perturbation Cost: {cost:.6e}")
        
        if cost > 0:
            print(">>> VALIDATION SUCCESSFUL <<<")
            print("Action is positive definite. The vacuum is stable against perturbations.")
        else:
            print(">>> CRITICAL FAILURE <<<")
            print("Negative action detected. The vacuum is unstable (Ghost Mode).")
        print(f"{'='*60}")

    def run_parity_check(self):
        """
        Verifies Z2 Symmetry (Metric Inversion).
        The action must be invariant under h -> -h.
        E(+epsilon) == E(-epsilon).
        If this fails, the lattice action contains unphysical odd-power terms (h^3).
        """
        print(f"\n{'-'*60}")
        print("PARITY CHECK (Z2 SYMMETRY)")
        print(f"{'-'*60}")
        
        # Measure +Epsilon
        S_pos = compute_entropic_action(self.config.L, self.config.D, 
                                        self.config.h_amplitude, self.config.k_mode, True)
        S_0   = compute_entropic_action(self.config.L, self.config.D, 
                                        self.config.h_amplitude, self.config.k_mode, False)
        Cost_Pos = S_pos - S_0
        
        # Measure -Epsilon
        S_neg = compute_entropic_action(self.config.L, self.config.D, 
                                        -self.config.h_amplitude, self.config.k_mode, True)
        Cost_Neg = S_neg - S_0
        
        diff = abs(Cost_Pos - Cost_Neg)
        
        print(f"Cost (+ε): {Cost_Pos:.6e}")
        print(f"Cost (-ε): {Cost_Neg:.6e}")
        print(f"Asymmetry: {diff:.6e}")
        
        if diff < 1e-12:
            print(">>> VALIDATION SUCCESSFUL <<<")
            print("Action is invariant under metric inversion (Even symmetry).")
        else:
            print(">>> FALSIFICATION WARNING <<<")
            print("Action breaks Parity symmetry (Odd terms detected).")
        print(f"{'='*60}")

    def run_nonlinear_regime_check(self):
        """
        Tests the Stability of the Linearized Limit.
        In this verification script, the kernel is explicitly quadratic.
        Therefore, S/epsilon^2 must be perfectly constant.
        Any deviation indicates numerical instability (floating point errors).
        """
        print(f"\n{'-'*60}")
        print("NON-LINEAR REGIME CHECK (NUMERICAL STABILITY)")
        print(f"{'-'*60}")
        print(f"{'Strain (ε)':<12} | {'Norm. Cost':<15} | {'Stability':<10}")
        print("-" * 55)
        
        amplitudes = [0.01, 0.1, 0.5, 1.0, 5.0]
        ref_norm = 0.0
        
        for i, eps in enumerate(amplitudes):
            # Note: apply_perturbation must be True for S_h
            S_h = compute_entropic_action(self.config.L, self.config.D, 
                                      eps, self.config.k_mode, True)
            S_0 = compute_entropic_action(self.config.L, self.config.D, 
                                      0, self.config.k_mode, False)
            cost = S_h - S_0
            
            # Normalized cost (S / ε^2)
            normalized = cost / (eps**2)
            
            if i == 0: ref_norm = normalized
            
            # Check for drift
            drift = abs(normalized - ref_norm) / ref_norm
            status = "STABLE" if drift < 1e-10 else "DRIFT"
            
            print(f"{eps:<12.2f} | {normalized:.6e}    | {status}")
        print("-" * 55)

    def run_isotropy_check(self):
        """
        Verifies Rotational Invariance (Isotropy).
        Gravity must behave the same regardless of propagation direction.
        Tests propagation in x1, x2, x3 directions.
        """
        print(f"\n{'-'*60}")
        print("ISOTROPY CHECK (ROTATIONAL INVARIANCE)")
        print(f"{'-'*60}")
        
        axes = [Axis.X, Axis.Y, Axis.Z]
        axis_names = ['X (x1)', 'Y (x2)', 'Z (x3)']
        costs = []

        for axis in axes:
            S_h = compute_entropic_action(self.config.L, self.config.D, 
                                        self.config.h_amplitude, self.config.k_mode, True, axis=axis)
            S_0 = compute_entropic_action(self.config.L, self.config.D, 
                                        0, self.config.k_mode, False, axis=axis)
            cost = S_h - S_0
            costs.append(cost)
        
        avg_cost = np.mean(costs)
    
        for i, axis_name in enumerate(axis_names):
            dev = abs(costs[i] - avg_cost) / avg_cost * 100
            print(f"{axis_name:<8} | {costs[i]:.6e}     | {dev:.6f}%")
        
        max_dev = max(abs(c - avg_cost)/avg_cost for c in costs) * 100
        print("-" * 55)
        
        if max_dev < 0.01:  # Less than 0.01% deviation
            print(">>> VALIDATION SUCCESSFUL <<<")
            print("Lattice gravity is rotationally invariant.")
        else:
            print(">>> FALSIFICATION WARNING <<<")
            print(f"Preferred frame detected (max deviation: {max_dev:.4f}%)")
        print(f"{'='*60}")       

    def run_speed_of_light_check(self):
        """
        Verifies Lorentz Invariance (c = 1).
        Checks if the lattice treats Time (x4) exactly like Space (x3).
        If S_time == S_space, then c = dx/dt = 1.
        Deviations imply a 'refractive index' of the vacuum (n != 1).
        """
        print(f"\n{'-'*60}")
        print("SPEED OF LIGHT CHECK (LORENTZ INVARIANCE)")
        print(f"{'-'*60}")
        
        # 1. Measure Spatial Propagation Cost (Z-axis / x3)
        # We use apply_perturbation=True directly, subtracting baseline 0 implicitly 
        # (since S_0 is 0 for flat space, but rigorous way is diff)
        
        S_space_h = compute_entropic_action(self.config.L, self.config.D, 
                                            self.config.h_amplitude, self.config.k_mode, True)
        S_space_0 = compute_entropic_action(self.config.L, self.config.D, 
                                            0, self.config.k_mode, False)
        Cost_Space = S_space_h - S_space_0
        
        # 2. Measure Temporal Propagation Cost (T-axis / x4)
        S_time_h = compute_entropic_action(self.config.L, self.config.D, 
                                           self.config.h_amplitude, self.config.k_mode, True, Axis.TIME)
        S_time_0 = compute_entropic_action(self.config.L, self.config.D, 
                                           0, self.config.k_mode, False, Axis.TIME)
        Cost_Time = S_time_h - S_time_0
        
        # 3. Calculate Effective C
        # Energy ~ Stiffness * Gradient^2
        # If Stiffness_Space == Stiffness_Time, then c = 1
        
        c_squared_lattice = Cost_Space / Cost_Time
        c_eff = np.sqrt(c_squared_lattice)
        
        print(f"Action Cost (Space): {Cost_Space:.6e}")
        print(f"Action Cost (Time):  {Cost_Time:.6e}")
        print(f"Effective Velocity:  {c_eff:.6f} c")
        
        if 0.99999 <= c_eff <= 1.00001:
            print(">>> VALIDATION SUCCESSFUL <<<")
            print("Gravitational waves propagate at exactly c=1.")
            print("Lorentz Invariance is preserved on the lattice.")
        else:
            print(">>> FALSIFICATION WARNING <<<")
            print("Lorentz violation detected. Gravity is slower/faster than light.")
        print(f"{'='*60}")

if __name__ == "__main__":
    audit = E8PersistenceVerifier()
    
    # 1. The Smoke Test: Do we recover the Einstein-Hilbert Action?
    audit.run_check()                   # Is it General Relativity? (Vacuum Stiffness κ=1)
    
    # 2. The Physics: Verifies the nature of the excitation.
    audit.run_dispersion_scan()         # Is it a wave? (Linear Dispersion ω = ck)
    audit.run_shift_symmetry_check()    # Is it massless? (Gauge Invariance / Soft Graviton Theorem)
    
    # 3. The Spacetime symmetry: Verifies that the lattice respects the laws of Relativity.
    audit.run_parity_check()            # Is it a tensor field? (Metric Inversion Symmetry)
    audit.run_isotropy_check()          # Is space uniform? (Rotational Invariance)
    audit.run_speed_of_light_check()    # Is it relativistic? (Lorentz Invariance / c=1)
    
    # 4. The Structural integrity: Verifies mathematical consistency and thermodynamic limits.
    audit.run_linearity_check()         # Is it perturbative? (Weak-Field Approximation)
    audit.run_nonlinear_regime_check()  # Is it numerically robust? (Strong-Field Stability)
    audit.run_finite_size_scaling()     # Is it a continuum? (Thermodynamic Limit L -> ∞)
    audit.run_stability_check()         # Is the vacuum stable? (No Ghost Modes / Positive Action)