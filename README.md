The $E_8$-Persistence Theory I: Universal Couplings and the Geometric Substrate

## Overview

This repository contains the source code, derivations, and LaTeX manuscripts for Persistence Theory I: Universal Couplings and the Geometric Substrate.

## Reproducibility

A core tenet of this work is that the physical constants are computational outputs. To verify the results found in the papers:

### 2. Generate the Constants
To regenerate the values for $\alpha^{-1}$, $\alpha_s$, $G_F$, etc., run:
```bash
python calculations/constants.py
```
To re-generate the latex output that is used in the paper
```bash
python calculations/constants.py --latex
```

## Simulations

This order follows the logical reconstruction of the universe: **(1) The Constants** $\to$ **(2) The Static Laws** $\to$ **(3) The Dynamic Laws**.

### Logical Execution Order

**1. `calculations/e8_impedance_calculator.py`** (The Constants)
> Performs a Monte Carlo diffusion audit on the projected $E_8$ lattice to derive the **Fine-Structure Constant** ($\alpha^{-1} \approx 137.036$) and **Manifold Friction** ($\eta \approx 0.994$). This confirms that fundamental couplings are not arbitrary inputs, but the geometric impedance of the substrate to information propagation.

**2. `calculations/capacity_conservation_gravity.py`** (The Static Limit)
> Simulates the relaxation of a static mass defect on a $96^4$ lattice to minimize Entropic Action. It numerically recovers the **Inverse-Square Law** ($\Phi \propto r^{-2}$) and verifies 4D Gauss's Law to within 6%, proving that the Newtonian potential emerges naturally from the conservation of channel capacity.

**3. `calculations/e8_gravity_killswitch.py`** (The Dynamic Limit)
> Injects transverse-traceless metric perturbations into the lattice to verify the emergence of the **Einstein-Hilbert Action**. It confirms that the gravitational carrier is a strictly massless ($m=0$), spin-2 Goldstone mode propagating at the speed of light ($c=1$), validating General Relativity as the hydrodynamic limit of the substrate.