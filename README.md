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
To run the lattice simulation to recover Einstein-Hilbert k^2 scaling and checking E(k) against Lattice Goldstone predictions
```bash
python calculations/e8_gravity_killswitch.py
```
