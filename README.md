
[![Build status](https://github.com/jjmoraa/Floquet/workflows/Tests/badge.svg)](https://github.com/jjmoraa/Floquet/actions?query=workflow%3A%22Tests%22)

# **stablib**

*A lightweight library for Floquet-based stability analysis of periodic dynamical systems.*

---

## Overview

**stablib** is a research-oriented library for performing **Floquet stability analysis** on linear time-periodic (LTP) systems of the form

[
\dot{x}(t) = A(t) x(t), \quad A(t+T)=A(t)
]

The library provides tools to compute:

* The **state transition (monodromy) matrix**
* **Floquet multipliers and exponents**
* **Modal growth/decay rates**
* **Natural frequencies of complex LTI systems**
* Some post-processing utilities for **mode interpretation and visualization**

This library is designed with **aeroelastic**, **hydrodynamic**, and **rotordynamic** applications in mind, but is general enough for any periodic linear system.

---

## Key Features

* Time-domain integration of LTP systems
* Robust computation of the monodromy matrix
* Extraction of Floquet multipliers and exponents
* Modal sorting and stability classification
* Clean, modular architecture for research workflows
* Pythonic

---

## Installation

Install stablib directly using pip:

pip install stablib

Ensure the package is available in your Python environment before use.

## Quick Start

... on the works

## Typical Workflow

1. Define the periodic system matrix `A(t)`
2. Integrate the variational equations over one period
3. Construct the monodromy matrix
4. Extract Floquet multipliers and exponents
5. Post-process modes (growth rates, frequencies, participation)

---

## Examples

The `examples/` directory includes:

* Canonical LTP systems for benchmarking (Mathiueu's oscillator, Rotation matrix)
* Aeroelastic-inspired periodic systems (5DOF Coleman's edgewise turbine model)
* Generalized A matrix input cases from simulation software (OpenFAST)

Each example is self-contained and intended to be run end-to-end.


## Applications

* Aeroelastic stability of rotating blades
* Rotor dynamics and periodic structures
* General linear periodic systems

---

## Citation

If you use **stablib** in academic work, please cite:

```bibtex
@software{stablib,
  title  = {stablib: A Floquet Stability Analysis Library},
  author = {Mora Amaro, J.; Branlard, E.; Riva, R.},
  year   = {2026},
  url    = {https://github.com/jjmoraa/Floquet}
}
```

---

## Contributing

Contributions are welcome. Please:

1. Fork the repository
2. Create a feature branch
3. Add tests where appropriate
4. Submit a pull request

---

## License

This project is released under the **MIT License**. See `LICENSE` for details.

---

## Contact

For questions, suggestions, or collaboration inquiries, feel free to open an issue or contact the maintainer directly.

