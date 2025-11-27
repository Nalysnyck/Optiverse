# Optiverse

Optiverse — a desktop application for one- and multi-dimensional numerical optimization with a PyQt6 GUI. The project implements many classic optimization methods (univariate and multivariate), provides visualization of functions and iterations, and supports saving/loading input and results.

**Key features**
- One-dimensional methods: Golden ratio, Dichotomy, Ternary search, Fibonacci, Newton–Raphson, Secant, Brent, Cubic search, Exhaustive search, Powell's 1D, SPI, Ridder, Nelder–Mead (1D), and others.
- Multidimensional methods: Newton, Modified Newton, Simplex, Hooke–Jeeves, Conjugate Gradient variants, Levenberg–Marquardt, BFGS, Broyden, Nelder–Mead (ND), Particle Swarm, Differential Evolution, Simulated Annealing, and more.
- Function parser and symbolic derivatives via SymPy.
- Interactive plotting using Matplotlib embedded in PyQt6.
- Save/load input configurations and export results as JSON.

**Prerequisites**
- Python 3.10+ (recommended)
- Windows (project tested on Windows; UI uses PyQt6)

**Required Python packages**
- PyQt6
- sympy
- numpy
- matplotlib

You can install required packages with pip (adjust for your environment):

```powershell
python -m pip install pyqt6 sympy numpy matplotlib
```

(If you use a virtual environment, activate it first.)

**Run the application**
From the project root, run:

```powershell
python app.py
```

This launches the PyQt6 GUI. The UI labels and messages are in Ukrainian.

**Run tests / sample runners**
There is a simple test script for algorithm runs in `tests/logic.py`. To run those tests (they exercise many optimization methods and print results):

```powershell
python tests/logic.py
```

Note: tests may take a while because they run many methods; they print summaries of passed/failed methods and timings.

**Where results and inputs are stored**
- Saved inputs: `resources/saved_input/` (JSON files)
- Output results: `resources/output/` (JSON files)

**Developer notes & tips**
- The function parsing uses `sympy.sympify` and supports `^` (caret) in expressions by converting where needed. Numeric evaluation is done with `float()` after `evalf()`.
- If you install packages but the GUI fails to start, check that PyQt6 is correctly installed and the environment is the one used to run the script.
- To speed up testing, edit `tests/logic.py` and limit the methods lists or reduce `max_iterations`.
