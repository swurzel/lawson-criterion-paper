# Progress toward Fusion Energy Breakeven and Gain as Measured against the Lawson Criteria

### Description
The purpose of this codebase is to generate the figures and tables for the papers:

S.E. Wurzel, S.C. Hsu, "Update: Progress toward fusion energy breakeven and gain as
measured against the Lawson criteria" arXiv:2505.03834 [physics.plasm-ph] (2025)
https://arxiv.org/abs/2505.03834

S.E. Wurzel, S.C. Hsu, "Progress toward fusion energy breakeven and gain as
measured against the Lawson criterion" Physics of Plasmas 29, 062103 (2022)
https://doi.org/10.1063/5.0083990

The code is intended to be run in a Jupyter notebook. We use juypytext to store the code as
a text-only .py file to facilitate clean diffs in source control.

On a modern laptop the entire notebook takes about 3 minutes to generate all plots and tables.

### Systemm Dependencies
- MacTex https://tug.org/mactex/mactex-download.html
- Python 3.11
- uv

### Installation
- Make sure the System Dependencies are already installed
- Install uv if you haven't already: https://docs.astral.sh/uv/getting-started/installation/
```bash
# Clone the repository
git clone git@github.com:swurzel/lawson-criterion-paper.git

# Navigate to the project directory
cd lawson-criterion-paper

# Install the dependencies with uv
uv sync

# Create the jupyter notebook from the .py file
uv run jupytext --to notebook lawson-criterion-paper.py

# Run JupyterLab and open lawson-criterion-paper.ipynb
uv run jupyter-lab lawson-criterion-paper.ipynb

# From the 'Run' menu select 'Run All Cells'

# The notebook should execute in full, generating all plots and tables.
# Alternatively, you can use VSCode based Jupyter integration (e.g., within Windsurf or Cursor).
```

### Credits
The cross sections for the reactions,

- T(d,n)4He (D + T --> n + α)
- D(d,p)T (D + D --> p + T)
- D(d,n)3He (D + D --> n + 3He)
- 3He(d,p)4He (D + 3He --> p + α)

are from:
H.S. Bosch and G.M. Hale 1992 Nucl. Fusion 32 611
https://doi.org/10.1088/0029-5515/32/4/I07

The cross sections for the reaction,

- 11B(p,4He)4He4He (p + 11B --> 3α)

is from
W.M. Nevins and R. Swain 2000 Nucl. Fusion 40 865
https://doi.org/10.1088/0029-5515/40/4/310

### License
MIT License, see LICENSE.

- Sam Wurzel, 21 August 2025
