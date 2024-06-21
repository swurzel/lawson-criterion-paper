# Progress toward Fusion Energy Breakeven and Gain as Measured against the Lawson Criterion

### Description
The purpose of this codebase is to generate the figures and tables in the paper:

S.E. Wurzel, S.C. Hsu, "Progress toward fusion energy breakeven and gain as
measured against the Lawson criterion" Physics of Plasmas 29, 062103 (2022)
https://doi.org/10.1063/5.0083990

The code is writen in Python 3 and uses juypytext to store the code as
a text-only .py file but is intended to be used in a jupyter notebook.

This code is not optimized for speed. Running the entire notebook may take
upward of 20 minutes. The main bottleneck is calculating large numbers of
fusion reactivities by integration of cross section over a maxwellian velocity
distribution. These could be cached, however for the sake of simplicity
there is currently no caching functionality.


### Systemm Dependencies
- MacTex https://tug.org/mactex/mainpage2024.html
- Python 3.X

### Installation
- Make sure the System Dependencies are already installed
```bash
# Clone the repository
git clone git@github.com:swurzel/lawson-criterion-paper.git

# Navigate to the project directory
cd lawson-criterion-paper

# Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows use `.venv\Scripts\activate`

# Install the dependencies
pip install -r requirements.txt

# Create the jupyter notebook from the .py file
jupytext --to notebook lawson-criterion-paper.py

# Run JupyterLab and open lawson-criterion-paper.ipynb
jupyter-lab lawson-criterion-paper.ipynb

# From the 'Run' menu select 'Run All Cells'

# The notebook should execute in full, generating all plots and tables.
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

- Sam Wurzel, 21 June 2024
