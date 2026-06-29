# Run `just` to list available recipes.
default:
    @just --list

# Regenerate all plots, tables, and the Lawson animation from the latest pkl data.
makeplots:
    uv run jupytext --to notebook lawson-criterion-paper.py
    uv run jupyter nbconvert --to notebook --execute --inplace \
        --ExecutePreprocessor.timeout=-1 \
        lawson-criterion-paper.ipynb
