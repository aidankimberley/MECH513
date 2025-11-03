#!/bin/bash
# Script to activate the virtual environment for MECH513 Assignment 3

echo "Activating virtual environment for MECH513 Assignment 3..."
source venv/bin/activate
echo "Virtual environment activated!"
echo "Python version: $(python --version)"
echo "Available CVXPY solvers: $(python -c 'import cvxpy; print(cvxpy.installed_solvers())')"
echo ""
echo "To run your code, use: python A3Q1_LMI_solver.py"
echo "To deactivate, type: deactivate"

