"""Compatibility imports for earlier examples. Implementations are in perturbations.py.

This file is not a baseline. random_noise in perturbations.py is the new baseline.
Old code such as 'from attacks import fgsm, pgd' continues to work.
"""
from perturbations import fgsm, pgd
