"""Compatibility entry point for old commands; prefer compare_attacks.py for new runs."""
from compare_attacks import main

if __name__ == "__main__":
    main(legacy=True)
