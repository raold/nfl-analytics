#!/usr/bin/env python3
"""Quick test of current package status."""

import subprocess
import sys

packages = ['mapie', 'linearmodels', 'river', 'torchuncertainty']

print("Checking package installation status...\n")

for pkg in packages:
    result = subprocess.run(
        ['python3', '-m', 'pip', 'show', pkg],
        capture_output=True,
        text=True
    )
    if result.returncode == 0:
        lines = result.stdout.strip().split('\n')
        version = next((l.split(': ')[1] for l in lines if l.startswith('Version:')), 'UNKNOWN')
        print(f"✓ {pkg:20s} - Already installed (v{version})")
    else:
        print(f"✗ {pkg:20s} - Not installed")

print("\nAttempting to import packages...\n")

for pkg in packages:
    try:
        __import__(pkg)
        print(f"✓ {pkg:20s} - Import successful")
    except ImportError as e:
        print(f"✗ {pkg:20s} - Import failed: {str(e)[:50]}")

print("\nDone.")
