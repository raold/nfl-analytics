#!/usr/bin/env python3
"""
Minimal dependency installation and verification script.
No external dependencies required.
"""

import subprocess
import sys
import os

# Change to project directory
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PACKAGES = {
    "mapie": "Phase 1: Hierarchical conformal prediction",
    "linearmodels": "Phase 2: Difference-in-Differences panel regression",
    "river": "Phase 4: Online learning and drift detection",
    "torchuncertainty": "Phase 5: Evidential deep learning",
}


def run_cmd(cmd):
    """Run shell command and return success/output."""
    try:
        result = subprocess.run(
            cmd, shell=True, capture_output=True, text=True, timeout=60
        )
        return result.returncode == 0, result.stdout.strip(), result.stderr.strip()
    except Exception as e:
        return False, "", str(e)


def main():
    print("\n" + "=" * 80)
    print("DISSERTATION ENHANCEMENT PLAN - DEPENDENCY CHECK & INSTALLATION")
    print("=" * 80 + "\n")

    # Verify environment
    print("Environment Check:")
    success, py_ver, _ = run_cmd("python3 --version")
    print(f"  Python: {py_ver if success else 'NOT FOUND'}")

    success, uv_ver, _ = run_cmd("uv --version")
    print(f"  uv: {uv_ver if success else 'NOT FOUND'}")
    print()

    # Results tracking
    status_map = {}
    version_map = {}

    # Check and install each package
    for pkg_name, description in PACKAGES.items():
        print("-" * 80)
        print(f"{pkg_name.upper()}")
        print(f"  {description}")
        print()

        # Check if installed
        success, output, _ = run_cmd(f"uv pip show {pkg_name}")

        if success and output:
            # Already installed - extract version
            for line in output.split("\n"):
                if line.startswith("Version:"):
                    ver = line.split("Version:")[1].strip()
                    version_map[pkg_name] = ver
                    status_map[pkg_name] = "INSTALLED"
                    print(f"  Status: Already installed (v{ver})")
                    break
        else:
            # Not installed - attempt installation
            print(f"  Status: Installing...")
            success, out, err = run_cmd(f"uv pip install {pkg_name}")

            if success:
                # Get version after install
                success_show, output_show, _ = run_cmd(f"uv pip show {pkg_name}")
                if success_show:
                    for line in output_show.split("\n"):
                        if line.startswith("Version:"):
                            ver = line.split("Version:")[1].strip()
                            version_map[pkg_name] = ver
                            status_map[pkg_name] = "NEWLY_INSTALLED"
                            print(f"  Status: Successfully installed (v{ver})")
                            break
            else:
                status_map[pkg_name] = "FAILED"
                print(f"  Status: Installation FAILED")
                if err:
                    print(f"  Error: {err[:200]}")

        print()

    # Conflict check
    print("-" * 80)
    print("DEPENDENCY CONFLICT CHECK")
    print()
    success, conflicts, _ = run_cmd("uv pip check")
    if success:
        print("  Status: No conflicts detected")
    else:
        print("  Status: Possible conflicts")
        if conflicts:
            print(f"  Details: {conflicts[:200]}")
    print()

    # Summary
    print("=" * 80)
    print("INSTALLATION SUMMARY")
    print("=" * 80)
    print()

    newly = sum(1 for s in status_map.values() if s == "NEWLY_INSTALLED")
    existing = sum(1 for s in status_map.values() if s == "INSTALLED")
    failed = sum(1 for s in status_map.values() if s == "FAILED")

    for pkg_name in PACKAGES.keys():
        status = status_map.get(pkg_name, "UNKNOWN")
        version = version_map.get(pkg_name, "N/A")
        symbol = "✓" if status in ["INSTALLED", "NEWLY_INSTALLED"] else "✗"
        status_label = (
            "NEW"
            if status == "NEWLY_INSTALLED"
            else "EXISTING"
            if status == "INSTALLED"
            else "FAILED"
        )
        print(f"  {symbol} {pkg_name:20s} {status_label:12s} v{version}")

    print()
    print(f"Summary: {newly} new, {existing} existing, {failed} failed")
    print()

    # Test imports
    print("=" * 80)
    print("IMPORT VERIFICATION")
    print("=" * 80)
    print()

    import_failures = []
    for pkg_name in PACKAGES.keys():
        if status_map.get(pkg_name) == "FAILED":
            print(f"  ⊘ {pkg_name}: SKIPPED (not installed)")
            continue

        success, _, err = run_cmd(f"python3 -c 'import {pkg_name}'")
        if success:
            print(f"  ✓ {pkg_name}: Import successful")
        else:
            print(f"  ✗ {pkg_name}: Import FAILED")
            import_failures.append(pkg_name)

    print()

    if failed > 0:
        print(f"ERROR: {failed} package(s) failed to install")
        return 1

    if import_failures:
        print(f"ERROR: {len(import_failures)} package(s) failed to import")
        return 1

    print("SUCCESS: All dependencies installed and verified")
    print()
    print("Next steps:")
    print("  1. Update requirements.txt with new packages")
    print("  2. Review Phase 1-5 dissertation enhancement documentation")
    print("  3. Begin Phase 1 implementation (MAPIE conformal prediction)")
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
