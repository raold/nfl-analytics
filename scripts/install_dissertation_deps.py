#!/usr/bin/env python3
"""
Install dissertation enhancement plan dependencies.

Phase 1: MAPIE (conformal prediction)
Phase 2: linearmodels (Difference-in-Differences)
Phase 4: river (online learning & drift detection)
Phase 5: torchuncertainty (evidential deep learning)
"""

import subprocess
import sys

# Required packages with details
PACKAGES = {
    "mapie": {
        "phase": 1,
        "purpose": "Hierarchical conformal wrappers for uncertainty quantification",
        "min_version": "0.6.0",
    },
    "linearmodels": {
        "phase": 2,
        "purpose": "Panel regression analysis of injury shocks (DiD methodology)",
        "min_version": "4.0.0",
    },
    "river": {
        "phase": 4,
        "purpose": "Online learning and concept drift detection in production models",
        "min_version": "0.9.0",
    },
    "torchuncertainty": {
        "phase": 5,
        "purpose": "Evidential deep learning with uncertainty decomposition",
        "min_version": "0.1.0",
    },
}


def run_command(cmd, check=True):
    """Run a shell command and return output."""
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            check=check,
        )
        return result.returncode == 0, result.stdout.strip(), result.stderr.strip()
    except Exception as e:
        return False, "", str(e)


def check_package_installed(package_name):
    """Check if a package is installed using uv pip show."""
    success, stdout, _ = run_command(f"uv pip show {package_name}", check=False)
    return success, stdout


def get_package_version(package_name):
    """Extract version from pip show output."""
    success, stdout, _ = check_package_installed(package_name)
    if success and "Version:" in stdout:
        for line in stdout.split("\n"):
            if line.startswith("Version:"):
                return line.split("Version:")[1].strip()
    return None


def install_package(package_name):
    """Install a package using uv pip install."""
    success, stdout, stderr = run_command(
        f"uv pip install {package_name} -q", check=False
    )
    return success, stdout, stderr


def check_conflicts():
    """Check for dependency conflicts using uv pip check."""
    success, stdout, stderr = run_command("uv pip check", check=False)
    return success, stdout, stderr


def test_import(package_name):
    """Test if a package can be imported."""
    cmd = f"python3 -c \"import {package_name}\""
    success, _, stderr = run_command(cmd, check=False)
    return success, stderr


def main():
    """Main installation workflow."""
    print("=" * 80)
    print("DISSERTATION ENHANCEMENT PLAN - DEPENDENCY INSTALLATION")
    print("=" * 80)
    print()

    # Check prerequisites
    print("Checking prerequisites...")
    success, python_version, _ = run_command("python3 --version", check=False)
    if not success:
        print("ERROR: Python 3 not found")
        sys.exit(1)
    print(f"Python: {python_version.strip()}")

    success, uv_version, _ = run_command("uv --version", check=False)
    if not success:
        print(
            "ERROR: uv not found. Install with: curl -LsSf https://astral.sh/uv/install.sh | sh"
        )
        sys.exit(1)
    print(f"uv: {uv_version.strip()}")
    print()

    # Track results
    results = {}
    versions = {}

    # Install each package
    for package_name, package_info in PACKAGES.items():
        phase = package_info["phase"]
        purpose = package_info["purpose"]

        print("=" * 80)
        print(f"PHASE {phase}: {package_name.upper()}")
        print("=" * 80)
        print(f"Purpose: {purpose}")
        print()

        # Check if already installed
        installed, _ = check_package_installed(package_name)

        if installed:
            pkg_version = get_package_version(package_name)
            versions[package_name] = pkg_version
            results[package_name] = "ALREADY_INSTALLED"
            print(f"✓ {package_name} already installed (v{pkg_version})")
        else:
            print(f"Installing {package_name}...")
            success, stdout, stderr = install_package(package_name)

            if success:
                pkg_version = get_package_version(package_name)
                versions[package_name] = pkg_version
                results[package_name] = "NEWLY_INSTALLED"
                print(f"✓ {package_name} installed successfully (v{pkg_version})")
            else:
                results[package_name] = "FAILED"
                print(f"✗ {package_name} installation failed")
                if stderr:
                    print(f"  Error: {stderr}")

        print()

    # Check for conflicts
    print("=" * 80)
    print("DEPENDENCY CONFLICT CHECK")
    print("=" * 80)
    print()

    success, stdout, stderr = check_conflicts()
    if success:
        print("✓ No dependency conflicts detected")
    else:
        print("⚠ WARNING: Potential dependency conflicts detected:")
        if stdout:
            print(stdout)
    print()

    # Summary
    print("=" * 80)
    print("INSTALLATION SUMMARY")
    print("=" * 80)
    print()

    newly_installed = 0
    already_installed = 0
    failed = 0

    for package_name, status in results.items():
        pkg_version = versions.get(package_name, "UNKNOWN")

        if status == "NEWLY_INSTALLED":
            print(f"NEW:      {package_name:20s} (v{pkg_version})")
            newly_installed += 1
        elif status == "ALREADY_INSTALLED":
            print(f"EXISTING: {package_name:20s} (v{pkg_version})")
            already_installed += 1
        elif status == "FAILED":
            print(f"FAILED:   {package_name:20s}")
            failed += 1

    print()
    print("Summary:")
    print(f"  Newly Installed:    {newly_installed}")
    print(f"  Already Installed:  {already_installed}")
    print(f"  Failed:             {failed}")
    print()

    if failed > 0:
        print("ERROR: Some packages failed to install")
        sys.exit(1)

    # Test imports
    print("=" * 80)
    print("VERIFICATION")
    print("=" * 80)
    print()
    print("Testing Python imports...")

    failed_imports = []
    for package_name in PACKAGES.keys():
        success, stderr = test_import(package_name)
        if success:
            print(f"✓ {package_name}: Import successful")
        else:
            print(f"✗ {package_name}: Import failed - {stderr}")
            failed_imports.append(package_name)

    if failed_imports:
        print()
        print(f"ERROR: {len(failed_imports)} package(s) failed to import")
        sys.exit(1)

    print()
    print("✓ All packages imported successfully")
    print()

    # Final message
    print("=" * 80)
    print("INSTALLATION COMPLETE")
    print("=" * 80)
    print()
    print("Next steps:")
    print("1. Update requirements.txt with new dependencies")
    print("2. Review Phase 1-5 dissertation enhancement documentation")
    print("3. Begin implementing MAPIE conformal prediction (Phase 1)")
    print()


if __name__ == "__main__":
    main()
