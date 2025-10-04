#!/usr/bin/env python3
"""
Smart dependency checker and installer for NFL Analytics.

Only installs packages that are missing, avoiding redundant installations.
"""

import subprocess
import sys
import importlib.util
from typing import List, Tuple, Dict
import json
import time

# Package mappings: import name -> pip package spec
PACKAGE_MAPPINGS = {
    # Core dependencies
    'redis': 'redis>=4.5.0',
    'aioredis': 'aioredis>=2.0.0',
    'psutil': 'psutil>=5.9.0',

    # Data science packages
    'numpy': 'numpy',
    'pandas': 'pandas',
    'polars': 'polars',
    'pyarrow': 'pyarrow',
    'scipy': 'scipy',
    'sklearn': 'scikit-learn',
    'xgboost': 'xgboost',

    # Database
    'sqlalchemy': 'SQLAlchemy',
    'psycopg': 'psycopg[binary]',
    'dotenv': 'python-dotenv',

    # Visualization
    'matplotlib': 'matplotlib',
    'meteostat': 'meteostat',

    # Testing
    'pytest': 'pytest',

    # Web/API
    'requests': 'requests',
}

# Optional packages (don't fail if missing)
OPTIONAL_PACKAGES = {
    'torch': 'torch',  # PyTorch - platform specific
    'tensorflow': 'tensorflow',  # TensorFlow - optional
}


def check_package_installed(package_name: str) -> bool:
    """Check if a Python package is installed."""
    try:
        # Try to import the package
        spec = importlib.util.find_spec(package_name)
        if spec is not None:
            return True
    except (ImportError, ModuleNotFoundError):
        pass

    # Check alternative import names
    alternative_names = {
        'sklearn': 'scikit-learn',
        'dotenv': 'python-dotenv',
        'psycopg': 'psycopg2',
    }

    if package_name in alternative_names:
        try:
            spec = importlib.util.find_spec(alternative_names[package_name].replace('-', '_'))
            if spec is not None:
                return True
        except:
            pass

    return False


def get_installed_packages() -> Dict[str, str]:
    """Get list of installed packages with versions."""
    try:
        result = subprocess.run(
            [sys.executable, '-m', 'pip', 'list', '--format=json'],
            capture_output=True,
            text=True,
            check=True
        )
        packages = json.loads(result.stdout)
        return {pkg['name'].lower(): pkg['version'] for pkg in packages}
    except Exception as e:
        print(f"⚠️ Failed to get installed packages: {e}")
        return {}


def install_package(package_spec: str) -> bool:
    """Install a single package."""
    print(f"📦 Installing {package_spec}...")
    try:
        subprocess.run(
            [sys.executable, '-m', 'pip', 'install', '--no-cache-dir', package_spec],
            check=True,
            capture_output=True
        )
        print(f"✅ Successfully installed {package_spec}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install {package_spec}: {e}")
        return False


def check_and_install_dependencies():
    """Check and install missing dependencies."""
    print("🔍 Checking Python dependencies...")

    # Get currently installed packages
    installed = get_installed_packages()
    print(f"📋 Found {len(installed)} installed packages")

    # Check required packages
    missing_required = []
    missing_optional = []

    # Check required packages
    for import_name, package_spec in PACKAGE_MAPPINGS.items():
        if check_package_installed(import_name):
            print(f"✅ {import_name:15} already installed")
        else:
            print(f"❌ {import_name:15} missing")
            missing_required.append(package_spec)

    # Check optional packages
    for import_name, package_spec in OPTIONAL_PACKAGES.items():
        if check_package_installed(import_name):
            print(f"✅ {import_name:15} already installed (optional)")
        else:
            print(f"⚠️ {import_name:15} missing (optional)")
            missing_optional.append(package_spec)

    # Install missing required packages
    if missing_required:
        print(f"\n📦 Installing {len(missing_required)} missing required packages...")
        failed = []

        for package in missing_required:
            if not install_package(package):
                failed.append(package)

        if failed:
            print(f"\n❌ Failed to install: {', '.join(failed)}")
            print("You may need to install these manually")
            return False
    else:
        print("\n✅ All required packages are installed")

    # Report optional packages
    if missing_optional:
        print(f"\n⚠️ Optional packages not installed: {', '.join(missing_optional)}")
        print("These are not required but may enable additional features")

    return True


def verify_redis_connection() -> bool:
    """Verify Redis server is accessible."""
    print("\n🔧 Checking Redis connection...")
    try:
        import redis
        r = redis.Redis(host='localhost', port=6379, socket_connect_timeout=1)
        r.ping()
        print("✅ Redis connection successful")
        return True
    except ImportError:
        print("❌ Redis package not installed")
        return False
    except Exception as e:
        print(f"⚠️ Redis not running or not accessible: {e}")
        print("Start Redis with: redis-server")
        return False


def check_hardware_support():
    """Check hardware capabilities."""
    print("\n🖥️ Checking hardware support...")

    # CPU info
    try:
        import psutil
        cpu_count = psutil.cpu_count()
        memory_gb = psutil.virtual_memory().total / (1024**3)
        print(f"✅ CPU: {cpu_count} cores")
        print(f"✅ Memory: {memory_gb:.1f} GB")
    except ImportError:
        print("⚠️ psutil not available, cannot detect hardware")

    # GPU info
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            print(f"✅ GPU: {gpu_name} ({gpu_memory:.1f} GB)")
        else:
            print("⚠️ No CUDA GPU detected")
    except ImportError:
        print("⚠️ PyTorch not installed, GPU detection unavailable")
    except Exception as e:
        print(f"⚠️ GPU detection failed: {e}")


def main():
    """Run all dependency checks."""
    print("🔍 NFL Analytics Dependency Checker")
    print("=" * 60)

    # Check and install dependencies
    deps_ok = check_and_install_dependencies()

    # Verify Redis
    redis_ok = verify_redis_connection()

    # Check hardware
    check_hardware_support()

    # Summary
    print("\n" + "=" * 60)
    print("📊 Summary")
    print("=" * 60)

    if deps_ok:
        print("✅ All required dependencies installed")
    else:
        print("❌ Some dependencies missing")

    if redis_ok:
        print("✅ Redis is accessible")
    else:
        print("⚠️ Redis not accessible (required for distributed mode)")

    print("\n🎯 Next steps:")
    if not redis_ok:
        print("1. Start Redis: redis-server")
    print("1. Run tests: python test_redis_setup.py")
    print("2. Start worker: python run_compute.py --redis-mode")

    return 0 if deps_ok else 1


if __name__ == "__main__":
    sys.exit(main())