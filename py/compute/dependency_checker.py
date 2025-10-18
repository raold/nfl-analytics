#!/usr/bin/env python3
"""
Smart dependency checker for NFL Analytics system.

Checks for required packages and services before attempting installation
or usage, providing graceful fallbacks and helpful error messages.
"""

import importlib
import json
import logging
import subprocess
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


class DependencyChecker:
    """Smart dependency checker with caching and graceful fallbacks."""

    def __init__(self, cache_file: str = "/tmp/nfl_analytics_deps.json"):
        self.cache_file = Path(cache_file)
        self.cache = self._load_cache()

    def _load_cache(self) -> dict:
        """Load dependency cache from file."""
        try:
            if self.cache_file.exists():
                with open(self.cache_file) as f:
                    return json.load(f)
        except Exception as e:
            logger.debug(f"Failed to load dependency cache: {e}")
        return {}

    def _save_cache(self):
        """Save dependency cache to file."""
        try:
            with open(self.cache_file, "w") as f:
                json.dump(self.cache, f, indent=2)
        except Exception as e:
            logger.debug(f"Failed to save dependency cache: {e}")

    def check_package(
        self, package_name: str, import_name: str = None, version_check: callable = None
    ) -> tuple[bool, str]:
        """
        Check if a Python package is available.

        Args:
            package_name: Name for pip install (e.g., 'scikit-learn')
            import_name: Name for import (e.g., 'sklearn'). Defaults to package_name
            version_check: Optional function to check version compatibility

        Returns:
            (is_available, message)
        """
        if import_name is None:
            import_name = package_name

        # Check cache first
        cache_key = f"package:{import_name}"
        if cache_key in self.cache:
            cached_result = self.cache[cache_key]
            logger.debug(f"Using cached result for {import_name}: {cached_result}")
            return cached_result["available"], cached_result["message"]

        try:
            # Try to import the package
            module = importlib.import_module(import_name)

            # Check version if specified
            if version_check:
                if not version_check(module):
                    message = f"{import_name} version incompatible"
                    self.cache[cache_key] = {"available": False, "message": message}
                    self._save_cache()
                    return False, message

            message = f"{import_name} available"
            self.cache[cache_key] = {"available": True, "message": message}
            self._save_cache()
            return True, message

        except ImportError as e:
            message = f"{import_name} not available: {str(e)}"
            self.cache[cache_key] = {"available": False, "message": message}
            self._save_cache()
            return False, message

    def check_redis_connection(self, host: str = "localhost", port: int = 6379) -> tuple[bool, str]:
        """Check Redis connection."""

        # Don't cache Redis connections (they can change)
        try:
            import redis

            r = redis.Redis(host=host, port=port, socket_timeout=2)
            r.ping()
            return True, f"Redis connected at {host}:{port}"
        except ImportError:
            return False, "Redis package not installed"
        except Exception as e:
            return False, f"Redis connection failed: {str(e)}"

    def check_system_command(self, command: str) -> tuple[bool, str]:
        """Check if system command is available."""
        cache_key = f"command:{command}"

        if cache_key in self.cache:
            cached_result = self.cache[cache_key]
            return cached_result["available"], cached_result["message"]

        try:
            subprocess.run([command, "--version"], capture_output=True, check=True, timeout=5)
            message = f"{command} command available"
            self.cache[cache_key] = {"available": True, "message": message}
            self._save_cache()
            return True, message
        except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
            message = f"{command} command not found"
            self.cache[cache_key] = {"available": False, "message": message}
            self._save_cache()
            return False, message

    def check_gpu_support(self) -> tuple[bool, str]:
        """Check GPU support availability."""
        cache_key = "gpu_support"

        # GPU support can change, but cache for short time
        if cache_key in self.cache:
            cached_result = self.cache[cache_key]
            return cached_result["available"], cached_result["message"]

        try:
            import torch

            if torch.cuda.is_available():
                gpu_count = torch.cuda.device_count()
                gpu_name = torch.cuda.get_device_name(0) if gpu_count > 0 else "Unknown"
                message = f"GPU available: {gpu_name} ({gpu_count} devices)"
                self.cache[cache_key] = {"available": True, "message": message}
            else:
                message = "PyTorch available but no CUDA devices"
                self.cache[cache_key] = {"available": False, "message": message}
        except ImportError:
            message = "PyTorch not available"
            self.cache[cache_key] = {"available": False, "message": message}

        self._save_cache()
        return self.cache[cache_key]["available"], self.cache[cache_key]["message"]

    def install_missing_packages(self, packages: list[str], dry_run: bool = False) -> bool:
        """Install missing packages using pip."""
        if not packages:
            return True

        if dry_run:
            logger.info(f"Would install: {packages}")
            return True

        try:
            logger.info(f"Installing missing packages: {packages}")
            cmd = [sys.executable, "-m", "pip", "install"] + packages
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

            if result.returncode == 0:
                logger.info("✅ Package installation successful")
                # Clear cache for installed packages
                self._clear_package_cache(packages)
                return True
            else:
                logger.error(f"❌ Package installation failed: {result.stderr}")
                return False

        except subprocess.TimeoutExpired:
            logger.error("❌ Package installation timed out")
            return False
        except Exception as e:
            logger.error(f"❌ Package installation error: {e}")
            return False

    def _clear_package_cache(self, packages: list[str]):
        """Clear cache for specific packages."""
        keys_to_remove = []
        for key in self.cache.keys():
            if key.startswith("package:"):
                package_name = key.split(":", 1)[1]
                if any(pkg in package_name for pkg in packages):
                    keys_to_remove.append(key)

        for key in keys_to_remove:
            del self.cache[key]

        self._save_cache()

    def get_comprehensive_status(self) -> dict[str, dict]:
        """Get comprehensive dependency status."""
        status = {"core_packages": {}, "optional_packages": {}, "services": {}, "hardware": {}}

        # Core packages
        core_packages = [
            ("redis", "redis"),
            ("psutil", "psutil"),
            ("numpy", "numpy"),
            ("pandas", "pandas"),
        ]

        for package_name, import_name in core_packages:
            available, message = self.check_package(package_name, import_name)
            status["core_packages"][package_name] = {
                "available": available,
                "message": message,
                "required": True,
            }

        # Optional packages
        optional_packages = [
            ("scikit-learn", "sklearn"),
            ("scipy", "scipy"),
            ("torch", "torch"),
            ("aioredis", "aioredis"),
        ]

        for package_name, import_name in optional_packages:
            available, message = self.check_package(package_name, import_name)
            status["optional_packages"][package_name] = {
                "available": available,
                "message": message,
                "required": False,
            }

        # Services
        redis_available, redis_message = self.check_redis_connection()
        status["services"]["redis"] = {
            "available": redis_available,
            "message": redis_message,
            "required": True,
        }

        # Hardware
        gpu_available, gpu_message = self.check_gpu_support()
        status["hardware"]["gpu"] = {
            "available": gpu_available,
            "message": gpu_message,
            "required": False,
        }

        return status

    def ensure_minimal_requirements(self, auto_install: bool = False) -> bool:
        """Ensure minimal requirements are met."""
        missing_packages = []

        # Check core requirements
        required_packages = [
            ("redis", "redis"),
            ("psutil", "psutil"),
            ("numpy", "numpy"),
            ("pandas", "pandas"),
        ]

        logger.info("🔍 Checking minimal requirements...")

        for package_name, import_name in required_packages:
            available, message = self.check_package(package_name, import_name)
            if not available:
                logger.warning(f"❌ Missing: {package_name}")
                missing_packages.append(package_name)
            else:
                logger.info(f"✅ Available: {package_name}")

        if missing_packages:
            if auto_install:
                logger.info(f"📦 Auto-installing missing packages: {missing_packages}")
                return self.install_missing_packages(missing_packages)
            else:
                logger.error(f"❌ Missing required packages: {missing_packages}")
                logger.error("Run: pip install " + " ".join(missing_packages))
                return False

        logger.info("✅ All minimal requirements satisfied")
        return True


# Global instance
dependency_checker = DependencyChecker()


def quick_check_redis_mode() -> bool:
    """Quick check if Redis mode is available."""
    redis_available, _ = dependency_checker.check_package("redis", "redis")
    psutil_available, _ = dependency_checker.check_package("psutil", "psutil")
    redis_conn, _ = dependency_checker.check_redis_connection()

    return redis_available and psutil_available and redis_conn


def get_missing_dependencies() -> list[str]:
    """Get list of missing core dependencies."""
    missing = []

    core_deps = [("redis", "redis"), ("psutil", "psutil"), ("numpy", "numpy"), ("pandas", "pandas")]

    for package_name, import_name in core_deps:
        available, _ = dependency_checker.check_package(package_name, import_name)
        if not available:
            missing.append(package_name)

    return missing


if __name__ == "__main__":
    # Test dependency checker
    print("🧪 Testing Dependency Checker")
    print("=" * 50)

    checker = DependencyChecker()

    # Test comprehensive status
    status = checker.get_comprehensive_status()

    for category, packages in status.items():
        print(f"\n{category.upper()}:")
        for package, info in packages.items():
            status_icon = "✅" if info["available"] else "❌"
            required_text = "(required)" if info["required"] else "(optional)"
            print(f"  {status_icon} {package} {required_text}: {info['message']}")

    # Test minimal requirements
    print("\n🔍 MINIMAL REQUIREMENTS CHECK:")
    minimal_ok = checker.ensure_minimal_requirements(auto_install=False)
    print(f"Status: {'✅ PASSED' if minimal_ok else '❌ FAILED'}")

    # Quick Redis check
    print("\n🔧 REDIS MODE CHECK:")
    redis_ok = quick_check_redis_mode()
    print(f"Redis Mode Available: {'✅ YES' if redis_ok else '❌ NO'}")

    print("\n✅ Dependency checker test completed")
