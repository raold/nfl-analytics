# Google Drive Cleanup - Quick Fix Guide

**Date:** 2025-11-14
**Priority:** P1 (CRITICAL - Blocks CI/CD)
**Estimated Time:** 15 minutes
**Impact:** Removes 5 orphaned imports from Nov 14 cleanup

---

## Problem Statement

After removing Google Drive sync code (3,500 lines on Nov 14, 2025), **5 orphaned references remain** that break imports and tests:

1. `tests/unit/test_sync_manager.py` - Entire file imports deleted module
2. `containers/test_redis_setup.py` - Imports `GoogleDriveSyncManager` (line 19)
3. `scripts/compute/run_compute.py` - Imports `GoogleDriveSyncManager` (line 63)

**Impact:** Test suite fails, scripts fail to import modules.

---

## Quick Fix (3 Steps)

### Step 1: Delete Orphaned Test File

```bash
# File: tests/unit/test_sync_manager.py
# Reason: Auto-generated stubs for deleted module (365 lines)
# Action: DELETE ENTIRE FILE

rm /Users/dro/rice/nfl-analytics/tests/unit/test_sync_manager.py
```

**Verification:**
```bash
ls -la tests/unit/test_sync_manager.py
# Should return: No such file or directory
```

---

### Step 2: Fix containers/test_redis_setup.py

**Remove 3 sections:**

#### Section 1: Import Statement (Line 19)
```python
# BEFORE (Line 19)
from sync_manager import GoogleDriveSyncManager

# AFTER
# DELETED (module no longer exists)
```

#### Section 2: Test Function (Lines 152-191)
```python
# BEFORE (Lines 152-191)
def test_sync_manager():
    """Test Google Drive sync manager."""
    print("\n🔄 Testing Sync Manager...")
    # ... 39 lines of test code ...
    return False

# AFTER
# DELETED (function tests deleted module)
```

#### Section 3: Test Registration (Line 235)
```python
# BEFORE (Line 235)
tests = [
    ("Redis Connection", test_redis_connection),
    ("Hardware Detection", test_hardware_detection),
    ("Redis Task Queue", test_redis_task_queue),
    ("Sync Manager", test_sync_manager),  # ← REMOVE THIS LINE
    ("Worker Integration", test_worker_integration)
]

# AFTER
tests = [
    ("Redis Connection", test_redis_connection),
    ("Hardware Detection", test_hardware_detection),
    ("Redis Task Queue", test_redis_task_queue),
    ("Worker Integration", test_worker_integration)
]
```

**Manual Edit:**
1. Open `/Users/dro/rice/nfl-analytics/containers/test_redis_setup.py`
2. Delete line 19
3. Delete lines 152-191 (entire `test_sync_manager()` function)
4. Delete `("Sync Manager", test_sync_manager),` from tests list (around line 235)

**Verification:**
```bash
grep -n "sync_manager\|GoogleDrive" containers/test_redis_setup.py
# Should return: 0 results
```

---

### Step 3: Fix scripts/compute/run_compute.py

**Remove 2 sections:**

#### Section 1: Import Statement (Line 63)
```python
# BEFORE (Line 63)
from sync_manager import GoogleDriveSyncManager

# AFTER
# DELETED (module no longer exists)
```

#### Section 2: Sync Manager Instantiation (Lines 561-598)
```python
# BEFORE (Lines 560-598)
        # Start sync manager if in Redis mode
        sync_manager = None
        if redis_mode and REDIS_AVAILABLE:
            try:
                redis_client = redis.Redis(host='localhost', port=6379, decode_responses=True)
                sync_manager = GoogleDriveSyncManager(
                    redis_client=redis_client,
                    sync_directory=Path("/data"),
                    machine_id=args.machine_id,
                    sync_interval=args.sync_interval
                )

                # Start background sync in a separate thread
                import threading
                sync_thread = threading.Thread(target=sync_manager.start_background_sync, daemon=True)
                sync_thread.start()
                print("🔄 Started Google Drive sync manager")

            except Exception as e:
                print(f"⚠️ Failed to start sync manager: {e}")

        # Create and run worker
        # ...
        try:
            worker.run()
        except KeyboardInterrupt:
            print("\n🛑 Shutting down compute system...")
            if sync_manager:
                print("🔄 Performing final sync...")
                sync_manager.full_sync()
            print("👋 Compute system stopped. Your hardware can cool down now!")

# AFTER (Lines 560-580)
        # Create and run worker
        if redis_mode and REDIS_AVAILABLE:
            worker = RedisComputeWorker(
                intensity=args.intensity,
                use_adaptive=args.adaptive,
                machine_id=args.machine_id,
                hardware_profile=args.hardware_profile
            )
        else:
            worker = ComputeWorker(intensity=args.intensity, use_adaptive=args.adaptive)

        try:
            worker.run()
        except KeyboardInterrupt:
            print("\n🛑 Shutting down compute system...")
            print("👋 Compute system stopped. Your hardware can cool down now!")
```

**Manual Edit:**
1. Open `/Users/dro/rice/nfl-analytics/scripts/compute/run_compute.py`
2. Delete line 63 (`from sync_manager import GoogleDriveSyncManager`)
3. Delete lines 561-579 (sync_manager instantiation block)
4. Delete lines 596-598 (sync_manager cleanup in KeyboardInterrupt handler)

**Verification:**
```bash
grep -n "sync_manager\|GoogleDrive" scripts/compute/run_compute.py
# Should return: 0 results
```

---

## Final Verification

**Run all validation checks:**

```bash
# 1. Verify no sync_manager imports in active code
grep -r "sync_manager\|GoogleDriveSyncManager" py/ tests/ scripts/ containers/ \
  --include="*.py" --exclude-dir=archive
# Expected: 0 results

# 2. Verify imports resolve (no ModuleNotFoundError)
cd /Users/dro/rice/nfl-analytics
python3 -c "import sys; sys.path.insert(0, 'py'); from py.compute.redis_worker import RedisComputeWorker; print('✅ Imports working')"

# 3. Run test suite (should skip deleted test)
pytest tests/unit -v -k "not sync_manager"
# Expected: All tests pass

# 4. Check linting
ruff check py/ scripts/ containers/
# Expected: No import errors
```

---

## Success Criteria

✅ All validation checks pass
✅ No import errors in production code
✅ Test suite runs successfully
✅ Zero references to `sync_manager` or `GoogleDriveSyncManager` (excluding archive/)

---

## Files Modified

1. ❌ **DELETED:** `tests/unit/test_sync_manager.py` (365 lines)
2. ✅ **MODIFIED:** `containers/test_redis_setup.py` (removed 3 sections, ~42 lines)
3. ✅ **MODIFIED:** `scripts/compute/run_compute.py` (removed 2 sections, ~39 lines)

**Total Lines Removed:** ~446 lines

---

## Rollback Plan

If issues arise, restore files from Git:

```bash
# Restore test file
git checkout HEAD -- tests/unit/test_sync_manager.py

# Restore test_redis_setup.py
git checkout HEAD -- containers/test_redis_setup.py

# Restore run_compute.py
git checkout HEAD -- scripts/compute/run_compute.py
```

**Note:** This will restore orphaned imports - only use if blocking production issue.

---

## Next Steps After Cleanup

1. **Commit changes:**
   ```bash
   git add tests/unit/test_sync_manager.py \
           containers/test_redis_setup.py \
           scripts/compute/run_compute.py

   git commit -m "fix: remove orphaned Google Drive sync references (5 imports)

   - Delete tests/unit/test_sync_manager.py (auto-generated stubs for deleted module)
   - Remove GoogleDriveSyncManager imports from test_redis_setup.py
   - Remove GoogleDriveSyncManager imports from run_compute.py

   Completes Google Drive removal started Nov 14 (3,500 lines removed).
   Now 100% clean - zero references to sync_manager in active code.

   Refs: docs/audits/2025_11_comprehensive_audit/10_legacy_code_report.md"
   ```

2. **Update CLAUDE.md** Recent Work section:
   ```markdown
   ### November 14, 2025: Google Drive Cleanup Complete
   - ✅ Removed 3,500 lines of Google Drive sync code
   - ✅ Removed 5 orphaned imports (446 lines)
   - ✅ 100% clean - zero references in active code
   ```

3. **Run full test suite:**
   ```bash
   pytest tests/unit tests/integration -v
   ```

---

**Completion Time:** 15 minutes
**Status:** Ready to execute
