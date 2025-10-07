#!/usr/bin/env python3
"""
Monitor running compute tasks and upgrade their intensity when round completes.
"""

import subprocess
import time
import sys
import os
import signal
from datetime import datetime

# Target processes to monitor
TARGET_PIDS = [64774, 52241]

def check_process_running(pid):
    """Check if a process is still running."""
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False

def get_process_cmd(pid):
    """Get the command line for a process."""
    try:
        result = subprocess.run(['ps', '-p', str(pid), '-o', 'command='],
                              capture_output=True, text=True)
        if result.returncode == 0:
            return result.stdout.strip()
    except:
        pass
    return None

def monitor_round_completion(pid):
    """Monitor process output to detect round completion."""
    print(f"📊 Monitoring PID {pid} for round completion...")

    # Use lsof to find open files and look for log patterns
    try:
        # Check if process is writing to any log files
        result = subprocess.run(['lsof', '-p', str(pid)],
                              capture_output=True, text=True)

        # For now, we'll wait for the process to complete
        # In production, we'd tail log files or check Redis task status
        return not check_process_running(pid)
    except:
        return False

def restart_with_high_intensity(original_cmd, pid):
    """Restart a command with high intensity."""
    # Replace intensity flag in command
    new_cmd = original_cmd.replace('--intensity low', '--intensity high')
    new_cmd = new_cmd.replace('--intensity medium', '--intensity high')

    # If no intensity flag, add it
    if '--intensity' not in new_cmd:
        new_cmd += ' --intensity high'

    print(f"🚀 Starting new process with high intensity:")
    print(f"   Command: {new_cmd}")

    # Start the new process in background
    subprocess.Popen(new_cmd, shell=True)
    print(f"✅ Process upgraded to high intensity")

def main():
    print("🔍 NFL Analytics Task Intensity Monitor")
    print("=" * 60)
    print(f"Monitoring PIDs: {TARGET_PIDS}")
    print()

    # Initial check
    process_info = {}
    for pid in TARGET_PIDS:
        if check_process_running(pid):
            cmd = get_process_cmd(pid)
            if cmd:
                process_info[pid] = {
                    'cmd': cmd,
                    'status': 'running',
                    'upgraded': False
                }
                print(f"✅ PID {pid} is running")
                print(f"   Command: {cmd[:80]}...")
            else:
                print(f"⚠️ PID {pid} running but couldn't get command")
        else:
            print(f"❌ PID {pid} is not running")

    if not process_info:
        print("\n❌ No target processes found running")
        return 1

    print("\n📊 Monitoring for round completion...")
    print("(Checking every 30 seconds)")

    # Monitor loop
    check_interval = 30  # seconds
    while process_info:
        time.sleep(check_interval)

        for pid, info in list(process_info.items()):
            if info['upgraded']:
                continue

            if not check_process_running(pid):
                print(f"\n🏁 PID {pid} has completed!")

                # Restart with high intensity
                restart_with_high_intensity(info['cmd'], pid)
                info['upgraded'] = True

                # Check if all processes upgraded
                if all(p['upgraded'] for p in process_info.values()):
                    print("\n✅ All processes upgraded to high intensity")
                    return 0
            else:
                print(f"⏳ PID {pid} still running...", end='\r')

    return 0

if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n\n⚠️ Monitor interrupted by user")
        sys.exit(0)