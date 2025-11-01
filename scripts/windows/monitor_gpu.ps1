# PowerShell GPU Monitoring Script for RTX 4090
# Run as: .\scripts\windows\monitor_gpu.ps1

param(
    [int]$RefreshRate = 2,
    [switch]$Detailed,
    [switch]$LogToFile
)

Write-Host "🎮 RTX 4090 GPU Monitor" -ForegroundColor Cyan
Write-Host "Press Ctrl+C to exit" -ForegroundColor Yellow
Write-Host ""

# Log file setup
$logFile = "logs\gpu_monitor_$(Get-Date -Format 'yyyyMMdd_HHmmss').csv"
if ($LogToFile) {
    "Timestamp,GPU_Util,Memory_Used_GB,Memory_Total_GB,Temp_C,Power_W" | Out-File $logFile
    Write-Host "Logging to: $logFile" -ForegroundColor Green
}

# Function to get GPU stats
function Get-GPUStats {
    $stats = nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw,fan.speed --format=csv,noheader,nounits
    $values = $stats -split ', '
    
    return @{
        Utilization = [float]$values[0]
        MemoryUsedMB = [float]$values[1]
        MemoryTotalMB = [float]$values[2]
        Temperature = [float]$values[3]
        PowerDraw = [float]$values[4]
        FanSpeed = [float]$values[5]
    }
}

# Function to get process information
function Get-GPUProcesses {
    $processes = nvidia-smi --query-compute-apps=pid,name,used_memory --format=csv,noheader,nounits
    $processList = @()
    
    if ($processes) {
        foreach ($proc in $processes -split "`n") {
            if ($proc.Trim()) {
                $parts = $proc -split ', '
                $processList += @{
                    PID = $parts[0]
                    Name = $parts[1]
                    MemoryMB = [float]$parts[2]
                }
            }
        }
    }
    return $processList
}

# Function to create utilization bar
function Get-UtilizationBar {
    param([float]$Percentage, [int]$Width = 20)
    
    $filled = [math]::Round($Percentage * $Width / 100)
    $empty = $Width - $filled
    
    $bar = "█" * $filled + "░" * $empty
    
    # Color based on utilization
    if ($Percentage -gt 80) {
        return "$bar" # Green (good utilization)
    } elseif ($Percentage -gt 50) {
        return "$bar" # Yellow
    } else {
        return "$bar" # Red (underutilized)
    }
}

# Main monitoring loop
$iteration = 0
while ($true) {
    Clear-Host
    $iteration++
    
    # Get current stats
    $stats = Get-GPUStats
    $memoryGB = $stats.MemoryUsedMB / 1024
    $memoryTotalGB = $stats.MemoryTotalMB / 1024
    $memoryPercent = ($stats.MemoryUsedMB / $stats.MemoryTotalMB) * 100
    
    # Header
    Write-Host "╔══════════════════════════════════════════════════════════╗" -ForegroundColor Cyan
    Write-Host "║          RTX 4090 GPU Monitor - $(Get-Date -Format 'HH:mm:ss')          ║" -ForegroundColor Cyan
    Write-Host "╚══════════════════════════════════════════════════════════╝" -ForegroundColor Cyan
    Write-Host ""
    
    # GPU Utilization
    $utilBar = Get-UtilizationBar -Percentage $stats.Utilization
    $utilColor = if ($stats.Utilization -gt 80) { "Green" } elseif ($stats.Utilization -gt 50) { "Yellow" } else { "Red" }
    Write-Host "GPU Utilization: " -NoNewline
    Write-Host ("{0,5:N1}%" -f $stats.Utilization) -ForegroundColor $utilColor -NoNewline
    Write-Host " [$utilBar]"
    
    # Memory Usage
    $memBar = Get-UtilizationBar -Percentage $memoryPercent
    $memColor = if ($memoryPercent -gt 70) { "Green" } elseif ($memoryPercent -gt 40) { "Yellow" } else { "Red" }
    Write-Host "Memory Usage:    " -NoNewline
    Write-Host ("{0,5:N1}/{1,5:N1} GB" -f $memoryGB, $memoryTotalGB) -ForegroundColor $memColor -NoNewline
    Write-Host " [$memBar]"
    
    # Temperature
    $tempColor = if ($stats.Temperature -gt 83) { "Red" } elseif ($stats.Temperature -gt 75) { "Yellow" } else { "Green" }
    Write-Host "Temperature:     " -NoNewline
    Write-Host ("{0,3:N0}°C" -f $stats.Temperature) -ForegroundColor $tempColor -NoNewline
    Write-Host ("  (Fan: {0,3:N0}%)" -f $stats.FanSpeed)
    
    # Power Draw
    Write-Host "Power Draw:      " -NoNewline
    Write-Host ("{0,3:N0}W" -f $stats.PowerDraw) -ForegroundColor White -NoNewline
    Write-Host " / 450W max"
    
    Write-Host ""
    
    # Detailed process information
    if ($Detailed) {
        $processes = Get-GPUProcesses
        if ($processes.Count -gt 0) {
            Write-Host "Active GPU Processes:" -ForegroundColor Yellow
            Write-Host "─────────────────────────────────────────" -ForegroundColor DarkGray
            foreach ($proc in $processes) {
                $procMemGB = $proc.MemoryMB / 1024
                Write-Host ("  PID {0,6}: {1,-20} {2,6:N2} GB" -f $proc.PID, $proc.Name, $procMemGB)
            }
            Write-Host ""
        }
    }
    
    # Training detection
    $trainingProcesses = Get-Process -ErrorAction SilentlyContinue | Where-Object {
        $_.ProcessName -eq "python" -or $_.ProcessName -eq "python3"
    }
    
    if ($trainingProcesses) {
        Write-Host "🔥 Training Jobs Detected:" -ForegroundColor Green
        foreach ($proc in $trainingProcesses) {
            $cpuPercent = 0
            try {
                $cpuPercent = (Get-Counter "\Process($($proc.ProcessName))\% Processor Time" -ErrorAction SilentlyContinue).CounterSamples[0].CookedValue
            } catch {}
            
            Write-Host ("  {0,-20} CPU: {1,5:N1}% Memory: {2,6:N0} MB" -f 
                $proc.ProcessName, 
                $cpuPercent,
                ($proc.WorkingSet64 / 1MB))
        }
        Write-Host ""
    }
    
    # Performance indicators
    Write-Host "Performance Status:" -ForegroundColor Cyan
    if ($stats.Utilization -gt 90) {
        Write-Host "  🚀 Excellent - GPU fully utilized!" -ForegroundColor Green
    } elseif ($stats.Utilization -gt 70) {
        Write-Host "  ✅ Good - Training efficiently" -ForegroundColor Green
    } elseif ($stats.Utilization -gt 50) {
        Write-Host "  ⚠️  Moderate - Could increase batch size" -ForegroundColor Yellow
    } else {
        Write-Host "  ⚡ Low - GPU underutilized or idle" -ForegroundColor Red
    }
    
    # Temperature warning
    if ($stats.Temperature -gt 83) {
        Write-Host "  ⚠️  WARNING: High temperature! Consider improving cooling" -ForegroundColor Red
    }
    
    # Log to file if requested
    if ($LogToFile) {
        $logEntry = "{0},{1},{2:N2},{3:N2},{4},{5}" -f 
            (Get-Date -Format "yyyy-MM-dd HH:mm:ss"),
            $stats.Utilization,
            $memoryGB,
            $memoryTotalGB,
            $stats.Temperature,
            $stats.PowerDraw
        $logEntry | Out-File $logFile -Append
    }
    
    # Footer
    Write-Host ""
    Write-Host ("─" * 60) -ForegroundColor DarkGray
    Write-Host "Iteration: $iteration | Refresh: ${RefreshRate}s | " -NoNewline -ForegroundColor DarkGray
    if ($LogToFile) {
        Write-Host "Logging: ON" -ForegroundColor Green -NoNewline
    } else {
        Write-Host "Logging: OFF" -ForegroundColor DarkGray -NoNewline
    }
    Write-Host " | Press Ctrl+C to exit" -ForegroundColor DarkGray
    
    Start-Sleep -Seconds $RefreshRate
}