# Cleanup script for the finalized strategies project
# This script removes unnecessary files and cleans up the project structure
# It creates backups of all removed files before deleting them

Write-Host "==============================================================="
Write-Host "             FINALIZED STRATEGIES CLEANUP SCRIPT               "
Write-Host "==============================================================="
Write-Host "This script will clean up the project by removing test files,"
Write-Host "duplicate scripts, backup files, and other unnecessary items."
Write-Host "All removed files will be backed up first."
Write-Host ""

# Create a backup directory for removed files (just to be safe)
$backupDir = "c:\vs code projects\finalized strategies\cleanup_backups"
$date = Get-Date -Format "yyyyMMdd_HHmmss"
$backupDir = Join-Path $backupDir $date
New-Item -ItemType Directory -Path $backupDir -Force | Out-Null
Write-Host "Created backup directory: $backupDir"

# Function to move a file to backup instead of deleting
# Arrays to track results
$removedFiles = @()
$notFoundFiles = @()
$errorFiles = @()

function BackupFile {
    param (
        [string]$filePath
    )
    
    if (Test-Path $filePath) {
        try {
            # Create subdirectories in the backup location if needed
            $relativePath = $filePath.Replace($baseDir, "").TrimStart("\")
            $destDir = Split-Path -Path (Join-Path $backupDir $relativePath) -Parent
            
            if (-not (Test-Path $destDir)) {
                New-Item -ItemType Directory -Path $destDir -Force | Out-Null
            }
            
            $fileName = Split-Path $filePath -Leaf
            $destPath = Join-Path $destDir $fileName
            
            # If file with same name exists, add a suffix
            $counter = 1
            while (Test-Path $destPath) {
                $fileName = [System.IO.Path]::GetFileNameWithoutExtension($filePath) + "_$counter" + [System.IO.Path]::GetExtension($filePath)
                $destPath = Join-Path $destDir $fileName
                $counter++
            }
            
            Copy-Item -Path $filePath -Destination $destPath
            Remove-Item -Path $filePath
            Write-Host "Backed up and removed: $filePath"
            
            # Track for summary
            $global:removedFiles += $filePath
        }
        catch {
            Write-Host "Error backing up file: $filePath - $($_.Exception.Message)" -ForegroundColor Red
            $global:errorFiles += $filePath
        }
    } else {
        Write-Host "File not found: $filePath" -ForegroundColor Yellow
        $global:notFoundFiles += $filePath
    }
}

function GenerateCleanupReport {
    $reportPath = Join-Path $baseDir "cleanup_report.md"
    
    $reportContent = @"
# Project Cleanup Report

## Summary
- **Date and Time:** $(Get-Date -Format "yyyy-MM-dd HH:mm:ss")
- **Total Files Removed:** $($removedFiles.Count)
- **Files Not Found:** $($notFoundFiles.Count)
- **Errors Encountered:** $($errorFiles.Count)
- **Backup Location:** $backupDir

## Files Successfully Removed

"@
    
    if ($removedFiles.Count -gt 0) {
        foreach ($file in $removedFiles) {
            $relativePath = $file.Replace($baseDir, "").TrimStart("\")
            $reportContent += "- $relativePath`n"
        }
    } else {
        $reportContent += "No files were removed.`n"
    }
    
    $reportContent += "`n## Files Not Found`n`n"
    
    if ($notFoundFiles.Count -gt 0) {
        foreach ($file in $notFoundFiles) {
            $relativePath = $file.Replace($baseDir, "").TrimStart("\")
            $reportContent += "- $relativePath`n"
        }
    } else {
        $reportContent += "All files were found.`n"
    }
    
    $reportContent += "`n## Files With Errors`n`n"
    
    if ($errorFiles.Count -gt 0) {
        foreach ($file in $errorFiles) {
            $relativePath = $file.Replace($baseDir, "").TrimStart("\")
            $reportContent += "- $relativePath`n"
        }
    } else {
        $reportContent += "No errors were encountered.`n"
    }
    
    $reportContent | Out-File -FilePath $reportPath -Encoding utf8
    Write-Host "Cleanup report generated: $reportPath" -ForegroundColor Green
}

# Base directory
$baseDir = "c:\vs code projects\finalized strategies"

# List of test files to remove
$testFiles = @(
    "simple_strategy_test.py",
    "test_trailing_stop_loss.py",
    "direct_test.py",
    "simple_test.py",
    "check_methods.py",
    "method_check.py",
    "verify_methods.py",
    "simple_check.py",
    "ultimate_simple_test.py",
    "final_test.py",
    "bare_minimum_test.py",
    "simple_test.py",
    "test_symbol_conversion.py",
    "test_fyers_option_format.py"
)

# List of duplicate run scripts to remove
$duplicateRunScripts = @(
    "run_simple_strategy.py",
    "run_standalone.py",
    "final_run.py",
    "run_fixed.py",
    "run_strategy_with_fixed_format.py",
    "run_strategy_with_improved_websocket.py"
)

# List of fix and debugging files to remove
$fixFiles = @(
    "direct_fix_trailing_stoploss.py",
    "comprehensive_syntax_fix.py",
    "enhanced_syntax_fix.py",
    "direct_comprehensive_fix.py",
    "patch_strategy.py",
    "standalone_strategy.py",
    "standalone_trailing_test.py",
    "fix_strategy_clean.py",
    "fix_all_issues.py",
    "fix_trailing_stoploss.py"
)

# List of batch files to remove
$batchFiles = @(
    "fix_and_run.bat",
    "run_tests_and_strategy.bat",
    "run_strategy_filtered.bat",
    "run_strategy_with_improved_websocket.bat",
    "run_strategy_secure.bat"
)

Write-Host ""
Write-Host "Starting cleanup process..." -ForegroundColor Green
Write-Host ""

# 1. Remove test files
Write-Host "Removing test files..." -ForegroundColor Cyan
foreach ($file in $testFiles) {
    $filePath = Join-Path $baseDir $file
    BackupFile $filePath
}

# 2. Remove duplicate run scripts
Write-Host ""
Write-Host "Removing duplicate run scripts..." -ForegroundColor Cyan
foreach ($file in $duplicateRunScripts) {
    $filePath = Join-Path $baseDir $file
    BackupFile $filePath
}

# 3. Remove fix and debugging files
Write-Host ""
Write-Host "Removing fix and debugging files..." -ForegroundColor Cyan
foreach ($file in $fixFiles) {
    $filePath = Join-Path $baseDir $file
    BackupFile $filePath
}

# 4. Remove batch files
Write-Host ""
Write-Host "Removing unnecessary batch files..." -ForegroundColor Cyan
foreach ($file in $batchFiles) {
    $filePath = Join-Path $baseDir $file
    BackupFile $filePath
}

# 5. Remove files with specific prefixes
Write-Host ""
Write-Host "Removing files with fix_ and debug_ prefixes..." -ForegroundColor Cyan
$fixPrefixFiles = Get-ChildItem -Path $baseDir -Filter "fix_*.py" -File -Recurse
foreach ($file in $fixPrefixFiles) {
    BackupFile $file.FullName
}

$debugPrefixFiles = Get-ChildItem -Path $baseDir -Filter "debug_*.py" -File -Recurse
foreach ($file in $debugPrefixFiles) {
    BackupFile $file.FullName
}

# 6. Remove backup files and other temporary files
Write-Host ""
Write-Host "Removing backup and temporary files..." -ForegroundColor Cyan

# .bak files in the root directory (not in the backups folder)
$bakFiles = Get-ChildItem -Path $baseDir -Filter "*.bak" -File -Recurse | Where-Object { $_.DirectoryName -notlike "*\backups*" }
foreach ($file in $bakFiles) {
    BackupFile $file.FullName
}

# Strategy backup files
$strategyBackupFiles = Get-ChildItem -Path $baseDir -Filter "strategy.py.*" -Recurse | Where-Object { $_.Name -ne "strategy.py" }
foreach ($file in $strategyBackupFiles) {
    BackupFile $file.FullName
}

# Files with backup in the name
$backupNameFiles = Get-ChildItem -Path $baseDir -Filter "*backup*" -File -Recurse
foreach ($file in $backupNameFiles) {
    BackupFile $file.FullName
}

# Files with .old extension
$oldFiles = Get-ChildItem -Path $baseDir -Filter "*.old" -File -Recurse
foreach ($file in $oldFiles) {
    BackupFile $file.FullName
}

# 7. Move the archive directory
Write-Host ""
Write-Host "Processing archive directory..." -ForegroundColor Cyan

if (Test-Path "$baseDir\archive") {
    # Create archive backup directory
    $archiveBackupDir = Join-Path $backupDir "archive"
    New-Item -ItemType Directory -Path $archiveBackupDir -Force | Out-Null
    
    # Copy all archive files to backup
    Copy-Item -Path "$baseDir\archive\*" -Destination $archiveBackupDir -Recurse
    
    # Remove the archive directory
    Remove-Item -Path "$baseDir\archive" -Recurse -Force
    Write-Host "Archive directory backed up and removed"
} else {
    Write-Host "Archive directory not found" -ForegroundColor Yellow
}

# 8. Clean up src directory backup files
Write-Host ""
Write-Host "Cleaning up src directory..." -ForegroundColor Cyan

$srcBackupFiles = Get-ChildItem -Path "$baseDir\src" -Filter "*.py.*" -File
foreach ($file in $srcBackupFiles) {
    BackupFile $file.FullName
}

# 9. Create a unified directory structure (optional)
# Note: This is commented out as it modifies core files - uncomment if you want this functionality
<#
Write-Host ""
Write-Host "Organizing remaining files into a cleaner structure..." -ForegroundColor Cyan

# Create main directories if they don't exist
$dirConfig = Join-Path $baseDir "config"
$dirDocs = Join-Path $baseDir "docs"
$dirSrc = Join-Path $baseDir "src"
$dirLogs = Join-Path $baseDir "logs"

if (-not (Test-Path $dirConfig)) {
    New-Item -ItemType Directory -Path $dirConfig -Force | Out-Null
}

if (-not (Test-Path $dirDocs)) {
    New-Item -ItemType Directory -Path $dirDocs -Force | Out-Null
}

# Create a docs directory and move documentation there
if ((Test-Path "$baseDir\README.md") -and (Test-Path $dirDocs)) {
    Copy-Item -Path "$baseDir\README.md" -Destination "$dirDocs\README.md"
}

if ((Test-Path "$baseDir\DIRECTORY_STRUCTURE.md") -and (Test-Path $dirDocs)) {
    Copy-Item -Path "$baseDir\DIRECTORY_STRUCTURE.md" -Destination "$dirDocs\DIRECTORY_STRUCTURE.md"
}
#>

# Generate a cleanup report
GenerateCleanupReport

# Done!
Write-Host ""
Write-Host "==============================================================="
Write-Host "Cleanup completed successfully!" -ForegroundColor Green
Write-Host "All removed files were backed up to: $backupDir"
Write-Host "A detailed cleanup report has been saved to: $baseDir\cleanup_report.md"
Write-Host "==============================================================="

Write-Host "Project cleanup completed successfully!"
Write-Host "All removed files have been backed up to: $backupDir"
