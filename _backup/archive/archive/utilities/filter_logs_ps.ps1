# Filter sensitive information from strategy.log
$logFile = "logs\strategy.log"
$content = Get-Content $logFile -Raw

# Replace sensitive information with placeholders
$content = $content -replace '(\[DEBUG\] get_fyers_client: client_id=)[^\s,]+', '$1***FILTERED***'
$content = $content -replace '(access_token_head=)[^\s,]+', '$1***FILTERED***'
$content = $content -replace '(token_combo=)[^\s,]+', '$1***FILTERED***'

# Write filtered content back to the file
$content | Set-Content $logFile

Write-Host "Sensitive information has been filtered from $logFile"
