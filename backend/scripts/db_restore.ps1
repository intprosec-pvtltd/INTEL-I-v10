param(
  [string]$DatabaseUrl = $env:DATABASE_URL,
  [Parameter(Mandatory=$true)][string]$InputFile
)
if ([string]::IsNullOrWhiteSpace($DatabaseUrl)) { throw "DATABASE_URL is required" }
if (!(Test-Path $InputFile)) { throw "Backup file not found: $InputFile" }
pg_restore --clean --if-exists --no-owner --dbname=$DatabaseUrl $InputFile
if ($LASTEXITCODE -ne 0) { throw "pg_restore failed" }
Write-Host "Restore completed from: $InputFile"
