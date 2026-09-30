param(
  [string]$DatabaseUrl = $env:DATABASE_URL,
  [string]$Output = "intel_i_backup_$(Get-Date -Format yyyyMMdd_HHmmss).dump"
)
if ([string]::IsNullOrWhiteSpace($DatabaseUrl)) { throw "DATABASE_URL is required" }
pg_dump $DatabaseUrl --format=custom --file=$Output
if ($LASTEXITCODE -ne 0) { throw "pg_dump failed" }
Write-Host "Backup created: $Output"
