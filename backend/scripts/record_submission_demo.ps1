param(
  [ValidateSet("government", "own-feed")]
  [string]$Mode = "government",
  [ValidateRange(60, 600)]
  [int]$DurationSeconds = 180,
  [string]$OutputDirectory = "submission-recordings"
)

$ErrorActionPreference = "Stop"
if (-not (Get-Command ffmpeg -ErrorAction SilentlyContinue)) {
  throw "ffmpeg is required and must be available on PATH."
}
New-Item -ItemType Directory -Force -Path $OutputDirectory | Out-Null
$Timestamp = (Get-Date).ToUniversalTime().ToString("yyyyMMddTHHmmssZ")
$Output = Join-Path $OutputDirectory "intel-i-$Mode-$Timestamp.mp4"

& ffmpeg -hide_banner -loglevel warning -y `
  -f gdigrab -framerate 30 -i desktop `
  -t $DurationSeconds -c:v libx264 -preset veryfast -crf 20 `
  -pix_fmt yuv420p -movflags +faststart $Output

if ($LASTEXITCODE -ne 0) { throw "ffmpeg recording failed with exit code $LASTEXITCODE" }
Write-Output $Output
