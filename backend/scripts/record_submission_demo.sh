#!/usr/bin/env bash
set -euo pipefail

mode="${1:-government}"
duration="${2:-180}"
output_dir="${3:-submission-recordings}"

if [[ "$mode" != "government" && "$mode" != "own-feed" ]]; then
  echo "Usage: $0 government|own-feed [duration-seconds] [output-directory]" >&2
  exit 2
fi
if ! [[ "$duration" =~ ^[0-9]+$ ]] || (( duration < 60 || duration > 600 )); then
  echo "Duration must be an integer from 60 to 600 seconds." >&2
  exit 2
fi
if [[ -z "${DISPLAY:-}" ]]; then
  echo "DISPLAY must point to the desktop session to record." >&2
  exit 2
fi
if ! command -v ffmpeg >/dev/null 2>&1; then
  echo "ffmpeg is required." >&2
  exit 2
fi

mkdir -p "$output_dir"
timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
output="$output_dir/intel-i-${mode}-${timestamp}.mp4"

ffmpeg -hide_banner -loglevel warning -y \
  -f x11grab -framerate 30 -video_size "${RECORDING_SIZE:-1920x1080}" -i "${DISPLAY}.0+0,0" \
  -t "$duration" -c:v libx264 -preset veryfast -crf 20 -pix_fmt yuv420p -movflags +faststart "$output"

echo "$output"
