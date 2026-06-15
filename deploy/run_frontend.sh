#!/usr/bin/env bash
# Launch the Czech Gradio web UI for MedSAM3 on this server.
# Prereqs: setup_ec2.sh / CPU setup done, HF logged in, LoRA weights downloaded,
#          and (on a CPU box) deploy/cpu_compat.sh already applied.
#
# Usage:  bash deploy/run_frontend.sh
#   -> prints a public https://....gradio.live link to share.
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

source ~/venv/bin/activate 2>/dev/null || true
pip install -q gradio

echo "Starting Gradio (model loads once, ~1 min)…"
nohup python3 gradio_app.py > ~/frontend.log 2>&1 &
echo "pid $!  — log: ~/frontend.log"
echo "Waiting for public URL…"
for i in $(seq 1 40); do
  url=$(grep -oE 'https://[a-z0-9]+\.gradio\.live' ~/frontend.log | head -1 || true)
  [ -n "$url" ] && { echo "PUBLIC URL: $url"; break; }
  grep -qi 'Traceback\|Error' ~/frontend.log && { echo "--- error ---"; tail -20 ~/frontend.log; break; }
  sleep 5
done
