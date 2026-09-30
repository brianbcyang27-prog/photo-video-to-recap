#!/usr/bin/env bash
# Run the pipeline across the options that matter and verify every render.
#
#   ./tools/sweep.sh          # the standard matrix
#
# Each combination is rendered for real and then measured by verify_output.py,
# which reads the rendered file rather than trusting the plan. A run that fails
# to render, or renders something that does not verify, is reported here.

cd "$(dirname "$0")/.." || exit 1
PY=.venv/bin/python
MEDIA="${SWEEP_MEDIA:-media}"
[ -x "$PY" ] || { echo "run ./setup.sh first"; exit 1; }

pass=0
fail=0
declare -a failed=()

run_case() {
  local label="$1"; shift
  local name="sweep_$(echo "$label" | tr -cd '[:alnum:]' | cut -c1-40)"
  local out="out/${name}.mp4"

  if ! "$PY" make_video.py --media "$MEDIA" --name "$name" "$@" >"/tmp/${name}.log" 2>&1; then
    printf '  %-42s RENDER FAILED\n' "$label"
    tail -3 "/tmp/${name}.log" | sed 's/^/      /'
    fail=$((fail + 1)); failed+=("$label (render)")
    return
  fi

  local result
  result=$("$PY" tools/verify_output.py "$out" 2>&1)
  local summary
  summary=$(echo "$result" | grep -E '^[0-9]+/[0-9]+ checks passed' | tail -1)
  local bad
  bad=$(echo "$result" | grep -c '^  FAIL')

  if [ -z "$summary" ]; then
    printf '  %-42s NO VERDICT\n' "$label"
    fail=$((fail + 1)); failed+=("$label (verdict)")
    return
  fi
  if [ "$bad" -eq 0 ]; then
    printf '  %-42s %s\n' "$label" "$summary"
    pass=$((pass + 1))
  else
    printf '  %-42s %s   (%s failed)\n' "$label" "$summary" "$bad"
    echo "$result" | grep '^  FAIL' | sed 's/^/      /'
    fail=$((fail + 1)); failed+=("$label ($bad checks)")
  fi
  rm -f "$out"
}

echo
echo "duration"
run_case "target 60s"   --target 60
run_case "target 90s"   --target 90
run_case "target 120s"  --target 120
run_case "target 300s"  --target 300

echo
echo "music"
run_case "mood cinematic"   --target 90 --mood cinematic
run_case "mood energetic"   --target 90 --mood energetic
run_case "mood calm"        --target 90 --mood calm
run_case "mood warm"        --target 90 --mood warm
run_case "no ducking"       --target 90 --no-duck
run_case "supplied track"   --target 150 --music-track music/test_track_103bpm.wav

echo
echo "framing"
run_case "fit crop"       --target 90 --fit crop
run_case "fit pad"        --target 90 --fit pad
run_case "no ken burns"   --target 90 --no-ken-burns
run_case "24 fps"         --target 90 --fps 24

echo
echo "structure"
run_case "order hook"     --target 90 --order hook
run_case "no titles"      --target 90 --no-titles
run_case "custom title"   --target 90 --title "Kyoto in Autumn"

echo
echo "=============================================================="
if [ "$fail" -eq 0 ]; then
  echo "  all $pass combinations passed"
else
  echo "  $pass passed, $fail FAILED:"
  for f in "${failed[@]}"; do echo "    - $f"; done
fi
echo "=============================================================="
exit $([ "$fail" -eq 0 ] && echo 0 || echo 1)
