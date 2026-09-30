#!/usr/bin/env bash
# One-time setup for the trip recap builder.
#
# Checks the tools the pipeline depends on, creates a virtual environment, and
# verifies the whole thing can import and run. Safe to re-run.

set -euo pipefail
cd "$(dirname "$0")"

BOLD=$'\033[1m'; DIM=$'\033[2m'; RED=$'\033[31m'; GREEN=$'\033[32m'
YELLOW=$'\033[33m'; RESET=$'\033[0m'

ok()   { printf "  ${GREEN}ok${RESET}    %s\n" "$1"; }
warn() { printf "  ${YELLOW}warn${RESET}  %s\n" "$1"; }
die()  { printf "  ${RED}fail${RESET}  %s\n" "$1"; exit 1; }
step() { printf "\n${BOLD}%s${RESET}\n" "$1"; }

printf "\n${BOLD}trip recap builder - setup${RESET}\n"

# ------------------------------------------------------------------ tools
step "checking required tools"

missing=()

if ! command -v ffmpeg >/dev/null 2>&1; then
  missing+=("ffmpeg")
else
  ver=$(ffmpeg -version 2>/dev/null | head -1 | cut -d' ' -f3)
  if ffmpeg -hide_banner -filters 2>/dev/null | grep -q ' sidechaincompress '; then
    ok "ffmpeg $ver"
  else
    die "this ffmpeg lacks the sidechaincompress filter, which the audio mix needs.
     install a full build, e.g.  brew install ffmpeg"
  fi
fi

command -v ffprobe >/dev/null 2>&1 || missing+=("ffprobe")
[ ${#missing[@]} -eq 0 ] && ok "ffprobe"

# exiftool is how capture time and GPS get read. Without it the pipeline still
# runs, but loses the single most valuable signal for a trip recap.
if command -v exiftool >/dev/null 2>&1; then
  ok "exiftool $(exiftool -ver 2>/dev/null)"
else
  warn "exiftool not found - capture times and GPS will fall back to file dates
       Install with:  brew install exiftool"
fi

# Needed only for HEIC/HEIF, which iPhones produce by default.
if command -v sips >/dev/null 2>&1; then
  ok "sips (HEIC support)"
else
  warn "sips not found - HEIC photos may not load (non-issue on macOS, which has it)"
fi

if [ ${#missing[@]} -gt 0 ]; then
  die "missing required tools: ${missing[*]}
     install with:  brew install ${missing[*]}"
fi

# -------------------------------------------------------------- python
step "checking python"

PY=""
for cand in python3.13 python3.12 python3.11 python3; do
  if command -v "$cand" >/dev/null 2>&1; then
    v=$("$cand" -c 'import sys; print("%d.%d" % sys.version_info[:2])')
    major=${v%%.*}; minor=${v##*.}
    if [ "$major" -eq 3 ] && [ "$minor" -ge 11 ]; then PY=$cand; break; fi
  fi
done
[ -n "$PY" ] || die "python 3.11 or newer is required and was not found"
ok "$PY ($("$PY" -V 2>&1))"

# --------------------------------------------------------- virtualenv
step "creating the virtual environment"

if [ ! -d .venv ]; then
  # --system-site-packages lets numpy/Pillow/scipy be reused from the system
  # interpreter instead of being downloaded again.
  "$PY" -m venv --system-site-packages .venv
  ok "created .venv"
else
  ok ".venv already exists"
fi

VPY=".venv/bin/python"
[ -x "$VPY" ] || die ".venv/bin/python is missing - delete .venv and re-run setup.sh"

# Only soundfile is genuinely required. numpy and Pillow are usually already
# present via --system-site-packages; opencv adds real face detection, which is
# the single most useful scoring signal on a trip, so it is worth installing.
step "installing python packages"
missing_pkgs=$("$VPY" - <<'PY'
import importlib.util
need = []
for mod in ("numpy", "PIL", "soundfile"):
    if importlib.util.find_spec(mod) is None:
        need.append(mod)
print(" ".join(need))
PY
)

if [ -n "$missing_pkgs" ]; then
  say "installing: $missing_pkgs"
  "$VPY" -m pip install --quiet --upgrade pip
  # shellcheck disable=SC2086
  "$VPY" -m pip install --quiet $missing_pkgs || die "pip install failed"
  ok "installed $missing_pkgs"
else
  ok "numpy, Pillow and soundfile already available"
fi

if ! "$VPY" -c "import cv2" >/dev/null 2>&1; then
  say "installing opencv (enables face detection)"
  if "$VPY" -m pip install --quiet opencv-python-headless >/dev/null 2>&1; then
    ok "opencv installed"
  else
    warn "opencv could not be installed; shots will be scored without face
       detection, which costs a little accuracy on photos of people"
  fi
else
  ok "opencv present (face detection active)"
fi

# ------------------------------------------------------------- self test
step "verifying the pipeline runs"

"$VPY" - <<'PY' || die "the pipeline failed its self-test"
import sys
mods = ["config", "util", "quality", "ingest", "analysis", "geo",
        "context", "music", "select", "render", "report"]
try:
    for m in mods:
        __import__(f"pipeline.{m}")
except Exception as exc:
    print(f"  import failed: {exc.__class__.__name__}: {exc}")
    sys.exit(1)
print(f"  ok    all {len(mods)} pipeline modules import")
PY

"$VPY" make_video.py --help >/dev/null 2>&1 \
  && ok "command line entry point works" \
  || die "make_video.py --help failed"

# -------------------------------------------------------------- folders
step "preparing folders"
mkdir -p media music out .cache
ok "media/  music/  out/  .cache/"

printf "\n"
cat <<EOF
${BOLD}ready${RESET}

  1. put your photos and videos in  ${BOLD}media/${RESET}
  2. run                          ${BOLD}./make_video.py${RESET}

The finished video, a report explaining every decision, and a machine-readable
edit list will be in ${BOLD}out/${RESET}. Nothing is uploaded anywhere.

  ./make_video.py --target 300     five minutes instead of ninety seconds
  ./make_video.py --dry-run        decide everything, render nothing

See README.md for the full option list and how the decisions are made.
EOF
printf "\n"
