#!/usr/bin/env bash
# Linux setup: a pinned native GDAL runtime and this checkout's separate uv env.
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ "$(uname -s)-$(uname -m)" != Linux-x86_64 ]]; then
  echo 'This native lock targets Linux x86_64. See README for other native installations.' >&2
  exit 1
fi
native_prefix="${NATIVE_PREFIX:-$PWD/.native}"
if [[ ! -x "$native_prefix/bin/gdal-config" ]]; then
  conda_executable="${CONDA_EXE:-conda}"
  "$conda_executable" create --yes --prefix "$native_prefix" --file native-linux-64.lock
fi
native_version="$("$native_prefix/bin/gdal-config" --version)"
if [[ "$native_version" != 3.13.0 ]]; then
  echo "Expected native GDAL 3.13.0; found $native_version" >&2
  exit 1
fi
PATH="$native_prefix/bin:$PATH" \
  GDAL_CONFIG="$native_prefix/bin/gdal-config" \
  LD_LIBRARY_PATH="$native_prefix/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
  uv sync --locked
python3 - "$native_prefix" <<'PY'
import pathlib, sys
path = pathlib.Path('.env')
text = path.read_text() if path.exists() else pathlib.Path('.env.example').read_text()
lines = [line for line in text.splitlines() if not line.startswith('NATIVE_PREFIX=')]
lines.append('NATIVE_PREFIX=' + str(pathlib.Path(sys.argv[1]).resolve()))
path.write_text('\n'.join(lines) + '\n')
PY
echo 'Python environment ready. The README contains the application launch command.'
