#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HEADER_ROOT="${ROOT_DIR}/include"
CVH_HEADER_ROOT="${HEADER_ROOT}/cvh"
ACTIVE_CODE_PATHS=(
  "${ROOT_DIR}/CMakeLists.txt"
  "${ROOT_DIR}/cmake"
  "${ROOT_DIR}/include"
  "${ROOT_DIR}/benchmark"
  "${ROOT_DIR}/scripts"
  "${ROOT_DIR}/test"
)

FORBIDDEN_INCLUDE_REGEX='#include[[:space:]]*[<"](test/|src/|backend/|libnpy/)'
FORBIDDEN_MODE_MACRO_REGEX='(^|[^[:alnum:]_])CVH_(LITE|NATIVE|FULL)([^[:alnum:]_]|$)'
RETIRED_FEATURE_MACRO_REGEX='CVH_ENABLE_(THREADS|FAST_MATH|PLATFORM_INTRINSICS|NATIVE_INTRINSICS|NATIVE_NEON(_AUTO)?|NATIVE_AVX2(_AUTO)?|OPENCV_INTRIN|DIRECT_INTRINSICS|DIRECT_NEON|DIRECT_AVX2)'

for required_header in \
  "${CVH_HEADER_ROOT}/cvh.h" \
  "${CVH_HEADER_ROOT}/core/mat.h" \
  "${CVH_HEADER_ROOT}/imgproc/imgproc.h" \
  "${CVH_HEADER_ROOT}/imgcodecs/imgcodecs.h" \
  "${CVH_HEADER_ROOT}/highgui/highgui.h"; do
  if [[ ! -f "${required_header}" ]]; then
    echo "Missing required public entry header: ${required_header}" >&2
    exit 1
  fi
done

if command -v rg >/dev/null 2>&1; then
  if rg -n "${FORBIDDEN_INCLUDE_REGEX}" "${HEADER_ROOT}" -g '!include/cvh/3rdparty/**'; then
    echo "Found forbidden include path in public headers." >&2
    exit 1
  fi
else
  if grep -RInE "${FORBIDDEN_INCLUDE_REGEX}" "${HEADER_ROOT}" --exclude-dir=3rdparty; then
    echo "Found forbidden include path in public headers." >&2
    exit 1
  fi
fi

if command -v rg >/dev/null 2>&1; then
  if rg -n \
      -e "${FORBIDDEN_MODE_MACRO_REGEX}" \
      -e "${RETIRED_FEATURE_MACRO_REGEX}" \
      "${ACTIVE_CODE_PATHS[@]}" -g '!include/cvh/3rdparty/**'; then
    echo "Found a removed build-mode or feature macro in active code." >&2
    exit 1
  fi
else
  if grep -RInE \
      "${FORBIDDEN_MODE_MACRO_REGEX}|${RETIRED_FEATURE_MACRO_REGEX}" \
      "${ACTIVE_CODE_PATHS[@]}" --exclude-dir=3rdparty; then
    echo "Found a removed build-mode or feature macro in active code." >&2
    exit 1
  fi
fi

echo "Public header boundary, dependency, and macro checks passed."
