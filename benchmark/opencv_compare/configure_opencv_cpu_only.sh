#!/usr/bin/env bash

set -euo pipefail

COMPARE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${COMPARE_DIR}/../.." && pwd)"
OPENCV_DIR="${CVH_OPENCV_DIR:-${ROOT_DIR}/../opencv}"
BUILD_DIR="${CVH_OPENCV_CPU_ONLY_BUILD_DIR:-${OPENCV_DIR}/build-cpu-only}"
BUILD_TYPE="${CVH_COMPARE_BUILD_TYPE:-Release}"
JOBS="${CVH_BUILD_JOBS:-$(sysctl -n hw.logicalcpu 2>/dev/null || getconf _NPROCESSORS_ONLN 2>/dev/null || printf '%s' 4)}"
CONFIGURE_ONLY=0

usage() {
  cat <<USAGE
Usage: $(basename "$0") [--configure-only] [--help]

Builds a separate upstream OpenCV core/imgproc configuration for CPU-only
GEMM attribution. It never modifies the normal build-slim directory.

Environment:
  CVH_OPENCV_DIR                 (default: ${OPENCV_DIR})
  CVH_OPENCV_CPU_ONLY_BUILD_DIR  (default: ${BUILD_DIR})
  CVH_COMPARE_BUILD_TYPE         (default: ${BUILD_TYPE})
  CVH_BUILD_JOBS                 (default: ${JOBS})
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --configure-only)
      CONFIGURE_ONLY=1
      shift
      ;;
    --help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown arg: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ ! -f "${OPENCV_DIR}/CMakeLists.txt" ]]; then
  echo "OpenCV source tree not found: ${OPENCV_DIR}" >&2
  exit 2
fi

cmake -S "${OPENCV_DIR}" -B "${BUILD_DIR}" \
  -DCMAKE_BUILD_TYPE="${BUILD_TYPE}" \
  -DBUILD_LIST=core,imgproc \
  -DBUILD_TESTS=OFF \
  -DBUILD_PERF_TESTS=OFF \
  -DBUILD_EXAMPLES=OFF \
  -DBUILD_opencv_apps=OFF \
  -DBUILD_JAVA=OFF \
  -DBUILD_opencv_python3=OFF \
  -DBUILD_opencv_python_bindings_generator=OFF \
  -DWITH_CAROTENE=OFF \
  -DWITH_EIGEN=OFF \
  -DWITH_IPP=OFF \
  -DWITH_KLEIDICV=OFF \
  -DWITH_LAPACK=OFF \
  -DWITH_OPENCL=OFF

if ! grep -Eq '^WITH_LAPACK:(BOOL|UNINITIALIZED)=OFF$' "${BUILD_DIR}/CMakeCache.txt"; then
  echo "CPU-only gate failed: WITH_LAPACK is not OFF" >&2
  exit 3
fi
if ! grep -Eq '^WITH_IPP:(BOOL|UNINITIALIZED)=OFF$' "${BUILD_DIR}/CMakeCache.txt"; then
  echo "CPU-only gate failed: WITH_IPP is not OFF" >&2
  exit 3
fi
if ! grep -Eq '^WITH_KLEIDICV:(BOOL|UNINITIALIZED)=OFF$' "${BUILD_DIR}/CMakeCache.txt"; then
  echo "CPU-only gate failed: WITH_KLEIDICV is not OFF" >&2
  exit 3
fi
if ! grep -Eq '^WITH_CAROTENE:(BOOL|UNINITIALIZED)=OFF$' "${BUILD_DIR}/CMakeCache.txt"; then
  echo "CPU-only gate failed: WITH_CAROTENE is not OFF" >&2
  exit 3
fi

if [[ "${CONFIGURE_ONLY}" != "1" ]]; then
  cmake --build "${BUILD_DIR}" \
    --target opencv_core opencv_imgproc \
    -j "${JOBS}"
fi

echo "opencv_cpu_only_ready: source=${OPENCV_DIR}, build=${BUILD_DIR}, config=${BUILD_DIR}, lapack=OFF, ipp=OFF, kleidicv=OFF, carotene=OFF, opencl=OFF"
