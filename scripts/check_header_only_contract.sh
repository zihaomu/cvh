#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TMP_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/cvh_header_only_contract.XXXXXX")"
PARALLELISM="${CVH_CI_PARALLEL:-2}"

cleanup() {
  if [[ "${CVH_KEEP_CONTRACT_TMP:-0}" != "1" ]]; then
    rm -rf "${TMP_ROOT}"
  else
    echo "Keeping contract check temp dir: ${TMP_ROOT}" >&2
  fi
}
trap cleanup EXIT

BUILD_DIR="${TMP_ROOT}/build"
DEFAULT_BUILD_DIR="${TMP_ROOT}/default-build"
INSTALL_DIR="${TMP_ROOT}/install"
HEADERS_CONSUMER_DIR="${TMP_ROOT}/consumer-headers"

require_public_exports() {
  local cmake_dir="$1"

  if ! grep -R "cvh::headers" "${cmake_dir}" >/dev/null; then
    echo "Installed package does not export cvh::headers." >&2
    return 1
  fi

  if grep -R "cvh::headers_fast" "${cmake_dir}" >/dev/null; then
    echo "Installed package still exports removed cvh::headers_fast." >&2
    return 1
  fi

  if ! grep -R "cvh::highgui" "${cmake_dir}" >/dev/null; then
    echo "Installed package does not export cvh::highgui." >&2
    return 1
  fi

  if grep -R -E "cvh::native|cvh::full|full_backend|cvh_native_backend" "${cmake_dir}" >/dev/null; then
    echo "Installed package exports legacy .cpp targets." >&2
    grep -R -n -E "cvh::native|cvh::full|full_backend|cvh_native_backend" "${cmake_dir}" >&2
    return 1
  fi

  if grep -R -E "xsimd|XSIMD|CVH_ENABLE_XSIMD|CVH_ENABLE_LEGACY_XSIMD|XSimd" "${cmake_dir}" >/dev/null; then
    echo "Installed CMake package exposes removed xsimd surface." >&2
    grep -R -n -E "xsimd|XSIMD|CVH_ENABLE_XSIMD|CVH_ENABLE_LEGACY_XSIMD|XSimd" "${cmake_dir}" >&2
    return 1
  fi
}

require_public_header_surface() {
  local install_dir="$1"
  local include_dir="${install_dir}/include/cvh"

  for required_header in \
    "${include_dir}/cvh.h" \
    "${include_dir}/core/mat.h" \
    "${include_dir}/imgproc/imgproc.h" \
    "${include_dir}/imgcodecs/imgcodecs.h" \
    "${include_dir}/highgui/highgui.h" \
    "${include_dir}/highgui/highgui.hpp"; do
    if [[ ! -f "${required_header}" ]]; then
      echo "Installed package is missing public header: ${required_header}" >&2
      return 1
    fi
  done

  if find "${install_dir}" -type f \
      \( -name '*.cpp' -o -name '*.cc' -o -name '*.cxx' -o -name '*.mm' \) \
      -print -quit | grep -q .; then
    echo "Installed header-only package contains compiled-language source files." >&2
    return 1
  fi
}

"${ROOT_DIR}/scripts/check_public_headers.sh"

if [[ -d "${ROOT_DIR}/src" ]] &&
    find "${ROOT_DIR}/src" -type f -print -quit | grep -q .; then
  echo "Pure header-only repository must not retain a src/ implementation tree." >&2
  exit 1
fi

cmake -S "${ROOT_DIR}" -B "${DEFAULT_BUILD_DIR}" >/dev/null
if ! grep -q '^CVH_BUILD_TESTS:BOOL=OFF$' \
    "${DEFAULT_BUILD_DIR}/CMakeCache.txt"; then
  echo "Default product configuration unexpectedly enables test executables." >&2
  exit 1
fi
if ! grep -q '^CVH_BUILD_BENCHMARKS:BOOL=OFF$' \
    "${DEFAULT_BUILD_DIR}/CMakeCache.txt"; then
  echo "Default product configuration unexpectedly enables benchmark executables." >&2
  exit 1
fi
cmake --build "${DEFAULT_BUILD_DIR}" --parallel "${PARALLELISM}" >/dev/null

cmake -S "${ROOT_DIR}" -B "${BUILD_DIR}" \
  -DCVH_BUILD_TESTS=ON \
  -DCVH_BUILD_BENCHMARKS=OFF \
  >/dev/null

cmake --build "${BUILD_DIR}" --parallel "${PARALLELISM}" --target \
  cvh_header_compile_smoke \
  cvh_core_header_odr_smoke \
  cvh_core_headers_compile_smoke \
  cvh_imgproc_header_odr_smoke \
  cvh_imgproc_headers_compile_smoke \
  cvh_imgcodecs_headers_compile_smoke \
  cvh_highgui_headers_compile_smoke \
  cvh_highgui_header_odr_smoke \
  cvh_aggregate_headers_compile_smoke \
  cvh_include_only_smoke \
  cvh_pipeline_smoke \
  cvh_resize_dispatch_smoke \
  >/dev/null

ctest --test-dir "${BUILD_DIR}" --output-on-failure \
  -R 'cvh_header_compile_smoke|cvh_core_header_odr_smoke|cvh_core_headers_compile_smoke|cvh_imgproc_header_odr_smoke|cvh_imgproc_headers_compile_smoke|cvh_imgcodecs_headers_compile_smoke|cvh_highgui_headers_compile_smoke|cvh_highgui_header_odr_smoke|cvh_aggregate_headers_compile_smoke|cvh_include_only_smoke|cvh_pipeline_smoke|cvh_resize_dispatch_smoke'

cmake --install "${BUILD_DIR}" --prefix "${INSTALL_DIR}" >/dev/null
require_public_exports "${INSTALL_DIR}/lib/cmake/cvh"
require_public_header_surface "${INSTALL_DIR}"

mkdir -p "${HEADERS_CONSUMER_DIR}"
cat > "${HEADERS_CONSUMER_DIR}/CMakeLists.txt" <<'EOF'
cmake_minimum_required(VERSION 3.16)
project(cvh_headers_consumer LANGUAGES CXX)

find_package(cvh CONFIG REQUIRED)

if(NOT TARGET cvh::headers)
    message(FATAL_ERROR "Missing cvh::headers target")
endif()
if(TARGET cvh::headers_fast)
    message(FATAL_ERROR "Removed cvh::headers_fast target is still exported")
endif()
if(NOT TARGET cvh::highgui)
    message(FATAL_ERROR "Missing cvh::highgui target")
endif()
if(TARGET cvh::native OR TARGET cvh::full OR TARGET cvh::full_backend)
    message(FATAL_ERROR "Installed package must not expose legacy .cpp targets")
endif()

add_executable(headers_consumer main.cpp)
target_link_libraries(headers_consumer PRIVATE cvh::headers)
target_compile_features(headers_consumer PRIVATE cxx_std_17)

add_executable(highgui_consumer highgui.cpp)
target_link_libraries(highgui_consumer PRIVATE cvh::highgui)
target_compile_features(highgui_consumer PRIVATE cxx_std_17)
EOF

cat > "${HEADERS_CONSUMER_DIR}/highgui.cpp" <<'EOF'
#include <cvh/highgui/highgui.h>

int main()
{
    cvh::namedWindow("installed_highgui");
    cvh::Mat image({2, 2}, CV_8UC3);
    image = 7;
    cvh::imshow("installed_highgui", image);
    const int key = cvh::waitKey(1);
    cvh::destroyWindow("installed_highgui");
    cvh::destroyAllWindows();
    return key == -1 ? 0 : 1;
}
EOF

cat > "${HEADERS_CONSUMER_DIR}/main.cpp" <<'EOF'
#include <cvh/cvh.h>
#include <cvh/core/simd/opencv_ui.h>

#include <cstring>

#if !CVH_ENABLE_OPTIMIZATION
#error "cvh::headers must enable validated CPU optimizations by default"
#endif

#if !CVH_DETAIL_HAVE_OPENCV_UI
#error "cvh::headers must provide the OpenCV UI capability by default"
#endif

int main()
{
    if (std::strcmp(cvh::detail::opencv_ui_backend_name(), "opencv_intrin") != 0)
    {
        return 1;
    }

    cvh::Mat src({2, 2}, CV_8UC1);
    src = 7;

    cvh::Mat dst;
    cvh::resize(src, dst, cvh::Size(1, 1), 0.0, 0.0, cvh::INTER_LINEAR);

    cvh::Mat a({2, 2}, CV_32F);
    cvh::Mat b({2, 2}, CV_32F);
    a = 2.0f;
    b = 3.0f;
    cvh::Mat sum;
    cvh::add(a, b, sum);
    cvh::Mat transposed = cvh::transpose(sum);
    cvh::Mat product = cvh::gemm(a, b);
    cvh::Mat expression = a + b;

    const bool core_ok =
        reinterpret_cast<const float*>(transposed.data)[0] == 5.0f &&
        reinterpret_cast<const float*>(product.data)[0] == 12.0f &&
        reinterpret_cast<const float*>(expression.data)[0] == 5.0f;
    const bool resize_ok =
        dst.dims == 2 && dst.type() == CV_8UC1 && dst.size[0] == 1 && dst.size[1] == 1;
    return core_ok && resize_ok ? 0 : 2;
}
EOF

cmake -S "${HEADERS_CONSUMER_DIR}" -B "${HEADERS_CONSUMER_DIR}/build" \
  -DCMAKE_PREFIX_PATH="${INSTALL_DIR}" \
  >/dev/null
cmake --build "${HEADERS_CONSUMER_DIR}/build" \
  --parallel "${PARALLELISM}" >/dev/null
"${HEADERS_CONSUMER_DIR}/build/headers_consumer"
cmake -E env CVH_HIGHGUI_HEADLESS=1 \
  "${HEADERS_CONSUMER_DIR}/build/highgui_consumer"

echo "Header-only contract check passed."
