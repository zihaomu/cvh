#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EXPECTATIONS="${ROOT_DIR}/test/ci/header_gate_expectations.json"
ARCHITECTURE="$(uname -m)"
PARALLELISM="${CVH_CI_PARALLEL:-2}"
OPTIMIZATION=ON
PROFILE=ui-on

BUILD_DIR="${ROOT_DIR}/build-ci-headers-ui"
REPORT_DIR="${BUILD_DIR}/test-reports"
CTEST_REPORT="${REPORT_DIR}/ctest.xml"
CORE_REPORT="${REPORT_DIR}/cvh_test_core.xml"
IMGPROC_REPORT="${REPORT_DIR}/cvh_test_imgproc.xml"

print_env_fingerprint() {
  echo "ci_headers_env_begin"
  echo "architecture: ${ARCHITECTURE}"
  echo "uname: $(uname -a)"
  echo "compiler: $(c++ --version | head -n 1)"
  echo "cmake: $(cmake --version | head -n 1)"
  echo "python: $(python3 --version)"
  echo "build_type: Release"
  echo "optimization: ${OPTIMIZATION}"
  echo "target_profiles: cvh::headers, cvh::highgui"
  echo "parallelism: ${PARALLELISM}"
  echo "ci_headers_env_end"
}

run_gtest_report() {
  local bin_path="$1"
  local report_path="$2"
  local tag="$3"

  if [[ ! -x "${bin_path}" ]]; then
    echo "Missing test binary: ${bin_path}" >&2
    return 2
  fi

  echo "${tag}_report_begin"
  local status
  set +e
  "${bin_path}" \
    --gtest_brief=1 \
    "--gtest_output=xml:${report_path}"
  status=$?
  set -e
  echo "${tag}_report: ${report_path}"
  echo "${tag}_status: ${status}"
  echo "${tag}_report_end"
  return "${status}"
}

print_env_fingerprint

echo "documentation_contract_begin"
"${ROOT_DIR}/scripts/check_docs.sh"
echo "documentation_contract_end"

echo "installed_header_contract_begin"
"${ROOT_DIR}/scripts/check_header_only_contract.sh"
echo "installed_header_contract_end"

python3 "${ROOT_DIR}/scripts/check_test_fixtures.py"

cmake -E remove_directory "${BUILD_DIR}"
cmake -S "${ROOT_DIR}" -B "${BUILD_DIR}" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCVH_BUILD_TESTS=ON \
  -DCVH_BUILD_BENCHMARKS=OFF \
  -DCVH_ENABLE_OPTIMIZATION="${OPTIMIZATION}"

echo "ci_headers_cmake_cache_begin"
if command -v rg >/dev/null 2>&1; then
  rg '^(CVH_BUILD_TESTS|CVH_BUILD_BENCHMARKS|CVH_ENABLE_OPTIMIZATION|CMAKE_BUILD_TYPE|CMAKE_CXX_COMPILER):' \
    "${BUILD_DIR}/CMakeCache.txt" || true
else
  grep -E '^(CVH_BUILD_TESTS|CVH_BUILD_BENCHMARKS|CVH_ENABLE_OPTIMIZATION|CMAKE_BUILD_TYPE|CMAKE_CXX_COMPILER):' \
    "${BUILD_DIR}/CMakeCache.txt" || true
fi
echo "ci_headers_cmake_cache_end"

cmake --build "${BUILD_DIR}" --parallel "${PARALLELISM}"

cmake -E make_directory "${REPORT_DIR}"
overall_status=0

if ! ctest \
  --test-dir "${BUILD_DIR}" \
  --output-on-failure \
  --output-junit "${CTEST_REPORT}"; then
  echo "CTest failed; continuing to collect machine-readable reports." >&2
  overall_status=1
fi

if ! run_gtest_report \
  "${BUILD_DIR}/cvh_test_core" \
  "${CORE_REPORT}" \
  "cvh_test_core_headers_${PROFILE}"; then
  overall_status=1
fi

if ! run_gtest_report \
  "${BUILD_DIR}/cvh_test_imgproc" \
  "${IMGPROC_REPORT}" \
  "cvh_test_imgproc_headers_${PROFILE}"; then
  overall_status=1
fi

if [[ -f "${CORE_REPORT}" && -f "${IMGPROC_REPORT}" ]]; then
  if ! python3 "${ROOT_DIR}/scripts/check_ci_test_reports.py" \
    --build-dir "${BUILD_DIR}" \
    --expectations "${EXPECTATIONS}" \
    --profile "${PROFILE}" \
    --architecture "${ARCHITECTURE}" \
    --core-report "${CORE_REPORT}" \
    --imgproc-report "${IMGPROC_REPORT}"; then
    overall_status=1
  fi
else
  echo "One or more GTest XML reports are missing." >&2
  overall_status=1
fi

exit "${overall_status}"
