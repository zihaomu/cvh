#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ARCHITECTURE="$(uname -m)"
PARALLELISM="${CVH_X86_CI_PARALLEL:-2}"
SANITIZER_PARALLELISM="${CVH_X86_SANITIZER_PARALLEL:-2}"
RELEASE_BUILD_DIR="${ROOT_DIR}/build-ci-x86-v3"
SANITIZER_BUILD_DIR="${ROOT_DIR}/build-ci-x86-sanitizers"
RELEASE_REPORT_DIR="${RELEASE_BUILD_DIR}/test-reports"
SANITIZER_REPORT_DIR="${SANITIZER_BUILD_DIR}/test-reports"
EXPECTATIONS="${ROOT_DIR}/test/ci/header_gate_expectations.json"
SANITIZER_FLAGS="-march=x86-64-v3 -fno-omit-frame-pointer -fsanitize=address,undefined"

case "${ARCHITECTURE}" in
  x86_64|amd64|AMD64)
    ;;
  *)
    echo "x86 correctness gate requires x86-64; found ${ARCHITECTURE}" >&2
    exit 2
    ;;
esac

if [[ -r /proc/cpuinfo ]]; then
  for required_flag in avx2 fma; do
    if ! grep -m1 -E '^flags[[:space:]]*:' /proc/cpuinfo |
        grep -qw "${required_flag}"; then
      echo "x86 correctness gate requires CPU flag: ${required_flag}" >&2
      exit 2
    fi
  done
fi

print_env_fingerprint() {
  echo "ci_x86_env_begin"
  echo "commit: $(git -C "${ROOT_DIR}" rev-parse HEAD)"
  echo "architecture: ${ARCHITECTURE}"
  echo "uname: $(uname -a)"
  echo "compiler: $(c++ --version | head -n 1)"
  echo "cmake: $(cmake --version | head -n 1)"
  echo "python: $(python3 --version)"
  echo "release_target: x86-64-v3"
  echo "release_parallelism: ${PARALLELISM}"
  echo "sanitizer_parallelism: ${SANITIZER_PARALLELISM}"
  if [[ -r /proc/cpuinfo ]]; then
    echo "cpu_model: $(grep -m1 -E '^model name[[:space:]]*:' /proc/cpuinfo | cut -d: -f2- | sed 's/^ //')"
    echo "cpu_flags: $(grep -m1 -E '^flags[[:space:]]*:' /proc/cpuinfo | cut -d: -f2- | sed 's/^ //')"
  fi
  echo "ci_x86_env_end"
}

run_gtest_report() {
  local bin_path="$1"
  local report_path="$2"
  local tag="$3"
  shift 3

  if [[ ! -x "${bin_path}" ]]; then
    echo "Missing test binary: ${bin_path}" >&2
    return 2
  fi

  echo "${tag}_report_begin"
  local status
  set +e
  "$@" "${bin_path}" \
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
python3 "${ROOT_DIR}/scripts/check_test_fixtures.py"

cmake -E remove_directory "${RELEASE_BUILD_DIR}"
cmake -S "${ROOT_DIR}" -B "${RELEASE_BUILD_DIR}" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_FLAGS=-march=x86-64-v3 \
  -DCVH_BUILD_TESTS=ON \
  -DCVH_BUILD_BENCHMARKS=OFF \
  -DCVH_ENABLE_OPTIMIZATION=ON
cmake --build "${RELEASE_BUILD_DIR}" --parallel "${PARALLELISM}"
cmake -E make_directory "${RELEASE_REPORT_DIR}"

overall_status=0
if ! ctest \
  --test-dir "${RELEASE_BUILD_DIR}" \
  --output-on-failure \
  --output-junit "${RELEASE_REPORT_DIR}/ctest.xml"; then
  overall_status=1
fi
if ! run_gtest_report \
  "${RELEASE_BUILD_DIR}/cvh_test_core" \
  "${RELEASE_REPORT_DIR}/cvh_test_core.xml" \
  "cvh_test_core_x86_v3" \
  env; then
  overall_status=1
fi
if ! run_gtest_report \
  "${RELEASE_BUILD_DIR}/cvh_test_imgproc" \
  "${RELEASE_REPORT_DIR}/cvh_test_imgproc.xml" \
  "cvh_test_imgproc_x86_v3" \
  env; then
  overall_status=1
fi
if [[ -f "${RELEASE_REPORT_DIR}/cvh_test_core.xml" &&
      -f "${RELEASE_REPORT_DIR}/cvh_test_imgproc.xml" ]]; then
  if ! python3 "${ROOT_DIR}/scripts/check_ci_test_reports.py" \
    --build-dir "${RELEASE_BUILD_DIR}" \
    --expectations "${EXPECTATIONS}" \
    --profile ui-on \
    --architecture "${ARCHITECTURE}" \
    --core-report "${RELEASE_REPORT_DIR}/cvh_test_core.xml" \
    --imgproc-report "${RELEASE_REPORT_DIR}/cvh_test_imgproc.xml"; then
    overall_status=1
  fi
else
  echo "One or more x86-v3 GTest XML reports are missing." >&2
  overall_status=1
fi

cmake -E remove_directory "${SANITIZER_BUILD_DIR}"
cmake -S "${ROOT_DIR}" -B "${SANITIZER_BUILD_DIR}" \
  -DCMAKE_BUILD_TYPE=Debug \
  "-DCMAKE_CXX_FLAGS=${SANITIZER_FLAGS}" \
  "-DCMAKE_EXE_LINKER_FLAGS=-fsanitize=address,undefined" \
  -DCVH_BUILD_TESTS=ON \
  -DCVH_BUILD_BENCHMARKS=OFF \
  -DCVH_ENABLE_OPTIMIZATION=ON
cmake --build "${SANITIZER_BUILD_DIR}" \
  --parallel "${SANITIZER_PARALLELISM}" \
  --target \
  cvh_test_core \
  cvh_test_imgproc \
  cvh_test_imgcodecs \
  cvh_opencv_intrin_x86_smoke
cmake -E make_directory "${SANITIZER_REPORT_DIR}"

SANITIZER_ENV=(
  env
  ASAN_OPTIONS=detect_leaks=1:halt_on_error=1:abort_on_error=1
  UBSAN_OPTIONS=halt_on_error=1:abort_on_error=1:print_stacktrace=1
)
for target in cvh_test_core cvh_test_imgproc cvh_test_imgcodecs; do
  if ! run_gtest_report \
    "${SANITIZER_BUILD_DIR}/${target}" \
    "${SANITIZER_REPORT_DIR}/${target}.xml" \
    "${target}_x86_sanitizers" \
    "${SANITIZER_ENV[@]}"; then
    overall_status=1
  fi
done
if ! "${SANITIZER_ENV[@]}" \
  "${SANITIZER_BUILD_DIR}/cvh_opencv_intrin_x86_smoke"; then
  overall_status=1
fi

exit "${overall_status}"
