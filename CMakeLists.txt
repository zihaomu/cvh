cmake_minimum_required(VERSION 3.16)

file(READ "${CMAKE_CURRENT_SOURCE_DIR}/VERSION.txt" CVH_PROJECT_VERSION_RAW)
string(STRIP "${CVH_PROJECT_VERSION_RAW}" CVH_PROJECT_VERSION)
if(NOT CVH_PROJECT_VERSION MATCHES "^[0-9]+\\.[0-9]+\\.[0-9]+$")
    message(FATAL_ERROR "VERSION.txt must be semantic version (MAJOR.MINOR.PATCH), got: '${CVH_PROJECT_VERSION}'")
endif()

project(cvh VERSION ${CVH_PROJECT_VERSION} LANGUAGES CXX)

# default to Release build
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

include(CTest)
include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

option(CVH_BUILD_TESTS "Build developer test targets (smoke + core)" OFF)
option(CVH_BUILD_BENCHMARKS "Build developer benchmark targets" OFF)
option(CVH_ENABLE_OPTIMIZATION "Enable validated CPU optimization paths in cvh headers" ON)
option(CVH_ENABLE_OPENCV_COMPARE "Build cvh vs OpenCV compare benchmark target" OFF)

if(CVH_ENABLE_OPENCV_COMPARE)
    find_package(OpenCV REQUIRED COMPONENTS core imgproc)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_library(cvh_headers INTERFACE)
add_library(cvh::headers ALIAS cvh_headers)
set_target_properties(cvh_headers PROPERTIES EXPORT_NAME headers)
target_include_directories(cvh_headers INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)
target_compile_features(cvh_headers INTERFACE cxx_std_17)
if(NOT CVH_ENABLE_OPTIMIZATION)
    target_compile_definitions(cvh_headers INTERFACE
        CVH_ENABLE_OPTIMIZATION=0
    )
endif()
message(STATUS
    "cvh_headers: CPU optimization=${CVH_ENABLE_OPTIMIZATION}")

set(CVH_OPENCV_INTRIN_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/include/cvh/3rdparty/opencv_intrin")
target_include_directories(cvh_headers INTERFACE
    $<BUILD_INTERFACE:${CVH_OPENCV_INTRIN_INCLUDE_DIR}>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/cvh/3rdparty/opencv_intrin>
)

add_library(cvh_highgui INTERFACE)
add_library(cvh::highgui ALIAS cvh_highgui)
set_target_properties(cvh_highgui PROPERTIES EXPORT_NAME highgui)
target_link_libraries(cvh_highgui INTERFACE cvh_headers)
target_compile_features(cvh_highgui INTERFACE cxx_std_17)

if(APPLE)
    find_library(CVH_APPKIT_FRAMEWORK AppKit)
    find_library(CVH_COREGRAPHICS_FRAMEWORK CoreGraphics)
    find_library(CVH_COREFOUNDATION_FRAMEWORK CoreFoundation)
    if(NOT CVH_APPKIT_FRAMEWORK OR
       NOT CVH_COREGRAPHICS_FRAMEWORK OR
       NOT CVH_COREFOUNDATION_FRAMEWORK)
        message(FATAL_ERROR
            "cvh::highgui requires AppKit, CoreGraphics, and CoreFoundation")
    endif()
    target_link_libraries(cvh_highgui INTERFACE
        ${CVH_APPKIT_FRAMEWORK}
        ${CVH_COREGRAPHICS_FRAMEWORK}
        ${CVH_COREFOUNDATION_FRAMEWORK}
    )
elseif(WIN32)
    target_link_libraries(cvh_highgui INTERFACE user32 gdi32)
elseif(CMAKE_SYSTEM_NAME STREQUAL "Linux" AND NOT ANDROID)
    find_package(X11 QUIET)
    if(X11_FOUND AND TARGET X11::X11)
        target_compile_definitions(cvh_highgui INTERFACE CVH_HIGHGUI_X11=1)
        target_include_directories(cvh_highgui SYSTEM INTERFACE
            ${X11_INCLUDE_DIR}
        )
        # Keep X11 optional for consumers that only use cvh::headers.
        # Linking by the system library name avoids making find_package(cvh)
        # itself depend on X11.
        target_link_libraries(cvh_highgui INTERFACE X11)
    else()
        target_compile_definitions(cvh_highgui INTERFACE CVH_HIGHGUI_FORCE_STUB=1)
        message(STATUS
            "cvh_highgui: X11 not found; exporting the unsupported-platform stub")
    endif()
else()
    target_compile_definitions(cvh_highgui INTERFACE CVH_HIGHGUI_FORCE_STUB=1)
endif()

if(CVH_BUILD_BENCHMARKS)
    add_executable(cvh_benchmark_core_mat_header
        benchmark/core_mat_header_benchmark.cpp
    )
    target_link_libraries(cvh_benchmark_core_mat_header PRIVATE cvh::headers)
    target_compile_features(cvh_benchmark_core_mat_header PRIVATE cxx_std_17)

    add_executable(cvh_benchmark_cvtcolor_bgr2gray_header
        benchmark/cvtcolor_bgr2gray_header_benchmark.cpp
    )
    target_link_libraries(cvh_benchmark_cvtcolor_bgr2gray_header PRIVATE cvh::headers)
    target_compile_features(cvh_benchmark_cvtcolor_bgr2gray_header PRIVATE cxx_std_17)

    add_executable(cvh_benchmark_resize_bilinear_header
        benchmark/resize_bilinear_header_benchmark.cpp
    )
    target_link_libraries(cvh_benchmark_resize_bilinear_header PRIVATE cvh::headers)
    target_compile_features(cvh_benchmark_resize_bilinear_header PRIVATE cxx_std_17)

    add_executable(cvh_benchmark_imgproc_header
        benchmark/imgproc_header_benchmark.cpp
    )
    target_link_libraries(cvh_benchmark_imgproc_header PRIVATE cvh::headers)
    target_compile_features(cvh_benchmark_imgproc_header PRIVATE cxx_std_17)

    if(CVH_ENABLE_OPENCV_COMPARE)
        add_library(cvh_benchmark_opencv_compare_backend_obj OBJECT
            benchmark/opencv_compare_opencv_backend.cpp
            benchmark/opencv_compare_phase1_opencv_backend.cpp
            benchmark/opencv_compare_phase2_opencv_backend.cpp
        )
        target_compile_features(cvh_benchmark_opencv_compare_backend_obj PRIVATE cxx_std_17)
        target_include_directories(cvh_benchmark_opencv_compare_backend_obj PRIVATE
            ${OpenCV_INCLUDE_DIRS}
        )

        add_executable(cvh_benchmark_opencv_compare
            benchmark/opencv_compare_header_benchmark.cpp
            benchmark/opencv_compare_phase1_header_benchmark.cpp
            benchmark/opencv_compare_phase2_header_benchmark.cpp
            $<TARGET_OBJECTS:cvh_benchmark_opencv_compare_backend_obj>
        )
        target_link_libraries(cvh_benchmark_opencv_compare PRIVATE cvh::headers ${OpenCV_LIBS})
        target_compile_features(cvh_benchmark_opencv_compare PRIVATE cxx_std_17)
        target_compile_definitions(cvh_benchmark_opencv_compare PRIVATE CVH_COMPARE_IMPL_NAME="cvh_auto")
        target_include_directories(cvh_benchmark_opencv_compare BEFORE PRIVATE
            ${OpenCV_INCLUDE_DIRS}
        )
        target_include_directories(cvh_benchmark_opencv_compare PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}
            ${CMAKE_CURRENT_SOURCE_DIR}/include
        )
    endif()
endif()

if(CVH_BUILD_TESTS)
    add_subdirectory(test/3rdparty/googletest EXCLUDE_FROM_ALL)
    include(test/cmake/CvhTest.cmake)
    include(test/core/sources.cmake)
    include(test/imgproc/sources.cmake)
    include(test/smoke/core_headers/sources.cmake)
    include(test/smoke/imgproc_headers/sources.cmake)
    include(test/smoke/imgcodecs_headers/sources.cmake)
    include(test/smoke/highgui_headers/sources.cmake)
    include(test/smoke/public_headers/sources.cmake)

    cvh_assert_all_test_sources_registered(
        "${CMAKE_CURRENT_SOURCE_DIR}/test/core"
        SOURCES
            ${CVH_CORE_TEST_SOURCES}
            ${CVH_CORE_ISA_TEST_SOURCES}
    )
    cvh_assert_all_test_sources_registered(
        "${CMAKE_CURRENT_SOURCE_DIR}/test/imgproc"
        SOURCES ${CVH_IMGPROC_TEST_SOURCES}
    )
    cvh_assert_public_header_compile_smoke(
        "${CMAKE_CURRENT_SOURCE_DIR}/include/cvh/core"
        SOURCE_DIR test/smoke/core_headers
        HEADERS ${CVH_CORE_PUBLIC_HEADERS}
        SOURCES ${CVH_CORE_HEADER_COMPILE_SMOKE_SOURCES}
    )
    cvh_assert_public_header_compile_smoke(
        "${CMAKE_CURRENT_SOURCE_DIR}/include/cvh/imgproc"
        SOURCE_DIR test/smoke/imgproc_headers
        HEADERS ${CVH_IMGPROC_PUBLIC_HEADERS}
        SOURCES ${CVH_IMGPROC_HEADER_COMPILE_SMOKE_SOURCES}
    )
    cvh_assert_public_header_compile_smoke(
        "${CMAKE_CURRENT_SOURCE_DIR}/include/cvh/imgcodecs"
        SOURCE_DIR test/smoke/imgcodecs_headers
        HEADERS ${CVH_IMGCODECS_PUBLIC_HEADERS}
        SOURCES ${CVH_IMGCODECS_HEADER_COMPILE_SMOKE_SOURCES}
    )
    cvh_assert_public_header_compile_smoke(
        "${CMAKE_CURRENT_SOURCE_DIR}/include/cvh/highgui"
        SOURCE_DIR test/smoke/highgui_headers
        HEADERS ${CVH_HIGHGUI_PUBLIC_HEADERS}
        SOURCES ${CVH_HIGHGUI_HEADER_COMPILE_SMOKE_SOURCES}
    )

    add_executable(cvh_header_compile_smoke test/smoke/cvh_header_compile_smoke.cpp)
    target_link_libraries(cvh_header_compile_smoke PRIVATE cvh::headers)

    add_executable(cvh_core_header_odr_smoke
        test/smoke/cvh_core_header_odr_main.cpp
        test/smoke/cvh_core_header_odr_peer.cpp
    )
    target_link_libraries(cvh_core_header_odr_smoke PRIVATE cvh::headers)

    add_executable(cvh_core_headers_compile_smoke
        ${CVH_CORE_HEADER_COMPILE_SMOKE_SOURCES}
    )
    target_link_libraries(cvh_core_headers_compile_smoke PRIVATE cvh::headers)
    target_compile_features(cvh_core_headers_compile_smoke PRIVATE cxx_std_17)

    add_executable(cvh_imgproc_header_odr_smoke
        test/smoke/cvh_imgproc_header_odr_main.cpp
        test/smoke/cvh_imgproc_header_odr_peer.cpp
    )
    target_link_libraries(cvh_imgproc_header_odr_smoke PRIVATE cvh::headers)

    add_executable(cvh_imgproc_headers_compile_smoke
        ${CVH_IMGPROC_HEADER_COMPILE_SMOKE_SOURCES}
    )
    target_link_libraries(cvh_imgproc_headers_compile_smoke PRIVATE cvh::headers)
    target_compile_features(cvh_imgproc_headers_compile_smoke PRIVATE cxx_std_17)

    add_executable(cvh_imgcodecs_headers_compile_smoke
        ${CVH_IMGCODECS_HEADER_COMPILE_SMOKE_SOURCES}
    )
    target_link_libraries(cvh_imgcodecs_headers_compile_smoke PRIVATE cvh::headers)
    target_compile_features(cvh_imgcodecs_headers_compile_smoke PRIVATE cxx_std_17)

    add_executable(cvh_highgui_headers_compile_smoke
        ${CVH_HIGHGUI_HEADER_COMPILE_SMOKE_SOURCES}
    )
    target_link_libraries(cvh_highgui_headers_compile_smoke PRIVATE cvh::highgui)
    target_compile_features(cvh_highgui_headers_compile_smoke PRIVATE cxx_std_17)

    add_executable(cvh_highgui_header_odr_smoke
        test/smoke/cvh_highgui_header_odr_main.cpp
        test/smoke/cvh_highgui_header_odr_peer.cpp
    )
    target_link_libraries(cvh_highgui_header_odr_smoke PRIVATE cvh::highgui)
    target_compile_features(cvh_highgui_header_odr_smoke PRIVATE cxx_std_17)

    add_executable(cvh_aggregate_headers_compile_smoke
        ${CVH_AGGREGATE_HEADER_COMPILE_SMOKE_SOURCES}
    )
    target_link_libraries(cvh_aggregate_headers_compile_smoke PRIVATE cvh::headers)
    target_compile_features(cvh_aggregate_headers_compile_smoke PRIVATE cxx_std_17)

    add_executable(cvh_include_only_smoke test/smoke/cvh_include_only_smoke.cpp)
    target_include_directories(cvh_include_only_smoke PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CVH_OPENCV_INTRIN_INCLUDE_DIR}
    )
    target_compile_features(cvh_include_only_smoke PRIVATE cxx_std_17)
    if(NOT CVH_ENABLE_OPTIMIZATION)
        target_compile_definitions(
            cvh_include_only_smoke
            PRIVATE CVH_ENABLE_OPTIMIZATION=0
        )
    endif()

    add_executable(cvh_pipeline_smoke test/smoke/cvh_pipeline_smoke.cpp)
    target_link_libraries(cvh_pipeline_smoke PRIVATE cvh::headers)

    add_executable(cvh_resize_dispatch_smoke test/smoke/cvh_resize_dispatch_smoke.cpp)
    target_link_libraries(cvh_resize_dispatch_smoke PRIVATE cvh::headers)

    if(CVH_ENABLE_OPTIMIZATION)
        add_executable(cvh_opencv_intrin_smoke test/smoke/cvh_opencv_intrin_smoke.cpp)
        target_link_libraries(cvh_opencv_intrin_smoke PRIVATE cvh::headers)
        target_compile_definitions(cvh_opencv_intrin_smoke PRIVATE CV_FORCE_SIMD128_CPP=1)

        if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|amd64|AMD64|i386|i686|x86|X86)$")
            add_executable(cvh_opencv_intrin_x86_smoke test/smoke/cvh_opencv_intrin_x86_smoke.cpp)
            target_link_libraries(cvh_opencv_intrin_x86_smoke PRIVATE cvh::headers)
            if(MSVC)
                target_compile_options(cvh_opencv_intrin_x86_smoke PRIVATE /arch:AVX2)
            else()
                target_compile_options(cvh_opencv_intrin_x86_smoke PRIVATE -mavx2)
            endif()

            include(CheckCXXSourceRuns)
            set(cvh_saved_required_flags "${CMAKE_REQUIRED_FLAGS}")
            if(MSVC)
                set(CMAKE_REQUIRED_FLAGS "${CMAKE_REQUIRED_FLAGS} /arch:AVX2")
            else()
                set(CMAKE_REQUIRED_FLAGS "${CMAKE_REQUIRED_FLAGS} -mavx2")
            endif()
            check_cxx_source_runs(
                [=[
                    #include <immintrin.h>
                    int main()
                    {
                        alignas(32) int values[8] = {};
                        const __m256i vector = _mm256_set1_epi32(7);
                        _mm256_store_si256(
                            reinterpret_cast<__m256i*>(values),
                            vector);
                        return values[7] == 7 ? 0 : 1;
                    }
                ]=]
                CVH_X86_AVX2_RUNTIME_SUPPORTED
            )
            set(CMAKE_REQUIRED_FLAGS "${cvh_saved_required_flags}")
            message(STATUS
                "cvh_opencv_intrin_x86_smoke: runtime AVX2=${CVH_X86_AVX2_RUNTIME_SUPPORTED}")
        endif()

        add_executable(cvh_cvtcolor_opencv_intrin_smoke test/smoke/cvh_cvtcolor_opencv_intrin_smoke.cpp)
        target_link_libraries(cvh_cvtcolor_opencv_intrin_smoke PRIVATE cvh::headers)

        add_executable(cvh_resize_opencv_intrin_smoke test/smoke/cvh_resize_opencv_intrin_smoke.cpp)
        target_link_libraries(cvh_resize_opencv_intrin_smoke PRIVATE cvh::headers)
    else()
        add_executable(
            cvh_optimization_disabled_smoke
            test/smoke/cvh_optimization_disabled_smoke.cpp
        )
        target_link_libraries(
            cvh_optimization_disabled_smoke
            PRIVATE cvh::headers
        )
    endif()

    add_executable(cvh_test_imgcodecs
        test/imgcodecs/imgcodecs_format_test.cpp
    )
    target_link_libraries(cvh_test_imgcodecs PRIVATE cvh::headers gtest_main)
    target_compile_features(cvh_test_imgcodecs PRIVATE cxx_std_17)
    target_compile_definitions(cvh_test_imgcodecs PRIVATE M_ROOT_PATH="${CMAKE_CURRENT_SOURCE_DIR}")
    target_include_directories(cvh_test_imgcodecs PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/test
    )

    add_executable(cvh_test_highgui
        test/highgui/highgui_contract_test.cpp
    )
    target_link_libraries(cvh_test_highgui PRIVATE cvh::highgui gtest_main)
    target_compile_features(cvh_test_highgui PRIVATE cxx_std_17)
    target_include_directories(cvh_test_highgui PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/test
    )

    add_executable(cvh_test_imgproc ${CVH_IMGPROC_TEST_SOURCES})
    target_link_libraries(cvh_test_imgproc PRIVATE cvh::headers gtest_main)
    target_compile_features(cvh_test_imgproc PRIVATE cxx_std_17)
    target_compile_definitions(cvh_test_imgproc PRIVATE M_ROOT_PATH="${CMAKE_CURRENT_SOURCE_DIR}")
    target_include_directories(cvh_test_imgproc PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/test
    )

    add_executable(cvh_test_core
        ${CVH_CORE_TEST_SOURCES}
        ${CVH_CORE_TEST_SUPPORT_SOURCES}
    )
    target_link_libraries(cvh_test_core PRIVATE cvh::headers gtest_main)
    target_compile_features(cvh_test_core PRIVATE cxx_std_17)
    target_compile_definitions(cvh_test_core PRIVATE M_ROOT_PATH="${CMAKE_CURRENT_SOURCE_DIR}")
    target_include_directories(cvh_test_core PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/test
    )

    add_executable(cvh_test_gemm_isa
        ${CVH_CORE_ISA_TEST_SOURCES}
    )
    target_link_libraries(
        cvh_test_gemm_isa
        PRIVATE cvh::headers gtest_main
    )
    target_compile_features(cvh_test_gemm_isa PRIVATE cxx_std_17)
    target_include_directories(cvh_test_gemm_isa PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/test
    )

    if(CVH_ENABLE_OPENCV_COMPARE)
        add_library(cvh_test_opencv_contract_backend_obj OBJECT
            test/opencv_contract/opencv_contract_backend.cpp
        )
        target_compile_features(cvh_test_opencv_contract_backend_obj PRIVATE cxx_std_17)
        target_include_directories(cvh_test_opencv_contract_backend_obj PRIVATE
            ${OpenCV_INCLUDE_DIRS}
            ${CMAKE_CURRENT_SOURCE_DIR}/test/opencv_contract
        )

        add_executable(cvh_test_opencv_contract_smoke
            test/opencv_contract/opencv_contract_smoke_test.cpp
            $<TARGET_OBJECTS:cvh_test_opencv_contract_backend_obj>
        )
        target_link_libraries(cvh_test_opencv_contract_smoke PRIVATE
            cvh::headers
            gtest_main
            ${OpenCV_LIBS}
        )
        target_compile_features(cvh_test_opencv_contract_smoke PRIVATE cxx_std_17)
        target_include_directories(cvh_test_opencv_contract_smoke PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/include
            ${CMAKE_CURRENT_SOURCE_DIR}/test/opencv_contract
        )
    endif()

    if(BUILD_TESTING)
        add_test(NAME cvh_header_compile_smoke COMMAND cvh_header_compile_smoke)
        add_test(NAME cvh_core_header_odr_smoke COMMAND cvh_core_header_odr_smoke)
        add_test(NAME cvh_core_headers_compile_smoke COMMAND cvh_core_headers_compile_smoke)
        add_test(NAME cvh_imgproc_header_odr_smoke COMMAND cvh_imgproc_header_odr_smoke)
        add_test(NAME cvh_imgproc_headers_compile_smoke COMMAND cvh_imgproc_headers_compile_smoke)
        add_test(NAME cvh_imgcodecs_headers_compile_smoke COMMAND cvh_imgcodecs_headers_compile_smoke)
        add_test(NAME cvh_highgui_headers_compile_smoke COMMAND cvh_highgui_headers_compile_smoke)
        add_test(NAME cvh_highgui_header_odr_smoke COMMAND cvh_highgui_header_odr_smoke)
        add_test(NAME cvh_aggregate_headers_compile_smoke COMMAND cvh_aggregate_headers_compile_smoke)
        add_test(NAME cvh_include_only_smoke COMMAND cvh_include_only_smoke)
        add_test(NAME cvh_pipeline_smoke COMMAND cvh_pipeline_smoke)
        add_test(NAME cvh_resize_dispatch_smoke COMMAND cvh_resize_dispatch_smoke)
        if(CVH_ENABLE_OPTIMIZATION)
            add_test(NAME cvh_opencv_intrin_smoke COMMAND cvh_opencv_intrin_smoke)
            if(TARGET cvh_opencv_intrin_x86_smoke)
                if("${CVH_X86_AVX2_RUNTIME_SUPPORTED}" STREQUAL "1")
                    add_test(
                        NAME cvh_opencv_intrin_x86_smoke
                        COMMAND cvh_opencv_intrin_x86_smoke
                    )
                else()
                    add_test(
                        NAME cvh_opencv_intrin_x86_smoke
                        COMMAND
                            ${CMAKE_COMMAND} -E echo
                            "cvh_opencv_intrin_x86_smoke: compile-only; runtime AVX2 unavailable"
                    )
                endif()
            endif()
            add_test(NAME cvh_cvtcolor_opencv_intrin_smoke COMMAND cvh_cvtcolor_opencv_intrin_smoke)
            add_test(NAME cvh_resize_opencv_intrin_smoke COMMAND cvh_resize_opencv_intrin_smoke)
        else()
            add_test(
                NAME cvh_optimization_disabled_smoke
                COMMAND cvh_optimization_disabled_smoke
            )
        endif()
        add_test(NAME cvh_test_imgcodecs COMMAND cvh_test_imgcodecs)
        add_test(NAME cvh_test_highgui COMMAND cvh_test_highgui)
        add_test(NAME cvh_test_imgproc COMMAND cvh_test_imgproc)
        add_test(NAME cvh_test_core COMMAND cvh_test_core)
        add_test(NAME cvh_test_gemm_isa COMMAND cvh_test_gemm_isa)
        set_tests_properties(
            cvh_test_core
            PROPERTIES LABELS "core;unit"
        )
        set_tests_properties(
            cvh_test_gemm_isa
            PROPERTIES LABELS "core;unit;isa"
        )
        set_tests_properties(
            cvh_core_headers_compile_smoke
            PROPERTIES LABELS "core;smoke"
        )
        set_tests_properties(
            cvh_test_imgproc
            PROPERTIES LABELS "imgproc;unit"
        )
        set_tests_properties(
            cvh_imgproc_headers_compile_smoke
            PROPERTIES LABELS "imgproc;smoke"
        )
        set_tests_properties(
            cvh_imgcodecs_headers_compile_smoke
            PROPERTIES LABELS "imgcodecs;smoke"
        )
        set_tests_properties(
            cvh_highgui_headers_compile_smoke
            cvh_highgui_header_odr_smoke
            PROPERTIES
                LABELS "highgui;smoke"
                ENVIRONMENT "CVH_HIGHGUI_HEADLESS=1"
        )
        set_tests_properties(
            cvh_test_highgui
            PROPERTIES
                LABELS "highgui;unit"
                ENVIRONMENT "CVH_HIGHGUI_HEADLESS=1"
        )
        set_tests_properties(
            cvh_aggregate_headers_compile_smoke
            PROPERTIES LABELS "public;smoke"
        )
        if(TARGET cvh_test_opencv_contract_smoke)
            add_test(NAME cvh_test_opencv_contract_smoke COMMAND cvh_test_opencv_contract_smoke)
        endif()
    endif()

endif()

set(CVH_INSTALL_CMAKEDIR "${CMAKE_INSTALL_LIBDIR}/cmake/cvh" CACHE STRING "Install path for the cvh CMake package")

install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

install(TARGETS cvh_headers cvh_highgui
    EXPORT cvhTargets
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

install(EXPORT cvhTargets
    NAMESPACE cvh::
    FILE cvhTargets.cmake
    DESTINATION ${CVH_INSTALL_CMAKEDIR}
)

configure_package_config_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/cvhConfig.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/cvhConfig.cmake"
    INSTALL_DESTINATION ${CVH_INSTALL_CMAKEDIR}
)

write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/cvhConfigVersion.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)

install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/cvhConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/cvhConfigVersion.cmake"
    DESTINATION ${CVH_INSTALL_CMAKEDIR}
)
