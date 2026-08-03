function(cvh_assert_all_test_sources_registered module_root)
    cmake_parse_arguments(CVH "" "" "SOURCES" ${ARGN})

    file(GLOB_RECURSE discovered_sources
        CONFIGURE_DEPENDS
        "${module_root}/*_test.cpp"
    )

    set(registered_sources)
    foreach(source IN LISTS CVH_SOURCES)
        if(IS_ABSOLUTE "${source}")
            set(absolute_source "${source}")
        else()
            get_filename_component(
                absolute_source
                "${source}"
                ABSOLUTE
                BASE_DIR "${CMAKE_CURRENT_SOURCE_DIR}"
            )
        endif()
        list(APPEND registered_sources "${absolute_source}")
    endforeach()

    list(LENGTH registered_sources registered_count)
    list(REMOVE_DUPLICATES registered_sources)
    list(LENGTH registered_sources unique_registered_count)
    if(NOT registered_count EQUAL unique_registered_count)
        message(FATAL_ERROR
            "Duplicate test source registered under ${module_root}")
    endif()

    list(SORT discovered_sources)
    list(SORT registered_sources)
    if(NOT "${discovered_sources}" STREQUAL "${registered_sources}")
        string(REPLACE ";" "\n  " discovered_text "${discovered_sources}")
        string(REPLACE ";" "\n  " registered_text "${registered_sources}")
        message(FATAL_ERROR
            "Test source manifest mismatch for ${module_root}\n"
            "Discovered:\n  ${discovered_text}\n"
            "Registered:\n  ${registered_text}")
    endif()
endfunction()

function(cvh_assert_public_header_compile_smoke header_root)
    cmake_parse_arguments(CVH "" "SOURCE_DIR" "HEADERS;SOURCES" ${ARGN})

    if(NOT CVH_SOURCE_DIR)
        message(FATAL_ERROR
            "cvh_assert_public_header_compile_smoke requires SOURCE_DIR")
    endif()

    file(GLOB discovered_headers
        CONFIGURE_DEPENDS
        "${header_root}/*.h"
    )
    list(FILTER discovered_headers EXCLUDE REGEX "\\.inl\\.h$")

    set(registered_headers)
    set(expected_sources)
    foreach(header IN LISTS CVH_HEADERS)
        if(IS_ABSOLUTE "${header}")
            set(absolute_header "${header}")
        else()
            get_filename_component(
                absolute_header
                "${header}"
                ABSOLUTE
                BASE_DIR "${CMAKE_CURRENT_SOURCE_DIR}"
            )
        endif()
        list(APPEND registered_headers "${absolute_header}")

        get_filename_component(header_name "${absolute_header}" NAME_WE)
        list(APPEND expected_sources
            "${CMAKE_CURRENT_SOURCE_DIR}/${CVH_SOURCE_DIR}/${header_name}_compile.cpp"
        )
    endforeach()
    list(APPEND expected_sources
        "${CMAKE_CURRENT_SOURCE_DIR}/${CVH_SOURCE_DIR}/main.cpp"
    )

    set(registered_sources)
    foreach(source IN LISTS CVH_SOURCES)
        if(IS_ABSOLUTE "${source}")
            set(absolute_source "${source}")
        else()
            get_filename_component(
                absolute_source
                "${source}"
                ABSOLUTE
                BASE_DIR "${CMAKE_CURRENT_SOURCE_DIR}"
            )
        endif()
        list(APPEND registered_sources "${absolute_source}")
    endforeach()

    list(SORT discovered_headers)
    list(SORT registered_headers)
    if(NOT "${discovered_headers}" STREQUAL "${registered_headers}")
        message(FATAL_ERROR
            "Public header inventory does not match ${header_root}/*.h")
    endif()

    list(SORT expected_sources)
    list(SORT registered_sources)
    if(NOT "${expected_sources}" STREQUAL "${registered_sources}")
        message(FATAL_ERROR
            "Public header compile-smoke source inventory is incomplete for ${header_root}")
    endif()
endfunction()
