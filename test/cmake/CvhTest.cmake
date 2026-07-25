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
