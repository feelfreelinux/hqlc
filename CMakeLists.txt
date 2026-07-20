cmake_minimum_required(VERSION 3.16)

project(hqlc C)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_library(hqlc STATIC
    src/hqlc.c
    src/mdct.c
    src/mdct_tables.c
    src/psy.c
    src/quant.c
    src/entropy.c
    src/entropy_tables.c
    src/tns.c
)

target_include_directories(hqlc PUBLIC include)

# Enables the unit tests from tests/ folder
option(HQLC_TESTS "Build Unity unit tests" OFF)

# Enables the encoder / decoder tool
option(HQLC_TOOLS "Build CLI tools" OFF)

if (HQLC_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

if (HQLC_TOOLS)
    add_executable(hqlc_cli src/hqlc_cli.c)
    set_target_properties(hqlc_cli PROPERTIES OUTPUT_NAME hqlc)
    target_include_directories(hqlc_cli PRIVATE external)
    target_link_libraries(hqlc_cli PRIVATE hqlc)

    find_library(MATH_LIBRARY m)
    if (MATH_LIBRARY)
        target_link_libraries(hqlc_cli PRIVATE ${MATH_LIBRARY})
    endif()
endif()
