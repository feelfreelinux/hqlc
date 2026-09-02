cmake_minimum_required(VERSION 3.16)

project(hqlc C)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Default to Release
if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

add_library(hqlc STATIC
    src/hqlc.c
    src/mdct.c
    src/mdct_tables.c
    src/ms.c
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

# Enables the -T mode in cli, used for rANS table training
option(HQLC_TRAIN_TABLES "Build rANS table-training histogram hooks" OFF)

if (HQLC_TRAIN_TABLES)
    target_compile_definitions(hqlc PUBLIC HQLC_TRAIN_TABLES)
endif()

if (HQLC_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

if (HQLC_TOOLS)
    add_executable(hqlc_cli src/hqlc_cli.c)
    set_target_properties(hqlc_cli PROPERTIES OUTPUT_NAME hqlc)
    target_include_directories(hqlc_cli PRIVATE external)
    target_link_libraries(hqlc_cli PRIVATE hqlc m)
endif()
