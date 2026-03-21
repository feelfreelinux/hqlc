# Thin wrapper: pull hqlc sources from the repo root into an ESP-IDF component.

set(HQLC_ROOT "${CMAKE_CURRENT_LIST_DIR}/../../..")

idf_component_register(
    SRCS
        "${HQLC_ROOT}/src/hqlc.c"
        "${HQLC_ROOT}/src/mdct.c"
        "${HQLC_ROOT}/src/mdct_tables.c"
        "${HQLC_ROOT}/src/psy.c"
        "${HQLC_ROOT}/src/quant.c"
        "${HQLC_ROOT}/src/entropy.c"
        "${HQLC_ROOT}/src/entropy_tables.c"
        "${HQLC_ROOT}/src/tns.c"
    INCLUDE_DIRS
        "${HQLC_ROOT}/include"
        "${CMAKE_CURRENT_LIST_DIR}/../main"
    LDFRAGMENTS
        "linker.lf"
)

target_compile_definitions(${COMPONENT_LIB} PUBLIC HQLC_BENCH)

# Ensure codec hot paths get -O2 regardless of global setting
target_compile_options(${COMPONENT_LIB} PRIVATE -O2)
