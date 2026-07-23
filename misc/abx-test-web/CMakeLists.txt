cmake_minimum_required(VERSION 3.16)
project(hqlc_abx C)

include(FetchContent)

set(HQLC_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../..")

add_library(hqlc_codec STATIC
    ${HQLC_ROOT}/src/hqlc.c
    ${HQLC_ROOT}/src/mdct.c
    ${HQLC_ROOT}/src/mdct_tables.c
    ${HQLC_ROOT}/src/psy.c
    ${HQLC_ROOT}/src/quant.c
    ${HQLC_ROOT}/src/entropy.c
    ${HQLC_ROOT}/src/entropy_tables.c
    ${HQLC_ROOT}/src/tns.c
    ${HQLC_ROOT}/src/ms.c
)
target_include_directories(hqlc_codec PUBLIC ${HQLC_ROOT}/include)

FetchContent_Declare(opus
    URL https://downloads.xiph.org/releases/opus/opus-1.6.tar.gz
    URL_HASH SHA256=b7637334527201fdfd6dd6a02e67aceffb0e5e60155bbd89175647a80301c92c
)
set(OPUS_INSTALL_CMAKE_CONFIG_MODULE OFF CACHE BOOL "" FORCE)
set(OPUS_INSTALL_PKG_CONFIG_MODULE OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(opus)

FetchContent_Declare(lame
    URL https://sourceforge.net/projects/lame/files/lame/3.100/lame-3.100.tar.gz/download
    URL_HASH SHA256=ddfe36cab873794038ae2c1210557ad34857a4b6bdc515785d1da9e175b1da1e
    DOWNLOAD_NAME lame-3.100.tar.gz
)
FetchContent_MakeAvailable(lame)

file(GLOB LAME_SRCS "${lame_SOURCE_DIR}/libmp3lame/*.c")
list(FILTER LAME_SRCS EXCLUDE REGEX "mpglib_interface\\.c$")

add_library(lame_enc STATIC ${LAME_SRCS})
target_include_directories(lame_enc
    PUBLIC "${lame_SOURCE_DIR}/include"
    PRIVATE "${lame_SOURCE_DIR}/libmp3lame" "${CMAKE_CURRENT_SOURCE_DIR}/src"
)
target_compile_definitions(lame_enc PRIVATE HAVE_CONFIG_H)

FetchContent_Declare(minimp3
    GIT_REPOSITORY https://github.com/lieff/minimp3.git
    GIT_TAG master
    GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(minimp3)

add_library(minimp3 INTERFACE)
target_include_directories(minimp3 INTERFACE "${minimp3_SOURCE_DIR}")

add_executable(abx_codecs src/wasm_glue.c)
target_link_libraries(abx_codecs PRIVATE hqlc_codec opus lame_enc minimp3 m)
target_include_directories(abx_codecs PRIVATE src)
target_compile_definitions(abx_codecs PRIVATE HAVE_CONFIG_H)

set_target_properties(abx_codecs PROPERTIES
    SUFFIX ".js"
    LINK_FLAGS "\
        -s EXPORTED_FUNCTIONS=[_roundtrip_hqlc,_roundtrip_opus,_roundtrip_mp3,_malloc,_free] \
        -s EXPORTED_RUNTIME_METHODS=[ccall,HEAP16] \
        -s ALLOW_MEMORY_GROWTH=1 \
        -s MODULARIZE=1 \
        -s EXPORT_NAME=createCodecModule \
        -s STACK_SIZE=1048576 \
        --no-entry"
)
