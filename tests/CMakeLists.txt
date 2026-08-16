add_library(unity STATIC ${CMAKE_SOURCE_DIR}/external/Unity/src/unity.c)
target_include_directories(unity PUBLIC ${CMAKE_SOURCE_DIR}/external/Unity/src)

set(HQLC_TESTS
    test_roundtrip   # end-to-end encode/decode integration
    test_mdct        # MDCT forward/inverse transform identity
    test_entropy     # bitstream, Rice, zigzag, rANS roundtrip + table invariants
    test_quant       # quantizer scale decomposition
    test_tns         # TNS filtering and BFP exponent tracking
)

foreach(TEST_NAME ${HQLC_TESTS})
    add_executable(${TEST_NAME} ${CMAKE_CURRENT_SOURCE_DIR}/${TEST_NAME}.c)
    target_link_libraries(${TEST_NAME} PRIVATE unity hqlc m)
    target_compile_options(${TEST_NAME} PRIVATE
        -fsanitize=address,undefined
        -fno-sanitize-recover=undefined
        -fno-omit-frame-pointer)
    target_link_options(${TEST_NAME} PRIVATE -fsanitize=address,undefined)
    add_test(NAME ${TEST_NAME} COMMAND ${TEST_NAME})
endforeach()

target_include_directories(test_entropy PRIVATE ${CMAKE_SOURCE_DIR}/src)
