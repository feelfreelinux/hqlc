add_library(unity STATIC ${CMAKE_SOURCE_DIR}/external/Unity/src/unity.c)
target_include_directories(unity PUBLIC ${CMAKE_SOURCE_DIR}/external/Unity/src)

set(HQLC_TESTS
    test_roundtrip   # end-to-end encode/decode integration
    test_mdct        # MDCT forward/inverse transform identity
    test_entropy     # bitstream, Rice, zigzag, rANS roundtrip + table invariants
)

foreach(TEST_NAME ${HQLC_TESTS})
    add_executable(${TEST_NAME} ${CMAKE_CURRENT_SOURCE_DIR}/${TEST_NAME}.c)
    target_link_libraries(${TEST_NAME} PRIVATE unity hqlc m)
    target_compile_options(${TEST_NAME} PRIVATE -fsanitize=address -fno-omit-frame-pointer)
    target_link_options(${TEST_NAME} PRIVATE -fsanitize=address)
    add_test(NAME ${TEST_NAME} COMMAND ${TEST_NAME})
endforeach()
