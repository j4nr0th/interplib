set(TEST_COMMON_FILES
        test_c/common/common.h
        test_c/common/common.c
)

add_executable(test_gll test_c/integration/test_gll.c ${TEST_COMMON_FILES})
target_link_libraries(test_gll PRIVATE interp m)
add_test(NAME gauss-legendre-lobato COMMAND test_gll)


add_executable(test_glg test_c/integration/test_glg.c ${TEST_COMMON_FILES})
target_link_libraries(test_glg PRIVATE interp m)
add_test(NAME gauss-legendre COMMAND test_glg)

add_executable(test_int_rule_registry test_c/integration/test_int_rule_registry.c ${TEST_COMMON_FILES})
target_link_libraries(test_int_rule_registry PRIVATE interp m)
add_test(NAME int_rule_registry COMMAND test_int_rule_registry)

add_executable(test_basis_set_registry test_c/basis/test_basis_set_registry.c ${TEST_COMMON_FILES})
target_link_libraries(test_basis_set_registry PRIVATE interp m)
add_test(NAME basis_set_registry COMMAND test_basis_set_registry)

add_executable(test_rw_lock test_c/common/test_rw_lock.c ${TEST_COMMON_FILES})
target_link_libraries(test_rw_lock PRIVATE interp)
add_test(NAME rw_lock COMMAND test_rw_lock)
