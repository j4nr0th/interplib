set(TEST_COMMON_FILES
        common/common.h
        common/common.c
)

# INTEGRATION TESTS

add_executable(test_gll integration/test_gll.c ${TEST_COMMON_FILES})
target_link_libraries(test_gll PRIVATE interp m)
add_test(NAME gauss-legendre-lobato COMMAND test_gll)

add_executable(test_glg integration/test_glg.c ${TEST_COMMON_FILES})
target_link_libraries(test_glg PRIVATE interp m)
add_test(NAME gauss-legendre COMMAND test_glg)

add_executable(test_int_rule_registry integration/test_int_rule_registry.c ${TEST_COMMON_FILES})
target_link_libraries(test_int_rule_registry PRIVATE interp m)
add_test(NAME int_rule_registry COMMAND test_int_rule_registry)

# BASIS TESTS

add_executable(test_basis_set_registry basis/test_basis_set_registry.c ${TEST_COMMON_FILES})
target_link_libraries(test_basis_set_registry PRIVATE interp m)
add_test(NAME basis_set_registry COMMAND test_basis_set_registry)

# COMMON TESTS

add_executable(test_rw_lock common/test_rw_lock.c ${TEST_COMMON_FILES})
target_link_libraries(test_rw_lock PRIVATE interp)
add_test(NAME rw_lock COMMAND test_rw_lock)

# POLYNOMIAL TESTS

add_executable(test_lagrange_transpose polynomials/test_lagrange_transpose.c ${TEST_COMMON_FILES})
target_link_libraries(test_lagrange_transpose PRIVATE interp m)
add_test(NAME lagrange_transpose COMMAND test_lagrange_transpose)

add_executable(test_lagrange_derivative_transpose polynomials/test_lagrange_derivative_transpose.c ${TEST_COMMON_FILES})
target_link_libraries(test_lagrange_derivative_transpose PRIVATE interp m)
add_test(NAME lagrange_derivative_transpose COMMAND test_lagrange_derivative_transpose)

# OPERATION TESTS

add_executable(test_nd_iterator_1d operations/test_nd_iterator_1d.c)
target_link_libraries(test_nd_iterator_1d PRIVATE interp m)
add_test(NAME nd_iterator_1d COMMAND test_nd_iterator_1d 120)

add_executable(test_nd_iterator_2d operations/test_nd_iterator_2d.c)
target_link_libraries(test_nd_iterator_2d PRIVATE interp m)
add_test(NAME nd_iterator_2d COMMAND test_nd_iterator_2d 12 40)

add_executable(test_nd_iterator_3d operations/test_nd_iterator_3d.c)
target_link_libraries(test_nd_iterator_3d PRIVATE interp m)
add_test(NAME nd_iterator_3d COMMAND test_nd_iterator_3d 13 7 17)
