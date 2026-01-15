# Unit tests for collector components

add_executable(sampler_test sampler_test.cpp)
target_link_libraries(sampler_test
    PRIVATE
        pyflare_collector_lib
        GTest::gtest
        GTest::gtest_main
)
gtest_discover_tests(sampler_test)

add_executable(batcher_test batcher_test.cpp)
target_link_libraries(batcher_test
    PRIVATE
        pyflare_collector_lib
        GTest::gtest
        GTest::gtest_main
)
gtest_discover_tests(batcher_test)
