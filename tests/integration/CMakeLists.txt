# PyFlare Integration Tests

include(GoogleTest)

# Pipeline integration tests
add_executable(pipeline_integration_tests
    pipeline_test.cpp
)

target_link_libraries(pipeline_integration_tests
    PRIVATE
        PyFlare::Processor
        PyFlare::Storage
        GTest::gtest_main
)

gtest_discover_tests(pipeline_integration_tests)
