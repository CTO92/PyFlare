# PyFlare Unit Tests

include(GoogleTest)

# Common tests
add_executable(common_tests
    common/logging_test.cpp
    common/config_test.cpp
    common/metrics_test.cpp
    common/thread_pool_test.cpp
)

target_link_libraries(common_tests
    PRIVATE
        PyFlare::Common
        GTest::gtest_main
)

gtest_discover_tests(common_tests)

# Drift detector tests
add_executable(drift_tests
    processor/drift_detector_test.cpp
)

target_link_libraries(drift_tests
    PRIVATE
        PyFlare::Processor
        GTest::gtest_main
)

gtest_discover_tests(drift_tests)

# Cost tracker tests
add_executable(cost_tests
    processor/cost_tracker_test.cpp
)

target_link_libraries(cost_tests
    PRIVATE
        PyFlare::Processor
        GTest::gtest_main
)

gtest_discover_tests(cost_tests)

# =============================================================================
# Phase 2 Tests
# =============================================================================

# PSI drift detector tests
add_executable(psi_detector_tests
    processor/drift/psi_detector_test.cpp
)

target_link_libraries(psi_detector_tests
    PRIVATE
        PyFlare::Processor
        GTest::gtest_main
)

gtest_discover_tests(psi_detector_tests)

# MMD drift detector tests
add_executable(mmd_detector_tests
    processor/drift/mmd_detector_test.cpp
)

target_link_libraries(mmd_detector_tests
    PRIVATE
        PyFlare::Processor
        GTest::gtest_main
)

gtest_discover_tests(mmd_detector_tests)

# Budget manager tests
add_executable(budget_manager_tests
    processor/cost/budget_manager_test.cpp
)

target_link_libraries(budget_manager_tests
    PRIVATE
        PyFlare::Processor
        PyFlare::Storage
        GTest::gtest_main
)

gtest_discover_tests(budget_manager_tests)

# LLM Judge evaluator tests
add_executable(llm_judge_tests
    processor/eval/llm_judge_test.cpp
)

target_link_libraries(llm_judge_tests
    PRIVATE
        PyFlare::Processor
        GTest::gtest_main
)

gtest_discover_tests(llm_judge_tests)

# Slice analyzer tests
add_executable(slice_analyzer_tests
    processor/rca/slice_analyzer_test.cpp
)

target_link_libraries(slice_analyzer_tests
    PRIVATE
        PyFlare::Processor
        PyFlare::Storage
        GTest::gtest_main
)

gtest_discover_tests(slice_analyzer_tests)
