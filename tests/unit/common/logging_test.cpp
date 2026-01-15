/// @file logging_test.cpp
/// @brief Tests for PyFlare logging utilities

#include <gtest/gtest.h>

#include "common/logging.h"

namespace pyflare {
namespace {

TEST(LoggingTest, InitializeLogging) {
    LogConfig config;
    config.name = "test-logger";
    config.level = LogLevel::kDebug;

    EXPECT_NO_THROW(InitLogging(config));
    EXPECT_NE(GetLogger(), nullptr);
}

TEST(LoggingTest, LogLevelChange) {
    InitLogging();

    EXPECT_NO_THROW(SetLogLevel(LogLevel::kWarn));
    EXPECT_NO_THROW(SetLogLevel(LogLevel::kDebug));
}

TEST(LoggingTest, LoggingMacros) {
    InitLogging();

    // These should not throw
    EXPECT_NO_THROW({
        PYFLARE_LOG_TRACE("Trace message: {}", 1);
        PYFLARE_LOG_DEBUG("Debug message: {}", 2);
        PYFLARE_LOG_INFO("Info message: {}", 3);
        PYFLARE_LOG_WARN("Warn message: {}", 4);
        PYFLARE_LOG_ERROR("Error message: {}", 5);
    });
}

TEST(LoggingTest, FlushLogs) {
    InitLogging();
    PYFLARE_LOG_INFO("Test message");
    EXPECT_NO_THROW(FlushLogs());
}

}  // namespace
}  // namespace pyflare
