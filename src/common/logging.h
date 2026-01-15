#pragma once

/// @file logging.h
/// @brief PyFlare logging utilities wrapping spdlog

#include <memory>
#include <string>
#include <string_view>

#include <spdlog/spdlog.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/sinks/rotating_file_sink.h>

namespace pyflare {

/// @brief Log levels matching spdlog levels
enum class LogLevel {
    kTrace = spdlog::level::trace,
    kDebug = spdlog::level::debug,
    kInfo = spdlog::level::info,
    kWarn = spdlog::level::warn,
    kError = spdlog::level::err,
    kCritical = spdlog::level::critical,
    kOff = spdlog::level::off
};

/// @brief Logging configuration
struct LogConfig {
    std::string name = "pyflare";
    LogLevel level = LogLevel::kInfo;
    std::string pattern = "[%Y-%m-%d %H:%M:%S.%e] [%n] [%^%l%$] [%t] %v";

    // File logging (optional)
    bool enable_file = false;
    std::string file_path = "pyflare.log";
    size_t max_file_size = 10 * 1024 * 1024;  // 10 MB
    size_t max_files = 5;
};

/// @brief Initialize the global logger with the given configuration
/// @param config Logging configuration
void InitLogging(const LogConfig& config = {});

/// @brief Get the global logger instance
/// @return Shared pointer to the logger
std::shared_ptr<spdlog::logger> GetLogger();

/// @brief Set the global log level
/// @param level Log level to set
void SetLogLevel(LogLevel level);

/// @brief Flush all log messages
void FlushLogs();

/// @brief Shutdown the logging system
void ShutdownLogging();

// Convenience macros for logging
#define PYFLARE_LOG_TRACE(...) SPDLOG_LOGGER_TRACE(::pyflare::GetLogger(), __VA_ARGS__)
#define PYFLARE_LOG_DEBUG(...) SPDLOG_LOGGER_DEBUG(::pyflare::GetLogger(), __VA_ARGS__)
#define PYFLARE_LOG_INFO(...) SPDLOG_LOGGER_INFO(::pyflare::GetLogger(), __VA_ARGS__)
#define PYFLARE_LOG_WARN(...) SPDLOG_LOGGER_WARN(::pyflare::GetLogger(), __VA_ARGS__)
#define PYFLARE_LOG_ERROR(...) SPDLOG_LOGGER_ERROR(::pyflare::GetLogger(), __VA_ARGS__)
#define PYFLARE_LOG_CRITICAL(...) SPDLOG_LOGGER_CRITICAL(::pyflare::GetLogger(), __VA_ARGS__)

}  // namespace pyflare
