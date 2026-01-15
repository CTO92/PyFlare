#include "logging.h"

#include <mutex>
#include <vector>

namespace pyflare {

namespace {

std::shared_ptr<spdlog::logger> g_logger;
std::once_flag g_init_flag;

}  // namespace

void InitLogging(const LogConfig& config) {
    std::call_once(g_init_flag, [&config]() {
        std::vector<spdlog::sink_ptr> sinks;

        // Console sink (always enabled)
        auto console_sink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
        console_sink->set_level(static_cast<spdlog::level::level_enum>(config.level));
        sinks.push_back(console_sink);

        // File sink (optional)
        if (config.enable_file) {
            auto file_sink = std::make_shared<spdlog::sinks::rotating_file_sink_mt>(
                config.file_path,
                config.max_file_size,
                config.max_files
            );
            file_sink->set_level(static_cast<spdlog::level::level_enum>(config.level));
            sinks.push_back(file_sink);
        }

        // Create logger with all sinks
        g_logger = std::make_shared<spdlog::logger>(config.name, sinks.begin(), sinks.end());
        g_logger->set_level(static_cast<spdlog::level::level_enum>(config.level));
        g_logger->set_pattern(config.pattern);

        // Register as default logger
        spdlog::set_default_logger(g_logger);

        // Flush on warn and above
        g_logger->flush_on(spdlog::level::warn);
    });
}

std::shared_ptr<spdlog::logger> GetLogger() {
    if (!g_logger) {
        InitLogging();
    }
    return g_logger;
}

void SetLogLevel(LogLevel level) {
    if (g_logger) {
        g_logger->set_level(static_cast<spdlog::level::level_enum>(level));
    }
}

void FlushLogs() {
    if (g_logger) {
        g_logger->flush();
    }
}

void ShutdownLogging() {
    if (g_logger) {
        g_logger->flush();
        spdlog::shutdown();
        g_logger.reset();
    }
}

}  // namespace pyflare
