#include "config.h"

#include <cstdlib>
#include <fstream>
#include <sstream>

#include <absl/strings/str_split.h>
#include <absl/strings/ascii.h>

#include "logging.h"

namespace pyflare {

namespace {

Config g_global_config;

}  // namespace

absl::StatusOr<Config> Config::LoadFromFile(const std::filesystem::path& path) {
    if (!std::filesystem::exists(path)) {
        return absl::NotFoundError(
            absl::StrCat("Configuration file not found: ", path.string()));
    }

    try {
        Config config;
        config.root_ = YAML::LoadFile(path.string());
        return config;
    } catch (const YAML::Exception& e) {
        return absl::InvalidArgumentError(
            absl::StrCat("Failed to parse YAML configuration: ", e.what()));
    }
}

absl::StatusOr<Config> Config::LoadFromString(std::string_view yaml_content) {
    try {
        Config config;
        config.root_ = YAML::Load(std::string(yaml_content));
        return config;
    } catch (const YAML::Exception& e) {
        return absl::InvalidArgumentError(
            absl::StrCat("Failed to parse YAML content: ", e.what()));
    }
}

Config Config::LoadFromEnvironment(std::string_view prefix) {
    Config config;

    // This is a simplified implementation
    // In production, you'd iterate through environment variables
    // For now, we support common configuration keys

    auto get_env = [&prefix](const char* suffix) -> std::optional<std::string> {
        std::string key = std::string(prefix) + suffix;
        const char* value = std::getenv(key.c_str());
        if (value != nullptr) {
            return std::string(value);
        }
        return std::nullopt;
    };

    // Collector settings
    if (auto val = get_env("COLLECTOR_GRPC_PORT")) {
        config.Set("collector.otlp.grpc.port", std::stoll(*val));
    }
    if (auto val = get_env("COLLECTOR_HTTP_PORT")) {
        config.Set("collector.otlp.http.port", std::stoll(*val));
    }

    // Kafka settings
    if (auto val = get_env("KAFKA_BROKERS")) {
        std::vector<std::string> brokers = absl::StrSplit(*val, ',');
        config.Set("kafka.brokers", brokers);
    }

    // ClickHouse settings
    if (auto val = get_env("CLICKHOUSE_HOST")) {
        config.Set("clickhouse.host", *val);
    }
    if (auto val = get_env("CLICKHOUSE_PORT")) {
        config.Set("clickhouse.port", std::stoll(*val));
    }

    // Qdrant settings
    if (auto val = get_env("QDRANT_HOST")) {
        config.Set("qdrant.host", *val);
    }

    // Log level
    if (auto val = get_env("LOG_LEVEL")) {
        config.Set("logging.level", *val);
    }

    return config;
}

void Config::Merge(const Config& other) {
    // Deep merge YAML nodes
    std::function<void(YAML::Node&, const YAML::Node&)> merge_nodes;
    merge_nodes = [&merge_nodes](YAML::Node& base, const YAML::Node& overlay) {
        if (overlay.IsMap()) {
            for (const auto& kv : overlay) {
                const std::string key = kv.first.as<std::string>();
                if (base[key] && base[key].IsMap() && kv.second.IsMap()) {
                    YAML::Node base_child = base[key];
                    merge_nodes(base_child, kv.second);
                } else {
                    base[key] = kv.second;
                }
            }
        }
    };

    merge_nodes(root_, other.root_);
}

std::optional<YAML::Node> Config::GetNestedNode(std::string_view key) const {
    std::vector<std::string> parts = absl::StrSplit(key, '.');
    YAML::Node current = root_;

    for (const auto& part : parts) {
        if (!current || !current.IsMap()) {
            return std::nullopt;
        }
        current = current[part];
    }

    if (!current || current.IsNull()) {
        return std::nullopt;
    }

    return current;
}

std::string Config::GetString(std::string_view key, std::string_view default_value) const {
    auto node = GetNestedNode(key);
    if (node && node->IsScalar()) {
        return node->as<std::string>();
    }
    return std::string(default_value);
}

int64_t Config::GetInt(std::string_view key, int64_t default_value) const {
    auto node = GetNestedNode(key);
    if (node && node->IsScalar()) {
        try {
            return node->as<int64_t>();
        } catch (...) {
            return default_value;
        }
    }
    return default_value;
}

double Config::GetDouble(std::string_view key, double default_value) const {
    auto node = GetNestedNode(key);
    if (node && node->IsScalar()) {
        try {
            return node->as<double>();
        } catch (...) {
            return default_value;
        }
    }
    return default_value;
}

bool Config::GetBool(std::string_view key, bool default_value) const {
    auto node = GetNestedNode(key);
    if (node && node->IsScalar()) {
        try {
            return node->as<bool>();
        } catch (...) {
            return default_value;
        }
    }
    return default_value;
}

std::vector<std::string> Config::GetStringList(std::string_view key) const {
    std::vector<std::string> result;
    auto node = GetNestedNode(key);
    if (node && node->IsSequence()) {
        for (const auto& item : *node) {
            if (item.IsScalar()) {
                result.push_back(item.as<std::string>());
            }
        }
    }
    return result;
}

bool Config::HasKey(std::string_view key) const {
    return GetNestedNode(key).has_value();
}

void Config::Set(std::string_view key, ConfigValue value) {
    std::vector<std::string> parts = absl::StrSplit(key, '.');

    YAML::Node* current = &root_;
    for (size_t i = 0; i < parts.size() - 1; ++i) {
        if (!(*current)[parts[i]]) {
            (*current)[parts[i]] = YAML::Node(YAML::NodeType::Map);
        }
        current = &(*current)[parts[i]];
    }

    std::visit([&](auto&& val) {
        using T = std::decay_t<decltype(val)>;
        if constexpr (std::is_same_v<T, std::vector<std::string>>) {
            YAML::Node seq(YAML::NodeType::Sequence);
            for (const auto& item : val) {
                seq.push_back(item);
            }
            (*current)[parts.back()] = seq;
        } else if constexpr (std::is_same_v<T, std::unordered_map<std::string, std::string>>) {
            YAML::Node map(YAML::NodeType::Map);
            for (const auto& [k, v] : val) {
                map[k] = v;
            }
            (*current)[parts.back()] = map;
        } else {
            (*current)[parts.back()] = val;
        }
    }, value);
}

nlohmann::json Config::ToJson() const {
    std::stringstream ss;
    ss << root_;
    // Convert YAML to JSON (simplified - in production use proper conversion)
    return nlohmann::json::parse(ss.str(), nullptr, false);
}

Config& GlobalConfig() {
    return g_global_config;
}

absl::Status InitGlobalConfig(
    const std::optional<std::filesystem::path>& config_path,
    std::string_view env_prefix
) {
    // Start with default configuration
    g_global_config = Config();

    // Load from file if provided
    if (config_path.has_value()) {
        auto file_config = Config::LoadFromFile(*config_path);
        if (!file_config.ok()) {
            return file_config.status();
        }
        g_global_config.Merge(*file_config);
    }

    // Overlay environment variables (highest priority)
    Config env_config = Config::LoadFromEnvironment(env_prefix);
    g_global_config.Merge(env_config);

    return absl::OkStatus();
}

}  // namespace pyflare
