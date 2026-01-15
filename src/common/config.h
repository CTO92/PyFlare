#pragma once

/// @file config.h
/// @brief PyFlare configuration management

#include <filesystem>
#include <optional>
#include <string>
#include <unordered_map>
#include <variant>
#include <vector>

#include <absl/status/statusor.h>
#include <nlohmann/json.hpp>
#include <yaml-cpp/yaml.h>

namespace pyflare {

/// @brief Configuration value that can hold different types
using ConfigValue = std::variant<
    bool,
    int64_t,
    double,
    std::string,
    std::vector<std::string>,
    std::unordered_map<std::string, std::string>
>;

/// @brief Configuration manager for loading and accessing configuration
class Config {
public:
    /// @brief Default constructor creates empty configuration
    Config() = default;

    /// @brief Load configuration from a YAML file
    /// @param path Path to the YAML configuration file
    /// @return Status indicating success or failure
    static absl::StatusOr<Config> LoadFromFile(const std::filesystem::path& path);

    /// @brief Load configuration from a YAML string
    /// @param yaml_content YAML content as a string
    /// @return Status indicating success or failure
    static absl::StatusOr<Config> LoadFromString(std::string_view yaml_content);

    /// @brief Load configuration from environment variables with a prefix
    /// @param prefix Environment variable prefix (e.g., "PYFLARE_")
    /// @return Configuration loaded from environment
    static Config LoadFromEnvironment(std::string_view prefix = "PYFLARE_");

    /// @brief Merge another configuration into this one (other takes precedence)
    /// @param other Configuration to merge
    void Merge(const Config& other);

    /// @brief Get a string value
    /// @param key Configuration key (supports dot notation, e.g., "collector.port")
    /// @param default_value Default value if key not found
    /// @return Configuration value or default
    std::string GetString(std::string_view key, std::string_view default_value = "") const;

    /// @brief Get an integer value
    /// @param key Configuration key
    /// @param default_value Default value if key not found
    /// @return Configuration value or default
    int64_t GetInt(std::string_view key, int64_t default_value = 0) const;

    /// @brief Get a double value
    /// @param key Configuration key
    /// @param default_value Default value if key not found
    /// @return Configuration value or default
    double GetDouble(std::string_view key, double default_value = 0.0) const;

    /// @brief Get a boolean value
    /// @param key Configuration key
    /// @param default_value Default value if key not found
    /// @return Configuration value or default
    bool GetBool(std::string_view key, bool default_value = false) const;

    /// @brief Get a list of strings
    /// @param key Configuration key
    /// @return List of strings or empty vector if not found
    std::vector<std::string> GetStringList(std::string_view key) const;

    /// @brief Check if a key exists
    /// @param key Configuration key
    /// @return True if key exists
    bool HasKey(std::string_view key) const;

    /// @brief Set a configuration value
    /// @param key Configuration key
    /// @param value Value to set
    void Set(std::string_view key, ConfigValue value);

    /// @brief Get the underlying YAML node for advanced access
    /// @return YAML node
    const YAML::Node& GetNode() const { return root_; }

    /// @brief Export configuration to JSON
    /// @return JSON representation
    nlohmann::json ToJson() const;

private:
    YAML::Node root_;

    /// @brief Navigate to a nested node using dot notation
    std::optional<YAML::Node> GetNestedNode(std::string_view key) const;
};

/// @brief Global configuration instance
Config& GlobalConfig();

/// @brief Initialize global configuration from file and environment
/// @param config_path Path to configuration file (optional)
/// @param env_prefix Environment variable prefix
/// @return Status indicating success or failure
absl::Status InitGlobalConfig(
    const std::optional<std::filesystem::path>& config_path = std::nullopt,
    std::string_view env_prefix = "PYFLARE_"
);

}  // namespace pyflare
