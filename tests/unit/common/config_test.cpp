/// @file config_test.cpp
/// @brief Tests for PyFlare configuration management

#include <gtest/gtest.h>

#include "common/config.h"

namespace pyflare {
namespace {

TEST(ConfigTest, LoadFromString) {
    const std::string yaml_content = R"(
collector:
  grpc_port: 4317
  http_port: 4318
kafka:
  brokers:
    - localhost:9092
    - localhost:9093
logging:
  level: debug
)";

    auto result = Config::LoadFromString(yaml_content);
    ASSERT_TRUE(result.ok()) << result.status().message();

    Config config = std::move(*result);

    EXPECT_EQ(config.GetInt("collector.grpc_port"), 4317);
    EXPECT_EQ(config.GetInt("collector.http_port"), 4318);
    EXPECT_EQ(config.GetString("logging.level"), "debug");

    auto brokers = config.GetStringList("kafka.brokers");
    ASSERT_EQ(brokers.size(), 2);
    EXPECT_EQ(brokers[0], "localhost:9092");
    EXPECT_EQ(brokers[1], "localhost:9093");
}

TEST(ConfigTest, DefaultValues) {
    Config config;

    EXPECT_EQ(config.GetString("nonexistent.key", "default"), "default");
    EXPECT_EQ(config.GetInt("nonexistent.key", 42), 42);
    EXPECT_EQ(config.GetDouble("nonexistent.key", 3.14), 3.14);
    EXPECT_EQ(config.GetBool("nonexistent.key", true), true);
}

TEST(ConfigTest, SetValues) {
    Config config;

    config.Set("test.string", std::string("value"));
    config.Set("test.int", static_cast<int64_t>(123));
    config.Set("test.bool", true);

    EXPECT_EQ(config.GetString("test.string"), "value");
    EXPECT_EQ(config.GetInt("test.int"), 123);
    EXPECT_EQ(config.GetBool("test.bool"), true);
}

TEST(ConfigTest, HasKey) {
    const std::string yaml_content = R"(
existing:
  key: value
)";

    auto result = Config::LoadFromString(yaml_content);
    ASSERT_TRUE(result.ok());

    Config config = std::move(*result);

    EXPECT_TRUE(config.HasKey("existing.key"));
    EXPECT_FALSE(config.HasKey("nonexistent.key"));
}

TEST(ConfigTest, MergeConfigs) {
    const std::string base_yaml = R"(
key1: value1
nested:
  a: 1
  b: 2
)";

    const std::string overlay_yaml = R"(
key2: value2
nested:
  b: 20
  c: 3
)";

    auto base_result = Config::LoadFromString(base_yaml);
    auto overlay_result = Config::LoadFromString(overlay_yaml);
    ASSERT_TRUE(base_result.ok());
    ASSERT_TRUE(overlay_result.ok());

    Config base = std::move(*base_result);
    Config overlay = std::move(*overlay_result);

    base.Merge(overlay);

    EXPECT_EQ(base.GetString("key1"), "value1");
    EXPECT_EQ(base.GetString("key2"), "value2");
    EXPECT_EQ(base.GetInt("nested.a"), 1);
    EXPECT_EQ(base.GetInt("nested.b"), 20);  // Overwritten
    EXPECT_EQ(base.GetInt("nested.c"), 3);   // Added
}

TEST(ConfigTest, InvalidYaml) {
    const std::string invalid_yaml = "{ invalid yaml [";

    auto result = Config::LoadFromString(invalid_yaml);
    EXPECT_FALSE(result.ok());
}

}  // namespace
}  // namespace pyflare
