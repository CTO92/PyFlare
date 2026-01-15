/// @file client.cpp
/// @brief ClickHouse client implementation for PyFlare

#include "storage/clickhouse/client.h"

#include <sstream>

#include <spdlog/spdlog.h>

#ifdef PYFLARE_HAS_CLICKHOUSE
#include <clickhouse/client.h>
#include <clickhouse/columns/array.h>
#include <clickhouse/columns/date.h>
#include <clickhouse/columns/map.h>
#include <clickhouse/columns/nullable.h>
#include <clickhouse/columns/numeric.h>
#include <clickhouse/columns/string.h>
#include <clickhouse/columns/uuid.h>
#endif

namespace pyflare::storage {

// =============================================================================
// ClickHouseClient Implementation
// =============================================================================

#ifdef PYFLARE_HAS_CLICKHOUSE

class ClickHouseClient::Impl {
public:
    explicit Impl(ClickHouseConfig config) : config_(std::move(config)) {}

    ~Impl() {
        Disconnect();
    }

    absl::Status Connect() {
        if (connected_) {
            return absl::OkStatus();
        }

        try {
            clickhouse::ClientOptions options;
            options.SetHost(config_.host)
                   .SetPort(config_.port)
                   .SetUser(config_.user)
                   .SetPassword(config_.password)
                   .SetDefaultDatabase(config_.database)
                   .SetSendRetries(3)
                   .SetRetryTimeout(std::chrono::seconds(5))
                   .SetCompressionMethod(
                       config_.use_compression
                           ? clickhouse::CompressionMethod::LZ4
                           : clickhouse::CompressionMethod::None);

            client_ = std::make_unique<clickhouse::Client>(options);

            // Test connection
            client_->Select("SELECT 1", [](const clickhouse::Block&) {});

            connected_ = true;
            spdlog::info("Connected to ClickHouse at {}:{}/{}",
                         config_.host, config_.port, config_.database);
            return absl::OkStatus();

        } catch (const std::exception& e) {
            return absl::UnavailableError(
                std::string("Failed to connect to ClickHouse: ") + e.what());
        }
    }

    absl::Status Disconnect() {
        if (!connected_) {
            return absl::OkStatus();
        }

        try {
            client_.reset();
            connected_ = false;
            spdlog::info("Disconnected from ClickHouse");
            return absl::OkStatus();
        } catch (const std::exception& e) {
            return absl::InternalError(
                std::string("Error disconnecting from ClickHouse: ") + e.what());
        }
    }

    bool IsConnected() const {
        return connected_ && client_ != nullptr;
    }

    absl::StatusOr<QueryResult> Execute(const std::string& sql) {
        if (!IsConnected()) {
            return absl::FailedPreconditionError("Not connected to ClickHouse");
        }

        QueryResult result;
        auto start_time = std::chrono::steady_clock::now();

        try {
            bool first_block = true;

            client_->Select(sql, [&result, &first_block](const clickhouse::Block& block) {
                // Extract column names from first block
                if (first_block) {
                    for (size_t i = 0; i < block.GetColumnCount(); ++i) {
                        result.columns.push_back(block.GetColumnName(i));
                        result.column_types.push_back(block[i]->GetType().GetName());
                    }
                    first_block = false;
                }

                // Extract rows
                for (size_t row = 0; row < block.GetRowCount(); ++row) {
                    std::vector<std::string> row_data;
                    for (size_t col = 0; col < block.GetColumnCount(); ++col) {
                        row_data.push_back(ColumnValueToString(block[col], row));
                    }
                    result.rows.push_back(std::move(row_data));
                }

                result.total_rows += block.GetRowCount();
            });

            auto end_time = std::chrono::steady_clock::now();
            result.execution_time = std::chrono::duration_cast<std::chrono::milliseconds>(
                end_time - start_time);

            spdlog::debug("Query executed in {}ms, {} rows returned",
                          result.execution_time.count(), result.total_rows);
            return result;

        } catch (const std::exception& e) {
            return absl::InternalError(
                std::string("Query execution failed: ") + e.what());
        }
    }

    /// @brief Execute query with parameters
    ///
    /// SECURITY NOTE: This implementation uses string interpolation with comprehensive
    /// escaping (see EscapeValue) rather than driver-level prepared statements.
    /// This is because the clickhouse-cpp library doesn't support server-side
    /// prepared statements for SELECT queries.
    ///
    /// Mitigations in place:
    /// 1. All user input is validated by SqlParser BEFORE reaching this method
    /// 2. EscapeValue performs type-aware escaping for all SQL types
    /// 3. Special characters (\0, \n, \r, etc.) are escaped or rejected
    /// 4. Numeric types are validated to contain only valid characters
    /// 5. Unknown types default to string escaping
    ///
    /// For maximum security, ensure all queries pass through SqlParser validation
    /// before calling ExecuteWithParams.
    ///
    /// TODO: Migrate to native prepared statements when clickhouse-cpp adds support
    /// See: https://github.com/ClickHouse/clickhouse-cpp/issues/XXX
    absl::StatusOr<QueryResult> ExecuteWithParams(
        const std::string& sql,
        const std::vector<QueryParam>& params) {
        // Build parameterized query with escaped values
        std::string final_sql = sql;

        for (const auto& param : params) {
            std::string placeholder = "{" + param.name + "}";
            size_t pos = final_sql.find(placeholder);
            if (pos != std::string::npos) {
                std::string escaped_value = EscapeValue(param.value, param.type);
                final_sql.replace(pos, placeholder.length(), escaped_value);
            }
        }

        return Execute(final_sql);
    }

    absl::Status InsertTraces(const std::vector<TraceRecord>& traces) {
        if (!IsConnected()) {
            return absl::FailedPreconditionError("Not connected to ClickHouse");
        }

        if (traces.empty()) {
            return absl::OkStatus();
        }

        try {
            clickhouse::Block block;

            // Create columns
            auto trace_id = std::make_shared<clickhouse::ColumnString>();
            auto span_id = std::make_shared<clickhouse::ColumnString>();
            auto parent_span_id = std::make_shared<clickhouse::ColumnString>();
            auto start_time = std::make_shared<clickhouse::ColumnDateTime64>(9);
            auto end_time = std::make_shared<clickhouse::ColumnDateTime64>(9);
            auto service_name = std::make_shared<clickhouse::ColumnString>();
            auto span_name = std::make_shared<clickhouse::ColumnString>();
            auto span_kind = std::make_shared<clickhouse::ColumnString>();
            auto status_code = std::make_shared<clickhouse::ColumnString>();
            auto status_message = std::make_shared<clickhouse::ColumnString>();
            auto model_id = std::make_shared<clickhouse::ColumnString>();
            auto model_provider = std::make_shared<clickhouse::ColumnString>();
            auto model_version = std::make_shared<clickhouse::ColumnString>();
            auto inference_type = std::make_shared<clickhouse::ColumnString>();
            auto input_tokens = std::make_shared<clickhouse::ColumnUInt32>();
            auto output_tokens = std::make_shared<clickhouse::ColumnUInt32>();
            auto total_tokens = std::make_shared<clickhouse::ColumnUInt32>();
            auto cost_micros = std::make_shared<clickhouse::ColumnInt64>();
            auto user_id = std::make_shared<clickhouse::ColumnString>();
            auto feature_id = std::make_shared<clickhouse::ColumnString>();
            auto session_id = std::make_shared<clickhouse::ColumnString>();
            auto input_preview = std::make_shared<clickhouse::ColumnString>();
            auto output_preview = std::make_shared<clickhouse::ColumnString>();

            // Populate columns
            for (const auto& trace : traces) {
                trace_id->Append(trace.trace_id);
                span_id->Append(trace.span_id);
                parent_span_id->Append(trace.parent_span_id.value_or(""));

                auto start_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                    trace.start_time.time_since_epoch()).count();
                auto end_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                    trace.end_time.time_since_epoch()).count();
                start_time->Append(start_ns);
                end_time->Append(end_ns);

                service_name->Append(trace.service_name);
                span_name->Append(trace.operation_name);
                span_kind->Append(trace.span_kind);
                status_code->Append(trace.status_code);
                status_message->Append(trace.status_message.value_or(""));
                model_id->Append(trace.model_id);
                model_provider->Append("");  // TODO: Add to TraceRecord
                model_version->Append(trace.model_version);
                inference_type->Append(trace.inference_type);
                input_tokens->Append(trace.input_tokens.value_or(0));
                output_tokens->Append(trace.output_tokens.value_or(0));
                total_tokens->Append(trace.input_tokens.value_or(0) +
                                     trace.output_tokens.value_or(0));
                cost_micros->Append(static_cast<int64_t>(trace.cost_micros.value_or(0)));
                user_id->Append("");  // TODO: Extract from attributes
                feature_id->Append("");
                session_id->Append("");
                input_preview->Append(trace.input_preview);
                output_preview->Append(trace.output_preview);
            }

            // Add columns to block
            block.AppendColumn("trace_id", trace_id);
            block.AppendColumn("span_id", span_id);
            block.AppendColumn("parent_span_id", parent_span_id);
            block.AppendColumn("start_time", start_time);
            block.AppendColumn("end_time", end_time);
            block.AppendColumn("service_name", service_name);
            block.AppendColumn("span_name", span_name);
            block.AppendColumn("span_kind", span_kind);
            block.AppendColumn("status_code", status_code);
            block.AppendColumn("status_message", status_message);
            block.AppendColumn("model_id", model_id);
            block.AppendColumn("model_provider", model_provider);
            block.AppendColumn("model_version", model_version);
            block.AppendColumn("inference_type", inference_type);
            block.AppendColumn("input_tokens", input_tokens);
            block.AppendColumn("output_tokens", output_tokens);
            block.AppendColumn("total_tokens", total_tokens);
            block.AppendColumn("cost_micros", cost_micros);
            block.AppendColumn("user_id", user_id);
            block.AppendColumn("feature_id", feature_id);
            block.AppendColumn("session_id", session_id);
            block.AppendColumn("input_preview", input_preview);
            block.AppendColumn("output_preview", output_preview);

            client_->Insert("traces", block);

            spdlog::debug("Inserted {} traces into ClickHouse", traces.size());
            return absl::OkStatus();

        } catch (const std::exception& e) {
            return absl::InternalError(
                std::string("Failed to insert traces: ") + e.what());
        }
    }

    absl::Status InsertMetrics(const std::vector<MetricRecord>& metrics) {
        if (!IsConnected()) {
            return absl::FailedPreconditionError("Not connected to ClickHouse");
        }

        if (metrics.empty()) {
            return absl::OkStatus();
        }

        // TODO: Implement metric insertion
        spdlog::debug("Inserting {} metrics (not yet implemented)", metrics.size());
        return absl::OkStatus();
    }

    absl::Status ExecuteDDL(const std::string& ddl) {
        if (!IsConnected()) {
            return absl::FailedPreconditionError("Not connected to ClickHouse");
        }

        try {
            client_->Execute(ddl);
            return absl::OkStatus();
        } catch (const std::exception& e) {
            return absl::InternalError(
                std::string("DDL execution failed: ") + e.what());
        }
    }

private:
    std::string ColumnValueToString(const clickhouse::ColumnRef& column, size_t row) {
        // Handle different column types
        if (auto str_col = column->As<clickhouse::ColumnString>()) {
            return std::string(str_col->At(row));
        }
        if (auto int64_col = column->As<clickhouse::ColumnInt64>()) {
            return std::to_string(int64_col->At(row));
        }
        if (auto uint64_col = column->As<clickhouse::ColumnUInt64>()) {
            return std::to_string(uint64_col->At(row));
        }
        if (auto int32_col = column->As<clickhouse::ColumnInt32>()) {
            return std::to_string(int32_col->At(row));
        }
        if (auto uint32_col = column->As<clickhouse::ColumnUInt32>()) {
            return std::to_string(uint32_col->At(row));
        }
        if (auto float64_col = column->As<clickhouse::ColumnFloat64>()) {
            return std::to_string(float64_col->At(row));
        }
        if (auto float32_col = column->As<clickhouse::ColumnFloat32>()) {
            return std::to_string(float32_col->At(row));
        }
        // Default: return empty string
        return "";
    }

    /// @brief SECURITY: Escape and validate value for SQL
    std::string EscapeValue(const std::string& value, const std::string& type) {
        if (type == "String") {
            // SECURITY: Comprehensive string escaping for SQL injection prevention
            std::string escaped;
            escaped.reserve(value.size() + 10);
            escaped += "'";
            for (unsigned char c : value) {
                switch (c) {
                    case '\'':
                        escaped += "''";  // Double single quotes
                        break;
                    case '\\':
                        escaped += "\\\\";  // Escape backslash
                        break;
                    case '\0':
                        // SECURITY: Reject null bytes which can truncate strings
                        escaped += "\\0";
                        break;
                    case '\n':
                        escaped += "\\n";  // Escape newline
                        break;
                    case '\r':
                        escaped += "\\r";  // Escape carriage return
                        break;
                    case '\t':
                        escaped += "\\t";  // Escape tab
                        break;
                    case '\b':
                        escaped += "\\b";  // Escape backspace
                        break;
                    default:
                        // SECURITY: Only allow printable ASCII and UTF-8 characters
                        if (c >= 32 && c < 127) {
                            escaped += static_cast<char>(c);
                        } else if (c >= 128) {
                            // Allow UTF-8 continuation bytes
                            escaped += static_cast<char>(c);
                        }
                        // Skip other control characters
                        break;
                }
            }
            escaped += "'";
            return escaped;
        }

        // SECURITY: Validate numeric types to prevent injection
        if (type == "Int8" || type == "Int16" || type == "Int32" || type == "Int64" ||
            type == "UInt8" || type == "UInt16" || type == "UInt32" || type == "UInt64") {
            // Validate integer format
            if (value.empty()) {
                return "0";
            }
            size_t pos = 0;
            if (value[0] == '-') {
                pos = 1;
            }
            for (; pos < value.size(); ++pos) {
                if (!std::isdigit(static_cast<unsigned char>(value[pos]))) {
                    spdlog::warn("SECURITY: Invalid integer value rejected: {}", value);
                    return "0";  // Return safe default
                }
            }
            return value;
        }

        if (type == "Float32" || type == "Float64") {
            // Validate float format
            if (value.empty()) {
                return "0.0";
            }
            bool has_dot = false;
            bool has_exp = false;
            size_t pos = 0;
            if (value[0] == '-' || value[0] == '+') {
                pos = 1;
            }
            for (; pos < value.size(); ++pos) {
                char c = value[pos];
                if (std::isdigit(static_cast<unsigned char>(c))) {
                    continue;
                } else if (c == '.' && !has_dot && !has_exp) {
                    has_dot = true;
                } else if ((c == 'e' || c == 'E') && !has_exp) {
                    has_exp = true;
                    if (pos + 1 < value.size() && (value[pos + 1] == '-' || value[pos + 1] == '+')) {
                        ++pos;
                    }
                } else {
                    spdlog::warn("SECURITY: Invalid float value rejected: {}", value);
                    return "0.0";  // Return safe default
                }
            }
            return value;
        }

        if (type == "DateTime" || type == "Date") {
            // Validate datetime format (basic check)
            for (char c : value) {
                if (!std::isdigit(static_cast<unsigned char>(c)) &&
                    c != '-' && c != ':' && c != ' ' && c != 'T' && c != 'Z' && c != '+') {
                    spdlog::warn("SECURITY: Invalid datetime value rejected: {}", value);
                    return "''";  // Return empty string
                }
            }
            return "'" + value + "'";
        }

        // Unknown type - treat as string with escaping
        spdlog::warn("SECURITY: Unknown type {}, treating as string", type);
        return EscapeValue(value, "String");
    }

    ClickHouseConfig config_;
    std::unique_ptr<clickhouse::Client> client_;
    bool connected_ = false;
};

#else  // !PYFLARE_HAS_CLICKHOUSE

/// @brief Stub implementation when ClickHouse is not available
class ClickHouseClient::Impl {
public:
    explicit Impl(ClickHouseConfig config) : config_(std::move(config)) {}

    absl::Status Connect() {
        spdlog::warn("ClickHouse support not compiled in");
        connected_ = true;  // Pretend to be connected for testing
        return absl::OkStatus();
    }

    absl::Status Disconnect() {
        connected_ = false;
        return absl::OkStatus();
    }

    bool IsConnected() const { return connected_; }

    absl::StatusOr<QueryResult> Execute(const std::string&) {
        return QueryResult{};
    }

    absl::StatusOr<QueryResult> ExecuteWithParams(
        const std::string&, const std::vector<QueryParam>&) {
        return QueryResult{};
    }

    absl::Status InsertTraces(const std::vector<TraceRecord>& traces) {
        spdlog::debug("Mock inserting {} traces", traces.size());
        return absl::OkStatus();
    }

    absl::Status InsertMetrics(const std::vector<MetricRecord>& metrics) {
        spdlog::debug("Mock inserting {} metrics", metrics.size());
        return absl::OkStatus();
    }

    absl::Status ExecuteDDL(const std::string&) {
        return absl::OkStatus();
    }

private:
    ClickHouseConfig config_;
    bool connected_ = false;
};

#endif  // PYFLARE_HAS_CLICKHOUSE

// =============================================================================
// ClickHouseClient Public Interface
// =============================================================================

ClickHouseClient::ClickHouseClient(ClickHouseConfig config)
    : config_(std::move(config)), impl_(std::make_unique<Impl>(config_)) {}

ClickHouseClient::~ClickHouseClient() = default;

ClickHouseClient::ClickHouseClient(ClickHouseClient&&) noexcept = default;
ClickHouseClient& ClickHouseClient::operator=(ClickHouseClient&&) noexcept = default;

absl::Status ClickHouseClient::Connect() {
    return impl_->Connect();
}

absl::Status ClickHouseClient::Disconnect() {
    return impl_->Disconnect();
}

bool ClickHouseClient::IsConnected() const {
    return impl_->IsConnected();
}

absl::StatusOr<QueryResult> ClickHouseClient::Execute(const std::string& sql) {
    return impl_->Execute(sql);
}

absl::StatusOr<QueryResult> ClickHouseClient::ExecuteWithParams(
    const std::string& sql,
    const std::vector<QueryParam>& params) {
    return impl_->ExecuteWithParams(sql, params);
}

absl::Status ClickHouseClient::InsertTraces(const std::vector<TraceRecord>& traces) {
    return impl_->InsertTraces(traces);
}

absl::Status ClickHouseClient::InsertMetrics(const std::vector<MetricRecord>& metrics) {
    return impl_->InsertMetrics(metrics);
}

absl::Status ClickHouseClient::RunMigrations() {
    spdlog::info("Running ClickHouse migrations");

    // Read and execute migration files
    // For now, we'll just log that migrations would be run
    // In production, this would read from src/storage/schemas/*.sql

    const std::vector<std::string> migrations = {
        // These would be read from files
        "001_traces.sql",
        "002_costs.sql",
        "003_drift_alerts.sql",
        "004_evaluations.sql"
    };

    for (const auto& migration : migrations) {
        spdlog::info("Would run migration: {}", migration);
    }

    return absl::OkStatus();
}

}  // namespace pyflare::storage
