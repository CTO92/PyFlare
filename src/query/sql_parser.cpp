/// @file sql_parser.cpp
/// @brief Secure SQL parser for PyFlare query language
///
/// SECURITY: This parser implements an allowlist-based approach to prevent
/// SQL injection attacks. Only known-safe query patterns are permitted.

#include <algorithm>
#include <cctype>
#include <regex>
#include <string>
#include <unordered_set>
#include <vector>

#include <absl/status/statusor.h>
#include <absl/strings/str_cat.h>
#include <absl/strings/ascii.h>

namespace pyflare::query {

/// @brief Allowed tables for queries (allowlist approach)
static const std::unordered_set<std::string> kAllowedTables = {
    "traces",
    "spans",
    "metrics",
    "logs",
    "models",
    "services",
    "cost_summary",
    "token_usage",
};

/// @brief Allowed aggregate functions
static const std::unordered_set<std::string> kAllowedFunctions = {
    "count", "sum", "avg", "min", "max", "uniq", "uniqexact",
    "quantile", "quantiles", "median", "stddev", "variance",
    "grouparray", "groupuniqarray", "topk",
    "tostartofminute", "tostartofhour", "tostartofday",
    "todate", "todatetime", "tostring", "toint64", "tofloat64",
    "lower", "upper", "length", "substring", "trim",
    "if", "multiif", "coalesce", "nullif",
    "any", "anylast", "argmax", "argmin",
};

/// @brief Parsed query representation
struct ParsedQuery {
    enum class Type {
        kSelect,
        kAggregate,
        kTimeSeries,
        kTopK,
        kDistinct
    };

    Type type = Type::kSelect;
    std::string table;
    std::vector<std::string> columns;
    std::string where_clause;
    std::string group_by;
    std::string order_by;
    size_t limit = 0;
    size_t offset = 0;
    bool is_valid = false;
    std::string error_message;
};

/// @brief Secure SQL parser with allowlist-based validation
class SqlParser {
public:
    /// @brief Maximum query length
    static constexpr size_t kMaxQueryLength = 65536;  // 64 KB

    /// @brief Maximum nesting depth for parentheses
    static constexpr int kMaxNestingDepth = 20;

    /// @brief Parse and validate SQL string
    absl::StatusOr<ParsedQuery> Parse(const std::string& sql) {
        ParsedQuery query;

        // Basic validation
        if (sql.empty()) {
            return absl::InvalidArgumentError("Empty query");
        }

        if (sql.size() > kMaxQueryLength) {
            return absl::InvalidArgumentError(
                absl::StrCat("Query exceeds maximum length of ", kMaxQueryLength));
        }

        // Check for dangerous patterns FIRST (before any parsing)
        auto danger_check = CheckDangerousPatterns(sql);
        if (!danger_check.ok()) {
            return danger_check;
        }

        // Normalize whitespace
        std::string normalized = NormalizeWhitespace(sql);

        // Must start with SELECT (only SELECT queries allowed)
        std::string upper_sql = ToUpper(normalized);
        if (upper_sql.size() < 6 || upper_sql.substr(0, 6) != "SELECT") {
            return absl::PermissionDeniedError(
                "Only SELECT queries are allowed");
        }

        // Extract and validate table name
        auto table_result = ExtractTableName(normalized);
        if (!table_result.ok()) {
            return table_result.status();
        }
        query.table = *table_result;

        // Validate table is in allowlist
        if (kAllowedTables.find(query.table) == kAllowedTables.end()) {
            return absl::PermissionDeniedError(
                absl::StrCat("Table '", query.table, "' is not allowed"));
        }

        // Validate functions used in query
        auto func_check = ValidateFunctions(normalized);
        if (!func_check.ok()) {
            return func_check;
        }

        // Validate identifiers (column names, etc.)
        auto ident_check = ValidateIdentifiers(normalized);
        if (!ident_check.ok()) {
            return ident_check;
        }

        query.is_valid = true;
        query.type = DetectQueryType(upper_sql);

        return query;
    }

    /// @brief Validate that query only accesses allowed tables
    absl::Status ValidateTables(
        const ParsedQuery& query,
        const std::vector<std::string>& allowed_tables) {

        std::unordered_set<std::string> allowed_set(
            allowed_tables.begin(), allowed_tables.end());

        if (allowed_set.find(query.table) == allowed_set.end()) {
            return absl::PermissionDeniedError(
                absl::StrCat("Access to table '", query.table, "' is not permitted"));
        }

        return absl::OkStatus();
    }

    /// @brief Sanitize a string value for logging only
    /// IMPORTANT: Always use parameterized queries - never concatenate user input
    static std::string SanitizeForLogging(const std::string& value) {
        std::string result;
        result.reserve(value.size());

        for (char c : value) {
            switch (c) {
                case '\'': result += "\\'"; break;
                case '\\': result += "\\\\"; break;
                case '\0': result += "\\0"; break;
                case '\n': result += "\\n"; break;
                case '\r': result += "\\r"; break;
                case '\t': result += "\\t"; break;
                default:
                    if (std::isprint(static_cast<unsigned char>(c))) {
                        result += c;
                    } else {
                        result += '?';
                    }
            }
        }
        return result;
    }

private:
    /// @brief Convert string to uppercase
    static std::string ToUpper(const std::string& s) {
        std::string result = s;
        std::transform(result.begin(), result.end(), result.begin(),
                      [](unsigned char c) { return std::toupper(c); });
        return result;
    }

    /// @brief Convert string to lowercase
    static std::string ToLower(const std::string& s) {
        std::string result = s;
        std::transform(result.begin(), result.end(), result.begin(),
                      [](unsigned char c) { return std::tolower(c); });
        return result;
    }

    /// @brief Check for dangerous SQL patterns
    absl::Status CheckDangerousPatterns(const std::string& sql) {
        std::string lower_sql = ToLower(sql);

        // List of dangerous keywords (must check with word boundaries)
        static const std::vector<std::string> dangerous_keywords = {
            "drop", "delete", "truncate", "alter", "create",
            "insert", "update", "replace", "attach", "detach",
            "rename", "exchange", "grant", "revoke", "system",
            "kill", "optimize", "mutation"
        };

        for (const auto& keyword : dangerous_keywords) {
            // Check for keyword with word boundaries
            size_t pos = 0;
            while ((pos = lower_sql.find(keyword, pos)) != std::string::npos) {
                // Check word boundary before
                bool valid_before = (pos == 0) ||
                    !std::isalnum(static_cast<unsigned char>(lower_sql[pos - 1]));

                // Check word boundary after
                size_t end_pos = pos + keyword.size();
                bool valid_after = (end_pos >= lower_sql.size()) ||
                    !std::isalnum(static_cast<unsigned char>(lower_sql[end_pos]));

                if (valid_before && valid_after) {
                    return absl::PermissionDeniedError(
                        "Query contains forbidden keyword");
                }
                pos++;
            }
        }

        // Check for multiple statements (semicolons followed by more content)
        size_t semicolon_pos = sql.find(';');
        if (semicolon_pos != std::string::npos) {
            // Check if there's meaningful content after semicolon
            for (size_t i = semicolon_pos + 1; i < sql.size(); ++i) {
                if (!std::isspace(static_cast<unsigned char>(sql[i]))) {
                    return absl::PermissionDeniedError(
                        "Multiple statements are not allowed");
                }
            }
        }

        // Check for SQL comments
        if (sql.find("--") != std::string::npos) {
            return absl::PermissionDeniedError(
                "SQL comments (--) are not allowed");
        }

        if (sql.find("/*") != std::string::npos) {
            return absl::PermissionDeniedError(
                "Block comments (/*) are not allowed");
        }

        // Check for system variable access
        if (sql.find("@@") != std::string::npos) {
            return absl::PermissionDeniedError(
                "System variable access is not allowed");
        }

        // Check for file operations
        if (lower_sql.find("into outfile") != std::string::npos ||
            lower_sql.find("into dumpfile") != std::string::npos ||
            lower_sql.find("load_file") != std::string::npos) {
            return absl::PermissionDeniedError(
                "File operations are not allowed");
        }

        return absl::OkStatus();
    }

    /// @brief Normalize whitespace in SQL
    std::string NormalizeWhitespace(const std::string& sql) {
        std::string result;
        result.reserve(sql.size());

        bool in_string = false;
        char string_char = '\0';
        bool last_was_space = true;

        for (size_t i = 0; i < sql.size(); ++i) {
            char c = sql[i];

            // Track string literals
            if (!in_string && (c == '\'' || c == '"')) {
                in_string = true;
                string_char = c;
                result += c;
                last_was_space = false;
            } else if (in_string && c == string_char) {
                // Check for escaped quote
                if (i + 1 < sql.size() && sql[i + 1] == string_char) {
                    result += c;
                    result += sql[++i];
                } else {
                    in_string = false;
                    result += c;
                }
                last_was_space = false;
            } else if (in_string) {
                result += c;
                last_was_space = false;
            } else if (std::isspace(static_cast<unsigned char>(c))) {
                if (!last_was_space) {
                    result += ' ';
                    last_was_space = true;
                }
            } else {
                result += c;
                last_was_space = false;
            }
        }

        // Trim trailing space
        if (!result.empty() && result.back() == ' ') {
            result.pop_back();
        }

        return result;
    }

    /// @brief Extract table name from SQL
    absl::StatusOr<std::string> ExtractTableName(const std::string& sql) {
        std::string upper_sql = ToUpper(sql);

        // Find FROM keyword
        size_t from_pos = upper_sql.find(" FROM ");
        if (from_pos == std::string::npos) {
            from_pos = upper_sql.find("\tFROM ");
        }
        if (from_pos == std::string::npos) {
            return absl::InvalidArgumentError("Could not find FROM clause");
        }

        // Skip "FROM " and any whitespace
        size_t table_start = from_pos + 6;  // " FROM " length
        while (table_start < sql.size() &&
               std::isspace(static_cast<unsigned char>(sql[table_start]))) {
            table_start++;
        }

        // Extract table name (alphanumeric + underscore)
        size_t table_end = table_start;
        while (table_end < sql.size()) {
            char c = sql[table_end];
            if (std::isalnum(static_cast<unsigned char>(c)) || c == '_') {
                table_end++;
            } else {
                break;
            }
        }

        if (table_end == table_start) {
            return absl::InvalidArgumentError("Could not extract table name");
        }

        std::string table = sql.substr(table_start, table_end - table_start);
        return ToLower(table);
    }

    /// @brief Validate all functions in query are allowed
    absl::Status ValidateFunctions(const std::string& sql) {
        // Find function calls (identifier followed by parenthesis)
        for (size_t i = 0; i < sql.size(); ++i) {
            if (sql[i] == '(') {
                // Look back for function name
                size_t end = i;
                while (end > 0 && std::isspace(static_cast<unsigned char>(sql[end - 1]))) {
                    end--;
                }

                size_t start = end;
                while (start > 0) {
                    char c = sql[start - 1];
                    if (std::isalnum(static_cast<unsigned char>(c)) || c == '_') {
                        start--;
                    } else {
                        break;
                    }
                }

                if (start < end) {
                    std::string func_name = ToLower(sql.substr(start, end - start));

                    // Skip SQL keywords that look like functions
                    static const std::unordered_set<std::string> sql_keywords = {
                        "select", "from", "where", "and", "or", "not", "in",
                        "between", "like", "is", "null", "as", "case", "when",
                        "then", "else", "end", "having", "order", "group", "by",
                        "limit", "offset", "join", "left", "right", "inner", "outer"
                    };

                    if (sql_keywords.find(func_name) != sql_keywords.end()) {
                        continue;
                    }

                    if (kAllowedFunctions.find(func_name) == kAllowedFunctions.end()) {
                        return absl::PermissionDeniedError(
                            absl::StrCat("Function '", func_name, "' is not allowed"));
                    }
                }
            }
        }

        return absl::OkStatus();
    }

    /// @brief Validate identifiers don't contain injection attempts
    absl::Status ValidateIdentifiers(const std::string& sql) {
        // Check for suspicious characters
        for (char c : sql) {
            if (c == '`' || c == '[' || c == ']' ||
                c == '{' || c == '}' || c == '|' || c == '\\') {
                return absl::PermissionDeniedError(
                    "Query contains suspicious characters");
            }
        }

        // Check parentheses balance and depth
        int paren_depth = 0;
        int max_depth = 0;
        for (char c : sql) {
            if (c == '(') {
                paren_depth++;
                max_depth = std::max(max_depth, paren_depth);
            } else if (c == ')') {
                paren_depth--;
            }

            if (paren_depth < 0) {
                return absl::InvalidArgumentError("Unbalanced parentheses");
            }
        }

        if (paren_depth != 0) {
            return absl::InvalidArgumentError("Unbalanced parentheses");
        }

        if (max_depth > kMaxNestingDepth) {
            return absl::PermissionDeniedError(
                "Query has excessive nesting depth");
        }

        return absl::OkStatus();
    }

    /// @brief Detect the type of query
    ParsedQuery::Type DetectQueryType(const std::string& upper_sql) {
        if (upper_sql.find("GROUP BY") != std::string::npos) {
            return ParsedQuery::Type::kAggregate;
        }
        if (upper_sql.find("DISTINCT") != std::string::npos) {
            return ParsedQuery::Type::kDistinct;
        }
        if (upper_sql.find("TOSTART") != std::string::npos) {
            return ParsedQuery::Type::kTimeSeries;
        }
        if (upper_sql.find("TOPK") != std::string::npos) {
            return ParsedQuery::Type::kTopK;
        }

        return ParsedQuery::Type::kSelect;
    }
};

/// @brief Global parser instance
static SqlParser g_parser;

/// @brief Validate a SQL query (exported function)
absl::StatusOr<ParsedQuery> ValidateQuery(const std::string& sql) {
    return g_parser.Parse(sql);
}

/// @brief Check if a query is safe to execute (exported function)
absl::Status IsQuerySafe(const std::string& sql) {
    auto result = g_parser.Parse(sql);
    if (!result.ok()) {
        return result.status();
    }
    return absl::OkStatus();
}

}  // namespace pyflare::query
