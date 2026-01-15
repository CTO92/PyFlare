#pragma once

/// @file error.h
/// @brief PyFlare error handling utilities using absl::Status

#include <string>
#include <string_view>

#include <absl/status/status.h>
#include <absl/status/statusor.h>
#include <absl/strings/str_cat.h>

namespace pyflare {

/// @brief Error codes specific to PyFlare
enum class ErrorCode {
    kOk = 0,
    kUnknown,
    kInvalidArgument,
    kNotFound,
    kAlreadyExists,
    kPermissionDenied,
    kResourceExhausted,
    kFailedPrecondition,
    kAborted,
    kOutOfRange,
    kUnimplemented,
    kInternal,
    kUnavailable,
    kDataLoss,

    // PyFlare-specific error codes
    kConnectionFailed,
    kSerializationError,
    kDeserializationError,
    kTimeout,
    kRateLimited,
    kConfigurationError,
    kValidationError,
};

/// @brief Convert PyFlare error code to absl::StatusCode
absl::StatusCode ToAbslCode(ErrorCode code);

/// @brief Create an OK status
inline absl::Status OkStatus() {
    return absl::OkStatus();
}

/// @brief Create an error status with the given code and message
absl::Status MakeError(ErrorCode code, std::string_view message);

/// @brief Create an internal error
inline absl::Status InternalError(std::string_view message) {
    return absl::InternalError(message);
}

/// @brief Create an invalid argument error
inline absl::Status InvalidArgumentError(std::string_view message) {
    return absl::InvalidArgumentError(message);
}

/// @brief Create a not found error
inline absl::Status NotFoundError(std::string_view message) {
    return absl::NotFoundError(message);
}

/// @brief Create an unavailable error
inline absl::Status UnavailableError(std::string_view message) {
    return absl::UnavailableError(message);
}

/// @brief Create a failed precondition error
inline absl::Status FailedPreconditionError(std::string_view message) {
    return absl::FailedPreconditionError(message);
}

// Macros for status checking and propagation

/// @brief Return if status is not OK
#define PYFLARE_RETURN_IF_ERROR(expr)                                          \
    do {                                                                        \
        auto _status = (expr);                                                  \
        if (!_status.ok()) {                                                    \
            return _status;                                                     \
        }                                                                       \
    } while (0)

/// @brief Assign or return if status is not OK
#define PYFLARE_ASSIGN_OR_RETURN(lhs, rhs)                                     \
    PYFLARE_ASSIGN_OR_RETURN_IMPL(                                             \
        PYFLARE_CONCAT(_status_or_, __LINE__), lhs, rhs)

#define PYFLARE_ASSIGN_OR_RETURN_IMPL(statusor, lhs, rhs)                      \
    auto statusor = (rhs);                                                      \
    if (!statusor.ok()) {                                                       \
        return statusor.status();                                               \
    }                                                                           \
    lhs = std::move(statusor).value()

#define PYFLARE_CONCAT(a, b) PYFLARE_CONCAT_IMPL(a, b)
#define PYFLARE_CONCAT_IMPL(a, b) a##b

/// @brief Check condition and return error if false
#define PYFLARE_CHECK_OR_RETURN(condition, error_status)                       \
    do {                                                                        \
        if (!(condition)) {                                                     \
            return (error_status);                                              \
        }                                                                       \
    } while (0)

}  // namespace pyflare
