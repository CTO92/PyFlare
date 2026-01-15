#include "error.h"

namespace pyflare {

absl::StatusCode ToAbslCode(ErrorCode code) {
    switch (code) {
        case ErrorCode::kOk:
            return absl::StatusCode::kOk;
        case ErrorCode::kInvalidArgument:
        case ErrorCode::kValidationError:
            return absl::StatusCode::kInvalidArgument;
        case ErrorCode::kNotFound:
            return absl::StatusCode::kNotFound;
        case ErrorCode::kAlreadyExists:
            return absl::StatusCode::kAlreadyExists;
        case ErrorCode::kPermissionDenied:
            return absl::StatusCode::kPermissionDenied;
        case ErrorCode::kResourceExhausted:
        case ErrorCode::kRateLimited:
            return absl::StatusCode::kResourceExhausted;
        case ErrorCode::kFailedPrecondition:
        case ErrorCode::kConfigurationError:
            return absl::StatusCode::kFailedPrecondition;
        case ErrorCode::kAborted:
            return absl::StatusCode::kAborted;
        case ErrorCode::kOutOfRange:
            return absl::StatusCode::kOutOfRange;
        case ErrorCode::kUnimplemented:
            return absl::StatusCode::kUnimplemented;
        case ErrorCode::kInternal:
        case ErrorCode::kSerializationError:
        case ErrorCode::kDeserializationError:
            return absl::StatusCode::kInternal;
        case ErrorCode::kUnavailable:
        case ErrorCode::kConnectionFailed:
            return absl::StatusCode::kUnavailable;
        case ErrorCode::kDataLoss:
            return absl::StatusCode::kDataLoss;
        case ErrorCode::kTimeout:
            return absl::StatusCode::kDeadlineExceeded;
        case ErrorCode::kUnknown:
        default:
            return absl::StatusCode::kUnknown;
    }
}

absl::Status MakeError(ErrorCode code, std::string_view message) {
    return absl::Status(ToAbslCode(code), message);
}

}  // namespace pyflare
