#pragma once

#include <memory>
#include <stdexcept>
#include <utility>

namespace augmatch {

// Non-throwing status values are used by the view-oriented API. Existing
// pointer APIs retain their historical exception contract.
enum class StatusCode {
  Ok = 0,
  InvalidArgument,
  InvalidView,
  InvalidMetadata,
  Unsupported,
  ExecutionError,
};

// A lightweight, non-throwing outcome: a code plus a static diagnostic
// message. Default-constructed Status is success.
struct Status {
  StatusCode code = StatusCode::Ok;
  const char* message = "ok";

  constexpr bool ok() const noexcept { return code == StatusCode::Ok; }
  constexpr explicit operator bool() const noexcept { return ok(); }
  static constexpr Status success() noexcept { return {StatusCode::Ok, "ok"}; }
};

// A small owning result used by APIs that must return either a value or a
// Status. shared_ptr keeps Result<T> valid when T is incomplete at a method
// declaration (Pipeline::load_from_* returns Result<Pipeline>).
template <typename T>
class Result {
 public:
  static Result success(T value) {
    return Result(std::make_shared<T>(std::move(value)), Status::success());
  }

  static Result failure(Status status) {
    if (status.ok()) status = {StatusCode::ExecutionError, "result has no value"};
    return Result(nullptr, status);
  }

  bool ok() const noexcept { return value_ != nullptr && status_.ok(); }
  explicit operator bool() const noexcept { return ok(); }
  Status status() const noexcept { return status_; }

  T& value() {
    if (!ok()) throw std::logic_error(status_.message);
    return *value_;
  }

  const T& value() const {
    if (!ok()) throw std::logic_error(status_.message);
    return *value_;
  }

 private:
  Result(std::shared_ptr<T> value, Status status)
      : value_(std::move(value)), status_(status) {}

  std::shared_ptr<T> value_;
  Status status_;
};

}  // namespace augmatch
