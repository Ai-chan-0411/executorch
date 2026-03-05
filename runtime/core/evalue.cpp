/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/runtime/core/evalue.h>

namespace executorch {
namespace runtime {
template <>
executorch::aten::ArrayRef<std::optional<executorch::aten::Tensor>>
BoxedEvalueList<std::optional<executorch::aten::Tensor>>::get() const {
  for (typename executorch::aten::ArrayRef<
           std::optional<executorch::aten::Tensor>>::size_type i = 0;
       i < wrapped_vals_.size();
       i++) {
    if (wrapped_vals_[i] == nullptr) {
      unwrapped_vals_[i] = executorch::aten::nullopt;
    } else {
      // Validate type at access time. MoveCall instructions can change the
      // type of values_ entries after initial validation (TOCTOU).
      ET_CHECK_MSG(
          wrapped_vals_[i]->isNone() || wrapped_vals_[i]->isTensor(),
          "BoxedEvalueList<optional<Tensor>> element %zu has wrong type "
          "(tag %u). Value may have been overwritten by MoveCall.",
          (size_t)i,
          static_cast<unsigned>(wrapped_vals_[i]->tag));
      unwrapped_vals_[i] =
          wrapped_vals_[i]->to<std::optional<executorch::aten::Tensor>>();
    }
  }
  return executorch::aten::ArrayRef<std::optional<executorch::aten::Tensor>>{
      unwrapped_vals_, wrapped_vals_.size()};
}
} // namespace runtime
} // namespace executorch
