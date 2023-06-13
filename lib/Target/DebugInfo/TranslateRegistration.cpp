//===- TranslateRegistration.cpp - Register translation -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Comb/CombDialect.h"
#include "circt/Dialect/Debug/DebugDialect.h"
#include "circt/Dialect/HW/HWDialect.h"
#include "circt/Dialect/SV/SVDialect.h"
#include "circt/Dialect/Seq/SeqDialect.h"
#include "circt/Target/DebugInfo.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Tools/mlir-translate/Translation.h"

using namespace mlir;

namespace circt {
namespace debug {

void registerHGLDDTranslation() {
  TranslateFromMLIRRegistration reg(
      "emit-hgldd", "emit HGLDD debug information",
      [](ModuleOp op, raw_ostream &output) { return emitHGLDD(op, output); },
      [](DialectRegistry &registry) {
        // clang-format off
        registry.insert<debug::DebugDialect>();
        registry.insert<hw::HWDialect>();
        registry.insert<comb::CombDialect>();
        registry.insert<seq::SeqDialect>();
        registry.insert<sv::SVDialect>();
        // clang-format on
      });
}

void registerTranslations() { registerHGLDDTranslation(); }

} // namespace debug
} // namespace circt
