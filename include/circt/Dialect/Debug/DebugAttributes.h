//===- DebugAttributes.h - Debug information attributes ---------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_DEBUG_DEBUGATTRIBUTES_H
#define CIRCT_DIALECT_DEBUG_DEBUGATTRIBUTES_H

#include "circt/Support/LLVM.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Location.h"

// Location attributes generated from `DebugAttributes.td`
#define GET_ATTRDEF_CLASSES
#include "circt/Dialect/Debug/DebugAttributes.h.inc"

#endif // CIRCT_DIALECT_DEBUG_DEBUGATTRIBUTES_H
