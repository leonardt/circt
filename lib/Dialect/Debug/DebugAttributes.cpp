//===- DebugAttributes.cpp - Debug information attributes -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Debug/DebugAttributes.h"
#include "circt/Dialect/Debug/DebugDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/Location.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace circt;
using namespace debug;

namespace {
void printAlias(AsmPrinter &odsPrinter, Attribute attr) {
  odsPrinter.printStrippedAttrOrType(attr);
}
ParseResult parseAlias(AsmParser &odsParser, Attribute &result) {
  return odsParser.parseAttribute(result);
}
} // namespace

// Dialect implementation generated from `DebugAttributes.td`
#define GET_ATTRDEF_CLASSES
#include "circt/Dialect/Debug/DebugAttributes.cpp.inc"

void DebugDialect::registerAttributes() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "circt/Dialect/Debug/DebugAttributes.cpp.inc"
      >();
}
