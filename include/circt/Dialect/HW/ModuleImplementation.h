//===- ModuleImplementation.h - Module-like Op utilities --------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file provides utility functions for implementing module-like
// operations, in particular, parsing, and printing common to module-like
// operations.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_HW_MODULEIMPLEMENTATION_H
#define CIRCT_DIALECT_HW_MODULEIMPLEMENTATION_H

#include "circt/Dialect/HW/HWTypes.h"
#include "circt/Support/LLVM.h"
#include "mlir/IR/DialectImplementation.h"

namespace circt {
namespace hw {

namespace module_like_impl {
/// Get the portname from an SSA value string, if said value name is not a
/// number.
StringAttr getPortNameAttr(MLIRContext *context, StringRef name);

/// This is a variant of mlir::parseFunctionSignature that allows names on
/// result arguments.
ParseResult parseModuleFunctionSignature(
    OpAsmParser &parser, 
    SmallVectorImpl<OpAsmParser::Argument> &args,
    SmallVectorImpl<Attribute> &argLocs,
    SmallVectorImpl<DictionaryAttr> &resultAttrs,
    SmallVectorImpl<Attribute> &resultLocs, TypeAttr &type);

/// Print a module signature with named results.
void printModuleSignature(OpAsmPrinter &p, Operation *op, ModuleType type, bool &needArgNamesAttr);

} // namespace module_like_impl
} // namespace hw
} // namespace circt

#endif // CIRCT_DIALECT_HW_MODULEIMPLEMENTATION_H
