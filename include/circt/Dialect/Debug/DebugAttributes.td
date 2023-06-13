//===- DebugAttributes.td - Debug information attributes ---*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_DEBUG_DEBUGATTRIBUTES_TD
#define CIRCT_DIALECT_DEBUG_DEBUGATTRIBUTES_TD

include "circt/Dialect/Debug/DebugDialect.td"
include "mlir/IR/AttrTypeBase.td"

// Base class for debug information attributes.
class DebugAttr<string name, string attrMnemonic, list<Trait> traits = []>
    : AttrDef<DebugDialect, name, traits> {
  let cppClassName = name;
  let mnemonic = attrMnemonic;
}

def MagicLoc : DebugAttr<"MagicLoc", "magic"> {
  let parameters = (ins "StringAttr":$name);
  let assemblyFormat = "`<` $name `>`";
}

def Variable : DebugAttr<"DIVariable", "variable"> {
  let parameters = (ins "StringAttr":$name, OptionalParameter<"Attribute">:$scope);
  let assemblyFormat = [{
    `<` `name` `` `:` $name `,` `scope` `` `:` custom<Alias>($scope) `>`
  }];
}

#endif // CIRCT_DIALECT_DEBUG_DEBUGATTRIBUTES_TD
