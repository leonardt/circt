//===- EmitHGLDD.cpp - HGLDD debug info emission --------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Debug/DebugAttributes.h"
#include "circt/Dialect/Debug/DebugDialect.h"
#include "circt/Target/DebugInfo.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/JSON.h"

#define DEBUG_TYPE "di"

using namespace circt;

struct DIVariable {
  StringAttr name;
  // TODO: type
  // TODO: source_loc
  // TODO: output_loc
  // TODO: value expr
};

struct DIHierarchy {
  /// The operation that generated this level of hierarchy.
  Operation *op = nullptr;
  /// The name of this level of hierarchy.
  StringAttr name;

  SmallVector<DIVariable, 0> variables;
  SmallVector<DIHierarchy *, 0> children;
};

struct DebugInfo {
  DebugInfo(Operation *op);

  SmallDenseMap<StringAttr, DIHierarchy *> moduleNodes;

  DIHierarchy &getOrCreateHierarchyForModule(StringAttr moduleName) {
    auto &slot = moduleNodes[moduleName];
    if (!slot) {
      slot = new (hierarchyAllocator.Allocate());
      slot->name = moduleName;
    }
    return *slot;
  }

  llvm::SpecificBumpPtrAllocator<DIHierarchy> hierarchyAllocator;
};

DebugInfo::DebugInfo(Operation *op) {
  // TODO: Traverse the op, collect all the debug info attributes, and combine
  // them into a different debug info data structure.
  LLVM_DEBUG(llvm::dbgs() << "Should collect DI now\n");

  op->walk([&](Operation *op) {
    LLVM_DEBUG(llvm::dbgs() << "- Visiting " << op->getName() << " at "
                            << op->getLoc() << "\n");
    if (auto moduleOp = dyn_cast<HWModuleOp>(op)) {
      auto &hierarchy =
          getOrCreateHierarchyForModule(moduleOp.getSymNameAttr());
      hierarchy.op = op;
    } else if (auto instOp = dyn_cast<InstanceOp>(op)) {
      auto &subhierarchy =
          getOrCreateHierarchyForModule(instOp.getModuleNameAttr());
    }
  });
}

LogicalResult debug::emitHGLDD(ModuleOp module, llvm::raw_ostream &os) {
  DebugInfo di(module);
  llvm::json::OStream json(os, 2);
  json.objectBegin();

  json.attributeObject("HGLDD", [&] {
    json.attribute("version", "1.0");
    json.attributeArray("file_info", [&] {});
    json.attribute("hdl_file_index", 42);
  });

  json.attributeObject("objects", [&] {});

  json.objectEnd();

  return success();
}
