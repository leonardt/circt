//===- EmitHGLDD.cpp - HGLDD debug info emission --------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Debug/DebugAttributes.h"
#include "circt/Dialect/Debug/DebugDialect.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Target/DebugInfo.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/JSON.h"

#define DEBUG_TYPE "di"

using namespace mlir;
using namespace circt;
using llvm::SmallMapVector;

struct DIInstance;

struct DIVariable {
  StringAttr name;
  LocationAttr loc;
  /// The SSA value representing the value of this variable.
  Value value = nullptr;
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

  SmallVector<DIVariable *, 0> variables;
  SmallVector<DIInstance *, 0> children;
};

struct DIInstance {
  /// The operation that generated this instance.
  Operation *op = nullptr;
  /// The name of this instance.
  StringAttr name;
  /// The instantiated module.
  DIHierarchy *hierarchy;
};

struct DebugInfo {
  DebugInfo(Operation *op);

  SmallMapVector<StringAttr, DIHierarchy *, 8> moduleNodes;

  DIHierarchy &getOrCreateHierarchyForModule(StringAttr moduleName) {
    auto &slot = moduleNodes[moduleName];
    if (!slot) {
      slot = new (hierarchyAllocator.Allocate()) DIHierarchy;
      slot->name = moduleName;
      LLVM_DEBUG(llvm::dbgs() << "- Created hierarchy " << moduleName << "\n");
    }
    return *slot;
  }

  llvm::SpecificBumpPtrAllocator<DIHierarchy> hierarchyAllocator;
  llvm::SpecificBumpPtrAllocator<DIInstance> instanceAllocator;
  llvm::SpecificBumpPtrAllocator<DIVariable> variableAllocator;
};

DebugInfo::DebugInfo(Operation *op) {
  // TODO: Traverse the op, collect all the debug info attributes, and combine
  // them into a different debug info data structure.
  LLVM_DEBUG(llvm::dbgs() << "Should collect DI now\n");

  op->walk([&](Operation *op) {
    LLVM_DEBUG(llvm::dbgs() << "- Visiting " << op->getName() << " at "
                            << op->getLoc() << "\n");
    if (auto moduleOp = dyn_cast<hw::HWModuleOp>(op)) {
      auto &hierarchy = getOrCreateHierarchyForModule(moduleOp.getNameAttr());
      hierarchy.op = op;

      // Add variables for each of the ports.
      auto outputValues =
          moduleOp.getBodyBlock()->getTerminator()->getOperands();
      for (auto &port : moduleOp.getPortList()) {
        auto value = port.isOutput() ? outputValues[port.argNum]
                                     : moduleOp.getArgument(port.argNum);
        auto *var = new (variableAllocator.Allocate()) DIVariable;
        var->name = port.name;
        var->loc = port.loc;
        var->value = value;
        hierarchy.variables.push_back(var);
      }
    } else if (auto instOp = dyn_cast<hw::InstanceOp>(op)) {
      auto parentModule = op->getParentOfType<hw::HWModuleOp>();
      if (!parentModule)
        return;
      auto &parentHierarchy =
          getOrCreateHierarchyForModule(parentModule.getNameAttr());
      auto &childHierarchy =
          getOrCreateHierarchyForModule(instOp.getModuleNameAttr().getAttr());
      auto *instance = new (instanceAllocator.Allocate()) DIInstance;
      instance->name = instOp.getInstanceNameAttr();
      instance->op = instOp;
      instance->hierarchy = &childHierarchy;
      parentHierarchy.children.push_back(instance);

      // TODO: What do we do with the port assignments? These should be tracked
      // somewhere.
    } else if (auto wireOp = dyn_cast<hw::WireOp>(op)) {
      auto parentModule = op->getParentOfType<hw::HWModuleOp>();
      if (!parentModule)
        return;
      auto *var = new (variableAllocator.Allocate()) DIVariable;
      var->name = wireOp.getNameAttr();
      var->loc = wireOp.getLoc();
      var->value = wireOp;
      getOrCreateHierarchyForModule(parentModule.getNameAttr())
          .variables.push_back(var);
    }
  });
}

static void encodeLoc(llvm::json::OStream &json, FileLineColLoc loc) {
  // TODO: Collect this into a `file_info` structure up front.
  json.attribute("file", loc.getFilename().getValue());
  if (auto line = loc.getLine()) {
    json.attribute("begin_line", line);
    json.attribute("end_line", line);
  }
  if (auto col = loc.getColumn()) {
    json.attribute("begin_column", col);
    json.attribute("end_column", col);
  }
}

static FileLineColLoc findBestLocation(Location loc, bool emitted) {
  if (auto nameLoc = dyn_cast<NameLoc>(loc)) {
    bool nameIsEmitted = nameLoc.getName() == "emitted";
    if (emitted == nameIsEmitted)
      return findBestLocation(nameLoc.getChildLoc(), false);
  } else if (auto fusedLoc = dyn_cast<FusedLoc>(loc)) {
    for (auto innerLoc : fusedLoc.getLocations())
      if (auto bestInnerLoc = findBestLocation(innerLoc, emitted))
        return bestInnerLoc;
  } else if (auto fileLoc = dyn_cast<FileLineColLoc>(loc)) {
    if (!emitted)
      return fileLoc;
  }
  return {};
}

static void findAndEncodeLoc(llvm::json::OStream &json, StringRef fieldName,
                             Location loc, bool emitted) {
  if (auto fileLoc = findBestLocation(loc, emitted))
    json.attributeObject(fieldName, [&] { encodeLoc(json, fileLoc); });
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

  json.attributeArray("objects", [&] {
    for (auto moduleNodeEntry : di.moduleNodes) {
      StringAttr moduleNameAttr;
      DIHierarchy *hierarchy;
      std::tie(moduleNameAttr, hierarchy) = moduleNodeEntry;
      json.objectBegin();
      json.attribute("kind", "module");
      json.attribute("obj_name", moduleNameAttr.getValue());
      // TODO: This should probably me `sv.verilogName`.
      json.attribute("module_name", moduleNameAttr.getValue());
      if (auto *op = hierarchy->op) {
        findAndEncodeLoc(json, "hgl_loc", op->getLoc(), false);
        findAndEncodeLoc(json, "hdl_loc", op->getLoc(), true);
      }
      json.attributeArray("port_vars", [&] {
        for (auto *var : hierarchy->variables) {
          json.objectBegin();
          json.attribute("var_name", var->name.getValue());
          findAndEncodeLoc(json, "hgl_loc", var->loc, false);
          findAndEncodeLoc(json, "hdl_loc", var->loc, true);
          if (auto value = var->value) {
            StringAttr portName;
            auto *defOp = value.getParentBlock()->getParentOp();
            auto module = dyn_cast<hw::HWModuleOp>(defOp);
            if (!module)
              module = defOp->getParentOfType<hw::HWModuleOp>();
            if (module) {
              if (auto arg = dyn_cast<BlockArgument>(value)) {
                portName = dyn_cast_or_null<StringAttr>(
                    module.getArgNames()[arg.getArgNumber()]);
              } else if (auto wireOp = value.getDefiningOp<hw::WireOp>()) {
                portName = wireOp.getNameAttr();
              } else {
                for (auto &use : value.getUses()) {
                  if (auto outputOp = dyn_cast<hw::OutputOp>(use.getOwner())) {
                    portName = dyn_cast_or_null<StringAttr>(
                        module.getResultNames()[use.getOperandNumber()]);
                    break;
                  }
                }
              }
            }
            if (auto intType = dyn_cast<IntegerType>(value.getType())) {
              json.attribute("type_name", "logic");
              if (intType.getIntOrFloatBitWidth() != 1) {
                json.attributeArray("packed_range", [&] {
                  json.value(intType.getIntOrFloatBitWidth() - 1);
                  json.value(0);
                });
              }
            }
            if (portName) {
              json.attributeObject("value", [&] {
                json.attribute("sig_name", portName.getValue());
              });
            }
          }
          json.objectEnd();
        }
      });
      json.attributeArray("children", [&] {
        for (auto *instance : hierarchy->children) {
          json.objectBegin();
          json.attribute("module_name", instance->hierarchy->name.getValue());
          json.attribute("inst_name", instance->name.getValue());
          if (auto *op = instance->op) {
            findAndEncodeLoc(json, "hgl_loc", op->getLoc(), false);
            findAndEncodeLoc(json, "hdl_loc", op->getLoc(), true);
          }
          json.objectEnd();
        }
      });
      json.objectEnd();
    }
  });

  json.objectEnd();

  return success();
}
