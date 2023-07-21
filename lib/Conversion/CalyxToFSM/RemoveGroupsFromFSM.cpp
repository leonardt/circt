//===- RemoveGroupsFromFSM.cpp - Remove Groups Pass -------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Contains the definitions of the Remove Groups pass.
//
//===----------------------------------------------------------------------===//

#include "../PassDetail.h"
#include "circt/Conversion/CalyxToFSM.h"
#include "circt/Dialect/Calyx/CalyxHelpers.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/FSM/FSMGraph.h"
#include "circt/Support/BackedgeBuilder.h"
#include "circt/Support/LLVM.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OperationSupport.h"
#include "llvm/ADT/STLExtras.h"

using namespace circt;
using namespace calyx;
using namespace mlir;
using namespace fsm;

namespace {

struct CalyxRemoveGroupsFromFSM
    : public CalyxRemoveGroupsFromFSMBase<CalyxRemoveGroupsFromFSM> {
  void runOnOperation() override;

  // Outlines the `fsm.machine` operation from within the `calyx.control`
  // operation to the module scope, and instantiates the FSM. By doing so, we
  // record the association between FSM outputs and group go signals as well as
  // FSM inputs, which are backedges to the group done signals.
  LogicalResult outlineMachine();

  /// Makes several modifications to the operations of a GroupOp:
  /// 1. Assign the 'done' signal of the component with the done_op of the top
  ///    level control group.
  /// 2. Append the 'go' signal of the component to guard of each assignment.
  /// 3. Replace all uses of GroupGoOp with the respective guard, and delete the
  ///    GroupGoOp.
  /// 4. Remove the GroupDoneOp.
  LogicalResult modifyGroupOperations();

  /// Inlines each group in the WiresOp.
  void inlineGroups();

  /// A handle to the machine under transformation.
  MachineOp machineOp;

  // A handle to the component op under transformation.
  ComponentOp componentOp;

  OpBuilder *b;
  BackedgeBuilder *bb;

  // A mapping between group names and their 'go' inputs generated by the FSM.
  DenseMap<StringAttr, Value> groupGoSignals;

  // A mapping between group names and their 'done' output wires sent to
  // the FSM.
  DenseMap<StringAttr, calyx::WireLibOp> groupDoneWires;
};

} // end anonymous namespace

LogicalResult CalyxRemoveGroupsFromFSM::modifyGroupOperations() {
  auto loc = componentOp.getLoc();
  for (auto group : componentOp.getWiresOp().getOps<GroupOp>()) {
    auto groupGo = group.getGoOp();
    if (groupGo)
      return emitError(loc)
             << "This pass does not need `calyx.group_go` operations.";

    auto groupDone = group.getDoneOp();
    if (!groupDone)
      return emitError(loc) << "Group " << group.getSymName()
                            << " does not have a `calyx.group_done` operation";

    // Update group assignments to guard with the group go signal.
    auto fsmGroupGo = groupGoSignals.find(group.getSymNameAttr());
    assert(fsmGroupGo != groupGoSignals.end() &&
           "Could not find FSM go signal for group");

    updateGroupAssignmentGuards(*b, group, fsmGroupGo->second);

    // Create a calyx wire for the group done signal, and assign it to the
    // expression of the group_done operation.
    auto doneWireIt = groupDoneWires.find(group.getSymNameAttr());
    assert(doneWireIt != groupDoneWires.end() &&
           "Could not find FSM done backedge for group");
    auto doneWire = doneWireIt->second;

    b->setInsertionPointToEnd(componentOp.getWiresOp().getBodyBlock());
    b->create<calyx::AssignOp>(loc, doneWire.getIn(), groupDone.getSrc(),
                               groupDone.getGuard());

    groupDone.erase();
  }
  return success();
}

/// Inlines each group in the WiresOp.
void CalyxRemoveGroupsFromFSM::inlineGroups() {
  auto &wiresRegion = componentOp.getWiresOp().getRegion();
  auto &wireBlocks = wiresRegion.getBlocks();
  auto lastBlock = wiresRegion.end();

  // Inline the body of each group as a Block into the WiresOp.
  wiresRegion.walk([&](GroupOp group) {
    wireBlocks.splice(lastBlock, group.getRegion().getBlocks());
    group->erase();
  });

  // Merge the operations of each Block into the first block of the WiresOp.
  auto firstBlock = wireBlocks.begin();
  for (auto it = firstBlock, e = lastBlock; it != e; ++it) {
    if (it == firstBlock)
      continue;
    firstBlock->getOperations().splice(firstBlock->end(), it->getOperations());
  }

  // Erase the (now) empty blocks.
  while (&wiresRegion.front() != &wiresRegion.back())
    wiresRegion.back().erase();
}

LogicalResult CalyxRemoveGroupsFromFSM::outlineMachine() {
  // Walk all operations within the machine and gather the SSA values which are
  // referenced in case they are not defined within the machine.
  // MapVector ensures determinism.
  llvm::MapVector<Value, SmallVector<Operation *>> referencedValues;
  machineOp.walk([&](Operation *op) {
    for (auto &operand : op->getOpOperands()) {
      if (auto barg = operand.get().dyn_cast<BlockArgument>()) {
        if (barg.getOwner()->getParentOp() == machineOp)
          continue;

        // A block argument defined outside of the machineOp.
        referencedValues[operand.get()].push_back(op);
      } else {
        auto *defOp = operand.get().getDefiningOp();
        auto machineOpParent = defOp->getParentOfType<MachineOp>();
        if (machineOpParent && machineOpParent == machineOp)
          continue;

        referencedValues[operand.get()].push_back(op);
      }
    }
  });

  // Add a new input to the machine for each referenced SSA value and replace
  // all uses of the value with the new input.
  DenseMap<Value, size_t> ssaInputIndices;
  auto machineOutputTypes = machineOp.getModuleType().getOutputTypes();
  auto currentInputs = machineOp.getModuleType().getInputTypes();
  llvm::SmallVector<Type> machineInputTypes(currentInputs);

  for (auto &[value, users] : referencedValues) {
    ssaInputIndices[value] = machineOp.getBody().getNumArguments();
    auto t = value.getType();
    auto arg = machineOp.getBody().addArgument(t, b->getUnknownLoc());
    machineInputTypes.push_back(t);
    for (auto *user : users) {
      for (auto &operand : user->getOpOperands()) {
        if (operand.get() == value)
          operand.set(arg);
      }
    }
  }
  // Update the machineOp type.
  machineOp.setType(b->getFunctionType(machineInputTypes, machineOutputTypes));

  // Move the machine to module scope
  machineOp->moveBefore(componentOp);
  size_t nMachineInputs = machineOp.getBody().getNumArguments();

  // Create an fsm.hwinstance in the Calyx component scope with backedges for
  // the group done inputs.
  auto groupDoneInputsAttr =
      machineOp->getAttrOfType<DictionaryAttr>(calyxToFSM::sGroupDoneInputs);
  auto groupGoOutputsAttr =
      machineOp->getAttrOfType<DictionaryAttr>(calyxToFSM::sGroupGoOutputs);
  if (!groupDoneInputsAttr || !groupGoOutputsAttr)
    return emitError(machineOp.getLoc())
           << "MachineOp does not have a " << calyxToFSM::sGroupDoneInputs
           << " or " << calyxToFSM::sGroupGoOutputs
           << " attribute. Was --materialize-calyx-to-fsm run before "
              "this pass?";

  b->setInsertionPointToStart(&componentOp.getBody().front());

  // Maintain a mapping between the FSM input index and the SSA value.
  // We do this to sanity check that all inputs occur in the expected order.
  DenseMap<size_t, Value> fsmInputMap;

  // First we inspect the groupDoneInputsAttr map and create backedges.
  for (auto &namedAttr : groupDoneInputsAttr.getValue()) {
    auto name = namedAttr.getName();
    auto idx = namedAttr.getValue().cast<IntegerAttr>();
    auto inputIdx = idx.cast<IntegerAttr>().getInt();
    if (fsmInputMap.count(inputIdx))
      return emitError(machineOp.getLoc())
             << "MachineOp has duplicate input index " << idx;

    // Create a wire for the group done input.
    b->setInsertionPointToStart(&componentOp.getBody().front());
    auto groupDoneWire = b->create<calyx::WireLibOp>(
        componentOp.getLoc(), name.str() + "_done", b->getI1Type());
    fsmInputMap[inputIdx] = groupDoneWire.getOut();
    groupDoneWires[name] = groupDoneWire;
  }

  // Then we inspect the top level go/done attributes.
  auto topLevelGoAttr =
      machineOp->getAttrOfType<IntegerAttr>(calyxToFSM::sFSMTopLevelGoIndex);
  if (!topLevelGoAttr)
    return emitError(machineOp.getLoc())
           << "MachineOp does not have a " << calyxToFSM::sFSMTopLevelGoIndex
           << " attribute.";
  fsmInputMap[topLevelGoAttr.getInt()] = componentOp.getGoPort();

  auto topLevelDoneAttr =
      machineOp->getAttrOfType<IntegerAttr>(calyxToFSM::sFSMTopLevelDoneIndex);
  if (!topLevelDoneAttr)
    return emitError(machineOp.getLoc())
           << "MachineOp does not have a " << calyxToFSM::sFSMTopLevelDoneIndex
           << " attribute.";

  // Then we inspect the external SSA values.
  for (auto [value, idx] : ssaInputIndices) {
    if (fsmInputMap.count(idx))
      return emitError(machineOp.getLoc())
             << "MachineOp has duplicate input index " << idx;
    fsmInputMap[idx] = value;
  }

  if (fsmInputMap.size() != nMachineInputs)
    return emitError(machineOp.getLoc())
           << "MachineOp has " << nMachineInputs
           << " inputs, but only recorded " << fsmInputMap.size()
           << " inputs. This either means that --materialize-calyx-to-fsm "
              "failed or that there is a mismatch in the MachineOp attributes.";

  // Convert the fsmInputMap to a list.
  llvm::SmallVector<Value> fsmInputs;
  for (size_t idx = 0; idx < nMachineInputs; ++idx) {
    auto it = fsmInputMap.find(idx);
    assert(it != fsmInputMap.end() && "Missing FSM input index");
    fsmInputs.push_back(it->second);
  }

  // Instantiate the FSM.
  auto fsmInstance = b->create<fsm::HWInstanceOp>(
      machineOp.getLoc(), machineOutputTypes, b->getStringAttr("controller"),
      machineOp.getSymNameAttr(), fsmInputs, componentOp.getClkPort(),
      componentOp.getResetPort());

  // Record the FSM output group go signals.
  for (auto namedAttr : groupGoOutputsAttr.getValue()) {
    auto name = namedAttr.getName();
    auto idx = namedAttr.getValue().cast<IntegerAttr>().getInt();
    groupGoSignals[name] = fsmInstance.getResult(idx);
  }

  // Assign FSM top level done to the component done.
  b->setInsertionPointToEnd(componentOp.getWiresOp().getBodyBlock());
  b->create<calyx::AssignOp>(machineOp.getLoc(), componentOp.getDonePort(),
                             fsmInstance.getResult(topLevelDoneAttr.getInt()));

  return success();
}

void CalyxRemoveGroupsFromFSM::runOnOperation() {
  componentOp = getOperation();
  auto *ctx = componentOp.getContext();
  auto builder = OpBuilder(ctx);
  builder.setInsertionPointToStart(&componentOp.getBody().front());
  auto backedgeBuilder = BackedgeBuilder(builder, componentOp.getLoc());
  b = &builder;
  bb = &backedgeBuilder;

  // Locate the FSM machine in the control op..
  auto machineOps = componentOp.getControlOp().getOps<fsm::MachineOp>();
  if (std::distance(machineOps.begin(), machineOps.end()) != 1) {
    emitError(componentOp.getLoc())
        << "Expected exactly one fsm.MachineOp in the control op";
    signalPassFailure();
    return;
  }
  machineOp = *machineOps.begin();

  if (failed(outlineMachine()) || failed(modifyGroupOperations())) {
    signalPassFailure();
    return;
  }

  inlineGroups();
}

std::unique_ptr<mlir::Pass> circt::createRemoveGroupsFromFSMPass() {
  return std::make_unique<CalyxRemoveGroupsFromFSM>();
}
