// RUN: circt-opt %s --verify-diagnostics | circt-opt | FileCheck %s

// CHECK: module

#dbg1 = #dbg.magic<"hello">
#dbg2 = #dbg.variable<name: "a", scope: #dbg1>

#loc1 = loc(fused<#dbg1>[unknown])
#loc2 = loc(fused<#dbg2>[unknown])

unrealized_conversion_cast to i1 loc(#loc1)
unrealized_conversion_cast to i1 loc(#loc2)
