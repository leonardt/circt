// RUN: circt-translate %s --emit-hgldd

#loc = loc("test/Target/DebugInfo/DebugDoodle.scala":0:0)
#loc1 = loc("test/Target/DebugInfo/DebugDoodle.scala":4:10)
#loc2 = loc("test/Target/DebugInfo/DebugDoodle.scala":5:11)
#loc3 = loc("test/Target/DebugInfo/DebugDoodle.scala":6:12)
#loc4 = loc("test/Target/DebugInfo/DebugDoodle.scala":8:5)
#loc5 = loc("test/Target/DebugInfo/DebugDoodle.scala":14:5)
#loc6 = loc("test/Target/DebugInfo/DebugDoodle.scala":21:10)
#loc7 = loc("test/Target/DebugInfo/DebugDoodle.scala":22:11)
#loc8 = loc("test/Target/DebugInfo/DebugDoodle.scala":23:12)
#loc9 = loc("test/Target/DebugInfo/DebugDoodle.scala":25:15)
#loc10 = loc("DebugDoodle.sv":42:10)
#loc11 = loc("DebugDoodle.sv":49:10)
module {
  hw.module @Foo(%a: i32 loc(#loc2)) -> (b: i32 loc(#loc3)) {
    %b0.y = hw.instance "b0" @Bar(x: %a: i32) -> (y: i32) {sv.namehint = "c"} loc(#loc4)
    %b1.y = hw.instance "b1" @Bar(x: %b0.y: i32) -> (y: i32) loc(#loc5)
    hw.output %b1.y : i32 loc(#loc1)
  } loc(fused[#loc1, "emitted"(#loc10)])
  hw.module private @Bar(%x: i32 loc(#loc7)) -> (y: i32 loc(#loc8)) {
    %0 = comb.mul %x, %x : i32 loc(#loc9)
    hw.output %0 : i32 loc(#loc6)
  } loc(fused[#loc6, "emitted"(#loc11)])
} loc(#loc)
