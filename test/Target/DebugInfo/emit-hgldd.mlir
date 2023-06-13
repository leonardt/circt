// RUN: circt-translate %s --emit-hgldd

#loc2 = loc("test/Target/DebugInfo/DebugDoodle.fir":5:11)
#loc3 = loc("test/Target/DebugInfo/DebugDoodle.fir":6:12)
#loc7 = loc("test/Target/DebugInfo/DebugDoodle.fir":22:11)
#loc8 = loc("test/Target/DebugInfo/DebugDoodle.fir":23:12)
module {
  hw.module @Foo(%a: i32 loc(#loc2)) -> (b: i32 loc(#loc3)) {
    %b0.y = hw.instance "b0" @Bar(x: %a: i32) -> (y: i32) {sv.namehint = "c"} loc(#loc4)
    %b1.y = hw.instance "b1" @Bar(x: %b0.y: i32) -> (y: i32) loc(#loc5)
    hw.output %b1.y : i32 loc(#loc1)
  } loc(#loc1)
  hw.module private @Bar(%x: i32 loc(#loc7)) -> (y: i32 loc(#loc8)) {
    %0 = comb.mul %x, %x : i32 loc(#loc9)
    hw.output %0 : i32 loc(#loc6)
  } loc(#loc6)
} loc(#loc)
#loc = loc("test/Target/DebugInfo/DebugDoodle.fir":0:0)
#loc1 = loc("test/Target/DebugInfo/DebugDoodle.fir":4:10)
#loc4 = loc("test/Target/DebugInfo/DebugDoodle.fir":8:5)
#loc5 = loc("test/Target/DebugInfo/DebugDoodle.fir":14:5)
#loc6 = loc("test/Target/DebugInfo/DebugDoodle.fir":21:10)
#loc9 = loc("test/Target/DebugInfo/DebugDoodle.fir":25:15)
