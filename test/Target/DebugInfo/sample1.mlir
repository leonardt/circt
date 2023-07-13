// RUN: circt-translate %s --emit-hgldd

#loc2 = loc("test/Target/DebugInfo/sample1.fir":4:11)
#loc3 = loc("test/Target/DebugInfo/sample1.fir":5:12)
#loc6 = loc("test/Target/DebugInfo/sample1.fir":11:11)
#loc7 = loc("test/Target/DebugInfo/sample1.fir":12:12)
#loc9 = loc("test/Target/DebugInfo/sample1.fir":18:11)
#loc10 = loc("test/Target/DebugInfo/sample1.fir":19:12)
module {
  hw.module private @Bob(%in: i16 loc(#loc2)) -> (out: i16 loc(#loc3)) {
    %c-1_i16 = hw.constant -1 : i16 loc(#loc4)
    %0 = comb.xor bin %in, %c-1_i16 {sv.namehint = "x"} : i16 loc(#loc4)
    hw.output %0 : i16 loc(#loc1)
  } loc(#loc1)
  hw.module private @Bob_1(%in: i16 loc(#loc6)) -> (out: i16 loc(#loc7)) {
    %c-1_i16 = hw.constant -1 : i16 loc(#loc4)
    %0 = comb.xor bin %in, %c-1_i16 {sv.namehint = "x"} : i16 loc(#loc4)
    hw.output %0 : i16 loc(#loc5)
  } loc(#loc5)
  hw.module @Top(%in: i16 loc(#loc9)) -> (out: i16 loc(#loc10)) {
    %b0.out = hw.instance "b0" @Bob(in: %in: i16) -> (out: i16) loc(#loc11)
    %b1.out = hw.instance "b1" @Bob_1(in: %b0.out: i16) -> (out: i16) loc(#loc11)
    hw.output %b1.out : i16 loc(#loc8)
  } loc(#loc8)
} loc(#loc)
#loc = loc("test/Target/DebugInfo/sample1.fir":0:0)
#loc1 = loc("test/Target/DebugInfo/sample1.fir":3:10)
#loc4 = loc("sample1.scala":26:11)
#loc5 = loc("test/Target/DebugInfo/sample1.fir":10:10)
#loc8 = loc("test/Target/DebugInfo/sample1.fir":17:10)
#loc11 = loc("sample1.scala":19:17)
