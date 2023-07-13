; RUN: true
circuit Top :
  module Bob :
    input in : UInt<16>
    output out : UInt<16>

    node x = not(in) @[sample1.scala 26:11]
    out <= x @[sample1.scala 15:45]

  module Bob_1 :
    input in : UInt<16>
    output out : UInt<16>

    node x = not(in) @[sample1.scala 26:11]
    out <= x @[sample1.scala 15:45]

  module Top :
    input in : UInt<16>
    output out : UInt<16>

    inst b0 of Bob @[sample1.scala 19:17]
    b0.in <= in @[sample1.scala 11:12]
    inst b1 of Bob_1 @[sample1.scala 19:17]
    b1.in <= b0.out @[sample1.scala 11:12]
    out <= b1.out @[sample1.scala 14:7]
