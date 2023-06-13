add_circt_dialect(Debug dbg)
add_circt_dialect_doc(Debug dbg)

set(LLVM_TARGET_DEFINITIONS DebugAttributes.td)
mlir_tablegen(DebugAttributes.h.inc -gen-attrdef-decls)
mlir_tablegen(DebugAttributes.cpp.inc -gen-attrdef-defs)
add_public_tablegen_target(CIRCTDebugAttributesIncGen)
add_dependencies(circt-headers CIRCTDebugAttributesIncGen)
