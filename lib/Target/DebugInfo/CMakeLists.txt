add_circt_translation_library(CIRCTTargetDebugInfo
  EmitHGLDD.cpp
  TranslateRegistration.cpp

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  CIRCTComb
  CIRCTDebug
  CIRCTHW
  CIRCTSeq
  CIRCTSupport
  CIRCTSV
  MLIRIR
  MLIRTranslateLib
)
