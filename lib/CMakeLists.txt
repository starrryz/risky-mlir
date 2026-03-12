add_mlir_library(RiskyPasses
  RiskySetupSPIRVABI.cpp
  RiskyFinalizeSPIRV.cpp
  RiskySanitizeTritonLinalg.cpp

  DEPENDS
  MLIRPass

  LINK_LIBS PUBLIC
  MLIRPass
  MLIRIR
  MLIRFuncDialect
  MLIRLinalgDialect
  MLIRBufferizationDialect
  MLIRMemRefDialect
  MLIRArithDialect
  MLIRGPUDialect
  MLIRSPIRVDialect
  MLIRParser
)
