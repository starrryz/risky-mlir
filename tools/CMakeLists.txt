add_llvm_executable(risky-opt
  risky-opt.cpp
)

target_link_libraries(risky-opt
  PRIVATE
  RiskyPasses
  MLIROptLib
  MLIRRegisterAllPasses
  MLIRRegisterAllDialects
  MLIRIR
  MLIRPass
  MLIRSupport
  MLIRParser
  MLIRFuncDialect
  MLIRLinalgDialect
  MLIRBufferizationDialect
  MLIRMemRefDialect
  MLIRArithDialect
  MLIRGPUDialect
  MLIRSPIRVDialect

  # Pipeline upstream pass libraries
  MLIRTransforms
  MLIRLinalgTransforms
  MLIRSCFDialect
  MLIRSCFTransforms
  MLIRGPUTransforms
  MLIRGPUToSPIRV
  MLIRSPIRVConversion
  MLIRSPIRVTransforms
  MLIRMemRefToSPIRV
  MLIRArithToSPIRV
  MLIRFuncToSPIRV
  MLIRReconcileUnrealizedCasts
  MLIRAffineDialect
  MLIRAffineToStandard
)