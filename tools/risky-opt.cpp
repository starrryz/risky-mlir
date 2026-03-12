#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVDialect.h"

#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Parser/Parser.h"
#include "mlir/InitAllPasses.h"
#include "mlir/InitAllDialects.h"

#include "risky/Passes.h"

int main(int argc, char **argv) {
  // Register all upstream MLIR passes (needed for parsePassPipeline).
  mlir::registerAllPasses();

  // Register our custom passes.
  risky::registerRiskyPasses();

  // Register the end-to-end pipeline.
  mlir::PassPipelineRegistration<>(
      "lower-risky-to-vortex-spirv",
      "End-to-end pipeline to lower Triton Linalg to Vortex OpenCL SPIR-V",
      [](mlir::OpPassManager &pm) {
        llvm::StringRef pipelineStr =
            "func.func(risky-sanitize-triton-linalg,"
              "convert-linalg-to-parallel-loops,"
              "scf-parallel-loop-tiling{parallel-loop-tile-sizes=1024},"
              "gpu-map-parallel-loops,"
              "convert-parallel-loops-to-gpu),"
            "canonicalize,"
            "gpu-kernel-outlining,"
            "risky-setup-spirv-abi,"
            "strip-debuginfo,"
            "convert-gpu-to-spirv{use-64bit-index=true},"
            "convert-memref-to-spirv{use-64bit-index=true},"
            "convert-arith-to-spirv,"
            "convert-func-to-spirv,"
            "reconcile-unrealized-casts,"
            "canonicalize,"
            "spirv.module(spirv-lower-abi-attrs,spirv-update-vce),"
            "risky-finalize-spirv";
        if (failed(mlir::parsePassPipeline(pipelineStr, pm))) {
          llvm::errs() << "Failed to parse risky pipeline\n";
        }
      });

  mlir::DialectRegistry registry;
  mlir::registerAllDialects(registry);

  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "Risky MLIR optimizer\n", registry));
}