#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"

#include "risky/Passes.h"

int main(int argc, char **argv) {
  // Register our custom passes.
  risky::registerRiskyPasses();

  mlir::DialectRegistry registry;
  registry.insert<mlir::func::FuncDialect,
                  mlir::arith::ArithDialect,
                  mlir::memref::MemRefDialect,
                  mlir::bufferization::BufferizationDialect,
                  mlir::linalg::LinalgDialect>();

  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "Risky MLIR optimizer\n", registry));
}