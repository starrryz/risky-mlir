#ifndef RISKY_PASSES_H
#define RISKY_PASSES_H

#include <memory>
#include "mlir/Pass/Pass.h"
#include "mlir/IR/BuiltinOps.h"

namespace mlir {
namespace func {
class FuncOp;
} // namespace func
} // namespace mlir

namespace risky {

/// Create the --risky-setup-spirv-abi pass.
std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>> createSetupSPIRVABIPass();

/// Create the --risky-finalize-spirv pass.
std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>> createFinalizeSPIRVPass();

/// Create the --risky-sanitize-triton-linalg pass.
std::unique_ptr<mlir::OperationPass<mlir::func::FuncOp>>
createSanitizeTritonLinalgPass();

/// Register the --risky-sanitize-triton-linalg pass.
void registerSanitizeTritonLinalgPass();

/// Register the --risky-finalize-spirv pass.
void registerFinalizeSPIRVPass();

/// Register all Risky passes.
void registerRiskyPasses();

} // namespace risky

#endif // RISKY_PASSES_H