#include "risky/Passes.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;

namespace {

struct SetupSPIRVABIPass
    : public PassWrapper<SetupSPIRVABIPass, OperationPass<ModuleOp>> {

  StringRef getArgument() const final { return "risky-setup-spirv-abi"; }

  StringRef getDescription() const final {
    return "Set up SPIR-V ABI attributes (placeholder)";
  }

  void runOnOperation() override {
    // Intentionally empty – placeholder pass.
  }
};

} // anonymous namespace

namespace risky {

std::unique_ptr<OperationPass<ModuleOp>> createSetupSPIRVABIPass() {
  return std::make_unique<SetupSPIRVABIPass>();
}

void registerRiskyPasses() {
  PassRegistration<SetupSPIRVABIPass>();
  registerSanitizeTritonLinalgPass();
}

} // namespace risky