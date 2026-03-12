#include "risky/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;

namespace {

struct SanitizeTritonLinalgPass
    : public PassWrapper<SanitizeTritonLinalgPass,
                         OperationPass<func::FuncOp>> {

  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(SanitizeTritonLinalgPass)

  StringRef getArgument() const final {
    return "risky-sanitize-triton-linalg";
  }

  StringRef getDescription() const final {
    return "Sanitize Triton-generated Linalg IR for SPIR-V lowering";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<linalg::LinalgDialect, memref::MemRefDialect,
                    bufferization::BufferizationDialect, arith::ArithDialect,
                    func::FuncDialect>();
  }

  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();
    MLIRContext *ctx = &getContext();

    // ================================================================
    // Action A: Signature fix — UnrankedMemRef -> 1-D dynamic MemRef
    // ================================================================
    FunctionType oldFuncType = funcOp.getFunctionType();
    SmallVector<Type> newInputTypes;
    bool sigChanged = false;

    for (unsigned i = 0; i < oldFuncType.getNumInputs(); ++i) {
      Type argType = oldFuncType.getInput(i);
      if (auto unrankedTy = dyn_cast<UnrankedMemRefType>(argType)) {
        auto newTy = MemRefType::get({ShapedType::kDynamic},
                                     unrankedTy.getElementType());
        newInputTypes.push_back(newTy);
        funcOp.getBody().getArgument(i).setType(newTy);
        sigChanged = true;
      } else {
        newInputTypes.push_back(argType);
      }
      // Strip all argument attributes (e.g. tt.divisibility).
      funcOp.setArgAttrs(i, DictionaryAttr::get(ctx, {}));
    }

    if (sigChanged) {
      auto newFuncType =
          FunctionType::get(ctx, newInputTypes, oldFuncType.getResults());
      funcOp.setFunctionType(newFuncType);
    }

    // ================================================================
    // Action C: Restructure linalg.generic  (Tensor -> MemRef)
    //   Collect first, then mutate — never erase inside a walk.
    // ================================================================
    SmallVector<linalg::GenericOp> genericOps;
    funcOp.walk([&](linalg::GenericOp op) { genericOps.push_back(op); });

    for (linalg::GenericOp genericOp : genericOps) {
      // --- Collect new `ins` operands ---
      SmallVector<Value> newIns;
      bool failed = false;
      for (Value input : genericOp.getInputs()) {
        auto toTensor =
            input.getDefiningOp<bufferization::ToTensorOp>();
        if (!toTensor) { failed = true; break; }

        Value allocVal = toTensor->getOperand(0);
        auto allocOp = allocVal.getDefiningOp<memref::AllocOp>();
        if (!allocOp) { failed = true; break; }

        // Find the memref.copy whose target is this alloc.
        Value reinterpretSrc;
        for (Operation *user : allocVal.getUsers()) {
          if (auto copyOp = dyn_cast<memref::CopyOp>(user)) {
            if (copyOp.getTarget() == allocVal) {
              reinterpretSrc = copyOp.getSource();
              break;
            }
          }
        }
        if (!reinterpretSrc) { failed = true; break; }
        newIns.push_back(reinterpretSrc);
      }
      if (failed) { signalPassFailure(); return; }

      // --- Collect new `outs` operands ---
      SmallVector<Value> newOuts;
      if (genericOp.getNumResults() == 0) { signalPassFailure(); return; }

      Value tensorResult = genericOp.getResult(0);
      bufferization::MaterializeInDestinationOp matOp = nullptr;
      for (Operation *user : tensorResult.getUsers()) {
        matOp = dyn_cast<bufferization::MaterializeInDestinationOp>(user);
        if (matOp) break;
      }
      if (!matOp) { signalPassFailure(); return; }
      newOuts.push_back(matOp.getDest());

      // --- Build the replacement linalg.generic (MemRef, void) ---
      // Insert right before the materialize op so all operands dominate.
      OpBuilder builder(matOp);

      SmallVector<AffineMap> indexingMaps = genericOp.getIndexingMapsArray();
      SmallVector<utils::IteratorType> iteratorTypes =
          genericOp.getIteratorTypesArray();

      auto newGenericOp = builder.create<linalg::GenericOp>(
          genericOp.getLoc(),
          /*resultTensorTypes=*/TypeRange{},
          /*inputs=*/newIns,
          /*outputs=*/newOuts,
          /*indexingMaps=*/indexingMaps,
          /*iteratorTypes=*/iteratorTypes,
          /*bodyBuild=*/nullptr);

      // Transfer the body region from old → new.
      newGenericOp.getRegion().takeBody(genericOp.getRegion());

      // Unlink the old generic's result from materialize, then erase both.
      tensorResult.replaceAllUsesWith(newOuts[0]);
      matOp->erase();
      genericOp->erase();
    }

    // ================================================================
    // Action B: Eliminate redundant bufferization / memref ops
    //   Collect then erase bottom-up (users before defs).
    // ================================================================
    SmallVector<Operation *> toErase;
    funcOp.walk([&](Operation *op) {
      if (isa<bufferization::MaterializeInDestinationOp,
              bufferization::ToTensorOp,
              memref::CopyOp,
              memref::AllocOp>(op)) {
        toErase.push_back(op);
      }
    });

    // Erase in reverse program order (bottom-up).
    for (auto it = toErase.rbegin(); it != toErase.rend(); ++it) {
      Operation *op = *it;
      for (Value result : op->getResults())
        result.dropAllUses();
      op->erase();
    }

    // ================================================================
    // Cleanup: deduplicate identical arith.index_cast ops
    // ================================================================
    DenseMap<Value, Operation *> indexCastMap;
    SmallVector<Operation *> deadCasts;

    funcOp.walk([&](arith::IndexCastOp castOp) {
      Value src = castOp.getIn();
      auto it = indexCastMap.find(src);
      if (it == indexCastMap.end()) {
        indexCastMap[src] = castOp;
      } else {
        castOp.getResult().replaceAllUsesWith(it->second->getResult(0));
        deadCasts.push_back(castOp);
      }
    });
    for (Operation *op : deadCasts)
      op->erase();
  }
};

} // anonymous namespace

namespace risky {

std::unique_ptr<OperationPass<func::FuncOp>>
createSanitizeTritonLinalgPass() {
  return std::make_unique<SanitizeTritonLinalgPass>();
}

void registerSanitizeTritonLinalgPass() {
  PassRegistration<SanitizeTritonLinalgPass>();
}

} // namespace risky