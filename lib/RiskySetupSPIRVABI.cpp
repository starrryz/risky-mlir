#include "risky/Passes.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"

// 前置要求的头文件
#include "mlir/Dialect/SPIRV/IR/SPIRVDialect.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVAttributes.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

// 额外需要的头文件
#include "mlir/Dialect/SPIRV/IR/SPIRVEnums.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVTypes.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/AsmParser/AsmParser.h"

using namespace mlir;

namespace {

struct SetupSPIRVABIPass
    : public PassWrapper<SetupSPIRVABIPass, OperationPass<ModuleOp>> {

  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(SetupSPIRVABIPass)

  StringRef getArgument() const final { return "risky-setup-spirv-abi"; }

  StringRef getDescription() const final {
    return "Set up SPIR-V ABI attributes for GPU kernels: erase host code, "
           "inject target env, rewrite memref types, and clean up unused ops";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<spirv::SPIRVDialect, gpu::GPUDialect,
                    func::FuncDialect, memref::MemRefDialect>();
  }

  void runOnOperation() override {
    ModuleOp moduleOp = getOperation();
    MLIRContext *ctx = &getContext();

    // ================================================================
    // Action A: 强制清理 Host 代码
    //   遍历 Module 下所有 func::FuncOp，如果不在 gpu.module 内部
    //   （即顶层 Host 函数），直接 erase。
    // ================================================================
    SmallVector<func::FuncOp> hostFuncsToErase;
    moduleOp.walk([&](func::FuncOp funcOp) {
      if (!funcOp->getParentOfType<gpu::GPUModuleOp>()) {
        hostFuncsToErase.push_back(funcOp);
      }
    });
    for (auto funcOp : hostFuncsToErase) {
      funcOp.erase();
    }

    // ================================================================
    // Action B: 注入 SPIR-V Target Environment
    //   为顶层 ModuleOp 添加 spirv.target_env 属性。
    //   使用 parseAttribute 解析标准 SPIR-V target env 字符串。
    // ================================================================
    {
      StringRef targetEnvStr =
          "#spirv.target_env<#spirv.vce<v1.0, "
          "[Kernel, Addresses, Int64], []>, "
          "#spirv.resource_limits<>>";
      Attribute targetEnvAttr = parseAttribute(targetEnvStr, ctx);
      if (!targetEnvAttr) {
        emitError(moduleOp.getLoc())
            << "failed to parse spirv.target_env attribute";
        return signalPassFailure();
      }
      moduleOp->setAttr("spirv.target_env", targetEnvAttr);
    }

    // ================================================================
    // Action C + D + E: 处理每个 gpu::GPUFuncOp
    // ================================================================
    moduleOp.walk([&](gpu::GPUFuncOp gpuFunc) {
      // ==============================================================
      // Action C: 动态提取并注入 ABI 属性
      //   读取 known_block_size (DenseI32ArrayAttr)，提取 3 个维度，
      //   构建 spirv::EntryPointABIAttr 并塞入 gpu::GPUFuncOp。
      // ==============================================================
      if (auto blockSizeAttr =
              gpuFunc->getAttrOfType<DenseI32ArrayAttr>("known_block_size")) {
        SmallVector<int32_t, 3> workgroupSize(blockSizeAttr.asArrayRef());
        auto abiAttr = spirv::EntryPointABIAttr::get(
            ctx, DenseI32ArrayAttr::get(ctx, workgroupSize),
            std::nullopt, std::nullopt);
        gpuFunc->setAttr("spirv.entry_point_abi", abiAttr);
      }

      // ==============================================================
      // Action D: 暴力替换 MemRef 签名与 Memory Space
      //   将 memref<1024xf32, strided<[1], offset: ?>> 降级为
      //   memref<?xf32, #spirv.storage_class<CrossWorkgroup>>
      // ==============================================================
      auto memSpace = spirv::StorageClassAttr::get(
          ctx, spirv::StorageClass::CrossWorkgroup);

      // D.1: 修改 gpu::GPUFuncOp 的 FunctionType 和 Block Argument 类型
      SmallVector<Type> newInputTypes;
      for (unsigned i = 0; i < gpuFunc.getNumArguments(); ++i) {
        Type argType = gpuFunc.getArgument(i).getType();
        if (auto memrefTy = dyn_cast<MemRefType>(argType)) {
          // 构建新类型: shape={?}, 保留原始 elementType, CrossWorkgroup
          auto newMemRefType = MemRefType::get(
              {ShapedType::kDynamic}, memrefTy.getElementType(),
              MemRefLayoutAttrInterface{}, memSpace);
          newInputTypes.push_back(newMemRefType);
          gpuFunc.getBody().getArgument(i).setType(newMemRefType);
        } else {
          newInputTypes.push_back(argType);
        }
      }

      auto oldFuncType = gpuFunc.getFunctionType();
      auto newFuncType =
          FunctionType::get(ctx, newInputTypes, oldFuncType.getResults());
      gpuFunc.setFunctionType(newFuncType);

      // D.2: 遍历内部的 memref::LoadOp 和 memref::StoreOp，
      //      强行更新它们的 MemRef 操作数类型（通过 block arg 已自动生效，
      //      此处做显式验证/兜底）。
      //      对于 LoadOp，其 result type 是 element type，不需要改变。
      //      MemRef 操作数类型已通过 block argument setType 自动更新。

      // ==============================================================
      // Action E: 清理无用的 GPU 维度读取 (模拟 DCE)
      //   遍历 gpu::ThreadIdOp, gpu::BlockIdOp, gpu::GridDimOp,
      //   gpu::BlockDimOp，如果 result use_empty() 则 erase。
      // ==============================================================
      SmallVector<Operation *> deadOps;
      gpuFunc.walk([&](Operation *op) {
        if (isa<gpu::ThreadIdOp, gpu::BlockIdOp,
                gpu::GridDimOp, gpu::BlockDimOp>(op)) {
          if (op->getResult(0).use_empty()) {
            deadOps.push_back(op);
          }
        }
      });
      for (auto *op : deadOps) {
        op->erase();
      }
    });
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