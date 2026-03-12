#include "risky/Passes.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"

// 前置要求的头文件
#include "mlir/Dialect/SPIRV/IR/SPIRVDialect.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"

// 额外需要的头文件
#include "mlir/Dialect/SPIRV/IR/SPIRVEnums.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVAttributes.h"

using namespace mlir;

namespace {

struct FinalizeSPIRVPass
    : public PassWrapper<FinalizeSPIRVPass, OperationPass<ModuleOp>> {

  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(FinalizeSPIRVPass)

  StringRef getArgument() const final { return "risky-finalize-spirv"; }

  StringRef getDescription() const final {
    return "Finalize SPIR-V module structure for OpenCL: hoist spirv.module "
           "out of gpu.module, rename, set VCE triple, inject EntryPoint "
           "and ExecutionMode";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<spirv::SPIRVDialect, gpu::GPUDialect>();
  }

  void runOnOperation() override {
    ModuleOp moduleOp = getOperation();
    MLIRContext *ctx = &getContext();

    // ================================================================
    // 阶段 1：剥离与提升 (Extract & Hoist)
    //   遍历顶层 ModuleOp，找到遗留的 gpu::GPUModuleOp。
    //   在其内部找到真正生成的 spirv::ModuleOp。
    //   将 spirv::ModuleOp 移动到外层 ModuleOp（置于 gpu::GPUModuleOp 之前）。
    //   然后将废弃的 gpu::GPUModuleOp erase()。
    // ================================================================
    spirv::ModuleOp spvModule = nullptr;
    gpu::GPUModuleOp gpuModule = nullptr;

    for (auto gpuMod : moduleOp.getOps<gpu::GPUModuleOp>()) {
      gpuModule = gpuMod;
      for (auto spvMod : gpuMod.getOps<spirv::ModuleOp>()) {
        spvModule = spvMod;
        break;
      }
      break;
    }

    if (!spvModule || !gpuModule) {
      moduleOp.emitError("RiskyFinalizeSPIRV: could not find gpu.module "
                         "containing spirv.module");
      return signalPassFailure();
    }

    // 将 spirv.module 移动到 gpu.module 之前（外层 ModuleOp 中）
    spvModule->moveBefore(gpuModule);

    // 删除废弃的 gpu.module
    gpuModule.erase();

    // ================================================================
    // 阶段 2：规范化命名与属性 (Rename & TargetEnv)
    //   将 spirv::ModuleOp 重命名为 "vecadd"。
    //   设置 vce_triple 属性满足 OpenCL 规范。
    //   删除 gpu.kernel 空壳函数，将真正的 FuncOp 重命名为 "vecadd"。
    // ================================================================

    // 2a: 重命名 spirv.module（动态：取第一个 spirv::FuncOp 的名字）
    //     先扫描找到真正的函数名，用于 module 命名
    std::string kernelName;
    for (auto func : spvModule.getOps<spirv::FuncOp>()) {
      if (!func->hasAttr("gpu.kernel")) {
        kernelName = func.getSymName().str();
        break;
      }
    }
    if (kernelName.empty()) {
      // fallback: 取任意 FuncOp 的名字
      for (auto func : spvModule.getOps<spirv::FuncOp>()) {
        kernelName = func.getSymName().str();
        break;
      }
    }
    if (kernelName.empty())
      kernelName = "kernel";
    spvModule.setSymName(kernelName);

    // 2b: 设置 VCE triple
    auto vceAttr = spirv::VerCapExtAttr::get(
        spirv::Version::V_1_0,
        {spirv::Capability::Kernel, spirv::Capability::Addresses,
         spirv::Capability::Int64},
        ArrayRef<spirv::Extension>{}, ctx);
    spvModule->setAttr(spirv::ModuleOp::getVCETripleAttrName(), vceAttr);

    // 2c: 处理内部的 spirv::FuncOp
    //     删除带有 gpu.kernel 属性的空壳函数，
    //     保留真正包含代码的 FuncOp。
    spirv::FuncOp realFunc = nullptr;
    SmallVector<spirv::FuncOp> shellFuncs;

    for (auto func : spvModule.getOps<spirv::FuncOp>()) {
      if (func->hasAttr("gpu.kernel")) {
        shellFuncs.push_back(func);
      } else {
        realFunc = func;
      }
    }

    // 删除空壳函数
    for (auto shell : shellFuncs) {
      shell.erase();
    }

    // ================================================================
    // 阶段 3：链接全局 ID (Link GlobalInvocationId)
    //   convert-gpu-to-spirv 生成的是 LocalInvocationId（因为 gpu.thread_id
    //   映射到 local），但 Vortex OpenCL 需要 GlobalInvocationId。
    //   先查找 GlobalInvocationId；如果没有，则查找 LocalInvocationId 并
    //   将其重命名为 GlobalInvocationId。
    // ================================================================
    StringRef globalVarName;
    spirv::GlobalVariableOp targetGlobalVar = nullptr;

    // 先尝试找 GlobalInvocationId
    for (auto globalVar : spvModule.getOps<spirv::GlobalVariableOp>()) {
      if (globalVar.getSymName() == "__builtin__GlobalInvocationId__") {
        targetGlobalVar = globalVar;
        break;
      }
    }

    // 如果没有，找 LocalInvocationId 并替换
    if (!targetGlobalVar) {
      for (auto globalVar : spvModule.getOps<spirv::GlobalVariableOp>()) {
        if (globalVar.getSymName() == "__builtin__LocalInvocationId__") {
          targetGlobalVar = globalVar;
          break;
        }
      }
      if (targetGlobalVar) {
        // 重命名 GlobalVariable: Local → Global
        StringRef oldName = "__builtin__LocalInvocationId__";
        StringRef newName = "__builtin__GlobalInvocationId__";
        targetGlobalVar.setSymName(newName);
        targetGlobalVar->setAttr("built_in",
            StringAttr::get(ctx, "GlobalInvocationId"));

        // 更新所有 spirv.mlir.addressof 引用
        spvModule.walk([&](spirv::AddressOfOp addrOf) {
          if (addrOf.getVariable() == oldName) {
            addrOf->setAttr("variable",
                FlatSymbolRefAttr::get(ctx, newName));
          }
        });
      }
    }

    if (!targetGlobalVar) {
      spvModule.emitError("RiskyFinalizeSPIRV: could not find "
                          "@__builtin__GlobalInvocationId__ or "
                          "@__builtin__LocalInvocationId__ global variable");
      return signalPassFailure();
    }
    globalVarName = targetGlobalVar.getSymName();

    // ================================================================
    // 阶段 4：注入入口点 (Inject EntryPoint & ExecutionMode)
    //   在 spirv::ModuleOp 的 Block 尾部 push_back：
    //   - spirv::EntryPointOp (Kernel, funcName, [GlobalVariable symbol])
    //   - spirv::ExecutionModeOp (funcName, ContractionOff)
    // ================================================================
    Block *spvBody = spvModule.getBody();
    OpBuilder builder(ctx);
    builder.setInsertionPointToEnd(spvBody);

    // 获取真正函数的名字用于 EntryPoint / ExecutionMode
    StringRef funcName = realFunc ? realFunc.getSymName() : "vecadd";
    auto globalSymRef = FlatSymbolRefAttr::get(ctx, globalVarName);

    // 插入 spirv.EntryPoint "Kernel" @<funcName>, @__builtin__GlobalInvocationId__
    builder.create<spirv::EntryPointOp>(
        spvModule.getLoc(),
        spirv::ExecutionModel::Kernel,
        funcName,
        ArrayAttr::get(ctx, {globalSymRef}));

    // 插入 spirv.ExecutionMode @<funcName> "ContractionOff"
    builder.create<spirv::ExecutionModeOp>(
        spvModule.getLoc(),
        funcName,
        spirv::ExecutionMode::ContractionOff,
        ArrayAttr::get(ctx, {}));

    // ================================================================
    // 阶段 5：清理外层 ModuleOp 属性
    //   移除 gpu.container_module 和 spirv.target_env，
    //   使输出更干净，便于后续 mlir-translate 序列化。
    // ================================================================
    moduleOp->removeAttr("gpu.container_module");
    moduleOp->removeAttr("spirv.target_env");
  }
};

} // anonymous namespace

namespace risky {

std::unique_ptr<OperationPass<ModuleOp>> createFinalizeSPIRVPass() {
  return std::make_unique<FinalizeSPIRVPass>();
}

void registerFinalizeSPIRVPass() {
  PassRegistration<FinalizeSPIRVPass>();
}

} // namespace risky
