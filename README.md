```bash
cd risky-mlir
mkdir -p build
cd build
cmake .. -G Ninja \
  -DLLVM_DIR=/root/llvm-7d5de303-ubuntu-x64/lib/cmake/llvm \
  -DMLIR_DIR=/root/llvm-7d5de303-ubuntu-x64/lib/cmake/mlir
```

__编译（Ninja）：__

```bash
cd risky-mlir/build && ninja
```
__运行测试：__

```bash
./risky-mlir/build/tools/risky-opt test_src/ttir_linalg.mlir --risky-sanitize-triton-linalg
```
