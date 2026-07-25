# Core 测试

`test/core` 验证 Mat 语义、公开基础算子、运行时行为、内部派发一致性和选定的
OpenCV upstream 兼容合同。

## 目录职责

- `mat/`：生命周期、ROI、layout、reshape、conversion、expression 和 channel。
- `operations/`：array、arithmetic、math、reduction、transpose 和 GEMM。
- `types/`：Point、Size、Scalar 等基础类型。
- `runtime/`：线程运行时与异常信息。
- `internal/`：scalar/UI dispatch 和私有 kernel 路径；最终结果仍通过公开入口验证。
- `upstream/`：已落地并执行的 OpenCV case 子集。
- `support/`：测试专用 reference、guard 和比较工具。
- `data/`：当前被消费的 NumPy fixture、manifest 和唯一生成入口。
- `../smoke/core_headers/`：每个顶层 Core 公共 `.h` 的独立 C++17 编译单元。

公开目录不 include `cvh/core/detail/*`。内部测试修改全局 dispatch mode 时必须通过
RAII guard 恢复，避免失败断言污染后续用例。

Core 公共头清单与 compile-smoke source 清单在配置期逐项比对。新增顶层公共
`.h` 而未新增对应 `<name>_compile.cpp` 会直接使配置失败；`.inl.h`、`detail/`
和 `simd/` 明确属于内部实现面。

## Fixture

```bash
uv run test/core/data/generators/generate_fixtures.py
```

生成器固定 NumPy 版本和随机种子。`data/manifest.json` 对每个 `.npy` 记录 hash、
shape、oracle 和 consumer；出现无管理 fixture 时生成命令会失败。

## 运行

```bash
cmake -S . -B build-core \
  -DCVH_BUILD_TESTS=ON \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build-core --target cvh_test_core -j
ctest --test-dir build-core -R '^cvh_test_core$' --output-on-failure
```

需要定位单一职责时可直接使用稳定 suite，例如：

```bash
./build-core/cvh_test_core --gtest_filter='MatLifecycleTest.*'
./build-core/cvh_test_core --gtest_filter='Reduction*'
```

OutputArray 两项 upstream case 不属于当前 Mat-only 公开 API，状态记录为
`OUT_OF_SCOPE`，不会以永久 skip 进入可执行测试。
