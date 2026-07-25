# 测试目录

测试按长期产品职责组织，不按开发阶段、版本号或历史任务命名。

## 分层

- `core/`：Mat、基础算子、类型、运行时、内部派发和 upstream 兼容子集。
- `imgproc/`：颜色、滤波、几何、强度、形态学等图像处理合同。
- `imgcodecs/`、`highgui/`：各自模块的功能和异常路径。
- `smoke/`：头文件独立编译、ODR、模式和最小 pipeline 检查。
- `opencv_contract/`：可选的 OpenCV 隔离差分测试。
- `upstream/`：OpenCV 原始 case 快照和状态 manifest，不直接参与编译。
- `support/`：core/imgproc 共用的测试状态 guard。
- `utils/`：跨模块测试工具。

public contract 只验证公开 API 的输出、边界和异常；需要强制 scalar/UI 路径或
观察 dispatch 的用例放在模块的 `internal/`。上游移植用例放在 `upstream/`，
并保留原 suite/case 关联。

## 构建与运行

```bash
cmake -S . -B build-tests \
  -DCVH_BUILD_TESTS=ON \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build-tests --target cvh_test_core cvh_test_imgproc -j
ctest --test-dir build-tests --output-on-failure
```

规范模块级 GTest target 只有 `cvh_test_core` 和 `cvh_test_imgproc`，不再注册同一
二进制的重复别名。`test/core/sources.cmake` 与
`test/imgproc/sources.cmake` 显式列出 source；配置阶段会审计遗漏、重复和
不存在的 `*_test.cpp`。

完整发布门禁应运行两种 header 配置：

```bash
./scripts/ci_headers_all.sh
CVH_CI_OPENCV_INTRIN=OFF ./scripts/ci_headers_all.sh
```

两者都构建默认 `all` 目标并运行完整 CTest；Core/Imgproc 的 XML、CTest
inventory 和 executed/failed/skipped 数量由
`test/ci/header_gate_expectations.json` 校验。

## 维护约束

1. 文件名表达稳定 API 或算法职责，不使用 `phase1`、版本号和任务编号。
2. 一个测试必须有可观察断言；仅打印、空调用和永久 skip 不属于有效测试。
3. 混合多个 API 的异常 case 应拆开，使失败能直接定位到 owner。
4. 公共测试不得依赖生产 `detail` helper 作为 oracle。
5. fixture 必须有固定生成器、hash、oracle 和 consumer。
6. upstream 的产品边界外 case 记为 `OUT_OF_SCOPE`，不注册成 `GTEST_SKIP`。

当前失败和产品边界记录见 `failing-tests.md`。
