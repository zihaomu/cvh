# Core 测试数据

`npy/` 只保存当前测试直接消费的 NumPy fixture。每个文件的 hash、shape、
oracle 和 consumer 都记录在 `manifest.json`，不允许保留“以后可能会用”的
无消费者数据。

在仓库根目录执行：

```bash
uv run test/core/data/generators/generate_fixtures.py
```

生成器使用固定随机种子，覆盖 NPY reader、transpose 和 GEMM 三类 fixture。
依赖版本由脚本内的 PEP 723 metadata 固定。执行后仓库应当没有数据差异；如果
`npy/` 出现未被生成器管理的文件，命令会失败。
