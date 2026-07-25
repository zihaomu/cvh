# OpenCV Core upstream 快照

这里保存从 OpenCV `modules/core/test` 抽取的原始 TEST 块，用于 Mat/channel
兼容性追踪。快照不参与本仓库测试编译，也不应手工修改。

状态由 `channel_manifest.json` 维护：

- `PASS`：已在 `test/core/upstream/mat_channel_upstream_test.cpp` 落地并执行。
- `OUT_OF_SCOPE`：不属于当前 Mat-only 公开 API，并注明重新评估条件。

重新生成：

```bash
python3 scripts/sync_opencv_core_channel_cases.py \
  --opencv-root /path/to/opencv \
  --repo-root .
```

manifest 只记录 upstream project/commit、相对 source/snapshot 路径、case ID、
状态和 hash，不记录本机 checkout 的绝对路径。
