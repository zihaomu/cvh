# OpenCV Imgproc upstream 快照

这里保存从 OpenCV `modules/imgproc/test` 抽取的原始 TEST 块，用于追踪本仓库的
21 个选定兼容 case。快照不参与测试编译，也不应手工修改。

当前 manifest 状态使用：

- `PASS`：已有可执行的本地 GTest consumer。
- `OUT_OF_SCOPE`：不属于当前公开 API，并注明重新评估条件。

重新生成：

```bash
python3 scripts/sync_opencv_imgproc_cases.py \
  --opencv-root /path/to/opencv \
  --repo-root .
```

`case_manifest.json` 记录稳定 ID、upstream commit、相对 source/snapshot 路径、
hash 和本地 consumer，不记录本机 checkout 的绝对路径。
