# Release Guide

## Scope

This guide defines the minimum release flow for `opencv-header-only`.

## Prerequisites

- Clean branch based on `main`
- CI green on:
  - `smoke`
  - `core_basic`
  - `imgproc_quick_gate`

## Release Steps

1. Update version:
   - Edit `VERSION.txt` (`MAJOR.MINOR.PATCH`).
2. Update `CHANGELOG.md`:
   - Move key items from `[Unreleased]` into a dated version section.
3. Validate locally:
   - `./scripts/ci_core_basic.sh`
   - `./scripts/ci_imgproc_quick_gate.sh`
4. Validate install/package:
   - Configure with install prefix.
   - Run `cmake --install ...`.
   - Confirm consumer can `find_package(opencv_header_only CONFIG REQUIRED)`.
5. Land PR to `main`.
6. Create git tag:
   - `git tag vX.Y.Z`
   - `git push origin vX.Y.Z`
7. Publish release notes on GitHub:
   - Copy highlights from `CHANGELOG.md`.

## Notes

- Keep `imgproc_full_gate` as manual deep scan unless explicitly promoted.
- Treat benchmark baselines and metadata as runner-class specific artifacts.
