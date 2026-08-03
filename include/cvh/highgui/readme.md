# HighGUI Module

## Responsibility

HighGUI is the optional header-only window and event-loop surface:

```cpp
#include <cvh/highgui/highgui.h>
```

Use it through the `cvh::highgui` CMake target. The target depends on
`cvh::headers` and propagates the required platform GUI system libraries.

## Platform Backends

- macOS: AppKit, CoreGraphics, and CoreFoundation;
- Windows: User32 and GDI;
- Linux: X11 when available, otherwise the unsupported-platform stub;
- headless tests: deterministic no-display behavior.

The implementations are inline headers. Linking platform libraries does not
create a cvh binary library.

HighGUI is excluded from `cvh/cvh.h` so compute-only and headless consumers do
not inherit GUI dependencies.

## Validation

- independent public-header compilation;
- multi-translation-unit ODR and shared inline state;
- lifecycle and argument contract tests;
- install-tree `cvh::highgui` consumer;
- headless execution in automated tests.
