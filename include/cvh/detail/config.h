#ifndef CVH_DETAIL_CONFIG_H
#define CVH_DETAIL_CONFIG_H

// Public policy switch for all validated CPU optimization paths. Keep this
// value consistent across every translation unit in a program.
#ifndef CVH_ENABLE_OPTIMIZATION
#define CVH_ENABLE_OPTIMIZATION 1
#endif

#if CVH_ENABLE_OPTIMIZATION != 0 && CVH_ENABLE_OPTIMIZATION != 1
#error "CVH_ENABLE_OPTIMIZATION must be 0 or 1"
#endif

// Internal compile-time capability: the vendored OpenCV Universal Intrinsics
// facade is available to cvh implementations. This is a detected result, not
// a consumer-facing configuration switch.
#define CVH_DETAIL_HAVE_OPENCV_UI CVH_ENABLE_OPTIMIZATION

#endif  // CVH_DETAIL_CONFIG_H
