#ifndef CVH_HIGHGUI_DETAIL_BACKEND_SELECT_HPP
#define CVH_HIGHGUI_DETAIL_BACKEND_SELECT_HPP

#if defined(CVH_HIGHGUI_FORCE_STUB) && CVH_HIGHGUI_FORCE_STUB
#include "backend_stub.hpp"
#elif defined(__APPLE__)
#include "backend_cocoa.hpp"
#elif defined(_WIN32)
#include "backend_win32.hpp"
#elif defined(__linux__) && !defined(__ANDROID__)
#if defined(CVH_HIGHGUI_X11) && CVH_HIGHGUI_X11
#include "backend_x11.hpp"
#elif defined(__has_include)
#if __has_include(<X11/Xlib.h>) && __has_include(<X11/Xutil.h>)
#include "backend_x11.hpp"
#else
#include "backend_stub.hpp"
#endif
#else
#include "backend_stub.hpp"
#endif
#else
#include "backend_stub.hpp"
#endif

#endif  // CVH_HIGHGUI_DETAIL_BACKEND_SELECT_HPP
