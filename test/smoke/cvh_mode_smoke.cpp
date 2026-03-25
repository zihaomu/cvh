#include "cvh/cvh.h"

#if defined(CVH_EXPECT_FULL)
#ifndef CVH_FULL
#error "Expected CVH_FULL to be enabled for full mode smoke target"
#endif
#ifdef CVH_LITE
#error "CVH_LITE must not be enabled together with CVH_FULL"
#endif
#else
#ifndef CVH_LITE
#error "Expected CVH_LITE default mode for header-only smoke target"
#endif
#ifdef CVH_FULL
#error "CVH_FULL must not be enabled for lite smoke target"
#endif
#endif

int main()
{
    return 0;
}
