#ifndef HQLC_BENCH_H
#define HQLC_BENCH_H

// When HQLC_BENCH is defined at compile time, the build must provide
// hqlc_bench_impl.h on the include path
// That header defines the stage enum, cycle counter, ctx struct, and
// the HQLC_BENCH_BEGIN/END macros.
//
// When HQLC_BENCH is not defined, the macros compile to nothing.

#ifdef HQLC_BENCH
#include "hqlc_bench_impl.h"
#else
#define HQLC_BENCH_BEGIN()    ((void)0)
#define HQLC_BENCH_END(stage) ((void)0)
#endif

#endif // HQLC_BENCH_H
