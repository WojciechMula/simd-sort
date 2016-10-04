#define RDTSC_START(cycles)                                             \
    do {                                                                \
        uint32_t cyc_high, cyc_low;                                     \
        __asm volatile("cpuid\n"                                        \
                       "rdtsc\n"                                        \
                       "mov %%edx, %0\n"                                \
                       "mov %%eax, %1" :                                \
                       "=r" (cyc_high),                                 \
                       "=r"(cyc_low) :                                  \
                       : /* no read only */                             \
                       "%rax", "%rbx", "%rcx", "%rdx" /* clobbers */    \
                       );                                               \
        (cycles) = ((uint64_t)cyc_high << 32) | cyc_low;                \
    } while (0)

#define RDTSC_STOP(cycles)                                              \
    do {                                                                \
        uint32_t cyc_high, cyc_low;                                     \
        __asm volatile("rdtscp\n"                                       \
                       "mov %%edx, %0\n"                                \
                       "mov %%eax, %1\n"                                \
                       "cpuid" :                                        \
                       "=r"(cyc_high),                                  \
                       "=r"(cyc_low) :                                  \
                       /* no read only registers */ :                   \
                       "%rax", "%rbx", "%rcx", "%rdx" /* clobbers */    \
                       );                                               \
        (cycles) = ((uint64_t)cyc_high << 32) | cyc_low;                \
    } while (0)

static __attribute__ ((noinline))
uint64_t rdtsc_overhead_func(uint64_t dummy) {
    return dummy;
}

uint64_t global_rdtsc_overhead = (uint64_t) UINT64_MAX;

#define RDTSC_SET_OVERHEAD(test, repeat)			                    \
  do {								                                    \
    uint64_t cycles_start, cycles_final, cycles_diff;		            \
    uint64_t min_diff = UINT64_MAX;				                        \
    for (int i = 0; i < repeat; i++) {			                        \
      __asm volatile("" ::: /* pretend to clobber */ "memory");	        \
      RDTSC_START(cycles_start);				                        \
      test;							                                    \
      RDTSC_STOP(cycles_final);                                         \
      cycles_diff = (cycles_final - cycles_start);		                \
      if (cycles_diff < min_diff) min_diff = cycles_diff;	            \
    }								                                    \
    global_rdtsc_overhead = min_diff;				                    \
    printf("rdtsc_overhead set to %d\n", (int)global_rdtsc_overhead);   \
  } while (0)
