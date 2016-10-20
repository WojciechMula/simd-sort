#include <cstdlib>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cassert>
#include <immintrin.h>

#include "cmdline.cpp"
#include "input_data.cpp"
#include "quicksort-all.cpp"
#include "avx2-altquicksort.h"
#include "avx2-nate-quicksort.cpp"


bool is_sorted(uint32_t* array, size_t n) {
    assert(n > 0);
    for (size_t i=1; i < n; i++) {
        if (array[i - 1] > array[i]) {
            printf("mismatch at %lu\n", i);
            return false;
        }
    }

    return true;
}


const size_t AVX512_REGISTER_SIZE = 16;


class Test {

    bool verbose;

public:
    Test(bool v = true) : verbose(v) {}

    template <typename SORT_FN>
    bool run(SORT_FN sort) {
        const size_t start = 2*AVX512_REGISTER_SIZE;
        const size_t end   = 256*AVX512_REGISTER_SIZE;

        if (verbose) {
            putchar('\n');
        }

        for (size_t size=start; size < end; size += 1) {

            if (verbose) {
                printf("%lu/%lu\r", size, end);
                fflush(stdout);
            }

            InputAscending  asc(size);
            InputDescending dsc(size);
            InputRandom     rnd(size);
            InputRandomFew  rndfew(size);

            if (!test(sort, asc)) {
                printf("failed for size %lu, intput ascending\n", size);
                return false;
            }

            if (!test(sort, dsc)) {
                printf("failed for size %lu, intput descending\n", size);
                return false;
            }

            if (!test(sort, rnd)) {
                printf("failed for size %lu, intput random\n", size);
                return false;
            }

            if (!test(sort, rndfew)) {
                printf("failed for size %lu, intput random few\n", size);
                return false;
            }
        } // for

        if (verbose) {
            putchar('\n');
        }

        return true;
    }


private:
    template <typename SORT_FN>
    bool test(SORT_FN sort, InputData& data) {
        sort(data.pointer(), 0, data.count() - 1);

        return is_sorted(data.pointer(), data.count());
    }
};


class Flags {
    public:
        bool avx2;
        bool avx2_alt;
        bool avx2_nate;
        bool avx512;
        bool avx512_buf;
        bool avx512_popcnt;
        bool avx512_bmi;

    public:
        Flags(const CommandLine& cmd) {

            enable_all(false);

            bool any_set = false;
            if (cmd.has("-avx2")) {
                avx2 = true;
                any_set = true;
            }

            if (cmd.has("-avx2-alt")) {
                avx2_alt = true;
                any_set = true;
            }

            if (cmd.has("-avx2-nate")) {
                avx2_nate = true;
                any_set = true;
            }

            if (cmd.has("-avx512")) {
                avx512 = true;
                any_set = true;
            }

            if (cmd.has("-avx512-buf")) {
                avx512_buf = true;
                any_set = true;
            }

            if (cmd.has("-avx512-popcnt")) {
                avx512_popcnt = true;
                any_set = true;
            }

            if (cmd.has("-avx512-bmi")) {
                avx512_bmi = true;
                any_set = true;
            }

            if (!any_set) {
                enable_all(true);
            }
        }

        void enable_all(bool val) {
            avx2          = val;
            avx2_nate     = val;
            avx2_alt      = val;
            avx512        = val;
            avx512_buf    = val;
            avx512_popcnt = val;
            avx512_bmi    = val;
        }
};


int main(int argc, char* argv[]) {

    CommandLine cmd(argc, argv);

    puts("Please wait, it might take a while...");
    puts("");

    bool verbose = cmd.has("-v") || cmd.has("--verbose");
    Flags flags(cmd);

    Test test(verbose);
    int ret = EXIT_SUCCESS;

#ifdef HAVE_AVX2_INSTRUCTIONS
    if (flags.avx2) {
        printf("AVX2 base version... "); fflush(stdout);
        if (test.run(qs::avx2::quicksort)) {
            puts("OK");
        } else {
            puts("FAILED");
            ret = EXIT_FAILURE;
        }
    }

    if (flags.avx2_alt) {
        printf("AVX2 alt version... "); fflush(stdout);
        if (test.run(wrapped_avx2_pivotonlast_sort)) {
            puts("OK");
        } else {
            puts("FAILED");
            ret = EXIT_FAILURE;
        }
    }

    if (flags.avx2_nate) {
        printf("AVX2 Nate's variant... "); fflush(stdout);
        if (test.run(nate::wrapped_avx2_pivotonlast_sort)) {
            puts("OK");
        } else {
            puts("FAILED");
            ret = EXIT_FAILURE;
        }
    }
#endif

#ifdef HAVE_AVX512F_INSTRUCTIONS

#ifdef POPCNT_LOOKUP
    prepare_lookup();
#endif

    if (flags.avx512) {
        printf("AVX512 base version... "); fflush(stdout);
        if (test.run(qs::avx512::quicksort)) {
            puts("OK");
        } else {
            puts("FAILED");
            ret = EXIT_FAILURE;
        }
    }

    if (flags.avx512_popcnt) {
        printf("AVX512 + popcnt version... "); fflush(stdout);
        if (test.run(qs::avx512::popcnt_quicksort)) {
            puts("OK");
        } else {
            puts("FAILED");
            ret = EXIT_FAILURE;
        }
    }

    if (flags.avx512_buf) {
        printf("AVX512 with aux buffers... "); fflush(stdout);
        if (test.run(qs::avx512::auxbuffer_quicksort)) {
            puts("OK");
        } else {
            puts("FAILED");
            ret = EXIT_FAILURE;
        }
    }

#if 0
    if (flags.avx512_bmi) {
        printf("AVX512 + bmi2 version ... "); fflush(stdout);
        if (test.run(qs::avx512::bmi2_quicksort)) {
            puts("OK");
        } else {
            puts("FAILED");
            ret = EXIT_FAILURE;
        }
    }
#endif
#endif // HAVE_AVX512F_INSTRUCTIONS

    return ret;
}
