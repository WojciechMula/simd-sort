#include <cstdlib>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cassert>
#include <immintrin.h>

#include "input_data.cpp"
#include "quicksort-all.cpp"
#include "avx2-altquicksort.h"


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
                printf("%d/%d\r", size, end);
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


int main(int argc, char* argv[]) {

    puts("Please wait, it might take a while...");
    puts("");

    bool verbose = false;
    for (int i=1; i < argc; i++) {
        if ((strcmp("-v", argv[i]) == 0) || (strcmp("--verbose", argv[i]) == 0)) {
            verbose = true;
            break;
        }
    }

    Test test(verbose);
    int ret = EXIT_SUCCESS;

#ifdef HAVE_AVX2_INSTRUCTIONS
    if (0) {
        printf("AVX2 base version... "); fflush(stdout);
        if (test.run(qs::avx2::quicksort)) {
            puts("OK");
        } else {
            puts("FAILED");
            ret = EXIT_FAILURE;
        }
    }

    if (1) {
        printf("AVX2 alt version... "); fflush(stdout);
        if (test.run(wrapped_avx2_pivotonlast_sort)) {
            puts("OK");
        } else {
            puts("FAILED");
            ret = EXIT_FAILURE;
        }
    }
#endif

#ifdef HAVE_AVX512F_INSTRUCTIONS
    if (1) {
        printf("AVX512 base version... "); fflush(stdout);
        if (test.run(qs::avx512::quicksort)) {
            puts("OK");
        } else {
            puts("FAILED");
            ret = EXIT_FAILURE;
        }
    }

    if (1) {
        printf("AVX512 + popcnt version... "); fflush(stdout);
        if (test.run(qs::avx512::popcnt_quicksort)) {
            puts("OK");
        } else {
            puts("FAILED");
            ret = EXIT_FAILURE;
        }
    }

    if (1) {
        printf("AVX512 with aux buffers... "); fflush(stdout);
        if (test.run(qs::avx512::auxbuffer_quicksort)) {
            puts("OK");
        } else {
            puts("FAILED");
            ret = EXIT_FAILURE;
        }
    }

#if 0
    {
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
