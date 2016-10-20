.SUFFIXES:
.PHONY: all clean

FLAGS=-std=c++11 -mbmi2 -Wall -pedantic -Wextra
FLAGS_AVX512=$(FLAGS) -mavx512f -DHAVE_AVX512F_INSTRUCTIONS -DHAVE_AVX2_INSTRUCTIONS
FLAGS_AVX2=$(FLAGS) -mavx2 -DHAVE_AVX2_INSTRUCTIONS

DEPS_SORT=partition.cpp \
          avx2-partition.cpp \
          avx2-quicksort.cpp \
          avx2-altquicksort.h \
          avx2-nate-quicksort.cpp \
          avx512-swap.cpp \
          avx512-partition.cpp \
          avx512-auxbuffer-partition.cpp \
          avx512-bmi2-partition.cpp \
          avx512-popcnt-partition.cpp \
          avx512-quicksort.cpp \
          avx512-sort-register.cpp \
          avx512-partition-register.cpp \
          quicksort.cpp

SPEED_DEPS=$(DEPS_SORT) speed.cpp gettime.cpp rdtsc.cpp runtime_stats.cpp
SPEED_FLAGS=-O3 -DNDEBUG

ALL=test speed test_avx2 speed_avx2 speed_stats speed_avx2_stats

all: $(ALL)

test: test.cpp input_data.cpp $(DEPS_SORT)
	$(CXX) $(FLAGS_AVX512) -fsanitize=address test.cpp -o $@

test_avx2: test.cpp input_data.cpp $(DEPS_SORT)
	#$(CXX) $(FLAGS_AVX2) -fsanitize=address test.cpp -o $@
	$(CXX) $(FLAGS_AVX2) test.cpp -o $@

speed: $(SPEED_DEPS)
	$(CXX) $(FLAGS_AVX512) $(SPEED_FLAGS) speed.cpp -o $@

speed_avx2: $(SPEED_DEPS)
	$(CXX) $(FLAGS_AVX2) $(SPEED_FLAGS) speed.cpp -o $@

speed_stats: $(SPEED_DEPS)
	$(CXX) $(FLAGS_AVX512) $(SPEED_FLAGS) -DWITH_RUNTIME_STATS speed.cpp -o $@

speed_avx2_stats: $(SPEED_DEPS)
	$(CXX) $(FLAGS_AVX2) $(SPEED_FLAGS) -DWITH_RUNTIME_STATS speed.cpp -o $@

run: test
	sde -cnl -- ./$^

run_avx2: test_avx2
	sde -cnl -- ./$^

clean:
	rm -f $(ALL)
