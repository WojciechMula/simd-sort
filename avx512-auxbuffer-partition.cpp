namespace qs {

    namespace avx512 {

        void memset_epi32(uint32_t* array, uint32_t w, size_t n) {
            
            const int N = 16;
            const __m512i word = _mm512_set1_epi32(w);

            for (size_t i=0; i < n/N; i++) {
                _mm512_storeu_si512(array + i*N, word);
            }

            for (size_t i=n/N * N; i < n; i++) {
                array[i] = w;
            }
        }

        void memcpy_epi32(uint32_t* dst, uint32_t* src, size_t n) {
            
            const int N = 16;

            for (size_t i=0; i < n/N; i++) {
                _mm512_storeu_si512(dst + i*N, _mm512_loadu_si512(src + i*N));
            }

            for (size_t i=n/N * N; i < n; i++) {
                dst[i] = src[i];
            }
        }

        // parition array[0..n-1]
        uint32_t FORCE_INLINE partition_auxbuffer_epi32(uint32_t* array, size_t n, uint32_t pv) {

            const int N = 16;
            const int AUX_COUNT = 1024; // 4kB

            static uint32_t gt_buf[AUX_COUNT + N];

            size_t lt_count = 0;
            size_t gt_count = 0;

            const __m512i pivot = _mm512_set1_epi32(pv);

            // 1. copy greater and less values into separate buffers
            for (size_t i=0; i < n / N; i++) {
                
                const __m512i v = _mm512_loadu_si512(array + i*N);

                const __mmask16 lt = _mm512_cmplt_epi32_mask(v, pivot);
                const __mmask16 gt = _mm512_cmpgt_epi32_mask(v, pivot);

                const __m512i less    = _mm512_maskz_compress_epi32(lt, v);
                const __m512i greater = _mm512_maskz_compress_epi32(gt, v);

                _mm512_storeu_si512(array  + lt_count, less);
                _mm512_storeu_si512(gt_buf + gt_count, greater);

                lt_count += _mm_popcnt_u32(lt);
                gt_count += _mm_popcnt_u32(gt);
            }

            for (size_t i=0; i < n % N; i++) {
                
                const uint32_t v = array[(n/N) * N + i];

                if (v < pv) {
                    array[lt_count++] = v;
                } else if (v > pv) {
                    gt_buf[gt_count++] = v;
                }
            }

            const size_t eq_count = n - (lt_count + gt_count);

            // 2. replace array with partially ordered data

            // 2.a. pivots
            memset_epi32(array + lt_count, pv, eq_count);

            // 2.b. all values greater than pivot
            memcpy_epi32(array + lt_count + eq_count, gt_buf, gt_count);

            // 3. index before the first pivot
            return lt_count;
        }

    } // namespace avx512

} // namespace qa
