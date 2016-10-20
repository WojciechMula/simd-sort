void scalar_partition_epi32(uint32_t* array, const uint32_t pivot, int& left, int& right) {

    while (left <= right) {

        while (array[left] < pivot) {
            left += 1;
        }

        while (array[right] > pivot) {
            right -= 1;
        }

        if (left <= right) {
            const uint32_t t = array[left];
            array[left]      = array[right];
            array[right]     = t;

            left  += 1;
            right -= 1;
        }
    }
}


int lomuto_partition_epi32(uint32_t* array, int lo, int hi) {

    const uint32_t pivot = array[(lo + hi)/2];
    const uint32_t hi_value = array[hi];

    array[(lo + hi)/2] = hi_value;
    array[hi] = pivot;

    int i = lo;
    for (int j=lo; j < hi; j++) {
        if (array[j] <= pivot) {
            const uint32_t t = array[i];
            array[i] = array[j];
            array[j] = t;
            i += 1;
        }
    }

    {
        const uint32_t t = array[i];
        array[i]  = array[hi];
        array[hi] = t;
        i += 1;
    }

    return i;
}
