class Histogram {
    size_t hist[256];

public:
    Histogram() {
        reset();
    }

    void hit(uint8_t v) {
        hist[v] += 1;
    }

    void reset() {
        memset(hist, 0, sizeof(hist));
    }

    bool empty() const {
        for (int i=0; i < 256; i++) {
            if (hist[i] > 0) {
                return false;
            }
        }
        
        return true;
    }

public:
    void print() const {
        size_t total = 0;
        for (int i=0; i < 256; i++) {
            total += hist[i];
        }

        if (total == 0) return;

        for (int i=0; i < 256; i++) {
            if (hist[i] == 0) continue;

            printf("%02x: %5lu (%5.2f%%)\n", i, hist[i], 100.0 * hist[i]/total);
        }
    }
};

class Statistics {
public:
    size_t partition_calls;
    size_t items_processed;
    size_t scalar__partition_calls;
    size_t scalar__items_processed;
    Histogram pvbyte_histogram;

public:
    Statistics() {
        reset();
    }

    void reset() {
        partition_calls = 0;
        items_processed = 0;
        scalar__partition_calls = 0;
        scalar__items_processed = 0;

        pvbyte_histogram.reset();
    }

    bool anything_collected() const {
        return (partition_calls > 0)
            || (items_processed > 0)
            || (scalar__partition_calls > 0)
            || (scalar__items_processed > 0)
            || (!pvbyte_histogram.empty());
    }
};


Statistics statistics;
