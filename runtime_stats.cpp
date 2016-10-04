class Statistics {
public:
    size_t partition_calls;
    size_t items_processed;
    size_t scalar__partition_calls;
    size_t scalar__items_processed;

public:
    Statistics() {
        reset();
    }

    void reset() {
        partition_calls = 0;
        items_processed = 0;
        scalar__partition_calls = 0;
        scalar__items_processed = 0;
    }

    bool anything_collected() const {
        return (partition_calls > 0)
            || (items_processed > 0)
            || (scalar__partition_calls > 0)
            || (scalar__items_processed > 0);
    }
};


Statistics statistics;
