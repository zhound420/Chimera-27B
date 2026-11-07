#include "ggml.h"
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <dlfcn.h>
static std::atomic<unsigned long> overrides{0};
static int cutoff() {
    static const int value = [] {
        const char * text = std::getenv("HYBRID_MMVQ_CUTOFF");
        const int n = text ? std::atoi(text) : 8;
        if (n < 1 || n > 8) std::abort();
        return n;
    }();
    return value;
}
bool ggml_cuda_should_use_mmvq(ggml_type type, int cc, int64_t columns) {
    using original_fn = bool (*)(ggml_type, int, int64_t);
    static auto original = reinterpret_cast<original_fn>(dlsym(RTLD_NEXT, "_Z25ggml_cuda_should_use_mmvq9ggml_typeil"));
    if (!original) std::abort();
    if (cc == 860 && columns > cutoff() && columns <= 8 &&
        (type == GGML_TYPE_Q4_K || type == GGML_TYPE_Q5_K || type == GGML_TYPE_Q6_K)) {
        overrides.fetch_add(1, std::memory_order_relaxed);
        return false;
    }
    return original(type, cc, columns);
}
__attribute__((destructor)) static void report() {
    if (overrides.load()) std::fprintf(stderr, "hybrid MMVQ cutoff=%d, MMQ dispatch overrides=%lu\n", cutoff(), overrides.load());
}
