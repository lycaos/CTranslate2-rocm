#include <ctranslate2/ops/awq/gemv.h>

namespace ctranslate2 {
  namespace ops {
    // AWQ uses CUDA-specific cuBLAS and inline assembly not compatible with ROCm/HIP
    template <Device D, typename T, typename WeightT>
    void GemvAwq::compute_gemv(const StorageView&,
                               const StorageView&,
                               const StorageView&,
                               const StorageView&,
                               StorageView&) const {
      throw std::runtime_error("AWQ GEMV is not supported on ROCm/HIP - requires CUDA-specific cuBLAS operations");
    }

    template <Device D, typename T, typename WeightT>
    void GemvAwq::compute_gemv2(const StorageView&,
                                const StorageView&,
                                const StorageView&,
                                const StorageView&,
                                StorageView&) const {
      throw std::runtime_error("AWQ GEMV2 is not supported on ROCm/HIP - requires CUDA-specific cuBLAS operations");
    }

#define DECLARE_IMPL(T)                                                 \
    template void                                                       \
    GemvAwq::compute_gemv<Device::CUDA, T, int>(                        \
      const StorageView&,                                               \
      const StorageView&,                                               \
      const StorageView&,                                               \
      const StorageView&,                                               \
      StorageView&) const;                                              \
    template void                                                       \
    GemvAwq::compute_gemv2<Device::CUDA, T, int>(                       \
      const StorageView&,                                               \
      const StorageView&,                                               \
      const StorageView&,                                               \
      const StorageView&,                                               \
      StorageView&) const;

    DECLARE_IMPL(float16_t)
  }
}
