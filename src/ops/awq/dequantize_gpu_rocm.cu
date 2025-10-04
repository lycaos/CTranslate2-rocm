#include <ctranslate2/ops/awq/dequantize_awq.h>

namespace ctranslate2 {
  namespace ops {
    // AWQ uses CUDA-specific inline assembly that is not compatible with ROCm/HIP
    // Provide stub implementation that throws runtime error
    template <Device D, typename InT, typename OutT>
    void DequantizeAwq::dequantize(const StorageView&,
                                   const StorageView&,
                                   const StorageView&,
                                   StorageView&) const {
      throw std::runtime_error("AWQ dequantize is not supported on ROCm/HIP - requires CUDA-specific inline assembly");
    }

#define DECLARE_IMPL(T)                                                 \
    template void                                                       \
    DequantizeAwq::dequantize<Device::CUDA, int, T>(                    \
      const StorageView&,                                               \
      const StorageView&,                                               \
      const StorageView&,                                               \
      StorageView&) const;

    DECLARE_IMPL(float16_t)
  }
}
