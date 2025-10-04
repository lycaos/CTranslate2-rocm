#include <ctranslate2/ops/awq/gemm.h>

namespace ctranslate2 {
  namespace ops {
    // AWQ uses CUDA-specific cuBLAS and inline assembly not compatible with ROCm/HIP
    template <Device D, typename T, typename WeightT>
    void GemmAwq::compute(const StorageView&,
                         const StorageView&,
                         const StorageView&,
                         const StorageView&,
                         StorageView&) const {
      throw std::runtime_error("AWQ GEMM is not supported on ROCm/HIP - requires CUDA-specific cuBLAS operations");
    }

#define DECLARE_IMPL(T)                                                 \
    template void                                                       \
    GemmAwq::compute<Device::CUDA, T, int>(                             \
      const StorageView&,                                               \
      const StorageView&,                                               \
      const StorageView&,                                               \
      const StorageView&,                                               \
      StorageView&) const;

    DECLARE_IMPL(float16_t)
  }
}
