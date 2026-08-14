// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// Native gather kernel for the DeepSeek V4 fp8_ds_mla paged K cache.
//
// Replaces the torch advanced-indexing gather in `gather_ds_mla_slots_torch`
// (cache_utils.py) with the pre-built XPU `index_select` / `gather_nd`
// kernels shipped in torch_xmlir's libxpuapi.so.
//
// NOTE: the shipped headers declare the kernel API under `xpytorch::xpu::api`,
// but libxpuapi.so actually exports the instantiations under
// `baidu::xpu::api` (an SDK header/library version skew). We bridge the two
// with a forward declaration + an opaque pointer cast; Context is only ever
// passed through, never dereferenced here, so the ABI is identical.
#include <torch/extension.h>
#include <ATen/ATen.h>

#include <vector>

// The shipped headers declare the kernel API under `xpytorch::xpu::api`, but
// libxpuapi.so actually exports the kernels under `baidu::xpu::api` (an SDK
// header/library version skew). Bridge with forward declarations; Context is
// only ever passed through (never dereferenced) so the ABI is opaque.
namespace baidu {
namespace xpu {
namespace api {

struct Context;

template <typename T>
struct VectorParam {
    const T* cpu;
    int64_t len;
    T* xpu;
};

Context* create_context();
void destroy_context(Context* ctx);

template <typename T, typename TID>
int index_select(Context* ctx, const T* x, const TID* index, T* y,
                 const std::vector<int64_t>& xshape, int64_t index_len, int64_t axis);

template <typename T, typename TID>
int gather_nd(Context* ctx, const T* x, const TID* index, T* y,
              const VectorParam<int64_t>& xshape, const std::vector<int64_t>& index_shape);

}  // namespace api
}  // namespace xpu
}  // namespace baidu

namespace {

// One Context per thread; the kernels run on the Context's default stream.
baidu::xpu::api::Context* get_ctx() {
    static thread_local baidu::xpu::api::Context* ctx =
        baidu::xpu::api::create_context();
    return ctx;
}

// Native index_select on axis 0: y[i, :] = x[index[i], :].
at::Tensor index_select_u8_impl(at::Tensor x, at::Tensor index) {
    TORCH_CHECK(x.dim() == 2, "x must be 2D");
    TORCH_CHECK(x.scalar_type() == at::kByte, "x must be uint8");
    TORCH_CHECK(index.dim() == 1, "index must be 1D");
    auto y = at::empty({index.numel(), x.size(1)}, x.options());
    auto* ctx = get_ctx();
    std::vector<int64_t> xshape = {x.size(0), x.size(1)};
    int ret = baidu::xpu::api::index_select<uint8_t, int64_t>(
        ctx, x.data_ptr<uint8_t>(), index.data_ptr<int64_t>(),
        y.data_ptr<uint8_t>(), xshape, index.numel(), /*axis=*/0);
    TORCH_CHECK(ret == 0, "index_select failed with ", ret);
    return y;
}

// Native gather_nd: y[i0..ik] = x[index[i0..ik, :]].
at::Tensor gather_nd_u8_impl(at::Tensor x, at::Tensor index) {
    TORCH_CHECK(x.dim() == 2, "x must be 2D");
    TORCH_CHECK(x.scalar_type() == at::kByte, "x must be uint8");
    TORCH_CHECK(index.dim() >= 2 && index.size(-1) == 2, "index must be [..., 2]");
    auto y = at::empty(index.sizes().slice(0, index.dim() - 1), x.options());
    auto* ctx = get_ctx();
    int64_t xs[2] = {x.size(0), x.size(1)};
    baidu::xpu::api::VectorParam<int64_t> xshape_vp{xs, 2, nullptr};
    std::vector<int64_t> idxshape = index.sizes().vec();
    int ret = baidu::xpu::api::gather_nd<uint8_t, int64_t>(
        ctx, x.data_ptr<uint8_t>(), index.data_ptr<int64_t>(),
        y.data_ptr<uint8_t>(), xshape_vp, idxshape);
    TORCH_CHECK(ret == 0, "gather_nd failed with ", ret);
    return y;
}

}  // namespace

TORCH_LIBRARY(ds_mla_gather, m) {
    m.def("index_select_u8(Tensor x, Tensor index) -> Tensor");
    m.def("gather_nd_u8(Tensor x, Tensor index) -> Tensor");
}

TORCH_LIBRARY_IMPL(ds_mla_gather, CUDA, m) {
    m.impl("index_select_u8", &index_select_u8_impl);
    m.impl("gather_nd_u8", &gather_nd_u8_impl);
}
