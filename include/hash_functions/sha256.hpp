#pragma once

#include <internal/config.hpp>
#include <tools/usm_smart_ptr.hpp>

/****************************** MACROS ******************************/
constexpr dword SHA256_BLOCK_SIZE = 32;            // SHA256 outputs a 32 byte digest

namespace hash::internal {
    class sha256_kernel;

    using namespace usm_smart_ptr;

    sycl::buffer<dword, 1> get_sha256_buffer();

    sycl::event launch_sha256_kernel(sycl::queue &q, sycl::event e, device_accessible_ptr<byte> indata, device_accessible_ptr<byte> outdata, dword inlen, dword n_batch);

    sycl::event
    launch_sha256_kernel(sycl::queue &q, sycl::event e, device_accessible_ptr<byte> indata, device_accessible_ptr<byte> outdata, dword inlen, dword n_batch, sycl::buffer<dword, 1> &buffer);

}