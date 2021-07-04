#pragma once

#include <sycl/sycl.hpp>

static inline sycl::event memcpy_with_dependency(sycl::queue &q, void *dest, const void *src, size_t numBytes, sycl::event depEvent) {
    return q.submit([=](sycl::handler &cgh) {
        cgh.depends_on(depEvent);
        cgh.memcpy(dest, src, numBytes);
    });
}

static inline sycl::event memcpy_with_dependency(sycl::queue &q, void *dest, const void *src, size_t numBytes, const std::vector<sycl::event> &depEvent) {
    return q.submit([&](sycl::handler &cgh) {
        cgh.depends_on(depEvent);
        cgh.memcpy(dest, src, numBytes);
    });
}

template<class T, int Dim>
using local_accessor = sycl::accessor<T, Dim, sycl::access::mode::read_write, sycl::access::target::local>;

template<class T, int Dim>
using constant_accessor = sycl::accessor<T, Dim, sycl::access::mode::read, sycl::access::target::constant_buffer>;