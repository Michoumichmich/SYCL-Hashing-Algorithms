#pragma once

#include <sycl_hash.hpp>
#include <iomanip>
#include <iostream>
#include <cstring>
#include <tools/sycl_queue_helpers.hpp>

template<bool strict = true>
static inline sycl::queue try_get_queue_with_device(const sycl::device &in_dev) {
    auto exception_handler = [](const sycl::exception_list &exceptions) {
        for (std::exception_ptr const &e : exceptions) {
            try {
                std::rethrow_exception(e);
            }
            catch (sycl::exception const &e) {
                std::cout << "Caught asynchronous SYCL exception: " << e.what() << std::endl;
            }
            catch (std::exception const &e) {
                std::cout << "Caught asynchronous STL exception: " << e.what() << std::endl;
            }
        }
    };

    sycl::device dev;
    sycl::queue q;
    try {
        dev = in_dev;
        q = sycl::queue(dev, exception_handler);
        if constexpr (strict) {
            if (dev.is_cpu() || dev.is_gpu()) { //Only CPU and GPU not host, dsp, fpga, ?...
                queue_tester(q);
            }
        }
    }
    catch (...) {
        dev = sycl::device(sycl::host_selector());
        q = sycl::queue(dev, exception_handler);
        std::cout << "Warning: Expected device not found! Fall back on: " << dev.get_info<sycl::info::device::name>() << std::endl;
    }
    return q;
}


void print_hex(byte *ptr, dword len) {
    for (size_t i = 0; i < len; ++i) // only the first block
        std::cout << std::hex << std::setfill('0') << std::setw(2) << (int) (ptr[i]) << " ";
    std::cout << std::dec << std::endl << std::endl;
}

void duplicate(byte *in, byte *out, dword item_len, dword count) {
    for (size_t i = 0; i < count; ++i) {
        std::memcpy(out + item_len * i, in, item_len);
    }
}

template<typename Func>
void for_all_workers(Func f) {
    std::vector<sycl::device> devices = sycl::device::get_devices();
    {
        for (auto &dev: devices) {
            std::cout << "Running on: " << dev.get_info<sycl::info::device::name>() << std::endl;
            f(hash::runners(1, hash::runner{try_get_queue_with_device(dev), 1}));
        }
    }
}

template<typename Func>
void for_all_workers_pairs(Func f) {
    std::vector<sycl::device> devices = sycl::device::get_devices();
    {
        for (auto &dev1: devices) {
            for (auto &dev2: devices) {
                std::cout << "Running on: " << dev1.get_info<sycl::info::device::name>() << " and: " << dev2.get_info<sycl::info::device::name>() << std::endl;
                f({{try_get_queue_with_device(dev1), 1},
                   {try_get_queue_with_device(dev2), 1}});
            }

        }
    }
}