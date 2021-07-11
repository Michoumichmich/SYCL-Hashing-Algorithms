#pragma once

#include <sycl/sycl.hpp>
#include <iostream>
#include "../internal/common.hpp"

#ifdef USING_COMPUTECPP
class queue_kernel_tester;
namespace cl::sycl::usm{
    using cl::sycl::experimental::usm::alloc;
}
#endif

/**
 * Selects a CUDA device (but returns sometimes an invalid one)
 */
class cuda_selector : public sycl::device_selector {
public:
    int operator()(const sycl::device &device) const override {
        //return device.get_platform().get_backend() == sycl::backend::cuda && device.get_info<sycl::info::device::is_available>() ? 1 : -1;
        return device.is_gpu() && (device.get_info<sycl::info::device::driver_version>().find("CUDA") != std::string::npos) ? 1 : -1;
    }
};


void queue_tester(sycl::queue &q);


/**
 * Tries to get a queue from a selector else returns the host device
 * @tparam strict if true will check whether the queue can run a trivial task which implied
 * that the translation unit needs to be compiler with support for the device you're selecting.
 */
template<bool strict = true, typename T>
inline sycl::queue try_get_queue(const T &selector) {
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
        dev = sycl::device(selector);
        q = sycl::queue(dev, exception_handler);

        try {
            if constexpr (strict) {
                if (dev.is_cpu() || dev.is_gpu()) { //Only CPU and GPU not host, dsp, fpga, ?...
                    queue_tester(q);
                }
            }
        } catch (...) {
            std::cerr << "Warning: " << dev.get_info<sycl::info::device::name>() << " found but not working! Fall back on: ";
            dev = sycl::device(sycl::host_selector());
            q = sycl::queue(dev, exception_handler);
            std::cerr << dev.get_info<sycl::info::device::name>() << '\n';
            return q;
        }
    }
    catch (...) {

        dev = sycl::device(sycl::host_selector());
        q = sycl::queue(dev, exception_handler);
        std::cerr << "Warning: Expected device not found! Fall back on: " << dev.get_info<sycl::info::device::name>() << '\n';
    }
    return q;
}

template<typename T, bool debug = false>
inline bool is_ptr_usable(const T *ptr, const sycl::queue &q) {
    try {
        sycl::get_pointer_device(ptr, q.get_context());
        sycl::usm::alloc alloc_type = sycl::get_pointer_type(ptr, q.get_context());
        if constexpr(debug) {
            std::cerr << "Allocated on:" << q.get_device().get_info<sycl::info::device::name>() << " USM type: ";
            switch (alloc_type) {
                case sycl::usm::alloc::host:
                    std::cerr << "alloc::host" << '\n';
                    break;
                case sycl::usm::alloc::device:
                    std::cerr << "alloc::device" << '\n';
                    break;
                case sycl::usm::alloc::shared:
                    std::cerr << "alloc::shared" << '\n';
                    break;
                case sycl::usm::alloc::unknown:
                    std::cerr << "alloc::unknown" << '\n';
                    break;
            }
        }
        return alloc_type == sycl::usm::alloc::shared // Shared memory is cool
               || alloc_type == sycl::usm::alloc::device // Device memory is perfect
               || (q.get_device().is_host() && alloc_type != sycl::usm::alloc::unknown) // If we're on the host, anything known is OK.
            // || (q.get_device().is_cpu() && alloc_type != sycl::usm::alloc::unknown) // ???? is accessing host mem from CPU fine ?
                ;
    } catch (...) {
        if constexpr (debug) {
            std::cerr << "Not allocated on:" << q.get_device().get_info<sycl::info::device::name>() << '\n';
        }
        return false;
    }
}


/**
 * Usefull for memory bound computation.
 * Returns CPU devices that represents different numa nodes.
 * @return
 */
inline hash::runners get_cpu_runners_numa() {
    try {
        sycl::device d{sycl::cpu_selector{}};
        auto numa_nodes = d.create_sub_devices<sycl::info::partition_property::partition_by_affinity_domain>(sycl::info::partition_affinity_domain::numa);
        hash::runners runners_;
        std::transform(numa_nodes.begin(), numa_nodes.end(), runners_.begin(), [](auto &dev) -> hash::runner { return {try_get_queue(dev), 1}; });
        return runners_;
    }
    catch (...) {
        return {{sycl::queue{sycl::host_selector{}}, 1}};
    }
}
