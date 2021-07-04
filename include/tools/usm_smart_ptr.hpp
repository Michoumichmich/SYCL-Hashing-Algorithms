#pragma once

#include <sycl/sycl.hpp>
#include <memory>
#include <utility>
#include <type_traits>

#ifdef USING_COMPUTECPP
namespace cl::sycl::usm {
    using cl::sycl::experimental::usm::alloc;
}
#endif


#include <tools/missing_implementations.hpp>


namespace usm_smart_ptr {
    using namespace sycl::usm;

    template<class T, sycl::usm::alloc Tag>
    /* template <sycl::usm::alloc location>
    concept host_accessible = location == sycl::usm::alloc::host || location == sycl::usm::alloc::shared; */
    struct usm_ptr {
        explicit usm_ptr(T *t) : val_(t) {}

        /**
         * Explicit conversion needed if the memory is not shared
         * @return
         */
        explicit operator T *() const noexcept { return val_; }
        //explicit(Tag != sycl::usm::alloc::shared) operator T *() const noexcept { return val_; }

    private:
        T *val_;
    };


    template<typename T>
    struct device_accessible_ptr {
        explicit device_accessible_ptr(T *p) : val_((T *) p) {};

        device_accessible_ptr(usm_ptr<T, alloc::shared> p) : val_((T *) p) {};

        device_accessible_ptr(usm_ptr<T, alloc::device> p) : val_((T *) p) {};

        //device_accessible_ptr(usm_ptr<T, alloc::host> p): val_((T*)p){};
        operator T *() const noexcept { return val_; }

    private:
        T *val_;
    };

    template<typename T>
    struct host_accessible_ptr {
        host_accessible_ptr(usm_ptr<T, alloc::shared> p) : val_((T *) p) {};

        host_accessible_ptr(usm_ptr<T, alloc::host> p) : val_((T *) p) {};

        operator T *() const noexcept { return val_; }

    private:
        T *val_;
    };


/**
 * SYCL USM Deleter. The std::unique_ptr deleter takes only the pointer
 * to delete as an argument so that's the only work-around.
 */
    template<typename T>
    struct usm_deleter {
        sycl::queue q_;

        explicit usm_deleter(sycl::queue q) : q_(std::move(q)) {}

        void operator()(T *ptr) const

        noexcept {
            if (ptr)
                sycl::free(ptr, q_);
        }
    };

/**
 * Wrapper for a std::unique_ptr that calls the SYCL deleter (sycl::free).
 * Also holds the number of elements allocated.
 */
    template<typename T, sycl::usm::alloc location>
    class usm_unique_ptr : public std::unique_ptr<T, usm_deleter<T>> {
    private:
        size_t count_;
    public:
        usm_unique_ptr(T *ptr, usm_deleter<T> deleter, size_t count)
                : std::unique_ptr<T, usm_deleter<T>>(ptr, deleter) { count_ = count; }

        [[nodiscard]] inline size_t alloc_size() const noexcept { return count_ * sizeof(T); }

        [[nodiscard]] inline size_t alloc_count() const noexcept { return count_; }

        [[nodiscard]] inline usm_ptr<T, location> get() const noexcept {
            return usm_ptr<T, location>(std::unique_ptr<T, usm_deleter<T>>::get());
        }

        [[nodiscard]] inline T *raw() const noexcept {
            return std::unique_ptr<T, usm_deleter<T>>::get();
        }

    };

/**
 * Builds a usm_unique_ptr pointer
 * @tparam location indicates where is the memory allocated (device, host, or shared)
 */
    template<typename T, sycl::usm::alloc location>
    usm_unique_ptr<T, location> make_unique_ptr(size_t count, sycl::queue &q) {
        //return usm_unique_ptr<T>(sycl::usm_allocator < T, location > {q}.allocate(count), usm_deleter<T>{q}, count);
        if constexpr(location == alloc::shared)
            return usm_unique_ptr<T, location>(sycl::malloc_shared<T>(count, q), usm_deleter<T>{q}, count);
        else if constexpr(location == alloc::host)
            return usm_unique_ptr<T, location>(sycl::malloc_host<T>(count, q), usm_deleter<T>{q}, count);
        else if constexpr(location == alloc::device)
            return usm_unique_ptr<T, location>(sycl::malloc_device<T>(count, q), usm_deleter<T>{q}, count);
        else static_assert(!std::is_same_v<T, T>, "Invalid template parameter.");
    }


    template<typename T, sycl::usm::alloc location>
    usm_unique_ptr<T, location> make_unique_ptr(sycl::queue &q, T *ptr) {
        return usm_unique_ptr<T, location>(ptr, usm_deleter<T>{q}, 1);
    }

    template<typename T, sycl::usm::alloc location>
    usm_unique_ptr<T, location> make_unique_ptr(sycl::queue &q) {
        return make_unique_ptr<T, location>(1, q);
    }


/**
 * Same interface as usm_unique_ptr
 * @tparam T
 */
    template<typename T, sycl::usm::alloc location>
    class usm_shared_ptr : public std::shared_ptr<T> {
    private:
        size_t count_;

    public:
        usm_shared_ptr(T *ptr, usm_deleter<T> deleter, size_t count) : std::shared_ptr<T>(ptr, deleter) { count_ = count; }

        [[nodiscard]] inline size_t alloc_size() const noexcept { return count_ * sizeof(T); }

        [[nodiscard]] inline size_t alloc_count() const noexcept { return count_; }

        [[nodiscard]] inline usm_ptr<T, location> get() const noexcept {
            return usm_ptr<T, location>(std::shared_ptr<T>::get());
        }

        [[nodiscard]] inline T *raw() const noexcept {
            return std::shared_ptr<T>::get();
        }

    };

    template<typename T, sycl::usm::alloc location>
    usm_shared_ptr<T, location> make_shared_ptr(size_t count, sycl::queue &q) {
        //return usm_shared_ptr<T>(sycl::usm_allocator < T, location > {q}.allocate(count), usm_deleter<T>{q}, count);
        if constexpr(location == alloc::shared)
            return usm_shared_ptr<T, location>(sycl::malloc_shared<T>(count, q), usm_deleter<T>{q}, count);
        else if constexpr(location == alloc::host)
            return usm_shared_ptr<T, location>(sycl::malloc_host<T>(count, q), usm_deleter<T>{q}, count);
        else if constexpr(location == alloc::device)
            return usm_shared_ptr<T, location>(sycl::malloc_device<T>(count, q), usm_deleter<T>{q}, count);
        else static_assert(!std::is_same_v<T, T>, "Invalid template parameter.");
    }

    template<typename T, sycl::usm::alloc location>
    usm_shared_ptr<T, location> make_shared_ptr(sycl::queue &q) {
        return make_shared_ptr<T, location>(1, q);
    }
}