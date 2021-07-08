#pragma once

#include <hash_functions/sha256.hpp>
#include <hash_functions/keccak.hpp>
#include <hash_functions/blake2b.hpp>
#include <hash_functions/md5.hpp>
#include <hash_functions/md2.hpp>
#include <hash_functions/sha1.hpp>

#include <type_traits>
#include <tools/missing_implementations.hpp>
#include <future>
#include <vector>

#include <handle.hpp>

namespace hash {

    /**
     * Defines the various types of hashes supported.
     */
    enum class method {
        sha256,
        keccak,
        blake2b,
        sha1,
        sha3,
        md5,
        md2
    };

    /**
     * A runner is composed of a queue and a double which
     * represents the device performance on a given algorithm
     * and a given confiiguration.
     */
    struct runner {
        sycl::queue q /** Queue to run on */;
        double d /** Queue performance on an algorithm */;
    };


    using runners = std::vector<runner>;

    /**
     * A dummy struct used to abort the compilation
     * if in a  'if constexpr' ladder nothing has
     * matched
     * @tparam M
     */
    template<method M>
    struct nothing_matched : std::false_type {
    };


    /**
     * Returns the size of the hash result, in bytes, produced by a hashing function.
     * @tparam M the hashing function we want
     * @return
     */
    template<method M, typename = std::enable_if_t<M != method::keccak && M != method::blake2b && M != method::sha3> >
    static constexpr size_t get_block_size() {
        if constexpr(M == method::sha256) {
            return SHA256_BLOCK_SIZE;
        } else if constexpr (M == method::md5) {
            return MD5_BLOCK_SIZE;
        } else if constexpr(M == method::md2) {
            return MD2_BLOCK_SIZE;
        } else if constexpr(M == method::sha1) {
            return SHA1_BLOCK_SIZE;
        } else {
            static_assert(nothing_matched<M>::value);
        }
    }

    /**
     * Returns the size of the hash result, in bytes, produced by a hashing function.
     * @tparam M the hashing function we want
     * @tparam n_outbit
     * @return
     */
    template<hash::method M, int n_outbit>
    static constexpr size_t get_block_size() {
        if constexpr (M == hash::method::keccak && (n_outbit == 128 || n_outbit == 224 || n_outbit == 256 || n_outbit == 288 || n_outbit == 384 || n_outbit == 512)) {
            return n_outbit >> 3;
        } else if constexpr (M == hash::method::sha3 && (n_outbit == 224 || n_outbit == 256 || n_outbit == 384 || n_outbit == 512)) {
            return n_outbit >> 3;
        } else if constexpr (M == hash::method::blake2b) {
            return n_outbit >> 3;
        } else {
            return get_block_size<M>();
        }
    }

    /**
     * Returns the name of a hash function
     * @tparam M
     * @tparam n_outbit
     * @return
     */
    template<method M, int n_outbit = 0>
    static std::string get_name() {
        if constexpr(M == method::sha256) {
            return {"sha256"};
        } else if constexpr(M == method::md5) {
            return {"md5"};
        } else if constexpr(M == method::md2) {
            return {"md2"};
        } else if constexpr(M == method::sha1) {
            return {"sha1"};
        } else if constexpr(M == method::keccak) {
            return {"keccak"};
        } else if constexpr(M == method::sha3) {
            return {"sha3"};
        } else if constexpr(M == method::blake2b) {
            return {"blake2b"};
        } else {
            static_assert(nothing_matched<M>::value);
        }
    }


    using namespace usm_smart_ptr;
    namespace internal {

        /**
         * Function used to launch the hashing kernel.
         * One must ensure that the memory can be read by the device.
         * @tparam M The hash function one wants to execute
         * @tparam n_outbit The size of the output if applicable.
         * @tparam buffers Pack of buffers/extra arguments which will be passed to the kernel.
         * Principally used to pass the constant buffers from the outside so the call is not blocking.
         * If no buffers are passed, but needed, the function is blocking on some SYCL implementations.
         * @return A SYCL event.
         */
        template<method M, int n_outbit, typename... buffers>
        [[nodiscard]] sycl::event
        dispatch_hash(sycl::queue &q, const sycl::event &e, device_accessible_ptr<byte> indata, device_accessible_ptr<byte> outdata, dword inlen, dword n_batch, const byte *key, dword keylen,
                      buffers... bufs) {
            if (n_batch == 0) return sycl::event{};
            if constexpr(M == method::sha256) {
                return launch_sha256_kernel(q, e, indata, outdata, inlen, n_batch, bufs...);
            } else if constexpr(M == method::md5) {
                return launch_md5_kernel(q, e, indata, outdata, inlen, n_batch);
            } else if constexpr(M == method::md2) {
                return launch_md2_kernel(q, e, indata, outdata, inlen, n_batch, bufs...);
            } else if constexpr(M == method::sha1) {
                return launch_sha1_kernel(q, e, indata, outdata, inlen, n_batch);
            } else if constexpr(M == method::keccak && (n_outbit == 128 || n_outbit == 224 || n_outbit == 256 || n_outbit == 288 || n_outbit == 384 || n_outbit == 512)) {
                return launch_keccak_kernel(false, q, e, indata, outdata, inlen, n_batch, n_outbit, bufs...);
            } else if constexpr(M == method::sha3 && (n_outbit == 224 || n_outbit == 256 || n_outbit == 384 || n_outbit == 512)) {
                return launch_keccak_kernel(true, q, e, indata, outdata, inlen, n_batch, n_outbit, bufs...);
            } else if constexpr (M == method::blake2b) {
                return launch_blake2b_kernel(q, e, indata, outdata, inlen, n_batch, n_outbit, key, keylen, bufs...);
            } else {
                static_assert(nothing_matched<M>::value);
            }
        }

        /**
         * This struct represents an item of work to be executed on one queue.
         */
        struct queue_work {
            sycl::queue q /** Queue that has to perform the work */;
            const byte *input_data /** Pointer to the input data -- not managed by SYCL */;
            byte *output_data /** Pointer to the output data -- not managed by SYCL */;
            size_t batch_size /** Number of hashes to compute on the given memory */;
            dword inlen /** Length of one batch to hash */;
        };

        /**
         * Launches a hash kernel asynchronously (if possible, depends on your implementation) is memory copy is needed
         * @tparam M type of hash to perform
         * @tparam n_outbit number of bits to output, if applicable
         * @tparam buffers types of the buffers to pass to the kernel
         * @param q_work contanins pointers, sizes, queue and offset provided by the user
         * @param inlen length of an element to hash
         * @param key ptr to the key if applicable
         * @param keylen number of bytes in the key
         * @param bufs variadic list of the buffers to pass to the kernel
         * @return a handle struct that holds the unique pointers to the data used by the device that's running and the event that indicates wheter a device fiinished running
         */
        template<hash::method M, int n_outbit, typename... buffers>
        [[nodiscard]] hash::handle_item hash_with_data_copy(hash::internal::queue_work q_work, const byte *key, dword keylen, buffers... bufs) {
            auto[q, in_ptr, out_ptr, batch_size, inlen] = std::move(q_work);
#ifdef USING_COMPUTECPP
            auto device_indata = hash::make_unique_ptr<byte, hash::alloc::device>(inlen * batch_size + (inlen ? 0 : 1), q); // TODO ComputeCpp runtime throws error when ptr size is 0
#else
            auto device_indata = hash::make_unique_ptr<byte, hash::alloc::device>(inlen * batch_size, q);
#endif
            auto device_outdata = hash::make_unique_ptr<byte, hash::alloc::device>(hash::get_block_size<M, n_outbit>() * batch_size, q);
            sycl::event memcpy_in_e = inlen ? q.memcpy(device_indata.raw(), in_ptr, inlen * batch_size) : sycl::event{};
            sycl::event submission_e = hash::internal::dispatch_hash<M, n_outbit>(q, memcpy_in_e, device_indata.get(), device_outdata.get(), inlen, batch_size, key, keylen, bufs...);
            sycl::event memcpy_out_e = memcpy_with_dependency(q, out_ptr, device_outdata.raw(), hash::get_block_size<M, n_outbit>() * batch_size, submission_e);
            return {std::move(device_indata), std::move(device_outdata), memcpy_out_e};
        }


        /**
         * Breaks the work space into batches which will run on the various queues. It takes into account the
         * device performance.
         * @return
         */
        template<method M, int n_outbit = 0>
        [[nodiscard]] static std::vector<queue_work> get_hash_queue_work_item(const runners &v, const byte *in, dword inlen, byte *out, dword n_batch) {
            size_t len = v.size();
            std::vector<queue_work> out_vector(len);
            std::vector<size_t> batch_offsets(len + 1);
            double coefs_sum = 0;
            for (const auto &elt : v) {
                coefs_sum += elt.d;
            }
            for (size_t i = 0, prev_offset = 0; i < len; ++i) {
                prev_offset = (size_t) ((double) n_batch * v[i].d / coefs_sum) + prev_offset;
                batch_offsets[i + 1] = prev_offset;
            }
            batch_offsets[len] = n_batch;

            for (size_t i = 0; i < len; ++i) {
                const byte *in_ptr = in + batch_offsets[i] * inlen;
                byte *out_ptr = out + batch_offsets[i] * get_block_size<M, n_outbit>();
                size_t batch_size = batch_offsets[i + 1] - batch_offsets[i];
                out_vector[i] = {v[i].q, in_ptr, out_ptr, batch_size, inlen};
            }
            return out_vector;
        }
    }


    /**
     * Computes synchronously a hash.
     * @tparam M Hash method
     * @param q Queue to run on
     * @param in Pointer to the input data in any memory accessible by the HOST. Contains an array of data.
     * @param inlen Size in bytes of one block to hash.
     * @param out Pointer to the output memory accessible by the HOST
     * @param n_batch Number of blocks to hash. In and Out pointers must have correct sizes.
     */
    template<method M, typename = std::enable_if_t<M != method::keccak && M != method::sha3 && M != method::blake2b> >
    static void compute(sycl::queue &q, const byte *in, dword inlen, byte *out, dword n_batch) {
        internal::hash_with_data_copy<M>({q, in, out, n_batch, inlen}, nullptr, 0).dev_e_.wait();
    }

    /**
     * Computes synchronously a hash.
     * @tparam M Hash method
     * @tparam n_outbit Number of bits to output
     * @param q Queue to run on
     * @param in Pointer to the input data in any memory accessible by the HOST. Contains an array of data.
     * @param inlen Size in bytes of one block to hash.
     * @param out Pointer to the output memory accessible by the HOST
     * @param n_batch Number of blocks to hash. In and Out pointers must have correct sizes.
     */
    template<method M, int n_outbit, typename = std::enable_if_t<M == method::keccak || M == method::sha3 >>
    static void compute(sycl::queue &q, const byte *in, dword inlen, byte *out, dword n_batch) {
        internal::hash_with_data_copy<M, n_outbit>({q, in, out, n_batch, inlen}, nullptr, 0).dev_e_.wait();
    }

    /**
     * Computes synchronously a hash.
     * @tparam M Hash method
     * @tparam n_outbit Number of bits to output
     * @param q Queue to run on
     * @param in Pointer to the input data in any memory accessible by the HOST. Contains an array of data.
     * @param inlen Size in bytes of one block to hash.
     * @param out Pointer to the output memory accessible by the HOST
     * @param n_batch Number of blocks to hash. In and Out pointers must have correct sizes.
     */
    template<method M, int n_outbit, typename = std::enable_if_t<M == method::blake2b>>
    static void compute(sycl::queue &q, const byte *in, dword inlen, byte *out, dword n_batch, byte *key, dword keylen) {
        internal::hash_with_data_copy<M, n_outbit>({q, in, out, n_batch, inlen}, key, keylen).dev_e_.wait();
    }


/**
 * Computes synchronously a hash.
 * This overload does not perform any memory operation. We assume memory is accessible in read and write by the
 * device attached to the queue.
 * @tparam M Hash method
 * @param q Queue to run on
 * @param in Pointer to the input data in any memory accessible by the QUEUE/CONTEXT PROVIDED. Contains an array of data.
 * @param inlen Size in bytes of one block to hash.
 * @param out Pointer to the output memory accessible by the QUEUE/CONTEXT PROVIDED
 * @param n_batch Number of blocks to hash. In and Out pointers must have correct sizes.
 */
    template<method M, typename = std::enable_if_t<M != method::keccak && M != method::sha3 && M != method::blake2b>>
    static void compute(sycl::queue &q, device_accessible_ptr<byte> indata, dword inlen, device_accessible_ptr<byte> outdata, dword n_batch) {
        internal::dispatch_hash<M, 0>(q, sycl::event{}, indata, outdata, inlen, n_batch, nullptr, 0).wait();
    }


    /**
     * Computes synchronously a hash.
     * This overload does not perform any memory operation. We assume memory is accessible in read and write by the
     * device attached to the queue.
     * @tparam M Hash method
     * @tparam n_outbit Number of bits to output
     * @param q Queue to run on
     * @param in Pointer to the input data in any memory accessible by the QUEUE/CONTEXT PROVIDED. Contains an array of data.
     * @param inlen Size in bytes of one block to hash.
     * @param out Pointer to the output memory accessible by the QUEUE/CONTEXT PROVIDED
     * @param n_batch Number of blocks to hash. In and Out pointers must have correct sizes.
     */
    template<method M, int n_outbit, typename = std::enable_if_t<M == method::keccak || M == method::sha3>>
    static void compute(sycl::queue &q, const device_accessible_ptr<byte> indata, dword inlen, device_accessible_ptr<byte> outdata, dword n_batch) {
        internal::dispatch_hash<M, n_outbit>(q, sycl::event{}, indata, outdata, inlen, n_batch, nullptr, 0).wait();
    }


    /**
     * Computes synchronously a hash.
     * This overload does not perform any memory operation. We assume memory is accessible in read and write by the
     * device attached to the queue.
     * @tparam M Hash method
     * @tparam n_outbit Number of bits to output
     * @param q Queue to run on
     * @param in Pointer to the input data in any memory accessible by the QUEUE/CONTEXT PROVIDED. Contains an array of data.
     * @param inlen Size in bytes of one block to hash.
     * @param out Pointer to the output memory accessible by the QUEUE/CONTEXT PROVIDED
     * @param n_batch Number of blocks to hash. In and Out pointers must have correct sizes.
     */
    template<method M, int n_outbit, typename = std::enable_if_t<M == method::blake2b >>
    static void compute(sycl::queue &q, const device_accessible_ptr<byte> indata, dword inlen, device_accessible_ptr<byte> outdata, dword n_batch, const byte *key, dword keylen) {
        internal::dispatch_hash<M, n_outbit>(q, sycl::event{}, indata, outdata, inlen, n_batch, key, keylen).wait();
    }


} //namespace hash::v_1




