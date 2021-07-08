#pragma once

#include <internal/handle.hpp>
#include <internal/common.hpp>

#include <tools/missing_implementations.hpp>

#include <type_traits>
#include <future>
#include <vector>


namespace hash {
    using namespace usm_smart_ptr;


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




