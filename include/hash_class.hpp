#pragma once

#include <utility>
#include "sycl_hash.hpp"
#include "handle.hpp"

namespace hash {
    /**
     * Base class for hashinig
     * @tparam M
     * @tparam n_outbit
     */
    template<hash::method M, int n_outbit = 0>
    class hasher {
    private:
        runners runners_;
    public:
        explicit hasher(runners v) : runners_(std::move(v)) {}

        handle hash(const byte *indata, dword inlen, byte *outdata, dword n_batch, byte *key, dword keylen) {
            size_t size = runners_.size();
            std::vector<handle_item> handles;
            handles.reserve(size);
            auto items = internal::get_hash_queue_work_item<M, n_outbit>(runners_, indata, inlen, outdata, n_batch);
            for (size_t i = 0; i < size; ++i) {
                handles.emplace_back(internal::hash_launcher<M, n_outbit>(items[i], inlen, key, keylen));
            }
            return handle(std::move(handles));
        }

        handle hash(const byte *indata, dword inlen, byte *outdata, dword n_batch) {
            return hash(indata, inlen, outdata, n_batch, nullptr, 0);
        }


    };


    /**
     * SHA 256 Specialization for the buffer management.
     */
    template<>
    class hasher<method::sha256> {

    private:
        runners runners_;
        std::vector<sycl::buffer<dword, 1> > buffers_;
    public:
        explicit hasher(const runners &v) : runners_(v) {
            size_t size = v.size();
            buffers_.reserve(size);
            for (size_t i = 0; i < size; ++i) {
                buffers_.emplace_back(internal::get_sha256_buffer());
            }
        }

        handle hash(const byte *indata, dword inlen, byte *outdata, dword n_batch) {
            size_t size = runners_.size();
            std::vector<handle_item> handles;
            handles.reserve(size);
            auto items = hash::internal::get_hash_queue_work_item<method::sha256, 0>(runners_, indata, inlen, outdata, n_batch);
            for (size_t i = 0; i < size; ++i) {
                handles.emplace_back(internal::hash_launcher<method::sha256, 0, sycl::buffer<dword, 1> >(items[i], inlen, nullptr, 0, buffers_[i]));
            }
            return handle(std::move(handles));
        }
    };


    /**
     * MD2 spec
     */
    template<>
    class hasher<method::md2> {

    private:
        runners runners_;
        std::vector<sycl::buffer<byte, 1> > buffers_;
    public:
        explicit hasher(const runners &v) : runners_(v) {
            size_t size = v.size();
            buffers_.reserve(size);
            for (size_t i = 0; i < size; ++i) {
                buffers_.emplace_back(internal::get_buf_md2_consts());
            }
        }

        handle hash(const byte *indata, dword inlen, byte *outdata, dword n_batch) {
            size_t size = runners_.size();
            std::vector<handle_item> handles;
            handles.reserve(size);
            auto items = internal::get_hash_queue_work_item<method::md2, 0>(runners_, indata, inlen, outdata, n_batch);
            for (size_t i = 0; i < size; ++i) {
                handles.emplace_back(internal::hash_launcher<method::md2, 0, sycl::buffer<byte, 1> >(items[i], inlen, nullptr, 0, buffers_[i]));
            }
            return handle(std::move(handles));
        }
    };

    /**
     * KECCAK spec
     */
    template<int n_outbit>
    class hasher<method::keccak, n_outbit> {

    private:
        runners runners_;
        std::vector<sycl::buffer<qword, 1> >
                buffers_;
    public:
        explicit hasher(const runners &v) : runners_(v) {
            size_t size = v.size();
            buffers_.reserve(size);
            for (size_t i = 0; i < size; ++i) {
                buffers_.emplace_back(internal::get_buf_keccak_consts());
            }
        }

        handle hash(const byte *indata, dword inlen, byte *outdata, dword n_batch) {
            size_t size = runners_.size();
            std::vector<handle_item> handles;
            handles.reserve(size);
            auto items = internal::get_hash_queue_work_item<method::keccak, n_outbit>(runners_, indata, inlen, outdata, n_batch);
            for (size_t i = 0; i < size; ++i) {
                handles.emplace_back(internal::hash_launcher<method::keccak, n_outbit, sycl::buffer<qword, 1> >(items[i], inlen, nullptr, 0, buffers_[i]));
            }
            return handle(std::move(handles));
        }
    };

    /**
     * SHA3 spec
     * @tparam n_outbit
     */
    template<int n_outbit>
    class hasher<method::sha3, n_outbit> {

    private:
        runners runners_;
        std::vector<sycl::buffer<qword, 1>> buffers_;
    public:
        explicit hasher(const runners &v) : runners_(v) {
            size_t size = v.size();
            buffers_.reserve(size);
            for (size_t i = 0; i < size; ++i) {
                buffers_.emplace_back(internal::get_buf_keccak_consts());
            }
        }

        handle hash(const byte *indata, dword inlen, byte *outdata, dword n_batch) {
            size_t size = runners_.size();
            std::vector<handle_item> handles;
            handles.reserve(size);
            auto items = internal::get_hash_queue_work_item<method::sha3, n_outbit>(runners_, indata, inlen, outdata, n_batch);
            for (size_t i = 0; i < size; ++i) {
                handles.emplace_back(internal::hash_launcher<method::sha3, n_outbit, sycl::buffer<qword, 1> >(items[i], inlen, nullptr, 0, buffers_[i]));
            }
            return handle(std::move(handles));
        }
    };


    /**
     * Blake 2B
     * @tparam n_outbit
     */
    template<int n_outbit>
    class hasher<method::blake2b, n_outbit> {

    private:
        hash::runners runners_;
        std::vector<sycl::buffer<qword, 1> > buffers_ivs_;
        std::vector<sycl::buffer<byte, 2> > buffers_sigmas_;
        std::vector<usm_shared_ptr<blake2b_ctx, alloc::device>> keyed_ctxts_;
    public:
        explicit hasher(const hash::runners &v, const byte *key, dword keylen) : runners_(v) {
            size_t size = v.size();
            buffers_ivs_.reserve(size);
            buffers_sigmas_.reserve(size);
            keyed_ctxts_.reserve(size);

            for (size_t i = 0; i < size; ++i) {
                keyed_ctxts_.emplace_back(internal::get_blake2b_ctx(runners_[i].q, key, keylen, n_outbit));
            }

            for (size_t i = 0; i < size; ++i) {
                buffers_ivs_.emplace_back(internal::get_buf_ivs());
                buffers_sigmas_.emplace_back(internal::get_buf_sigmas());
            }
        }

        handle hash(const byte *indata, dword inlen, byte *outdata, dword n_batch) {
            size_t size = runners_.size();
            std::vector<handle_item> handles;
            handles.reserve(2 * size);
            auto items = internal::get_hash_queue_work_item<method::blake2b, n_outbit>(runners_, indata, inlen, outdata, n_batch);
            for (size_t i = 0; i < size; ++i) {
                handles.emplace_back(internal::hash_launcher<method::blake2b, n_outbit>(items[i], inlen, nullptr, 0, buffers_ivs_[i], buffers_sigmas_[i], keyed_ctxts_[i].get()));
            }
            return handle(std::move(handles));
        }
    };


    using md2 = hasher<hash::method::md2>;
    using md5 = hasher<hash::method::md5>;
    using sha1 = hasher<hash::method::sha1>;
    using sha256 = hasher<hash::method::sha256>;

    template<int n_outbit>
    using keccak = hasher<hash::method::keccak, n_outbit>;

    template<int n_outbit>
    using sha3 = hasher<hash::method::sha3, n_outbit>;

    template<int n_outbit>
    using blake2b = hasher<hash::method::blake2b, n_outbit>;
}
