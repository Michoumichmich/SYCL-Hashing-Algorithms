/**
    Copyright 2021 Codeplay Software Ltd.

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use these files except in compliance with the License.
    You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

    For your convenience, a copy of the License has been included in this
    repository.

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.

    @author Michel Migdal.
 */


#pragma once

#include <SYCL/sycl.hpp>
#include <type_traits>
#include <array>
#include <tools/intrinsics.hpp>


namespace sbb {

    namespace runtime_idx_detail {

        template<typename T, typename array_t, int N, int idx_max = N - 1>
        static inline constexpr void runtime_index_wrapper_internal_store_byte(array_t &arr, const uint &word_idx, const uint8_t &byte_in, const uint &byte_idx) {
            static_assert(idx_max >= 0 && idx_max < N);
#pragma unroll
            for (uint i = 0; i < N; ++i) {
                arr[i] = (word_idx == i) ? set_byte(arr[i], byte_in, byte_idx) : arr[i];
            }
        }


        template<typename T, typename array_t, int N, int idx_max = N - 1>
        [[nodiscard]] static inline constexpr T runtime_index_wrapper_internal_read_copy(const array_t &arr, const uint &i) {
            static_assert(idx_max >= 0 && idx_max < N);
            if constexpr (idx_max == 0 || N == 1) {
                return arr[0];
            } else {
                if (i == idx_max) {
                    return arr[idx_max];
                } else {
                    return runtime_index_wrapper_internal_read_copy<T, array_t, N, idx_max - 1>(arr, i);
                }
            }
        }


    }


/**
 * STD::ARRAY
 */

    template<typename T, size_t N>
    static inline constexpr uint8_t runtime_index_wrapper_store_byte(std::array<T, N> &array, const uint &i, const uint8_t &val, const uint &byte_idx) {
        runtime_idx_detail::runtime_index_wrapper_internal_store_byte<T, std::array<T, N>, N>(array, i, (T) val, byte_idx);
        return val;
    }


    template<typename T, size_t N>
    [[nodiscard]] static inline constexpr T runtime_index_wrapper(const std::array<T, N> &array, const uint &i) {
        return runtime_idx_detail::runtime_index_wrapper_internal_read_copy<T, std::array<T, N>, N>(array, i);
    }


}