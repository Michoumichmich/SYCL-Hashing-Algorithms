/**
 * Tests from https://github.com/B-Con/crypto-algorithms
 */

#include "tests_helpers.hpp"
#include <gtest/gtest.h>
#include "hash_class.hpp"

constexpr size_t loop_count = 229;


template<hash::method M, int ... args>
void
run_test(hash::runners &q, byte *input, size_t in_len, byte *expected_hash, size_t n_blocks, const byte *key = nullptr, qword keylen = 0) {
    byte *all_out = (byte *) malloc(hash::get_block_size<M, args...>() * n_blocks);
    byte *all_data = (byte *) malloc(in_len * n_blocks);
    duplicate(input, all_data, in_len, n_blocks);
    if constexpr(M == hash::method::blake2b) {
        hash::hasher<M, args...> hasher(q, key, keylen);
        hasher.hash(all_data, in_len, all_out, n_blocks).wait();
    } else {
        hash::hasher<M, args...> hasher(q);
        hasher.hash(all_data, in_len, all_out, n_blocks).wait();
    }

    for (size_t i = 0; i < n_blocks; ++i) {
        ASSERT_TRUE(!memcmp(expected_hash, all_out + hash::get_block_size<M, args...>() * i, hash::get_block_size<M, args...>()));
    }
    free(all_out);
    free(all_data);
}

void sha1_test(hash::runners &q, size_t count) {
    byte text1[] = {
            "abc"
    };
    byte text2[] = {"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq"};
    byte text3[] = {"sha1_on_sycl"};
    byte hash1[SHA1_BLOCK_SIZE] = {0xa9, 0x99, 0x3e, 0x36, 0x47, 0x06, 0x81, 0x6a, 0xba, 0x3e, 0x25, 0x71, 0x78, 0x50, 0xc2, 0x6c, 0x9c, 0xd0, 0xd8, 0x9d};
    byte hash2[SHA1_BLOCK_SIZE] = {0x84, 0x98, 0x3e, 0x44, 0x1c, 0x3b, 0xd2, 0x6e, 0xba, 0xae, 0x4a, 0xa1, 0xf9, 0x51, 0x29, 0xe5, 0xe5, 0x46, 0x70, 0xf1};
    byte hash3[SHA1_BLOCK_SIZE] = {0x26, 0x2e, 0xbf, 0xbd, 0x22, 0x78, 0x55, 0xbe, 0x7c, 0xe0, 0x53, 0x74, 0x14, 0x1e, 0x8c, 0xa3, 0x50, 0x42, 0xb8, 0x14};
    run_test<hash::method::sha1>(q, text1, strlen((char *) text1), hash1, count);
    run_test<hash::method::sha1>(q, text2, strlen((char *) text2), hash2, count);
    run_test<hash::method::sha1>(q, text3, strlen((char *) text3), hash3, count);
}

void md5_test(hash::runners &q, size_t count) {
    byte text1[] = {
            ""
    };
    byte text2[] = {"abc"};
    byte hash1[MD5_BLOCK_SIZE] = {0xd4, 0x1d, 0x8c, 0xd9, 0x8f, 0x00, 0xb2, 0x04, 0xe9, 0x80, 0x09, 0x98, 0xec, 0xf8, 0x42, 0x7e};
    byte hash2[MD5_BLOCK_SIZE] = {0x90, 0x01, 0x50, 0x98, 0x3c, 0xd2, 0x4f, 0xb0, 0xd6, 0x96, 0x3f, 0x7d, 0x28, 0xe1, 0x7f, 0x72};
    run_test<hash::method::md5>(q, text1, strlen((char *) text1), hash1, count);
    run_test<hash::method::md5>(q, text2, strlen((char *) text2), hash2, count);
}

void sha256_test(hash::runners &q, size_t n_blocks) {
    byte text1[] = {
            "abc"
    };
    byte text2[] = {"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq"};
    byte hash1[SHA256_BLOCK_SIZE] = {
            0xba, 0x78, 0x16, 0xbf, 0x8f, 0x01, 0xcf, 0xea,
            0x41, 0x41, 0x40, 0xde, 0x5d, 0xae, 0x22, 0x23,
            0xb0, 0x03, 0x61, 0xa3, 0x96, 0x17, 0x7a, 0x9c,
            0xb4, 0x10, 0xff, 0x61, 0xf2, 0x00, 0x15, 0xad};
    byte hash2[SHA256_BLOCK_SIZE] = {
            0x24, 0x8d, 0x6a, 0x61, 0xd2, 0x06, 0x38, 0xb8,
            0xe5, 0xc0, 0x26, 0x93, 0x0c, 0x3e, 0x60, 0x39,
            0xa3, 0x3c, 0xe4, 0x59, 0x64, 0xff, 0x21, 0x67,
            0xf6, 0xec, 0xed, 0xd4, 0x19, 0xdb, 0x06, 0xc1};
    run_test<hash::method::sha256>(q, text1, strlen((char *) text1), hash1, n_blocks);
    run_test<hash::method::sha256>(q, text2, strlen((char *) text2), hash2, n_blocks);
}

void md2_test(hash::runners &q, size_t count) {
    byte text1[] = {
            "abc"
    };
    byte text2[] = {"abcdefghijklmnopqrstuvwxyz"};
    byte hash1[MD2_BLOCK_SIZE] = {0xda, 0x85, 0x3b, 0x0d, 0x3f, 0x88, 0xd9, 0x9b, 0x30, 0x28, 0x3a, 0x69, 0xe6, 0xde, 0xd6, 0xbb};
    byte hash2[MD2_BLOCK_SIZE] = {0x4e, 0x8d, 0xdf, 0xf3, 0x65, 0x02, 0x92, 0xab, 0x5a, 0x41, 0x08, 0xc3, 0xaa, 0x47, 0x94, 0x0b};
    run_test<hash::method::md2>(q, text1, strlen((char *) text1), hash1, count);
    run_test<hash::method::md2>(q, text2, strlen((char *) text2), hash2, count);
}

void keccak_test(hash::runners &q, size_t count) {
    byte text1[] = {
            "abc"
    };
    byte hash1[hash::get_block_size<hash::method::keccak, 256>()] = {
            0x4e, 0x03, 0x65, 0x7a, 0xea, 0x45, 0xa9, 0x4f,
            0xc7, 0xd4, 0x7b, 0xa8, 0x26, 0xc8, 0xd6, 0x67,
            0xc0, 0xd1, 0xe6, 0xe3, 0x3a, 0x64, 0xa0, 0x36,
            0xec, 0x44, 0xf5, 0x8f, 0xa1, 0x2d, 0x6c, 0x45};
    run_test<hash::method::keccak, 256>(q, text1, 3, hash1, count);
}

void sha3_test(hash::runners &q, size_t count) {
    byte text1[] = {
            "abc"
    };
    byte hash1[hash::get_block_size<hash::method::sha3, 384>()] = {
            0xec, 0x01, 0x49, 0x82, 0x88, 0x51, 0x6f, 0xc9,
            0x26, 0x45, 0x9f, 0x58, 0xe2, 0xc6, 0xad, 0x8d,
            0xf9, 0xb4, 0x73, 0xcb, 0x0f, 0xc0, 0x8c, 0x25,
            0x96, 0xda, 0x7c, 0xf0, 0xe4, 0x9b, 0xe4, 0xb2,
            0x98, 0xd8, 0x8c, 0xea, 0x92, 0x7a, 0xc7, 0xf5,
            0x39, 0xf1, 0xed, 0xf2, 0x28, 0x37, 0x6d, 0x25};
    run_test<hash::method::sha3, 384>(q, text1, 3, hash1, count);
}

void blake2b_test(hash::runners &q, size_t count) {
    byte text1[] = {
            "abc"
    };
    byte key1[] = {"def"};
    byte hash1[hash::get_block_size<hash::method::blake2b, 512>()] = {
            0x95, 0x6f, 0x2f, 0x56, 0xe2, 0x30, 0x8b, 0x97,
            0x12, 0x0b, 0xb9, 0xf5, 0x0e, 0xef, 0xaa, 0x5c,
            0x6a, 0x5a, 0xe4, 0x23, 0x8a, 0x37, 0x2e, 0x30,
            0x8a, 0xeb, 0x82, 0x4d, 0x31, 0x66, 0xd8, 0x69,
            0xc9, 0xa9, 0xba, 0x32, 0x22, 0x6d, 0x33, 0xba,
            0x08, 0x1b, 0x23, 0x5f, 0xc4, 0x5c, 0x03, 0x85,
            0x2b, 0x26, 0x2d, 0x97, 0xce, 0x13, 0x01, 0x8c,
            0x55, 0xed, 0x30, 0x4d, 0x30, 0x2c, 0x86, 0xb5};
    run_test<hash::method::blake2b, 512>(q, text1, 3, hash1, count, key1, 3);


    constexpr int KAT_LENGTH = 256;
    constexpr int blake2b_keylen = 64;
    byte key[blake2b_keylen];
    byte buf[KAT_LENGTH];
    for (size_t i = 0; i < blake2b_keylen; ++i) {
        key[i] = (uint8_t) i;
    }

    for (size_t i = 0; i < KAT_LENGTH; ++i) {
        buf[i] = (uint8_t) i;
    }

    byte hash2[6][hash::get_block_size<hash::method::blake2b, 512>()] = {
            {
                    0x10, 0xEB, 0xB6, 0x77, 0x00, 0xB1, 0x86, 0x8E,
                    0xFB, 0x44, 0x17, 0x98, 0x7A, 0xCF, 0x46, 0x90,
                    0xAE, 0x9D, 0x97, 0x2F, 0xB7, 0xA5, 0x90, 0xC2,
                    0xF0, 0x28, 0x71, 0x79, 0x9A, 0xAA, 0x47, 0x86,
                    0xB5, 0xE9, 0x96, 0xE8, 0xF0, 0xF4, 0xEB, 0x98,
                    0x1F, 0xC2, 0x14, 0xB0, 0x05, 0xF4, 0x2D, 0x2F,
                    0xF4, 0x23, 0x34, 0x99, 0x39, 0x16, 0x53, 0xDF,
                    0x7A, 0xEF, 0xCB, 0xC1, 0x3F, 0xC5, 0x15, 0x68
            },
            {
                    0x96, 0x1F, 0x6D, 0xD1, 0xE4, 0xDD, 0x30, 0xF6,
                    0x39, 0x01, 0x69, 0x0C, 0x51, 0x2E, 0x78, 0xE4,
                    0xB4, 0x5E, 0x47, 0x42, 0xED, 0x19, 0x7C, 0x3C,
                    0x5E, 0x45, 0xC5, 0x49, 0xFD, 0x25, 0xF2, 0xE4,
                    0x18, 0x7B, 0x0B, 0xC9, 0xFE, 0x30, 0x49, 0x2B,
                    0x16, 0xB0, 0xD0, 0xBC, 0x4E, 0xF9, 0xB0, 0xF3,
                    0x4C, 0x70, 0x03, 0xFA, 0xC0, 0x9A, 0x5E, 0xF1,
                    0x53, 0x2E, 0x69, 0x43, 0x02, 0x34, 0xCE, 0xBD
            },
            {
                    0xDA, 0x2C, 0xFB, 0xE2, 0xD8, 0x40, 0x9A, 0x0F,
                    0x38, 0x02, 0x61, 0x13, 0x88, 0x4F, 0x84, 0xB5,
                    0x01, 0x56, 0x37, 0x1A, 0xE3, 0x04, 0xC4, 0x43,
                    0x01, 0x73, 0xD0, 0x8A, 0x99, 0xD9, 0xFB, 0x1B,
                    0x98, 0x31, 0x64, 0xA3, 0x77, 0x07, 0x06, 0xD5,
                    0x37, 0xF4, 0x9E, 0x0C, 0x91, 0x6D, 0x9F, 0x32,
                    0xB9, 0x5C, 0xC3, 0x7A, 0x95, 0xB9, 0x9D, 0x85,
                    0x74, 0x36, 0xF0, 0x23, 0x2C, 0x88, 0xA9, 0x65
            },
            {
                    0x33, 0xD0, 0x82, 0x5D, 0xDD, 0xF7, 0xAD, 0xA9,
                    0x9B, 0x0E, 0x7E, 0x30, 0x71, 0x04, 0xAD, 0x07,
                    0xCA, 0x9C, 0xFD, 0x96, 0x92, 0x21, 0x4F, 0x15,
                    0x61, 0x35, 0x63, 0x15, 0xE7, 0x84, 0xF3, 0xE5,
                    0xA1, 0x7E, 0x36, 0x4A, 0xE9, 0xDB, 0xB1, 0x4C,
                    0xB2, 0x03, 0x6D, 0xF9, 0x32, 0xB7, 0x7F, 0x4B,
                    0x29, 0x27, 0x61, 0x36, 0x5F, 0xB3, 0x28, 0xDE,
                    0x7A, 0xFD, 0xC6, 0xD8, 0x99, 0x8F, 0x5F, 0xC1
            },
            {
                    0xBE, 0xAA, 0x5A, 0x3D, 0x08, 0xF3, 0x80, 0x71,
                    0x43, 0xCF, 0x62, 0x1D, 0x95, 0xCD, 0x69, 0x05,
                    0x14, 0xD0, 0xB4, 0x9E, 0xFF, 0xF9, 0xC9, 0x1D,
                    0x24, 0xB5, 0x92, 0x41, 0xEC, 0x0E, 0xEF, 0xA5,
                    0xF6, 0x01, 0x96, 0xD4, 0x07, 0x04, 0x8B, 0xBA,
                    0x8D, 0x21, 0x46, 0x82, 0x8E, 0xBC, 0xB0, 0x48,
                    0x8D, 0x88, 0x42, 0xFD, 0x56, 0xBB, 0x4F, 0x6D,
                    0xF8, 0xE1, 0x9C, 0x4B, 0x4D, 0xAA, 0xB8, 0xAC
            },
            {
                    0x09, 0x80, 0x84, 0xB5, 0x1F, 0xD1, 0x3D, 0xEA,
                    0xE5, 0xF4, 0x32, 0x0D, 0xE9, 0x4A, 0x68, 0x8E,
                    0xE0, 0x7B, 0xAE, 0xA2, 0x80, 0x04, 0x86, 0x68,
                    0x9A, 0x86, 0x36, 0x11, 0x7B, 0x46, 0xC1, 0xF4,
                    0xC1, 0xF6, 0xAF, 0x7F, 0x74, 0xAE, 0x7C, 0x85,
                    0x76, 0x00, 0x45, 0x6A, 0x58, 0xA3, 0xAF, 0x25,
                    0x1D, 0xC4, 0x72, 0x3A, 0x64, 0xCC, 0x7C, 0x0A,
                    0x5A, 0xB6, 0xD9, 0xCA, 0xC9, 0x1C, 0x20, 0xBB
            }
    };


    for (size_t i = 0; i < 6; ++i) {
        run_test<hash::method::blake2b, 512>(q, buf, i, hash2[i], count, key, blake2b_keylen);
    }

}


TEST(Hash_Test, Blake2b) {
    for_all_workers([](auto q) {
        blake2b_test(q, loop_count);
    });
}

TEST(Hash_Test_Pairs, Blake2b) {
    for_all_workers_pairs([](hash::runners q) {
        blake2b_test(q, loop_count);
    });
}

TEST(Hash_Test, Keccak) {
    for_all_workers([](auto q) {
        keccak_test(q, loop_count);
    });
}

TEST(Hash_Test_Pairs, Keccak) {
    for_all_workers_pairs([](hash::runners q) {
        keccak_test(q, loop_count);
    });
}

TEST(Hash_Test, SHA3) {
    for_all_workers([](auto q) {
        sha3_test(q, loop_count);
    });
}

TEST(Hash_Test_Pairs, SHA3) {
    for_all_workers_pairs([](hash::runners q) {
        sha3_test(q, loop_count);
    });
}

TEST(Hash_Test, SHA256) {
    for_all_workers([](auto q) {
        sha256_test(q, loop_count);
    });
}

TEST(Hash_Test_Pairs, SHA256) {
    for_all_workers_pairs([](hash::runners q) {
        sha256_test(q, loop_count);
    });
}

TEST(Hash_Test, SHA1) {
    for_all_workers([](auto q) {
        sha1_test(q, loop_count);
    });
}

TEST(Hash_Test_Pairs, SHA1) {
    for_all_workers_pairs([](hash::runners q) {
        sha1_test(q, loop_count);
    });
}

TEST(Hash_Test, MD2) {
    for_all_workers([](auto q) {
        md2_test(q, loop_count);
    });
}

TEST(Hash_Test_Pairs, MD2) {
    for_all_workers_pairs([](hash::runners q) {
        md2_test(q, loop_count);
    });
}

TEST(Hash_Test, MD5) {
    for_all_workers([](auto q) {
        md5_test(q, loop_count);
    });
}

TEST(Hash_Test_Pairs, MD5) {
    for_all_workers_pairs([](hash::runners q) {
        md5_test(q, loop_count);
    });
}