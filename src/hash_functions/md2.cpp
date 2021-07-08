#include <hash_functions/md2.hpp>
#include <internal/determine_kernel_config.hpp>

#include <cstring>
#include <utility>


using namespace usm_smart_ptr;
using namespace hash;

struct md2_ctx {
    int len;
    byte data[16];
    byte state[48];
    byte checksum[16];
};

/**************************** VARIABLES *****************************/
static const byte GLOBAL_MD2_CONSTS[256] =
        {41, 46, 67, 201, 162, 216, 124, 1, 61, 54, 84, 161, 236, 240, 6,
         19, 98, 167, 5, 243, 192, 199, 115, 140, 152, 147, 43, 217, 188, 76,
         130, 202, 30, 155, 87, 60, 253, 212, 224, 22, 103, 66, 111, 24, 138,
         23, 229, 18, 190, 78, 196, 214, 218, 158, 222, 73, 160, 251, 245, 142,
         187, 47, 238, 122, 169, 104, 121, 145, 21, 178, 7, 63, 148, 194, 16,
         137, 11, 34, 95, 33, 128, 127, 93, 154, 90, 144, 50, 39, 53, 62,
         204, 231, 191, 247, 151, 3, 255, 25, 48, 179, 72, 165, 181, 209, 215,
         94, 146, 42, 172, 86, 170, 198, 79, 184, 56, 210, 150, 164, 125, 182,
         118, 252, 107, 226, 156, 116, 4, 241, 69, 157, 112, 89, 100, 113, 135,
         32, 134, 91, 207, 101, 230, 45, 168, 2, 27, 96, 37, 173, 174, 176,
         185, 246, 28, 70, 97, 105, 52, 64, 126, 15, 85, 71, 163, 35, 221,
         81, 175, 58, 195, 92, 249, 206, 186, 197, 234, 38, 44, 83, 13, 110,
         133, 40, 132, 9, 211, 223, 205, 244, 65, 129, 77, 82, 106, 220, 55,
         200, 108, 193, 171, 250, 36, 225, 123, 8, 12, 189, 177, 74, 120, 136,
         149, 139, 227, 99, 232, 109, 233, 203, 213, 254, 59, 0, 29, 57, 242,
         239, 183, 14, 102, 88, 208, 228, 166, 119, 114, 248, 235, 117, 75, 10,
         49, 68, 80, 180, 143, 237, 31, 26, 219, 153, 141, 51, 159, 17, 131,
         20};

/*********************** FUNCTION DEFINITIONS ***********************/
static inline void md2_transform(md2_ctx *ctx, const byte *data, const constant_accessor<byte, 1> &consts) {
    dword t;

    for (int j = 0; j < 16; ++j) {
        ctx->state[j + 32] = (ctx->state[j + 16] = data[j]) ^ ctx->state[j];
    }

    t = 0;
    for (dword j = 0; j < 18; ++j) {
        for (dword k = 0; k < 48; ++k) {
            t = ctx->state[k] ^= consts[t];
        }
        t = (t + j) & 0xFF;
    }

    t = ctx->checksum[15];
    for (int j = 0; j < 16; ++j) {
        t = ctx->checksum[j] ^= consts[data[j] ^ t];
    }
}

static inline void md2_init(md2_ctx *ctx) {
    memset(ctx, 0, sizeof(*ctx));
}

static inline void md2_update(md2_ctx *ctx, const byte *data, size_t len, const constant_accessor<byte, 1> &consts) {
    for (size_t i = 0; i < len; ++i) {
        ctx->data[ctx->len] = data[i];
        ctx->len++;
        if (ctx->len == MD2_BLOCK_SIZE) {
            md2_transform(ctx, ctx->data, consts);
            ctx->len = 0;
        }
    }
}

static inline void md2_final(md2_ctx *ctx, byte *hash, const constant_accessor<byte, 1> &consts) {
    long int to_pad = MD2_BLOCK_SIZE - ctx->len;
    if (to_pad > 0)memset(ctx->data + ctx->len, (byte) to_pad, (dword) to_pad);
    md2_transform(ctx, ctx->data, consts);
    md2_transform(ctx, ctx->checksum, consts);
    memcpy(hash, ctx->state, MD2_BLOCK_SIZE);
}

static inline void kernel_md2_hash(byte *indata, dword inlen, byte *outdata, dword n_batch, dword thread, const constant_accessor<byte, 1> &consts) {
    if (thread >= n_batch) {
        return;
    }
    byte *in = indata + thread * inlen;
    byte *out = outdata + thread * MD2_BLOCK_SIZE;
    md2_ctx ctx;
    md2_init(&ctx);
    md2_update(&ctx, in, inlen, consts);
    md2_final(&ctx, out, consts);
}

namespace hash::internal {

    sycl::buffer<byte, 1> get_buf_md2_consts() {
        sycl::buffer<byte, 1> buf{GLOBAL_MD2_CONSTS, sycl::range<1>(256)};
        buf.set_final_data(nullptr);
        buf.set_write_back(false);
        return buf;
    }


    sycl::event
    launch_md2_kernel(sycl::queue &q, sycl::event e, const device_accessible_ptr<byte> indata, device_accessible_ptr<byte> outdata, dword inlen, dword n_batch, sycl::buffer<byte, 1> &buf_md2_consts) {
        auto config = get_kernel_sizes(q, n_batch);
        return q.submit([&](sycl::handler &cgh) {
            cgh.depends_on(e);
            auto const_ptr = buf_md2_consts.get_access<sycl::access::mode::read, sycl::access::target::constant_buffer>(cgh);
            cgh.parallel_for<class md2_kernel>(
                    sycl::nd_range<1>(sycl::range<1>(config.block) * sycl::range<1>(config.wg_size), sycl::range<1>(config.wg_size)),
                    [=](sycl::nd_item<1> item) {
                        kernel_md2_hash(indata, inlen, outdata, n_batch, item.get_global_linear_id(), const_ptr);
                    });
        });
    }

    sycl::event launch_md2_kernel(sycl::queue &q, sycl::event e, const device_accessible_ptr<byte> indata, device_accessible_ptr<byte> outdata, dword inlen, dword n_batch) {
        auto buf = get_buf_md2_consts();
        return launch_md2_kernel(q, std::move(e), indata, outdata, inlen, n_batch, buf);
    }

}

