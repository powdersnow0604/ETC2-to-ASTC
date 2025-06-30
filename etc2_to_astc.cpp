#define NOMINMAX
#include "etc2_to_astc.h"
#include <etcpak/Tables.hpp>
#include <etcpak/ForceInline.hpp>
#include <astcenc/astcenc_internal.h>
#include <astcenc/astcenc_internal_entry.h>
#include <astcenc/astcenc.h>
#include <fstream>

#if defined __SSE4_1__ || defined __AVX2__ || defined _MSC_VER
#ifdef _MSC_VER
#include <intrin.h>
#include <Windows.h>
#define _bswap(x) _byteswap_ulong(x)
#define _bswap64(x) _byteswap_uint64(x)
#else
#include <x86intrin.h>
#endif
#endif

#ifndef _bswap
#define _bswap(x) __builtin_bswap32(x)
#define _bswap64(x) __builtin_bswap64(x)
#endif


static uint8_t table59T58H[8] = { 3,6,11,16,23,32,41,64 };

static int8_t matrix_ls[3][16] = {
    {-6, -6, -6, -6, -2, -2, -2, -2,  2,  2,  2,  2,  6,  6,  6,  6},
    {-6, -2,  2,  6, -6, -2,  2,  6, -6, -2,  2,  6, -6, -2,  2,  6},
    {23, 17, 11,  5, 17, 11,  5, -1, 11,  5, -1, -7,  5, -1, -7, -13}
};


/**
 * @brief Reverse bits in a byte.
 *
 * @param p   The value to reverse.
 *
 * @return The reversed result.
 */
static inline int bitrev8(int p)
{
    p = ((p & 0x0F) << 4) | ((p >> 4) & 0x0F);
    p = ((p & 0x33) << 2) | ((p >> 2) & 0x33);
    p = ((p & 0x55) << 1) | ((p >> 1) & 0x55);
    return p;
}

// ASTCENC 의 code 를 가져온 것인데, 주석상으로는 8 bit 까지만 쓸 수 있다고 적혀있으나,
// 구현을 보면 16 bit 까지 쓸 수 있으며, ASTCENC 의 소스코드에서도 16 bit 까지 활용함
/**
 * @brief Write up to 8 bits at an arbitrary bit offset.
 *
 * The stored value is at most 8 bits, but can be stored at an offset of between 0 and 7 bits so
 * may span two separate bytes in memory.
 *
 * @param         value       The value to write.
 * @param         bitcount    The number of bits to write, starting from LSB.
 * @param         bitoffset   The bit offset to store at, between 0 and 7.
 * @param[in,out] ptr         The data pointer to write to.
 */
static inline void write_bits(
    int value,
    int bitcount,
    int bitoffset,
    uint8_t *ptr)
{
    int mask = (1 << bitcount) - 1;
    value &= mask;
    ptr += bitoffset >> 3;
    bitoffset &= 7;
    value <<= bitoffset;
    mask <<= bitoffset;
    mask = ~mask;

    ptr[0] &= mask;
    ptr[0] |= value;
    ptr[1] &= mask >> 8;
    ptr[1] |= value >> 8;
}

static etcpak_force_inline int32_t expand6(uint32_t value)
{
    return (value << 2) | (value >> 4);
}

static etcpak_force_inline int32_t expand7(uint32_t value)
{
    return (value << 1) | (value >> 6);
}

static void matrix_multiply(int32_t* dst, int32_t* src, int32_t* matrix, 
                           size_t m, size_t k, size_t n,
                           size_t ld_dst, size_t ld_src, size_t ld_matrix)
{
    // dst[m][n] = src[m][k] * matrix[k][n]
    for (size_t i = 0; i < m; ++i)
    {
        for (size_t k_idx = 0; k_idx < k; ++k_idx)
        {
            int32_t src_val = src[i * ld_src + k_idx];
            for (size_t j = 0; j < n; ++j)
            {
                dst[i * ld_dst + j] += src_val * matrix[k_idx * ld_matrix + j];
            }
        }
    }
}

static float dot_product(float* vec1, float* vec2, int dim)
{
    float sum = 0;
    for (int i = 0; i < dim; ++i)
    {
        sum += vec1[i] * vec2[i];
    }
    return sum;
}

static uint8_t clamp6(int32_t x)
{
    return (x < 0) ? 0 : (x > 64) ? 64 : x;
}

// column-major → row-major for a matrix of 2-bit elements
// src : (rows × cols × 2) bits in column-major order
// dst : caller-allocated, at least (rows*cols+3)/4 bytes
static void col2row2bit(const uint8_t* src, uint8_t* dst,
    size_t rows, size_t cols)
{
    auto get2 = [](const uint8_t* buf, size_t i) -> uint8_t {
        return (buf[i >> 2] >> ((i & 3) << 1)) & 0x3;  // 4 items/byte
        };
    auto set2 = [](uint8_t* buf, size_t i, uint8_t v) {
        size_t byte = i >> 2;
        uint8_t shift = (i & 3) << 1;
        buf[byte] = (buf[byte] & ~(0x3 << shift)) | ((v & 0x3) << shift);
        };

    for (size_t r = 0; r < rows; ++r)
        for (size_t c = 0; c < cols; ++c) {
            size_t srcIdx = c * rows + r;   // column-major index
            size_t dstIdx = r * cols + c;   // row-major index
            set2(dst, dstIdx, get2(src, srcIdx));
        }
}

// column-major → row-major for a matrix of 1-bit elements
// src : rows*cols bits, column-major packed (LSB-first inside each byte)
// dst : caller-allocated, at least (rows*cols+7)/8 bytes
static void col2row1bit(const uint8_t* src, uint8_t* dst,
                        size_t rows, size_t cols)
{
    auto get1 = [](const uint8_t* b, size_t i) -> uint8_t {
        return (b[i >> 3] >> (i & 7)) & 1;
    };
    auto set1 = [](uint8_t* b, size_t i, uint8_t v) {
        size_t byte = i >> 3;
        uint8_t bit = 1u << (i & 7);
        if (v) b[byte] |=  bit;
        else   b[byte] &= ~bit;
    };

    for (size_t r = 0; r < rows; ++r)
        for (size_t c = 0; c < cols; ++c) {
            size_t srcIdx = c * rows + r;   // column-major
            size_t dstIdx = r * cols + c;   // row-major
            set1(dst, dstIdx, get1(src, srcIdx));
        }
}

static etcpak_force_inline uint64_t ConvertByteOrder(uint64_t d)
{
    uint32_t word[2];
    memcpy(word, &d, 8);
    word[0] = _bswap(word[0]);
    word[1] = _bswap(word[1]);
    memcpy(&d, word, 8);
    return d;
}

/**
 * @brief Compute bit-mismatch for partitioning in 2-partition mode.
 *
 * @param a   The texel assignment bitvector for the block.
 * @param b   The texel assignment bitvector for the partition table.
 *
 * @return    The number of bit mismatches.
 */
static inline uint8_t partition_mismatch2(
	const uint64_t a[2],
	const uint64_t b[2]
) {
	int v1 = popcount(a[0] ^ b[0]) + popcount(a[1] ^ b[1]);
	int v2 = popcount(a[0] ^ b[1]) + popcount(a[1] ^ b[0]);

	// Divide by 2 because XOR always counts errors twice, once when missing
	// in the expected position, and again when present in the wrong partition
	return static_cast<uint8_t>(std::min(v1, v2) / 2);
}

/**
 * @brief Use counting sort on the mismatch array to sort partition candidates.
 *
 * @param      partitioning_count   The number of packed partitionings.
 * @param      mismatch_count       Partitioning mismatch counts, in index order.
 * @param[out] partition_ordering   Partition index values, in mismatch order.
 *
 * @return The number of active partitions in this selection.
 */
static unsigned int get_partition_ordering_by_mismatch_bits(
	unsigned int texel_count,
	unsigned int partitioning_count,
	const uint8_t mismatch_count[BLOCK_MAX_PARTITIONINGS],
	uint16_t partition_ordering[BLOCK_MAX_PARTITIONINGS]
) {
	promise(partitioning_count > 0);
	uint16_t mscount[BLOCK_MAX_KMEANS_TEXELS] { 0 };

	// Create the histogram of mismatch counts
	for (unsigned int i = 0; i < partitioning_count; i++)
	{
		mscount[mismatch_count[i]]++;
	}

	// Create a running sum from the histogram array
	// Indices store previous values only; i.e. exclude self after sum
	uint16_t sum = 0;
	for (unsigned int i = 0; i < texel_count; i++)
	{
		uint16_t cnt = mscount[i];
		mscount[i] = sum;
		sum += cnt;
	}

	// Use the running sum as the index, incrementing after read to allow
	// sequential entries with the same count
	for (unsigned int i = 0; i < partitioning_count; i++)
	{
		unsigned int idx = mscount[mismatch_count[i]]++;
		partition_ordering[idx] = static_cast<uint16_t>(i);
	}

	return partitioning_count;
}

static etcpak_force_inline void single_color(uint8_t *dst)
{
    uint8_t weightbuf[16]{0};
    
    for (int i = 0; i < 16; i++)
    {
        dst[i] = static_cast<uint8_t>(bitrev8(weightbuf[15 - i]));
    }

    const uint16_t block_mode = 0x0042;

    const unsigned int partition_count = 1;
    const uint8_t color_format = 8;                   // RGB direct
    const quant_method color_quant_method = QUANT_256;

    write_bits(block_mode, 11, 0, dst);
    write_bits(partition_count - 1, 2, 11, dst);
    write_bits(color_format, 4, 13, dst);


    uint8_t color_values[6]{ 255, 255, 255, 255, 255, 255};

    encode_ise(color_quant_method, 6, (uint8_t *)color_values, dst, 17);
}

static etcpak_force_inline void DecodeT( uint64_t block, uint8_t* dst, v2i& weight_block_size, block_size_descriptor& bsd)
{
    /*      Store weights       */
    uint32_t idx = (block >> 48);

    uint8_t weightbuf[16]{0};

    // column major to row major
    col2row1bit((uint8_t *)&idx, weightbuf, weight_block_size.x, weight_block_size.y);

    for (int i = 0; i < 16; i++)
    {
        dst[i] = static_cast<uint8_t>(bitrev8(weightbuf[15 - i]));
    }

    /*      End       */

    /*      Find Best Partitioning       */
    // bitmap[0]: c2, c3
    // bitmap[1]: c0, c1
    uint64_t bitmap[2];

    idx = (block >> 32) & 0xFFFF;
    col2row1bit((uint8_t *)&idx, (uint8_t *)bitmap, weight_block_size.x, weight_block_size.y);

    bitmap[1] = (~(bitmap[0])) & 0xFFFF;

    int partition_count = 2;
    uint8_t mismatch_counts[BLOCK_MAX_PARTITIONINGS] = {0};
    uint16_t partition_ordering[BLOCK_MAX_PARTITIONINGS] = {0};

    unsigned int active_count = bsd.partitioning_count_selected[partition_count - 1];

    for (unsigned int i = 0; i < active_count; i++)
    {
        mismatch_counts[i] = partition_mismatch2(bitmap, bsd.coverage_bitmaps_2[i]);
    }

    get_partition_ordering_by_mismatch_bits(
        bsd.texel_count,
        bsd.partitioning_count_selected[partition_count - 1],
        mismatch_counts, partition_ordering);

    uint16_t partition_index = bsd.partitionings[partition_ordering[0]].partition_index;

    int v1 = popcount(bitmap[0] ^ bsd.coverage_bitmaps_2[partition_ordering[0]][0]) + popcount(bitmap[1] ^ bsd.coverage_bitmaps_2[partition_ordering[0]][1]);
	int v2 = popcount(bitmap[0] ^ bsd.coverage_bitmaps_2[partition_ordering[0]][1]) + popcount(bitmap[1] ^ bsd.coverage_bitmaps_2[partition_ordering[0]][0]);

    // If v1 > v2, then the partition 0 should be c0, c1
    // If v1 < v2, then the partition 0 should be c2, c3
    bool ordering_is_reversed = (v1 > v2);
    /*      End       */

    const quant_method color_quant_method = QUANT_96; //4x4 weight block 만 상정
    uint8_t color_values[2][6]{0};
    const uint8_t *pack_table = color_uquant_to_scrambled_pquant_tables[color_quant_method - QUANT_6];

    const auto r0 = ( block >> 24 ) & 0x1B;
    const auto rh0 = ( r0 >> 3 ) & 0x3;
    const auto rl0 = r0 & 0x3;
    const auto g0 = ( block >> 20 ) & 0xF;
    const auto b0 = ( block >> 16 ) & 0xF;

    const auto r1 = ( block >> 12 ) & 0xF;
    const auto g1 = ( block >> 8 ) & 0xF;
    const auto b1 = ( block >> 4 ) & 0xF;

    const auto cr0 = ( ( rh0 << 6 ) | ( rl0 << 4 ) | ( rh0 << 2 ) | rl0);
    const auto cg0 = ( g0 << 4 ) | g0;
    const auto cb0 = ( b0 << 4 ) | b0;

    const auto cr1 = ( r1 << 4 ) | r1;
    const auto cg1 = ( g1 << 4 ) | g1;
    const auto cb1 = ( b1 << 4 ) | b1;

    const auto codeword_hi = ( block >> 2 ) & 0x3;
    const auto codeword_lo = block & 0x1;
    const auto codeword = ( codeword_hi << 1 ) | codeword_lo;

    const auto c2r = clampu8( cr1 + table59T58H[codeword] );
    const auto c2g = clampu8( cg1 + table59T58H[codeword] );
    const auto c2b = clampu8( cb1 + table59T58H[codeword] );

    const auto c3r = clampu8( cr1 - table59T58H[codeword] );
    const auto c3g = clampu8( cg1 - table59T58H[codeword] );
    const auto c3b = clampu8( cb1 - table59T58H[codeword] );

    if(ordering_is_reversed)
    {
        color_values[0][0] = pack_table[cr0];
        color_values[0][1] = pack_table[cr1];
        color_values[0][2] = pack_table[cg0];
        color_values[0][3] = pack_table[cg1];
        color_values[0][4] = pack_table[cb0];
        color_values[0][5] = pack_table[cb1];

        color_values[1][0] = pack_table[c3r];
        color_values[1][1] = pack_table[c2r];
        color_values[1][2] = pack_table[c3g];
        color_values[1][3] = pack_table[c2g];
        color_values[1][4] = pack_table[c3b];
        color_values[1][5] = pack_table[c2b];
    }
    else
    {
        color_values[0][0] = pack_table[c3r];
        color_values[0][1] = pack_table[c2r];
        color_values[0][2] = pack_table[c3g];
        color_values[0][3] = pack_table[c2g];
        color_values[0][4] = pack_table[c3b];
        color_values[0][5] = pack_table[c2b];

        color_values[1][0] = pack_table[cr0];
        color_values[1][1] = pack_table[cr1];
        color_values[1][2] = pack_table[cg0];
        color_values[1][3] = pack_table[cg1];
        color_values[1][4] = pack_table[cb0];
        color_values[1][5] = pack_table[cb1];
    }

    // Block mode for
    //  weight range [0, 3]
    //  Low-precision
    //  Weight block width 4
    //  Weight block height 4
    //  No dual plane
    const uint16_t block_mode = 0x0042;
    const uint8_t color_format = 8;                   // RGB direct

    write_bits(block_mode, 11, 0, dst);
    write_bits(partition_count - 1, 2, 11, dst);
    write_bits(partition_index, PARTITION_INDEX_BITS, 13, dst);
    write_bits(color_format << 2, 6, 13 + PARTITION_INDEX_BITS, dst);

    encode_ise(color_quant_method, 12, (uint8_t *)color_values, dst,
               19 + PARTITION_INDEX_BITS);
}

static etcpak_force_inline void DecodeH( uint64_t block, uint8_t* dst, v2i& weight_block_size, block_size_descriptor& bsd)
{
    /*      Store weights       */
    uint32_t idx = (block >> 32) & 0xFFFF;

    uint8_t weightbuf[16]{0};

    // column major to row major
    col2row1bit((uint8_t *)&idx, weightbuf, weight_block_size.x, weight_block_size.y);

    for (int i = 0; i < 16; i++)
    {
        dst[i] = static_cast<uint8_t>(bitrev8(weightbuf[15 - i]));
    }

    /*      End       */

    /*      Find Best Partitioning       */
    // bitmap[0]: r0, g0, b0
    // bitmap[1]: r1, g1, b1
    uint64_t bitmap[2];

    idx = (block >> 48);
    col2row1bit((uint8_t *)&idx, (uint8_t *)bitmap, weight_block_size.x, weight_block_size.y);

    bitmap[1] = (~(bitmap[0])) & 0xFFFF;

    int partition_count = 2;
    uint8_t mismatch_counts[BLOCK_MAX_PARTITIONINGS] = {0};
    uint16_t partition_ordering[BLOCK_MAX_PARTITIONINGS] = {0};

    unsigned int active_count = bsd.partitioning_count_selected[partition_count - 1];

    for (unsigned int i = 0; i < active_count; i++)
    {
        mismatch_counts[i] = partition_mismatch2(bitmap, bsd.coverage_bitmaps_2[i]);
    }

    get_partition_ordering_by_mismatch_bits(
        bsd.texel_count,
        bsd.partitioning_count_selected[partition_count - 1],
        mismatch_counts, partition_ordering);

    uint16_t partition_index = bsd.partitionings[partition_ordering[0]].partition_index;

    int v1 = popcount(bitmap[0] ^ bsd.coverage_bitmaps_2[partition_ordering[0]][0]) + popcount(bitmap[1] ^ bsd.coverage_bitmaps_2[partition_ordering[0]][1]);
	int v2 = popcount(bitmap[0] ^ bsd.coverage_bitmaps_2[partition_ordering[0]][1]) + popcount(bitmap[1] ^ bsd.coverage_bitmaps_2[partition_ordering[0]][0]);

    // If v1 > v2, then the partition 0 should be r1, g1, b1
    // If v1 < v2, then the partition 0 should be r0, g0, b0
    bool ordering_is_reversed = (v1 > v2);
    /*      End       */

    const quant_method color_quant_method = QUANT_96; //4x4 weight block 만 상정
    uint8_t color_values[2][6]{0};
    const uint8_t *pack_table = color_uquant_to_scrambled_pquant_tables[color_quant_method - QUANT_6];

    const auto r0444 = ( block >> 27 ) & 0xF;
    const auto g0444 = ( ( block >> 20 ) & 0x1 ) | ( ( ( block >> 24 ) & 0x7 ) << 1 );
    const auto b0444 = ( ( block >> 15 ) & 0x7 ) | ( ( ( block >> 19 ) & 0x1 ) << 3 );

    const auto r1444 = ( block >> 11 ) & 0xF;
    const auto g1444 = ( block >> 7 ) & 0xF;
    const auto b1444 = ( block >> 3 ) & 0xF;

    const auto r0 = ( r0444 << 4 ) | r0444;
    const auto g0 = ( g0444 << 4 ) | g0444;
    const auto b0 = ( b0444 << 4 ) | b0444;

    const auto r1 = ( r1444 << 4 ) | r1444;
    const auto g1 = ( g1444 << 4 ) | g1444;
    const auto b1 = ( b1444 << 4 ) | b1444;

    const auto codeword_hi = ( ( block & 0x1 ) << 1 ) | ( ( block & 0x4 ) );
    const auto c0 = ( r0444 << 8 ) | ( g0444 << 4 ) | ( b0444 << 0 );
    const auto c1 = ( block >> 3 ) & ( ( 1 << 12 ) - 1 );
    const auto codeword_lo = ( c0 >= c1 ) ? 1 : 0;
    const auto codeword = codeword_hi | codeword_lo;

    if(ordering_is_reversed)
    {
        color_values[0][0] = pack_table[clampu8( r1 - table59T58H[codeword] )];
        color_values[0][1] = pack_table[clampu8( r1 + table59T58H[codeword] )];
        color_values[0][2] = pack_table[clampu8( g1 - table59T58H[codeword] )];
        color_values[0][3] = pack_table[clampu8( g1 + table59T58H[codeword] )];
        color_values[0][4] = pack_table[clampu8( b1 - table59T58H[codeword] )];
        color_values[0][5] = pack_table[clampu8( b1 + table59T58H[codeword] )];

        color_values[1][0] = pack_table[clampu8( r0 - table59T58H[codeword] )];
        color_values[1][1] = pack_table[clampu8( r0 + table59T58H[codeword] )];
        color_values[1][2] = pack_table[clampu8( g0 - table59T58H[codeword] )];
        color_values[1][3] = pack_table[clampu8( g0 + table59T58H[codeword] )];
        color_values[1][4] = pack_table[clampu8( b0 - table59T58H[codeword] )];
        color_values[1][5] = pack_table[clampu8( b0 + table59T58H[codeword] )];
    }
    else
    {
        color_values[0][0] = pack_table[clampu8( r0 - table59T58H[codeword] )];
        color_values[0][1] = pack_table[clampu8( r0 + table59T58H[codeword] )];
        color_values[0][2] = pack_table[clampu8( g0 - table59T58H[codeword] )];
        color_values[0][3] = pack_table[clampu8( g0 + table59T58H[codeword] )];
        color_values[0][4] = pack_table[clampu8( b0 - table59T58H[codeword] )];
        color_values[0][5] = pack_table[clampu8( b0 + table59T58H[codeword] )];

        color_values[1][0] = pack_table[clampu8( r1 - table59T58H[codeword] )];
        color_values[1][1] = pack_table[clampu8( r1 + table59T58H[codeword] )];
        color_values[1][2] = pack_table[clampu8( g1 - table59T58H[codeword] )];
        color_values[1][3] = pack_table[clampu8( g1 + table59T58H[codeword] )];
        color_values[1][4] = pack_table[clampu8( b1 - table59T58H[codeword] )];
        color_values[1][5] = pack_table[clampu8( b1 + table59T58H[codeword] )];
    }

    // Block mode for
    //  weight range [0, 3]
    //  Low-precision
    //  Weight block width 4
    //  Weight block height 4
    //  No dual plane
    const uint16_t block_mode = 0x0042;
    const uint8_t color_format = 8;                   // RGB direct

    write_bits(block_mode, 11, 0, dst);
    write_bits(partition_count - 1, 2, 11, dst);
    write_bits(partition_index, PARTITION_INDEX_BITS, 13, dst);
    write_bits(color_format << 2, 6, 13 + PARTITION_INDEX_BITS, dst);

    encode_ise(color_quant_method, 12, (uint8_t *)color_values, dst,
               19 + PARTITION_INDEX_BITS);
}

static etcpak_force_inline void DecodePlanar( uint64_t block, uint8_t* dst, v2i& weight_block_size, block_size_descriptor& bsd)
{
    const auto bv = expand6((block >> ( 0 + 32)) & 0x3F);
    const auto gv = expand7((block >> ( 6 + 32)) & 0x7F);
    const auto rv = expand6((block >> (13 + 32)) & 0x3F);

    const auto bh = expand6((block >> (19 + 32)) & 0x3F);
    const auto gh = expand7((block >> (25 + 32)) & 0x7F);

    const auto rh0 = (block >> (32 - 32)) & 0x01;
    const auto rh1 = ((block >> (34 - 32)) & 0x1F) << 1;
    const auto rh = expand6(rh0 | rh1);

    const auto bo0 = (block >> (39 - 32)) & 0x07;
    const auto bo1 = ((block >> (43 - 32)) & 0x3) << 3;
    const auto bo2 = ((block >> (48 - 32)) & 0x1) << 5;
    const auto bo = expand6(bo0 | bo1 | bo2);
    const auto go0 = (block >> (49 - 32)) & 0x3F;
    const auto go1 = ((block >> (56 - 32)) & 0x01) << 6;
    const auto go = expand7(go0 | go1);
    const auto ro = expand6((block >> (57 - 32)) & 0x3F);

    int32_t decoded_colors[16][3]{0};


    for( int j=0; j<4; j++ )
    {
        for( int i=0; i<4; i++ )
        {
            const uint32_t r = (i * (rh - ro) + j * (rv - ro) + 4 * ro + 2) >> 2;
            const uint32_t g = (i * (gh - go) + j * (gv - go) + 4 * go + 2) >> 2;
            const uint32_t b = (i * (bh - bo) + j * (bv - bo) + 4 * bo + 2) >> 2;
            if( ( ( r | g | b ) & ~0xFF ) == 0 )
            {
                decoded_colors[j*4+i][0] = r;
                decoded_colors[j*4+i][1] = g;
                decoded_colors[j*4+i][2] = b;
            }
            else
            {
                const auto rc = clampu8( r );
                const auto gc = clampu8( g );
                const auto bc = clampu8( b );
                decoded_colors[j*4+i][0] = rc;
                decoded_colors[j*4+i][1] = gc;
                decoded_colors[j*4+i][2] = bc;
            }
        }
    }

    int32_t abc[3][3]{0};
    matrix_multiply((int32_t*)abc, (int32_t*)matrix_ls, (int32_t*)decoded_colors, 3, 16, 3, 3, 16, 3);

    float abc_f[3][3]{0};
    float denom = 1.f / 80.f;
    for (int i = 0; i < 3; ++i)
    {
        for (int j = 0; j < 3; ++j)
        {
            abc_f[i][j] = abc[i][j] * denom;
        }
    }

    float d[3] = {abc_f[0][0] * 3 + abc_f[1][0] * 3, abc_f[0][1] * 3 + abc_f[1][1] * 3, abc_f[0][2] * 3 + abc_f[1][2] * 3};

    const quant_method color_quant_method = QUANT_256; // (4x4 weight block 만 상정)
    uint8_t color_values[6]{0};
    const uint8_t *pack_table = color_uquant_to_scrambled_pquant_tables[color_quant_method - QUANT_6];

    color_values[0] = pack_table[clampu8(abc_f[2][0])];
    color_values[1] = pack_table[clampu8(d[0] + abc_f[2][0])];
    color_values[2] = pack_table[clampu8(abc_f[2][1])];
    color_values[3] = pack_table[clampu8(d[1] + abc_f[2][1])];
    color_values[4] = pack_table[clampu8(abc_f[2][2])];
    color_values[5] = pack_table[clampu8(d[2] + abc_f[2][2])];

    denom = 1.f / (dot_product(d, d, 3));

    uint8_t weight_values[16];
    uint8_t weightbuf[16]{0};
    quant_method weight_quant_method = QUANT_4;
    const auto& qat = quant_and_xfer_tables[weight_quant_method];
    float weight_quant_levels = static_cast<float>(get_quant_level(weight_quant_method));


    float tmp[3];
    for (int i = 0; i < 4; ++i){
        for (int j = 0; j < 4; ++j){
            tmp[0] = abc_f[0][0] * i + abc_f[1][0] * j;
            tmp[1] = abc_f[0][1] * i + abc_f[1][1] * j;
            tmp[2] = abc_f[0][2] * i + abc_f[1][2] * j;

            float uqw = static_cast<float>(clamp6((uint8_t)(dot_product(tmp, d, 3) * denom)));
            float qw = (uqw / 64.0f) * (weight_quant_levels - 1.0f);
			int qwi = static_cast<int>(qw + 0.5f);
			weight_values[i * 4 + j] = qat.scramble_map[qwi];
        }
    }

    encode_ise(weight_quant_method, 16, weight_values, weightbuf, 0);

    for (int i = 0; i < 16; i++)
	{
		dst[i] = static_cast<uint8_t>(bitrev8(weightbuf[15 - i]));
	}

    // Block mode for
    //  weight range [0, 3]
    //  Low-precision
    //  Weight block width 4
    //  Weight block height 4
    //  No dual plane
    const uint16_t block_mode = 0x0042;

    const unsigned int partition_count = 1;
    const uint8_t color_format = 8;                   // RGB direct

    write_bits(block_mode, 11, 0, dst);
	write_bits(partition_count - 1, 2, 11, dst);

    write_bits(color_format, 4, 13, dst);

    encode_ise(color_quant_method, 6, (uint8_t *)color_values, dst, 17);
}

static etcpak_force_inline void DecodeRGBPart(uint64_t d, uint8_t *dst, v2i& block_size, v2i& weight_block_size, block_size_descriptor& bsd)
{
    d = ConvertByteOrder(d);

    uint32_t br[2], bg[2], bb[2];

    if (d & 0x2)
    {
        int32_t dr, dg, db;

        uint32_t r0 = (d & 0xF8000000) >> 27;
        uint32_t g0 = (d & 0x00F80000) >> 19;
        uint32_t b0 = (d & 0x0000F800) >> 11;

        dr = (int32_t(d) << 5) >> 29;
        dg = (int32_t(d) << 13) >> 29;
        db = (int32_t(d) << 21) >> 29;

        int32_t r1 = int32_t(r0) + dr;
        int32_t g1 = int32_t(g0) + dg;
        int32_t b1 = int32_t(b0) + db;

        // T mode
        if ((r1 < 0) || (r1 > 31))
        {
            //DecodeT(d, dst, weight_block_size, bsd);
            single_color(dst);
            return;
        }

        // H mode
        if ((g1 < 0) || (g1 > 31))
        {
            // DecodeH(d, dst, weight_block_size, bsd);
            single_color(dst);
            return;
        }

        // P mode
        if ((b1 < 0) || (b1 > 31))
        {
            // DecodePlanar(d, dst, weight_block_size, bsd);
            single_color(dst);
            return;
        }

        br[0] = (r0 << 3) | (r0 >> 2);
        br[1] = (r1 << 3) | (r1 >> 2);
        bg[0] = (g0 << 3) | (g0 >> 2);
        bg[1] = (g1 << 3) | (g1 >> 2);
        bb[0] = (b0 << 3) | (b0 >> 2);
        bb[1] = (b1 << 3) | (b1 >> 2);
    }
    else
    {
        br[0] = ((d & 0xF0000000) >> 24) | ((d & 0xF0000000) >> 28);
        br[1] = ((d & 0x0F000000) >> 20) | ((d & 0x0F000000) >> 24);
        bg[0] = ((d & 0x00F00000) >> 16) | ((d & 0x00F00000) >> 20);
        bg[1] = ((d & 0x000F0000) >> 12) | ((d & 0x000F0000) >> 16);
        bb[0] = ((d & 0x0000F000) >> 8) | ((d & 0x0000F000) >> 12);
        bb[1] = ((d & 0x00000F00) >> 4) | ((d & 0x00000F00) >> 8);
    }

    /*single_color(dst);
    return;*/

    unsigned int tcw[2];
    tcw[0] = (d & 0xE0) >> 5;
    tcw[1] = (d & 0x1C) >> 2;

    /*      Store weights       */
    uint32_t b1 = (d >> 32) & 0xFFFF;
    uint32_t b2 = (d >> 48);

    b1 = (b1 | (b1 << 8)) & 0x00FF00FF;
    b1 = (b1 | (b1 << 4)) & 0x0F0F0F0F;
    b1 = (b1 | (b1 << 2)) & 0x33333333;
    b1 = (b1 | (b1 << 1)) & 0x55555555;

    b2 = (b2 | (b2 << 8)) & 0x00FF00FF;
    b2 = (b2 | (b2 << 4)) & 0x0F0F0F0F;
    b2 = (b2 | (b2 << 2)) & 0x33333333;
    b2 = (b2 | (b2 << 1)) & 0x55555555;

    uint32_t idx = b1 | (b2 << 1);
    uint8_t weightbuf[16]{0};

    // column major to row major
    col2row2bit((uint8_t *)&idx, weightbuf, weight_block_size.x, weight_block_size.y);

    // Flip weight index (4x4 weight block 만 상정)
    uint32_t *weight = (uint32_t *)weightbuf;
    const uint32_t MSB_MASK = 0xaaaaaaaa;
    uint32_t toggle = ((~(*weight)) & MSB_MASK) >> 1;
    *weight ^= toggle;
    *weight = ~(*weight); // index 0 -> lowest value, index 3 -> highest value

    for (int i = 0; i < 16; i++)
    {
        dst[i] = static_cast<uint8_t>(bitrev8(weightbuf[15 - i]));
    }
    /*      End       */

    // Block mode for
    //  weight range [0, 3]
    //  Low-precision
    //  Weight block width 4
    //  Weight block height 4
    //  No dual plane
    const uint16_t block_mode = 0x0042;

    const unsigned int partition_count = 2;
    const uint16_t partition_index_4x2 = 136; // (4x4 weight block 만 상정)
    const uint16_t partition_index_2x4 = 28;  // (4x4 weight block 만 상정)
    const unsigned int PARTITION_INDEX_BITS = 10;
    const uint8_t color_format = 8;                   // RGB direct
    const quant_method color_quant_method = QUANT_40; // (4x4 weight block 만 상정)

    write_bits(block_mode, 11, 0, dst);
    write_bits(partition_count - 1, 2, 11, dst);

    if (d & 0x1) // 4x2 subblock
    {
        write_bits(partition_index_4x2, PARTITION_INDEX_BITS, 13, dst);
    }
    else // 2x4 subblock
    {
        write_bits(partition_index_2x4, PARTITION_INDEX_BITS, 13, dst);
    }

    write_bits(color_format << 2, 6, 13 + PARTITION_INDEX_BITS, dst);

    // [0][0] : R low
    // [0][1] : R high
    // [0][2] : G low
    // [0][3] : G high
    // [0][4] : B low
    // [0][5] : B high
    // Same for partition 1
    uint8_t color_values[2][6]{0};
    const uint8_t *pack_table = color_uquant_to_scrambled_pquant_tables[color_quant_method - QUANT_6];

    // partition 0
    const auto tbl0 = g_table[tcw[0]];
    unsigned int base0_rl = br[0] + tbl0[3];
    unsigned int base0_gl = bg[0] + tbl0[3];
    unsigned int base0_bl = bb[0] + tbl0[3];

    unsigned int base0_rh = br[0] + tbl0[1];
    unsigned int base0_gh = bg[0] + tbl0[1];
    unsigned int base0_bh = bb[0] + tbl0[1];

    if (((base0_rl | base0_gl | base0_bl) & ~0xFF) != 0)
    {
        base0_rl = clampu8(base0_rl);
        base0_gl = clampu8(base0_gl);
        base0_bl = clampu8(base0_bl);
    }

    if (((base0_rh | base0_gh | base0_bh) & ~0xFF) != 0)
    {
        base0_rh = clampu8(base0_rh);
        base0_gh = clampu8(base0_gh);
        base0_bh = clampu8(base0_bh);
    }

    color_values[0][0] = pack_table[base0_rl];
    color_values[0][1] = pack_table[base0_rh];
    color_values[0][2] = pack_table[base0_gl];
    color_values[0][3] = pack_table[base0_gh];
    color_values[0][4] = pack_table[base0_bl];
    color_values[0][5] = pack_table[base0_bh];

    // partition 1
    const auto tbl1 = g_table[tcw[1]];
    unsigned int base1_rl = br[1] + tbl1[3];
    unsigned int base1_gl = bg[1] + tbl1[3];
    unsigned int base1_bl = bb[1] + tbl1[3];

    unsigned int base1_rh = br[1] + tbl1[1];
    unsigned int base1_gh = bg[1] + tbl1[1];
    unsigned int base1_bh = bb[1] + tbl1[1];

    if (((base1_rl | base1_gl | base1_bl) & ~0xFF) != 0)
    {
        base1_rl = clampu8(base1_rl);
        base1_gl = clampu8(base1_gl);
        base1_bl = clampu8(base1_bl);
    }

    if (((base1_rh | base1_gh | base1_bh) & ~0xFF) != 0)
    {
        base1_rh = clampu8(base1_rh);
        base1_gh = clampu8(base1_gh);
        base1_bh = clampu8(base1_bh);
    }

    color_values[1][0] = pack_table[base1_rl];
    color_values[1][1] = pack_table[base1_rh];
    color_values[1][2] = pack_table[base1_gl];
    color_values[1][3] = pack_table[base1_gh];
    color_values[1][4] = pack_table[base1_bl];
    color_values[1][5] = pack_table[base1_bh];

    // Encode the color components
    int valuecount_to_encode = 12;

    encode_ise(color_quant_method, valuecount_to_encode, (uint8_t *)color_values, dst,
               19 + PARTITION_INDEX_BITS);
}

static void DecodeRGB(const uint64_t *src, uint8_t *dst, v2i& src_size, v2i& astc_img_size, v2i& block_size, v2i& weight_block_size, float quality)
{
    // ------------------------------------------------------------------------
	// Initialize the default configuration for the block size and quality
	astcenc_config config;
	astcenc_error status;
	status = astcenc_config_init(ASTCENC_PRF_LDR, block_size.x, block_size.y, 1, quality, 0, &config);
	if (status != ASTCENC_SUCCESS)
	{
		printf("ERROR: Codec config init failed: %s\n", astcenc_get_error_string(status));
		return;
	}

	// ------------------------------------------------------------------------
	// Create a context based on the configuration
	astcenc_context* context;
	status = astcenc_context_alloc(&config, 1, &context);
	if (status != ASTCENC_SUCCESS)
	{
		printf("ERROR: Codec context alloc failed: %s\n", astcenc_get_error_string(status));
		return;
	}

    astcenc_contexti* ctx = &context->context;
	block_size_descriptor& bsd = *(ctx->bsd);
	// ------------------------------------------------------------------------

    const int etc2_block_size_x = src_size.x / 4;
    const int etc2_block_size_y = src_size.y / 4;
    const int total_blocks = etc2_block_size_x * etc2_block_size_y;

    #pragma omp parallel for
    for (int block_idx = 0; block_idx < total_blocks; block_idx++)
    {
        int y = block_idx / etc2_block_size_y;
        int x = block_idx % etc2_block_size_y;
        
        uint64_t d = src[block_idx];
        uint8_t *block_dst = dst + (block_idx << 4);
        DecodeRGBPart(d, block_dst, block_size, weight_block_size, bsd);
    }


    // for (int y = 0; y < src_size.y / 4; y++)
    // {
    //     for (int x = 0; x < src_size.x / 4; x++)
    //     {
    //         uint64_t d = *src++;
    //         DecodeRGBPart(d, dst, src_size.x, block_size, weight_block_size);
    //         dst += 16;
    //     }
    // }

    astcenc_context_free(context);
}

void BlockData::transcodeETC2toASTC(uint8_t *astc, float quality)
{
    v2i block_size = {4, 4};
    v2i weight_block_size = {4, 4};
    v2i astc_img_size = {m_size.x / block_size.x, m_size.y / block_size.y};

    const uint64_t *src = (const uint64_t *)(m_data + m_dataOffset);

    switch (m_type)
    {
    case Etc1:
    case Etc2_RGB:
        ::DecodeRGB(src, astc, m_size, astc_img_size, block_size, weight_block_size, quality);
        break;
    default:
        assert(false);
    }
}

/* ============================================================================
    ASTC compressed file loading
============================================================================ */
struct astc_header
{
    uint8_t magic[4];
    uint8_t block_x;
    uint8_t block_y;
    uint8_t block_z;
    uint8_t dim_x[3]; // dims = dim[0] + (dim[1] << 8) + (dim[2] << 16)
    uint8_t dim_y[3]; // Sizes are given in texels;
    uint8_t dim_z[3]; // block count is inferred
};

static const uint32_t ASTC_MAGIC_ID = 0x5CA1AB13;

/**
 * @brief Print a formatted string to stderr.
 */
template <typename... _Args>
static inline void print_error(
    const char *format,
    _Args... args)
{
    fprintf(stderr, format, args...);
}

int store_cimage(
    const astc_compressed_image &img,
    const char *filename)
{
    astc_header hdr;
    hdr.magic[0] = ASTC_MAGIC_ID & 0xFF;
    hdr.magic[1] = (ASTC_MAGIC_ID >> 8) & 0xFF;
    hdr.magic[2] = (ASTC_MAGIC_ID >> 16) & 0xFF;
    hdr.magic[3] = (ASTC_MAGIC_ID >> 24) & 0xFF;

    hdr.block_x = static_cast<uint8_t>(img.block_x);
    hdr.block_y = static_cast<uint8_t>(img.block_y);
    hdr.block_z = static_cast<uint8_t>(img.block_z);

    hdr.dim_x[0] = img.dim_x & 0xFF;
    hdr.dim_x[1] = (img.dim_x >> 8) & 0xFF;
    hdr.dim_x[2] = (img.dim_x >> 16) & 0xFF;

    hdr.dim_y[0] = img.dim_y & 0xFF;
    hdr.dim_y[1] = (img.dim_y >> 8) & 0xFF;
    hdr.dim_y[2] = (img.dim_y >> 16) & 0xFF;

    hdr.dim_z[0] = img.dim_z & 0xFF;
    hdr.dim_z[1] = (img.dim_z >> 8) & 0xFF;
    hdr.dim_z[2] = (img.dim_z >> 16) & 0xFF;

    std::ofstream file(filename, std::ios::out | std::ios::binary);
    if (!file)
    {
        print_error("ERROR: File open failed '%s'\n", filename);
        return 1;
    }

    file.write(reinterpret_cast<char *>(&hdr), sizeof(astc_header));
    file.write(reinterpret_cast<char *>(img.data), img.data_len);
    return 0;
}