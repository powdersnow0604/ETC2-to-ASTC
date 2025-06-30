#ifndef __ETC2_TO_ASTC_H__
#define __ETC2_TO_ASTC_H__

#include <etcpak/BlockData.hpp>

/**
 * @brief The payload stored in a compressed ASTC image.
 */
struct astc_compressed_image
{
	/** @brief The block width in texels. */
	unsigned int block_x;

	/** @brief The block height in texels. */
	unsigned int block_y;

	/** @brief The block depth in texels. */
	unsigned int block_z;

	/** @brief The image width in texels. */
	unsigned int dim_x;

	/** @brief The image height in texels. */
	unsigned int dim_y;

	/** @brief The image depth in texels. */
	unsigned int dim_z;

	/** @brief The binary data payload. */
	uint8_t* data;

	/** @brief The binary data length in bytes. */
	size_t data_len;

    astc_compressed_image(unsigned int block_x, unsigned int block_y, unsigned int block_z, unsigned int dim_x, unsigned int dim_y, unsigned int dim_z)
        : block_x(block_x), block_y(block_y), block_z(block_z), dim_x(dim_x), dim_y(dim_y), dim_z(dim_z), data(nullptr)
    {
        data_len = (dim_x / block_x) * (dim_y / block_y) * (dim_z / block_z) * 16;
        data = (uint8_t*)malloc(data_len);
    }

    ~astc_compressed_image()
    {
        if (data)
        {
            free(data);
        }
    }
};


int store_cimage(
	const astc_compressed_image& img,
	const char* filename
);

#endif