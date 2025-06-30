#include <etcpak/BlockData.hpp>
#include <astcenc/astcenc_internal.h>
#include "etc2_to_astc.h"



int main()
{
	const char* in_path = "backpack.pvr";
	const char* out_path = "backpack.astc";

	auto bd = std::make_shared<BlockData>(in_path);

	auto& imgsize = bd->Size();
	astc_compressed_image cimg(4, 4, 1, imgsize.x, imgsize.y, 1);

	bd->transcodeETC2toASTC(cimg.data, ASTCENC_PRE_MEDIUM);

	store_cimage(cimg, out_path);

	return 0;
}