// stb single-header implementations (image decode/encode) - replaces OpenCV
// imgcodecs so the portable build avoids the GDAL/DB/poppler dependency closure.
#ifdef _WIN32
// stb's stdio wrappers otherwise call the narrow Windows fopen API.  The
// runner stores CLI paths as UTF-8, so enable the built-in UTF-8 -> UTF-16
// conversion for both readers and writers.
#define STBI_WINDOWS_UTF8
#define STBIW_WINDOWS_UTF8
#endif
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_write.h"
