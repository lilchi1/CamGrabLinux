// FrameCallback — shared typedef for NV12 frame delivery callback.
#pragma once

#include <cstdint>
#include <functional>

using FrameCallback = std::function<void(uint8_t* yPlane, uint8_t* uvPlane,
                                         int width, int height,
                                         int strideY, int strideUV,
                                         int64_t pts)>;
