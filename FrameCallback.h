// FrameCallback.h — Единый тип колбэка для доставки NV12 кадров.
#pragma once

#include <cstdint>
#include <functional>

// Колбэк вызывается декодером при каждом декодированном кадре.
// Параметры:
//   yPlane   — указатель на Y-плоскость NV12
//   uvPlane  — указатель на UV-плоскость NV12
//   width    — ширина кадра в пикселях
//   height   — высота кадра в пикселях
//   strideY  — шаг строки Y-плоскости (в байтах)
//   strideUV — шаг строки UV-плоскости (в байтах)
//   pts      — временная метка кадра (в единицах времени)
using FrameCallback = std::function<void(uint8_t* yPlane, uint8_t* uvPlane,
                                         int width, int height,
                                         int strideY, int strideUV,
                                         int64_t pts)>;
