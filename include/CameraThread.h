// CameraThread.h — Главный поток камеры: RTSP → декодирование → отображение.
#pragma once

#include <string>

void cameraThread(std::string url, int camIdx);
