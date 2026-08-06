// Detection.h — Результат детекции (YOLO) в координатах исходного кадра.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

// Прямоугольник детекции, координаты — пиксели исходного (декодированного)
// кадра до препроцессинга. Ограничены границами кадра.
struct Detection {
    float x1 = 0, y1 = 0;   // левый-верхний угол
    float x2 = 0, y2 = 0;   // правый-нижний угол
    float confidence = 0;   // уверенность (макс. классовая вероятность)
    int classId = -1;       // индекс класса
};

using Detections = std::vector<Detection>;

// Загрузка списка имён классов из файла (по одному имени на строку).
std::vector<std::string> loadClassNames(const std::string& path);
