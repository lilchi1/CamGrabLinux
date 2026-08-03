// Log.h — Логирование и вывод статуса камер.
#pragma once

#include <string>

// Текущее время в формате "YYYY-MM-DD HH:MM:SS"
std::string getCurrentTimestamp();

// Потокобезопасный вывод лога с меткой времени
void logWrite(const std::string& level, const std::string& url, const std::string& msg);

// Печать статуса всех камер
void printAllStatus();
