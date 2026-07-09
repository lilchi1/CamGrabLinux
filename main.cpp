// Entry point. Spawns a thread per camera, manages dynamic add at runtime.
#include "headers.h"

volatile std::sig_atomic_t g_running = 1;

static void signalHandler(int) {
    g_running = 0;
}

int main() {
    std::signal(SIGINT, signalHandler);
    std::signal(SIGTERM, signalHandler);

    std::cout << "===========================================" << std::endl;
    std::cout << "     Jetson RTSP Software Decoder" << std::endl;
    std::cout << "     FFmpeg + X11 Multimedia" << std::endl;
    std::cout << "===========================================" << std::endl;

    std::cout << "Enter RTSP URLs (comma-separated)." << std::endl;
    std::cout << "  Example: rtsp://admin:pass@192.168.1.100:554/stream" << std::endl;
    std::cout << "URLs: ";

    std::string input;
    std::getline(std::cin, input);

    std::vector<std::string> urls;
    std::stringstream ss(input);
    std::string item;
    while (std::getline(ss, item, ',')) {
        item.erase(std::remove_if(item.begin(), item.end(), ::isspace), item.end());
        if (!item.empty()) urls.push_back(item);
    }

    std::vector<std::thread> threads;

    for (auto& u : urls) {
        int idx;
        {
            std::lock_guard<std::mutex> lock(g_camMtx);
            idx = (int)g_cams.size();
            g_cams.push_back(CamInfo{u, 0, 0, 0, 0, false});
            g_camRunning.push_back(new std::atomic<bool>(true));
        }
        threads.emplace_back(cameraThread, u, idx);
    }
    printAllStatus();

    // Main loop: monitor cameras, allow adding new ones
    while (g_running) {
        bool anyAlive = false;
        {
            std::lock_guard<std::mutex> lock(g_camMtx);
            for (auto* f : g_camRunning)
                if (f && f->load()) { anyAlive = true; break; }
        }

        // If all cameras closed, offer reconnect or exit
        if (!anyAlive) {
            std::cout << "\nAll cameras closed." << std::endl;
            std::cout << "[1] Connect new camera" << std::endl;
            std::cout << "[2] Exit" << std::endl;
            std::cout << "Choice: ";
            std::string opt;
            std::getline(std::cin, opt);
            if (opt != "1") break;
        }

        // Prompt for new camera URL
        std::cout << "Enter new RTSP URL (or 'exit' to quit): ";
        std::string newUrl;
        std::getline(std::cin, newUrl);
        if (newUrl == "exit" || newUrl == "q" || !g_running) break;
        if (newUrl.empty()) { printAllStatus(); continue; }

        int idx;
        {
            std::lock_guard<std::mutex> lock(g_camMtx);
            idx = (int)g_cams.size();
            g_cams.push_back(CamInfo{newUrl, 0, 0, 0, 0, false});
            g_camRunning.push_back(new std::atomic<bool>(true));
        }
        threads.emplace_back(cameraThread, newUrl, idx);
        printAllStatus();
    }

    // Shutdown: signal all threads, join, cleanup
    {
        std::lock_guard<std::mutex> lock(g_camMtx);
        for (auto* f : g_camRunning) if (f) *f = false;
    }
    for (auto& t : threads) if (t.joinable()) t.join();
    {
        std::lock_guard<std::mutex> lock(g_camMtx);
        for (auto* f : g_camRunning) delete f;
        g_camRunning.clear();
        g_cams.clear();
    }

    std::cout << "Exiting." << std::endl;
    return 0;
}
