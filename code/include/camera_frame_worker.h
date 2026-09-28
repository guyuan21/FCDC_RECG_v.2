// Latest-frame camera capture worker.
// Owns cv::VideoCapture on a background thread and exposes only a copied latest
// frame to keep the main recognition loop from blocking on camera I/O.

#ifndef CAMERA_FRAME_WORKER_H
#define CAMERA_FRAME_WORKER_H

#include <stdint.h>

#include <atomic>
#include <mutex>
#include <thread>

#include <opencv2/opencv.hpp>

class CameraFrameWorker {
public:
    CameraFrameWorker() = default;
    ~CameraFrameWorker();

    bool start(int width, int height, cv::Mat *first_frame = nullptr);
    void stop();

    bool is_running() const;
    bool latest(cv::Mat &frame, uint64_t *seq = nullptr);

private:
    void run();

    cv::VideoCapture cap_;
    std::thread worker_;
    mutable std::mutex mutex_;
    cv::Mat latest_frame_;
    uint64_t frame_seq_ = 0;
    std::atomic<bool> running_{false};
    std::atomic<bool> stop_requested_{false};
};

#endif // CAMERA_FRAME_WORKER_H
