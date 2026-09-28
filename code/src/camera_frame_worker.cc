#include "camera_frame_worker.h"

#include <stdio.h>
#include <utility>
#include <unistd.h>

CameraFrameWorker::~CameraFrameWorker()
{
    stop();
}

bool CameraFrameWorker::start(int width, int height, cv::Mat *first_frame)
{
    stop();

    if (!cap_.open(0)) {
        printf("[Camera] open failed\n");
        return false;
    }
    cap_.set(cv::CAP_PROP_FRAME_WIDTH, width);
    cap_.set(cv::CAP_PROP_FRAME_HEIGHT, height);

    cv::Mat probe;
    cap_ >> probe;
    if (probe.empty()) {
        printf("[Camera] first frame empty, check sensor/ISP\n");
        cap_.release();
        return false;
    }

    {
        std::lock_guard<std::mutex> lock(mutex_);
        latest_frame_ = probe.clone();
        frame_seq_ = 1;
        stop_requested_ = false;
        running_ = true;
    }
    if (first_frame) probe.copyTo(*first_frame);

    try {
        worker_ = std::thread(&CameraFrameWorker::run, this);
    } catch (...) {
        stop_requested_.store(true);
        running_.store(false);
        cap_.release();
        return false;
    }
    return true;
}

void CameraFrameWorker::stop()
{
    if (!running_.load() && !cap_.isOpened()) return;
    stop_requested_.store(true);
    if (worker_.joinable()) worker_.join();
    cap_.release();
    {
        std::lock_guard<std::mutex> lock(mutex_);
        latest_frame_.release();
        frame_seq_ = 0;
    }
    running_.store(false);
    stop_requested_.store(false);
}

bool CameraFrameWorker::is_running() const
{
    return running_.load();
}

bool CameraFrameWorker::latest(cv::Mat &frame, uint64_t *seq)
{
    std::lock_guard<std::mutex> lock(mutex_);
    if (latest_frame_.empty()) return false;
    latest_frame_.copyTo(frame);
    if (seq) *seq = frame_seq_;
    return true;
}

void CameraFrameWorker::run()
{
    cv::Mat frame;
    while (!stop_requested_.load()) {
        cap_ >> frame;
        if (frame.empty()) {
            usleep(5000);
            continue;
        }
        {
            std::lock_guard<std::mutex> lock(mutex_);
            std::swap(latest_frame_, frame);
            ++frame_seq_;
        }
    }
}
