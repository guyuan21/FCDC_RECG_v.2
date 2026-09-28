// =============================================================================
// main.cc  —  RV1106 人脸考勤与权限管理主程序
// Platform : Luckfox Pico Pro Max (RV1106)  |  Display: 2.8" TFT (240×320)
//
// 功能：
//   1. SCRFD 人脸检测（640×640）
//   2. MobileFace/ArcFace 人脸特征提取与比对（多样本均值策略）
//   3. 工人考勤打卡（冷却去重 + CSV日志）
//   4. 门禁权限判断
//   5. 2.8" TFT LCD 实时显示（240×320，SPI）
//   6. RV1106 协处理器 NPU 全程零拷贝推理
// =============================================================================

#ifndef RV1106_1103
#define RV1106_1103
#endif

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <time.h>
#include <unistd.h>
#include <signal.h>
#include <sys/time.h>
#include <sys/stat.h>
#include <ctype.h>
#include <dirent.h>
#include <errno.h>

#include <string>
#include <vector>
#include <sstream>
#include <algorithm>
#include <array>
#include <cmath>
#include <atomic>
#include <condition_variable>
#include <mutex>
#include <thread>
#include <opencv2/opencv.hpp>

#include "DEV_Config.h"
#include "LCD_Driver.h"
#include "GUI_Paint.h"
#include "AS608.h"
#include "attendance_service.h"
#include "camera_frame_worker.h"
#include "employee_management.h"
#include "uart_control.h"
#include "face_models.h"
#include "servo_gimbal_tracker.h"
#include "yolov5.h"

extern UWORD *BlackImage;

// =============================================================================
// 配置常量
// LCD_Driver.h: LCD_2_8_WIDTH=240, LCD_2_8_HEIGHT=320
// =============================================================================
#define DISP_W          240
#define DISP_H          320
#define RETINA_W        640
#define RETINA_H        640

#define CAM_W           640
#define CAM_H           480
#define YOLO_MODEL_PATH "./model/yolov5n.rknn"
#define UART_CONTROL_DEV "/dev/ttyS2"
#define UART_UPLOAD_DEV "/dev/ttyS4"
#define FINGERPRINT_DEV "/dev/ttyS1"
#define FINGERPRINT_BAUDRATE 115200
#define DUTY_INFER_INTERVAL_MS 800
#define LCD_FLUSH_INTERVAL_MS 25
#define LCD_SPI_SPEED_HZ 35000000u
#define LCD_STATUS_BAR_H 20
#define LCD_EVENT_PANEL_H 34
#define FACE_DETECT_INTERVAL_MS 250
#define FINGERPRINT_POLL_INTERVAL_MS 350
#define FINGERPRINT_DEFAULT_PAGE_NUM 300
#define FACE_RECOG_INTERVAL_MS 150
#define FACE_RECOG_CACHE_TTL_MS 700
#define FACE_PROCESS_MAX_FACES 1
#define ENROLL_FACE_SAMPLE_TARGET 2
#define FACE_FEATURE_FUSION_MIN_FRAMES 2
#define FACE_FEATURE_FUSION_MAX_FRAMES 3
#define FACE_QUALITY_MIN_SHARPNESS 5.0
#define FPS_SMOOTH_ALPHA 0.20f
#define APP_VERBOSE_LOG 0
#define EMPLOYEE_CHECKIN_TIMEOUT_S 15
#define EMPLOYEE_ENROLL_TIMEOUT_S 30

// 多帧确认：连续 FACE_CONFIRM_FRAMES 帧识别同一人才触发打卡
// 避免单帧误识别
static const int CONFIRM_FRAMES = FACE_CONFIRM_FRAMES;

static void init_app_timezone()
{
    const char *configured_tz = getenv("ATTENDANCE_TIMEZONE");
    if (configured_tz && configured_tz[0]) {
        setenv("TZ", configured_tz, 1);
    } else {
        setenv("TZ", "CST-8", 1);
    }
    tzset();
}

static bool env_flag_enabled(const char *name)
{
    const char *value = getenv(name);
    if (!value || !value[0]) return false;
    return strcmp(value, "1") == 0 ||
           strcasecmp(value, "true") == 0 ||
           strcasecmp(value, "yes") == 0 ||
           strcasecmp(value, "on") == 0;
}

struct UiBanner {
    std::string title;
    std::string line1;
    std::string line2;
    time_t      updated_at = 0;
};

// =============================================================================
// LCD 显示辅助
// =============================================================================

static std::string format_time_text(time_t ts)
{
    char buf[32];
    struct tm tm_info;
    if (!localtime_r(&ts, &tm_info)) return "N/A";
    strftime(buf, sizeof(buf), "%Y-%m-%d %H:%M:%S", &tm_info);
    return std::string(buf);
}

static void set_ui_banner(UiBanner &banner,
                          const std::string &title,
                          const std::string &line1,
                          const std::string &line2)
{
    banner.title = title;
    banner.line1 = line1;
    banner.line2 = line2;
    banner.updated_at = time(nullptr);
}

static std::string trim_ascii_spaces(const std::string &text)
{
    size_t start = 0;
    while (start < text.size() && isspace((unsigned char)text[start])) start++;

    size_t end = text.size();
    while (end > start && isspace((unsigned char)text[end - 1])) end--;
    return text.substr(start, end - start);
}

static std::string ascii_printable_only(const std::string &text)
{
    std::string out;
    for (size_t i = 0; i < text.size(); ++i) {
        unsigned char ch = (unsigned char)text[i];
        if (ch >= 32 && ch <= 126) out.push_back((char)ch);
    }
    return out;
}

static std::string compact_employee_label(const std::string &name, uint32_t employee_id)
{
    std::string ascii_name = trim_ascii_spaces(ascii_printable_only(name));
    if (!ascii_name.empty()) {
        return ascii_name + "  ID " + std::to_string(employee_id);
    }
    return "ID " + std::to_string(employee_id);
}

static std::string trim_to_fit(const std::string &text, double scale, int max_w)
{
    std::string out = ascii_printable_only(text);
    if (out.empty()) return out;

    int baseline = 0;
    while (!out.empty()) {
        cv::Size size = cv::getTextSize(out, cv::FONT_HERSHEY_SIMPLEX, scale, 1, &baseline);
        if (size.width <= max_w) return out;
        out.pop_back();
    }
    return out;
}

static std::string compact_banner_title(const UiBanner &banner)
{
    const std::string &title = banner.title;
    const std::string &line1 = banner.line1;
    bool success = title.find("成功") != std::string::npos ||
                   title.find("OK") != std::string::npos;

    if (title.find("人脸采集") != std::string::npos) return "CAPTURE";
    if (title.find("指纹录入中") != std::string::npos) return "FINGERPRINT";
    if (title.find("身份不匹配") != std::string::npos) return "MISMATCH";
    if (title.find("录入") != std::string::npos) return success ? "ENROLL OK" : "ENROLL";
    if (title.find("签到") != std::string::npos ||
        title.find("签入") != std::string::npos) return success ? "CHECK IN OK" : "CHECK IN";
    if (title.find("签退") != std::string::npos) return success ? "CHECK OUT OK" : "CHECK OUT";
    if (title.find("确认中") != std::string::npos) return "VERIFY";
    if (title.find("进入在岗检测") != std::string::npos ||
        title.find("在岗检测中") != std::string::npos) return "ON DUTY";
    if (title.find("未检测到在岗") != std::string::npos) return "OFF DUTY";
    if (title.find("失败") != std::string::npos ||
        title.find("异常") != std::string::npos) return "ERROR";
    if (title.find("模型未就绪") != std::string::npos) return "MODEL OFF";
    if (title.find("待机") != std::string::npos) return "READY";
    if (line1.find("YOLO") != std::string::npos) return "DUTY";

    std::string ascii = ascii_printable_only(title);
    return ascii.empty() ? "READY" : ascii;
}

static std::string compact_banner_detail(const UiBanner &banner, bool uart_ready)
{
    std::string line = trim_ascii_spaces(ascii_printable_only(banner.line1));
    std::string detail = trim_ascii_spaces(ascii_printable_only(banner.line2));

    const std::string &title = banner.title;
    if (title.find("待机") != std::string::npos) {
        return uart_ready ? "WAIT CMD / FACE" : "UART TX OFF";
    }
    if (title.find("签到请求") != std::string::npos) {
        return "FACE OR FINGER";
    }
    if (title.find("进入在岗检测") != std::string::npos) {
        return "WAIT CHECK OUT";
    }

    if (line.empty()) line = detail;
    if (line.empty()) line = uart_ready ? "TX OK" : "TX OFF";
    if (!line.empty() && line[0] == '#') line = "ID " + trim_ascii_spaces(line.substr(1));
    return line;
}

static cv::Scalar banner_accent_color(const UiBanner &banner, bool uart_ready)
{
    const std::string &title = banner.title;
    if (title.find("失败") != std::string::npos ||
        title.find("异常") != std::string::npos ||
        title.find("不匹配") != std::string::npos) {
        return cv::Scalar(0, 70, 255);
    }
    if (title.find("录入中") != std::string::npos ||
        title.find("确认中") != std::string::npos) {
        return cv::Scalar(0, 185, 255);
    }
    if (title.find("成功") != std::string::npos ||
        title.find("在岗") != std::string::npos) {
        return cv::Scalar(0, 220, 110);
    }
    return uart_ready ? cv::Scalar(0, 200, 220) : cv::Scalar(0, 150, 255);
}

static void draw_text_line(cv::Mat &bgr, const std::string &text, int x, int y,
                           const cv::Scalar &color, double scale, int max_w)
{
    std::string clipped = trim_to_fit(text, scale, max_w);
    if (clipped.empty()) return;
    cv::putText(bgr, clipped, cv::Point(x, y),
                cv::FONT_HERSHEY_SIMPLEX, scale, color, 1, cv::LINE_AA);
}

static void draw_text_line_right(cv::Mat &bgr, const std::string &text, int right, int y,
                                 const cv::Scalar &color, double scale, int max_w)
{
    std::string clipped = trim_to_fit(text, scale, max_w);
    if (clipped.empty()) return;

    int baseline = 0;
    cv::Size text_size = cv::getTextSize(clipped, cv::FONT_HERSHEY_SIMPLEX,
                                         scale, 1, &baseline);
    int x = std::max(2, right - text_size.width);
    cv::putText(bgr, clipped, cv::Point(x, y),
                cv::FONT_HERSHEY_SIMPLEX, scale, color, 1, cv::LINE_AA);
}

// 绘制顶部状态栏（时间(到分) + 主循环/LCD刷新速率）
static void draw_status_bar(cv::Mat &bgr, float loop_fps, float lcd_fps)
{
    char time_text[16];
    time_t now = time(nullptr);
    struct tm tm_info;
    if (localtime_r(&now, &tm_info)) {
        if (strftime(time_text, sizeof(time_text), "%H:%M", &tm_info) <= 0) {
            snprintf(time_text, sizeof(time_text), "--:--");
        } else {
            time_text[sizeof(time_text) - 1] = '\0';
        }
    } else {
        snprintf(time_text, sizeof(time_text), "--:--");
    }

    char fps_text[32];
    snprintf(fps_text, sizeof(fps_text), "L%.0f D%.0f", loop_fps, lcd_fps);

    cv::rectangle(bgr, cv::Point(0, 0), cv::Point(DISP_W - 1, LCD_STATUS_BAR_H - 1),
                  cv::Scalar(8, 10, 12), cv::FILLED);
    cv::line(bgr, cv::Point(0, LCD_STATUS_BAR_H - 1),
             cv::Point(DISP_W - 1, LCD_STATUS_BAR_H - 1), cv::Scalar(42, 52, 58), 1);
    draw_text_line(bgr, time_text, 6, 15, cv::Scalar(220, 245, 230), 0.43, 70);

    int baseline = 0;
    cv::Size fps_size = cv::getTextSize(fps_text, cv::FONT_HERSHEY_SIMPLEX, 0.34, 1, &baseline);
    int fps_x = std::max(86, DISP_W - fps_size.width - 6);
    cv::putText(bgr, fps_text, cv::Point(fps_x, 14),
                cv::FONT_HERSHEY_SIMPLEX, 0.34, cv::Scalar(170, 210, 255), 1, cv::LINE_AA);
}

static void draw_event_panel(cv::Mat &bgr, const UiBanner &banner, bool uart_ready)
{
    const int panel_h = LCD_EVENT_PANEL_H;
    const int top = DISP_H - panel_h;
    const int split_x = 118;
    const int text_y = top + 22;

    cv::rectangle(bgr, cv::Point(0, top), cv::Point(DISP_W - 1, DISP_H - 1),
                  cv::Scalar(15, 17, 20), cv::FILLED);
    cv::line(bgr, cv::Point(0, top), cv::Point(DISP_W - 1, top),
             cv::Scalar(48, 58, 64), 1);

    cv::Scalar accent = banner_accent_color(banner, uart_ready);
    cv::rectangle(bgr, cv::Point(0, top + 1), cv::Point(3, DISP_H - 1), accent, cv::FILLED);
    cv::circle(bgr, cv::Point(13, top + 17), 3, accent, cv::FILLED);
    cv::line(bgr, cv::Point(split_x, top + 6),
             cv::Point(split_x, DISP_H - 6), cv::Scalar(44, 52, 58), 1);

    draw_text_line(bgr, compact_banner_title(banner), 24, text_y,
                   cv::Scalar(240, 245, 245), 0.35, split_x - 30);
    draw_text_line_right(bgr, compact_banner_detail(banner, uart_ready),
                         DISP_W - 8, text_y,
                         cv::Scalar(178, 216, 248), 0.31,
                         DISP_W - split_x - 12);
}

static std::string compact_face_label(const char *label, bool is_known, bool has_perm)
{
    if (!is_known) return "STRANGER";

    std::string text = label ? label : "";
    size_t id_sep = text.rfind('#');
    if (id_sep != std::string::npos && id_sep + 1 < text.size()) {
        text = "ID " + text.substr(id_sep + 1);
    } else {
        text = ascii_printable_only(text);
        if (text.empty()) text = "KNOWN";
    }

    text += has_perm ? " OK" : " DENY";
    return text;
}

enum FaceBoxVisualState {
    FACE_BOX_DETECT_ONLY = 0,
    FACE_BOX_ACCEPTED,
    FACE_BOX_REJECTED,
};

// 绘制检测框 + 识别结果标签
static void draw_face_result(cv::Mat &bgr,
                              const object_detect_result *det,
                              const char *label,
                              FaceBoxVisualState visual_state,
                              bool is_known,
                              bool has_perm,
                              float dist)
{
    if (!det) return;

    const int status_h = LCD_STATUS_BAR_H;
    const int panel_top = DISP_H - LCD_EVENT_PANEL_H;
    int left = std::max(0, std::min(det->box.left, DISP_W - 1));
    int top = std::max(status_h, std::min(det->box.top, panel_top - 1));
    int right = std::max(0, std::min(det->box.right, DISP_W - 1));
    int bottom = std::max(status_h, std::min(det->box.bottom, panel_top - 1));
    if (right <= left || bottom <= top) return;

    cv::Scalar box_color(245, 245, 245);
    cv::Scalar text_color(245, 245, 245);
    switch (visual_state) {
    case FACE_BOX_ACCEPTED:
        box_color = cv::Scalar(0, 230, 90);
        text_color = cv::Scalar(200, 255, 210);
        break;
    case FACE_BOX_REJECTED:
        box_color = cv::Scalar(0, 60, 255);
        text_color = cv::Scalar(210, 210, 255);
        break;
    case FACE_BOX_DETECT_ONLY:
    default:
        break;
    }

    int corner = std::max(6, std::min(16, std::min(right - left, bottom - top) / 4));
    cv::line(bgr, cv::Point(left, top), cv::Point(left + corner, top), box_color, 2);
    cv::line(bgr, cv::Point(left, top), cv::Point(left, top + corner), box_color, 2);
    cv::line(bgr, cv::Point(right, top), cv::Point(right - corner, top), box_color, 2);
    cv::line(bgr, cv::Point(right, top), cv::Point(right, top + corner), box_color, 2);
    cv::line(bgr, cv::Point(left, bottom), cv::Point(left + corner, bottom), box_color, 2);
    cv::line(bgr, cv::Point(left, bottom), cv::Point(left, bottom - corner), box_color, 2);
    cv::line(bgr, cv::Point(right, bottom), cv::Point(right - corner, bottom), box_color, 2);
    cv::line(bgr, cv::Point(right, bottom), cv::Point(right, bottom - corner), box_color, 2);

    if (!is_known) return;

    char info[48];
    std::string face_label = compact_face_label(label, is_known, has_perm);
    snprintf(info, sizeof(info), "%s", face_label.c_str());

    const double scale = 0.31;
    int baseline = 0;
    std::string clipped = trim_to_fit(info, scale, DISP_W - 6);
    cv::Size text_size = cv::getTextSize(clipped, cv::FONT_HERSHEY_SIMPLEX, scale, 1, &baseline);
    int text_x = std::max(2, std::min(left, DISP_W - text_size.width - 4));
    int box_y = (top - 16 >= status_h) ? top - 16 : std::min(bottom + 3, panel_top - 14);
    cv::rectangle(bgr,
                  cv::Point(text_x - 2, box_y),
                  cv::Point(text_x + text_size.width + 3, box_y + 13),
                  cv::Scalar(0, 0, 0), cv::FILLED);
    cv::putText(bgr, clipped, cv::Point(text_x, box_y + 10),
                cv::FONT_HERSHEY_SIMPLEX, scale, text_color, 1, cv::LINE_AA);
}

// 刷新 LCD（BGR→BGR565 转换 + SPI 传输，使用预分配缓冲）
static void lcd_flush(cv::Mat &bgr, cv::Mat &bgr565)
{
    cv::cvtColor(bgr, bgr565, cv::COLOR_BGR2BGR565);
    uint32_t *pairs = reinterpret_cast<uint32_t *>(bgr565.data);
    const size_t pair_count = ((size_t)DISP_W * (size_t)DISP_H) / 2u;
    for (size_t i = 0; i < pair_count; ++i) {
        uint32_t v = pairs[i];
        pairs[i] = ((v & 0x00ff00ffu) << 8) | ((v & 0xff00ff00u) >> 8);
    }
    LCD_Display((UWORD *)bgr565.data);
}

class LcdRefreshWorker {
public:
    ~LcdRefreshWorker()
    {
        stop();
    }

    bool start()
    {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (running_) return true;
            pending_bgr_.create(DISP_H, DISP_W, CV_8UC3);
            has_pending_ = false;
            stop_requested_ = false;
            running_ = true;
            lcd_fps_x10_.store(0);
        }

        try {
            worker_ = std::thread(&LcdRefreshWorker::run, this);
        } catch (...) {
            std::lock_guard<std::mutex> lock(mutex_);
            running_ = false;
            stop_requested_ = false;
            return false;
        }
        return true;
    }

    void stop()
    {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (!running_) return;
            stop_requested_ = true;
            has_pending_ = false;
        }
        cond_.notify_one();
        if (worker_.joinable()) worker_.join();

        std::lock_guard<std::mutex> lock(mutex_);
        running_ = false;
        stop_requested_ = false;
        has_pending_ = false;
        lcd_fps_x10_.store(0);
    }

    bool submit(const cv::Mat &bgr)
    {
        if (bgr.empty() || bgr.cols != DISP_W || bgr.rows != DISP_H || bgr.type() != CV_8UC3) {
            return false;
        }

        std::unique_lock<std::mutex> lock(mutex_, std::try_to_lock);
        if (!lock.owns_lock() || !running_ || stop_requested_) {
            return false;
        }

        bgr.copyTo(pending_bgr_);
        has_pending_ = true;
        lock.unlock();
        cond_.notify_one();
        return true;
    }

    float lcd_fps() const
    {
        return (float)lcd_fps_x10_.load() / 10.f;
    }

private:
    void run()
    {
        cv::Mat local_bgr(DISP_H, DISP_W, CV_8UC3);
        cv::Mat local_bgr565(DISP_H, DISP_W, CV_8UC2);
        struct timeval last_flush;
        memset(&last_flush, 0, sizeof(last_flush));
        float smoothed_fps = 0.f;

        while (true) {
            {
                std::unique_lock<std::mutex> lock(mutex_);
                cond_.wait(lock, [&]() { return stop_requested_ || has_pending_; });
                if (stop_requested_) break;
                std::swap(local_bgr, pending_bgr_);
                has_pending_ = false;
            }

            lcd_flush(local_bgr, local_bgr565);

            struct timeval now;
            gettimeofday(&now, nullptr);
            if (last_flush.tv_sec != 0 || last_flush.tv_usec != 0) {
                long delta_ms = (now.tv_sec - last_flush.tv_sec) * 1000L +
                                (now.tv_usec - last_flush.tv_usec) / 1000L;
                if (delta_ms > 0) {
                    float instant = 1000.f / (float)delta_ms;
                    smoothed_fps = smoothed_fps <= 0.f
                        ? instant
                        : smoothed_fps * 0.80f + instant * 0.20f;
                    lcd_fps_x10_.store(std::max(0, (int)(smoothed_fps * 10.f + 0.5f)));
                }
            }
            last_flush = now;
        }
    }

    mutable std::mutex mutex_;
    std::condition_variable cond_;
    std::thread worker_;
    cv::Mat pending_bgr_;
    bool running_ = false;
    bool stop_requested_ = false;
    bool has_pending_ = false;
    std::atomic<int> lcd_fps_x10_{0};
};

static bool resize_cover_to_mat(const cv::Mat &src, cv::Mat &dst,
                                image_transform_t *transform,
                                cv::Mat &resize_buf)
{
    if (src.empty() || dst.empty()) return false;

    int dw = dst.cols;
    int dh = dst.rows;
    float scale = std::max((float)dw / (float)src.cols,
                           (float)dh / (float)src.rows);
    int rw = std::max(1, (int)(src.cols * scale + 0.5f));
    int rh = std::max(1, (int)(src.rows * scale + 0.5f));
    int lpad = (dw - rw) / 2;
    int tpad = (dh - rh) / 2;

    resize_buf.create(rh, rw, CV_8UC3);
    cv::resize(src, resize_buf, cv::Size(rw, rh), 0, 0, cv::INTER_LINEAR);

    int dst_x = std::max(0, lpad);
    int dst_y = std::max(0, tpad);
    int src_x = std::max(0, -lpad);
    int src_y = std::max(0, -tpad);
    int copy_w = std::min(dw - dst_x, rw - src_x);
    int copy_h = std::min(dh - dst_y, rh - src_y);
    if (copy_w <= 0 || copy_h <= 0) return false;

    cv::Mat dst_roi = dst(cv::Rect(dst_x, dst_y, copy_w, copy_h));
    cv::Mat src_roi = resize_buf(cv::Rect(src_x, src_y, copy_w, copy_h));
    src_roi.copyTo(dst_roi);

    if (transform) {
        memset(transform, 0, sizeof(*transform));
        transform->src_w = src.cols;
        transform->src_h = src.rows;
        transform->dst_w = dst.cols;
        transform->dst_h = dst.rows;
        transform->resized_w = rw;
        transform->resized_h = rh;
        transform->pad_left = lpad;
        transform->pad_top = tpad;
        transform->scale = scale;
    }
    return true;
}

static bool source_to_display_coordinates(const image_transform_t &transform,
                                          int *x, int *y)
{
    if (!x || !y || transform.scale <= 0.f ||
        transform.dst_w <= 0 || transform.dst_h <= 0) {
        return false;
    }

    float px = (float)*x * transform.scale + (float)transform.pad_left;
    float py = (float)*y * transform.scale + (float)transform.pad_top;
    *x = std::max(0, std::min((int)(px + 0.5f), transform.dst_w - 1));
    *y = std::max(0, std::min((int)(py + 0.5f), transform.dst_h - 1));
    return true;
}

static void draw_centered_text(cv::Mat &bgr, const std::string &text, int baseline,
                               double scale, const cv::Scalar &color)
{
    int text_baseline = 0;
    cv::Size text_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX,
                                         scale, 1, &text_baseline);
    int x = std::max(2, (bgr.cols - text_size.width) / 2);
    cv::putText(bgr, text, cv::Point(x, baseline),
                cv::FONT_HERSHEY_SIMPLEX, scale, color, 1);
}

static void lcd_show_exit_screen(cv::Mat &bgr)
{
    cv::Mat bgr565(DISP_H, DISP_W, CV_8UC2);
    bgr.setTo(cv::Scalar(0, 0, 0));
    cv::rectangle(bgr, cv::Point(0, 0), cv::Point(DISP_W - 1, 28),
                  cv::Scalar(20, 80, 20), cv::FILLED);
    draw_centered_text(bgr, "ATTENDANCE", 20, 0.50, cv::Scalar(220, 255, 220));

    draw_centered_text(bgr, "APP STOPPED", 122, 0.58, cv::Scalar(0, 255, 0));
    draw_centered_text(bgr, "Safe exit", 168, 0.42, cv::Scalar(200, 220, 200));
    draw_centered_text(bgr, format_time_text(time(nullptr)), 212, 0.40,
                       cv::Scalar(180, 180, 180));
    lcd_flush(bgr, bgr565);
}

static bool lcd_start_display()
{
    if (DEV_ModuleInit() != 0) {
        printf("[LCD] GPIO init fail, display disabled.\n");
        return false;
    }

    LCD_SCAN_DIR scan = R2L_D2U;
    LCD_Init(scan, 800);
    uint32_t lcd_spi_speed = LCD_SPI_SPEED_HZ;
    const char *lcd_spi_mhz = getenv("ATTENDANCE_LCD_SPI_MHZ");
    if (lcd_spi_mhz && lcd_spi_mhz[0]) {
        char *end = nullptr;
        long mhz = strtol(lcd_spi_mhz, &end, 10);
        if (end != lcd_spi_mhz && mhz >= 10 && mhz <= 60) {
            lcd_spi_speed = (uint32_t)mhz * 1000u * 1000u;
        }
    }
    int speed_ret = DEV_HARDWARE_SPI_setSpeed(lcd_spi_speed);
#if APP_VERBOSE_LOG
    printf("[LCD] SPI speed request=%u actual=%u ret=%d\n",
           lcd_spi_speed, DEV_HARDWARE_SPI_getSpeed(), speed_ret);
#else
    (void)speed_ret;
#endif
    Paint_CreatImage();
    Paint_Clear(WHITE);
    LCD_BL_1;
    LCD_Display(BlackImage);
#if APP_VERBOSE_LOG
    printf("[LCD] 2.8\" TFT ready  %dx%d\n", DISP_W, DISP_H);
#endif
    return true;
}

static void lcd_stop_display()
{
    LCD_BL_0;
}

// =============================================================================
// 多帧确认状态
// =============================================================================
struct ConfirmState {
    std::string pending_name;
    int         confirm_count = 0;
};

struct FaceRecognizeCache {
    bool valid = false;
    struct timeval last_run = {0, 0};
    std::string label = "STRANGER";
    const FaceEntry *match = nullptr;
    bool is_known = false;
    bool has_perm = false;
    bool compared = false;
    float dist = -1.f;
    bool fusion_valid = false;
    int fusion_worker_id = 0;
    std::string fusion_label;
    int fusion_count = 0;
    int fusion_next = 0;
    std::array<std::array<float, FACENET_FEAT_DIM>, FACE_FEATURE_FUSION_MAX_FRAMES> fusion_feats = {};
};

enum DetectState {
    DETECT_STATE_FACE_ACTIVE = 0,
    DETECT_STATE_DUTY_ACTIVE = 1,
    DETECT_STATE_SLEEP = 2,
};

enum RecognitionPurpose {
    RECOGNITION_PURPOSE_CHECKIN = 0,
    RECOGNITION_PURPOSE_CHECKOUT = 1,
};

struct FingerprintRuntime {
    AS608 sensor;
    bool ready = false;
    uint16_t page_num = FINGERPRINT_DEFAULT_PAGE_NUM;
    struct timeval last_poll = {0, 0};
};

struct DutyRuntime {
    time_t started_at = 0;
    time_t last_eval_at = 0;
    int    on_duty_seconds = 0;
    int    total_seconds = 0;
    bool   last_on_duty = false;
    float  best_score = 0.f;
};

struct FpsCounter {
    struct timeval last_frame = {0, 0};
    float value = 0.f;
};

static long elapsed_ms(const struct timeval &a, const struct timeval &b)
{
    return (b.tv_sec - a.tv_sec) * 1000L + (b.tv_usec - a.tv_usec) / 1000L;
}

static bool timeval_is_zero(const struct timeval &tv)
{
    return tv.tv_sec == 0 && tv.tv_usec == 0;
}

static float update_loop_fps(FpsCounter &counter)
{
    struct timeval now;
    gettimeofday(&now, nullptr);

    if (timeval_is_zero(counter.last_frame)) {
        counter.last_frame = now;
        return counter.value;
    }

    long delta_ms = elapsed_ms(counter.last_frame, now);
    counter.last_frame = now;
    if (delta_ms <= 0) return counter.value;

    float instant = 1000.f / (float)delta_ms;
    if (counter.value <= 0.f) {
        counter.value = instant;
    } else {
        counter.value = counter.value * (1.f - FPS_SMOOTH_ALPHA) + instant * FPS_SMOOTH_ALPHA;
    }
    return counter.value;
}

static bool should_flush_lcd(struct timeval &last_flush)
{
    struct timeval now;
    gettimeofday(&now, nullptr);
    if (timeval_is_zero(last_flush)) {
        last_flush = now;
        return true;
    }
    if (elapsed_ms(last_flush, now) < LCD_FLUSH_INTERVAL_MS) {
        return false;
    }
    last_flush = now;
    return true;
}

static std::string format_duration_text(int seconds)
{
    if (seconds < 0) seconds = 0;
    char buf[32];
    snprintf(buf, sizeof(buf), "%02d:%02d:%02d",
             seconds / 3600, (seconds / 60) % 60, seconds % 60);
    return std::string(buf);
}

static void reset_confirm_states(ConfirmState *confirm_states, int count)
{
    if (!confirm_states || count <= 0) return;
    for (int i = 0; i < count; ++i) {
        confirm_states[i].pending_name.clear();
        confirm_states[i].confirm_count = 0;
    }
}

static void reset_face_recognize_cache(FaceRecognizeCache *cache, int count)
{
    if (!cache || count <= 0) return;
    for (int i = 0; i < count; ++i) {
        cache[i] = FaceRecognizeCache();
    }
}

static void set_face_detect_only_cache(FaceRecognizeCache &cache,
                                       const struct timeval &now)
{
    cache.valid = true;
    cache.last_run = now;
    cache.label = "FACE";
    cache.match = nullptr;
    cache.is_known = false;
    cache.has_perm = false;
    cache.compared = false;
    cache.dist = -1.f;
    cache.fusion_valid = false;
    cache.fusion_worker_id = 0;
    cache.fusion_label.clear();
    cache.fusion_count = 0;
    cache.fusion_next = 0;
}

static std::string build_face_match_label(const FaceEntry *match)
{
    if (!match) return "STRANGER";

    char face_buf[96];
    snprintf(face_buf, sizeof(face_buf), "%s#%d",
             match->name.c_str(), match->worker_id);
    return std::string(face_buf);
}

static const FaceEntry *find_face_entry_by_worker_id(int worker_id)
{
    std::vector<FaceEntry> &db = get_face_db();
    for (const auto &entry : db) {
        if (entry.worker_id == worker_id) return &entry;
    }
    return nullptr;
}

struct FingerprintPageBinding {
    int worker_id = 0;
    uint16_t page_id = 0;
};

struct FingerprintPageMap {
    std::string path;
    std::vector<FingerprintPageBinding> entries;
};

static std::string path_join_simple(const std::string &base,
                                    const std::string &name)
{
    if (base.empty() || base == ".") return "./" + name;
    if (base[base.size() - 1] == '/') return base + name;
    return base + "/" + name;
}

static bool filesystem_path_is_directory(const std::string &path)
{
    struct stat st;
    return !path.empty() &&
           stat(path.c_str(), &st) == 0 &&
           S_ISDIR(st.st_mode);
}

static std::string parent_dir_path(const std::string &path)
{
    if (path.empty()) return ".";
    size_t slash = path.find_last_of('/');
    if (slash == std::string::npos) return ".";
    if (slash == 0) return "/";
    return path.substr(0, slash);
}

static bool remove_path_recursive(const std::string &path)
{
    struct stat st;
    if (path.empty()) return false;
    if (lstat(path.c_str(), &st) != 0) {
        return errno == ENOENT;
    }

    if (S_ISDIR(st.st_mode)) {
        DIR *dir = opendir(path.c_str());
        if (!dir) {
            printf("[Reset] open dir failed: %s (%s)\n",
                   path.c_str(), strerror(errno));
            return false;
        }

        bool ok = true;
        struct dirent *entry = nullptr;
        while ((entry = readdir(dir)) != nullptr) {
            if (strcmp(entry->d_name, ".") == 0 ||
                strcmp(entry->d_name, "..") == 0) {
                continue;
            }
            if (!remove_path_recursive(path_join_simple(path, entry->d_name))) {
                ok = false;
            }
        }
        closedir(dir);

        if (rmdir(path.c_str()) != 0 && errno != ENOENT) {
            printf("[Reset] remove dir failed: %s (%s)\n",
                   path.c_str(), strerror(errno));
            ok = false;
        }
        return ok;
    }

    if (unlink(path.c_str()) != 0 && errno != ENOENT) {
        printf("[Reset] remove file failed: %s (%s)\n",
               path.c_str(), strerror(errno));
        return false;
    }
    return true;
}

static bool clear_face_database_storage(const char *db_path)
{
    if (!db_path || !db_path[0]) return true;

    std::string path = db_path;
    if (filesystem_path_is_directory(path)) {
        DIR *dir = opendir(path.c_str());
        if (!dir) {
            printf("[Reset] open face db dir failed: %s (%s)\n",
                   path.c_str(), strerror(errno));
            return false;
        }

        bool ok = true;
        struct dirent *entry = nullptr;
        while ((entry = readdir(dir)) != nullptr) {
            if (strcmp(entry->d_name, ".") == 0 ||
                strcmp(entry->d_name, "..") == 0) {
                continue;
            }
            if (!remove_path_recursive(path_join_simple(path, entry->d_name))) {
                ok = false;
            }
        }
        closedir(dir);
        return ok;
    }

    FILE *fp = fopen(path.c_str(), "w");
    if (!fp) {
        printf("[Reset] truncate face db failed: %s (%s)\n",
               path.c_str(), strerror(errno));
        return false;
    }
    if (fclose(fp) != 0) {
        printf("[Reset] close face db failed: %s (%s)\n",
               path.c_str(), strerror(errno));
        return false;
    }
    remove_path_recursive(path_join_simple(parent_dir_path(path), ".face_feature_cache"));
    return true;
}

static std::string fingerprint_map_storage_path(const char *db_path)
{
    std::string base = (db_path && db_path[0]) ? db_path : ".";
    if (!filesystem_path_is_directory(base)) {
        base = parent_dir_path(base);
    }
    return path_join_simple(base, "fingerprint_map.txt");
}

static bool valid_fingerprint_page(uint16_t page_id, uint16_t capacity)
{
    return capacity > 0 && page_id > 0 && page_id < capacity;
}

static bool fingerprint_page_map_get_page(const FingerprintPageMap &map,
                                          int worker_id,
                                          uint16_t *page_id)
{
    if (!page_id || worker_id <= 0) return false;
    for (const auto &entry : map.entries) {
        if (entry.worker_id == worker_id) {
            *page_id = entry.page_id;
            return true;
        }
    }
    return false;
}

static bool fingerprint_page_map_get_worker(const FingerprintPageMap &map,
                                            uint16_t page_id,
                                            int *worker_id)
{
    if (!worker_id || page_id == 0) return false;
    for (const auto &entry : map.entries) {
        if (entry.page_id == page_id) {
            *worker_id = entry.worker_id;
            return true;
        }
    }
    return false;
}

static bool fingerprint_page_map_page_used(const FingerprintPageMap &map,
                                           uint16_t page_id,
                                           int ignore_worker_id = 0)
{
    for (const auto &entry : map.entries) {
        if (entry.page_id == page_id &&
            entry.worker_id != ignore_worker_id) {
            return true;
        }
    }
    return false;
}

static bool fingerprint_page_map_set(FingerprintPageMap &map,
                                     int worker_id,
                                     uint16_t page_id,
                                     uint16_t capacity)
{
    if (worker_id <= 0 ||
        !valid_fingerprint_page(page_id, capacity) ||
        fingerprint_page_map_page_used(map, page_id, worker_id)) {
        return false;
    }

    for (auto &entry : map.entries) {
        if (entry.worker_id == worker_id) {
            entry.page_id = page_id;
            return true;
        }
    }

    FingerprintPageBinding binding;
    binding.worker_id = worker_id;
    binding.page_id = page_id;
    map.entries.push_back(binding);
    return true;
}

static bool fingerprint_page_map_remove_worker(FingerprintPageMap &map,
                                               int worker_id)
{
    for (auto it = map.entries.begin(); it != map.entries.end(); ++it) {
        if (it->worker_id == worker_id) {
            map.entries.erase(it);
            return true;
        }
    }
    return false;
}

static bool fingerprint_page_map_allocate(const FingerprintPageMap &map,
                                          uint16_t capacity,
                                          uint16_t *page_id)
{
    if (!page_id || capacity == 0) return false;
    for (uint32_t page = 1; page < (uint32_t)capacity; ++page) {
        if (!fingerprint_page_map_page_used(map, (uint16_t)page)) {
            *page_id = (uint16_t)page;
            return true;
        }
    }
    return false;
}

static bool save_fingerprint_page_map(const FingerprintPageMap &map)
{
    if (map.path.empty()) return false;

    std::string tmp_path = map.path + ".tmp";
    FILE *fp = fopen(tmp_path.c_str(), "w");
    if (!fp) {
        printf("[AS608] fingerprint map open failed: %s\n", tmp_path.c_str());
        return false;
    }

    fprintf(fp, "# employee_id fingerprint_page\n");
    std::vector<FingerprintPageBinding> sorted = map.entries;
    std::sort(sorted.begin(), sorted.end(),
              [](const FingerprintPageBinding &a,
                 const FingerprintPageBinding &b) {
                  if (a.page_id != b.page_id) return a.page_id < b.page_id;
                  return a.worker_id < b.worker_id;
              });
    for (const auto &entry : sorted) {
        fprintf(fp, "%d %u\n", entry.worker_id, entry.page_id);
    }

    if (fclose(fp) != 0) {
        unlink(tmp_path.c_str());
        printf("[AS608] fingerprint map close failed: %s\n", tmp_path.c_str());
        return false;
    }

    if (rename(tmp_path.c_str(), map.path.c_str()) != 0) {
        unlink(tmp_path.c_str());
        printf("[AS608] fingerprint map save failed: %s\n", map.path.c_str());
        return false;
    }
    return true;
}

static bool load_fingerprint_page_map(FingerprintPageMap &map,
                                      const char *db_path,
                                      uint16_t capacity)
{
    map.path = fingerprint_map_storage_path(db_path);
    map.entries.clear();

    bool map_exists = access(map.path.c_str(), R_OK) == 0;
    bool needs_save = false;
    if (map_exists) {
        FILE *fp = fopen(map.path.c_str(), "r");
        if (!fp) {
            printf("[AS608] fingerprint map read failed: %s\n", map.path.c_str());
            return false;
        }

        char line[128];
        while (fgets(line, sizeof(line), fp)) {
            if (line[0] == '#' || line[0] == '\n') continue;

            long worker = 0;
            unsigned long page = 0;
            uint16_t existing_page = 0;
            if (sscanf(line, "%ld %lu", &worker, &page) != 2 ||
                worker <= 0 ||
                worker > 2147483647L ||
                page > 65535ul ||
                !valid_fingerprint_page((uint16_t)page, capacity) ||
                !find_face_entry_by_worker_id((int)worker) ||
                fingerprint_page_map_page_used(map, (uint16_t)page) ||
                fingerprint_page_map_get_page(map, (int)worker, &existing_page)) {
                needs_save = true;
                continue;
            }

            FingerprintPageBinding binding;
            binding.worker_id = (int)worker;
            binding.page_id = (uint16_t)page;
            map.entries.push_back(binding);
        }
        fclose(fp);
    } else {
        std::vector<FaceEntry> &db = get_face_db();
        for (size_t i = 0; i < db.size(); ++i) {
            uint32_t page = (uint32_t)i + 1u;
            if (db[i].worker_id <= 0 ||
                page > 65535u ||
                !valid_fingerprint_page((uint16_t)page, capacity) ||
                fingerprint_page_map_page_used(map, (uint16_t)page)) {
                continue;
            }

            FingerprintPageBinding binding;
            binding.worker_id = db[i].worker_id;
            binding.page_id = (uint16_t)page;
            map.entries.push_back(binding);
        }
        needs_save = !map.entries.empty();
        printf("[AS608] fingerprint map legacy init entries=%zu path=%s\n",
               map.entries.size(), map.path.c_str());
    }

    if (needs_save && !save_fingerprint_page_map(map)) {
        return false;
    }

    printf("[AS608] fingerprint map ready entries=%zu path=%s\n",
           map.entries.size(), map.path.c_str());
    return true;
}

static bool resolve_fingerprint_worker_input(const FingerprintPageMap &map,
                                             int input_id,
                                             int *worker_id)
{
    if (!worker_id || input_id <= 0) return false;

    if (find_face_entry_by_worker_id(input_id)) {
        *worker_id = input_id;
        return true;
    }

    std::vector<FaceEntry> &db = get_face_db();
    if ((size_t)input_id <= db.size()) {
        *worker_id = db[(size_t)input_id - 1u].worker_id;
        return true;
    }

    uint16_t page_id = 0;
    if (fingerprint_page_map_get_page(map, input_id, &page_id)) {
        *worker_id = input_id;
        return true;
    }

    return false;
}

static const FaceEntry *find_face_entry_by_fingerprint_page(const FingerprintPageMap &map,
                                                            uint16_t page_id)
{
    int worker_id = 0;
    if (!fingerprint_page_map_get_worker(map, page_id, &worker_id)) {
        return nullptr;
    }
    return find_face_entry_by_worker_id(worker_id);
}

struct DeleteEmployeeResult {
    bool found = false;
    bool face_ok = false;
    bool fingerprint_ok = true;
    bool fingerprint_attempted = false;
    bool fingerprint_page_found = false;
    bool map_ok = true;
    uint16_t fingerprint_page = 0;
    uint8_t fingerprint_ensure = AS608::kTimeout;
    std::string name;
};

struct ResetDatabaseResult {
    bool face_ok = false;
    bool map_ok = false;
    bool fingerprint_ok = false;
    bool fingerprint_attempted = false;
    uint8_t fingerprint_ensure = AS608::kTimeout;
    size_t face_entries_before = 0;
};

static ResetDatabaseResult reset_runtime_database(FingerprintRuntime &fingerprint,
                                                  FingerprintPageMap &fingerprint_map,
                                                  const char *db_path,
                                                  bool require_fingerprint = true)
{
    ResetDatabaseResult result;
    std::vector<FaceEntry> &db = get_face_db();
    result.face_entries_before = db.size();

    result.face_ok = clear_face_database_storage(db_path);
    db.clear();

    fingerprint_map.entries.clear();
    result.map_ok = save_fingerprint_page_map(fingerprint_map);

    if (fingerprint.ready) {
        result.fingerprint_attempted = true;
        result.fingerprint_ensure = fingerprint.sensor.empty();
        result.fingerprint_ok = result.fingerprint_ensure == AS608::kOk;
    } else {
        result.fingerprint_ok = !require_fingerprint;
    }

    printf("[Reset] database reset face=%s map=%s fingerprint=%s entries_before=%zu\n",
           result.face_ok ? "ok" : "fail",
           result.map_ok ? "ok" : "fail",
           result.fingerprint_ok ? "ok" :
               (result.fingerprint_attempted ? "fail" : "skipped"),
           result.face_entries_before);
    return result;
}

static DeleteEmployeeResult delete_employee_records(uint32_t employee_id,
                                                    FingerprintRuntime &fingerprint,
                                                    FingerprintPageMap &fingerprint_map,
                                                    const char *db_path)
{
    DeleteEmployeeResult result;
    const FaceEntry *entry = find_face_entry_by_worker_id((int)employee_id);
    if (!entry) return result;

    result.found = true;
    result.name = entry->name;
    result.fingerprint_page_found = fingerprint_page_map_get_page(
        fingerprint_map, (int)employee_id, &result.fingerprint_page);

    if (fingerprint.ready && result.fingerprint_page_found) {
        result.fingerprint_attempted = true;
        result.fingerprint_ok = fingerprint.sensor.remove(
            result.fingerprint_page, 1, &result.fingerprint_ensure);
        printf("[EMP-PROTO] DELETE_EMPLOYEE fingerprint %s employee=%u page=%u ensure=0x%02X %s\n",
               result.fingerprint_ok ? "ok" : "fail", employee_id,
               result.fingerprint_page, result.fingerprint_ensure,
               fingerprint.sensor.ensureMessage(result.fingerprint_ensure));
    } else {
        result.fingerprint_ok = !result.fingerprint_page_found;
        printf("[EMP-PROTO] DELETE_EMPLOYEE fingerprint skipped employee=%u ready=%d page_found=%d\n",
               employee_id, fingerprint.ready ? 1 : 0,
               result.fingerprint_page_found ? 1 : 0);
    }

    if (!result.fingerprint_ok) {
        printf("[EMP-PROTO] DELETE_EMPLOYEE aborted before face delete employee=%u fingerprint=fail\n",
               employee_id);
        return result;
    }

    result.face_ok = remove_face_employee((int)employee_id, db_path);
    if (result.face_ok) {
        bool removed = fingerprint_page_map_remove_worker(fingerprint_map, (int)employee_id);
        if (removed && !save_fingerprint_page_map(fingerprint_map)) {
            result.map_ok = false;
        }
    }

    printf("[EMP-PROTO] DELETE_EMPLOYEE records employee=%u face=%s fingerprint=%s map=%s\n",
           employee_id, result.face_ok ? "ok" : "fail",
           result.fingerprint_ok ? "ok" : "fail",
           result.map_ok ? "ok" : "fail");
    return result;
}

static bool fingerprint_target_for_input_id(FingerprintPageMap &fingerprint_map,
                                            int input_id,
                                            uint16_t capacity,
                                            bool allocate_if_missing,
                                            int *worker_id,
                                            uint16_t *page_id)
{
    if (!worker_id || !page_id || input_id <= 0 || capacity == 0) return false;

    int resolved_worker_id = 0;
    if (!resolve_fingerprint_worker_input(fingerprint_map, input_id, &resolved_worker_id)) {
        return false;
    }

    if (fingerprint_page_map_get_page(fingerprint_map, resolved_worker_id, page_id)) {
        *worker_id = resolved_worker_id;
        return true;
    }

    if (!allocate_if_missing ||
        !fingerprint_page_map_allocate(fingerprint_map, capacity, page_id)) {
        return false;
    }
    *worker_id = resolved_worker_id;
    return true;
}

static bool fingerprint_poll_due(FingerprintRuntime &fingerprint)
{
    struct timeval now;
    gettimeofday(&now, nullptr);
    if (timeval_is_zero(fingerprint.last_poll)) {
        fingerprint.last_poll = now;
        return true;
    }
    if (elapsed_ms(fingerprint.last_poll, now) < FINGERPRINT_POLL_INTERVAL_MS) {
        return false;
    }
    fingerprint.last_poll = now;
    return true;
}

static bool init_fingerprint_runtime(FingerprintRuntime &fingerprint,
                                     const char *dev_path,
                                     int baudrate,
                                     uint16_t page_num)
{
    fingerprint.ready = false;
    fingerprint.page_num = page_num > 0 ? page_num : FINGERPRINT_DEFAULT_PAGE_NUM;
    memset(&fingerprint.last_poll, 0, sizeof(fingerprint.last_poll));

    if (!dev_path || !dev_path[0]) return false;
    if (!fingerprint.sensor.openDevice(dev_path, baudrate)) return false;

    AS608::SysParam param;
    uint8_t ensure = fingerprint.sensor.readSysParam(&param);
    if (ensure != AS608::kOk) {
        printf("[AS608] read sys param failed: %s (0x%02X)\n",
               fingerprint.sensor.ensureMessage(ensure), ensure);
        fingerprint.sensor.closeDevice();
        return false;
    }

    if (param.max_templates > 0) fingerprint.page_num = param.max_templates;
    fingerprint.ready = true;
    printf("[AS608] ready: %s baud=%d capacity=%u\n",
           dev_path, baudrate, fingerprint.page_num);
    return true;
}

static bool consume_fingerprint_manage_request(FingerprintManageRequest *request)
{
    return uart_control_consume_fingerprint_request(request);
}

static void process_fingerprint_manage_request(FingerprintRuntime &fingerprint,
                                               FingerprintPageMap &fingerprint_map,
                                               UiBanner &banner)
{
    FingerprintManageRequest request;
    if (!consume_fingerprint_manage_request(&request)) return;

    if (!fingerprint.ready) {
        set_ui_banner(banner, "指纹模块不可用", "UART1/AS608 未连接", "无法执行指纹管理");
        printf("[AS608] command ignored, sensor not ready.\n");
        return;
    }

    uint8_t ensure = AS608::kTimeout;
    if (request.type == FingerprintManageType::ENROLL) {
        int input_id = request.worker_id;
        uint16_t page_id = 0;
        int resolved_worker_id = 0;
        if (!fingerprint_target_for_input_id(fingerprint_map,
                                             input_id,
                                             fingerprint.page_num,
                                             true,
                                             &resolved_worker_id,
                                             &page_id)) {
            char line[96];
            snprintf(line, sizeof(line), "编号/工号:%d 无可用页号", input_id);
            set_ui_banner(banner, "指纹录入失败", line, "检查人脸库或容量");
            printf("[AS608] enroll rejected input=%d capacity=%u\n",
                   input_id, fingerprint.page_num);
            return;
        }
        request.worker_id = resolved_worker_id;
        request.page_id = page_id;

        char line[96];
        snprintf(line, sizeof(line), "编号:%d 工号:%d 页:%u",
                 input_id, request.worker_id, request.page_id);
        set_ui_banner(banner, "指纹录入中", line, "等待 AS608 采集");
        printf("[AS608] enroll start input=%d worker=%d page=%u\n",
               input_id, request.worker_id, request.page_id);

        bool map_save_failed = false;
        bool ok = fingerprint.sensor.enroll(request.page_id, &ensure);
        if (ok) {
            ok = fingerprint_page_map_set(fingerprint_map,
                                          request.worker_id,
                                          request.page_id,
                                          fingerprint.page_num) &&
                 save_fingerprint_page_map(fingerprint_map);
            if (!ok) {
                map_save_failed = true;
                fingerprint.sensor.remove(request.page_id, 1, nullptr);
            }
        }
        const char *result_text = ok ? "录入成功" :
            (map_save_failed ? "映射保存失败" : fingerprint.sensor.ensureMessage(ensure));
        snprintf(line, sizeof(line), "工号:%d 页:%u %s",
                 request.worker_id, request.page_id, result_text);
        set_ui_banner(banner, ok ? "指纹录入成功" : "指纹录入失败", line, "继续等待打卡");
        printf("[AS608] enroll %s worker=%d page=%u ensure=0x%02X %s\n",
               ok ? "ok" : "fail", request.worker_id, request.page_id, ensure,
               result_text);
        return;
    }

    if (request.type == FingerprintManageType::DELETE_ONE) {
        int input_id = request.worker_id;
        uint16_t page_id = 0;
        int resolved_worker_id = 0;
        if (!fingerprint_target_for_input_id(fingerprint_map,
                                             input_id,
                                             fingerprint.page_num,
                                             false,
                                             &resolved_worker_id,
                                             &page_id)) {
            char line[96];
            snprintf(line, sizeof(line), "编号/工号:%d 无对应页号", input_id);
            set_ui_banner(banner, "指纹删除失败", line, "检查人脸库或容量");
            printf("[AS608] delete rejected input=%d capacity=%u\n",
                   input_id, fingerprint.page_num);
            return;
        }
        request.worker_id = resolved_worker_id;
        request.page_id = page_id;

        bool map_save_failed = false;
        bool ok = fingerprint.sensor.remove(request.page_id, request.count, &ensure);
        if (ok) {
            fingerprint_page_map_remove_worker(fingerprint_map, request.worker_id);
            ok = save_fingerprint_page_map(fingerprint_map);
            map_save_failed = !ok;
        }
        const char *result_text = ok ? "删除成功" :
            (map_save_failed ? "映射保存失败" : fingerprint.sensor.ensureMessage(ensure));
        char line[96];
        snprintf(line, sizeof(line), "工号:%d 页:%u %s",
                 request.worker_id, request.page_id, result_text);
        set_ui_banner(banner, ok ? "指纹删除成功" : "指纹删除失败", line, "继续等待打卡");
        printf("[AS608] delete %s worker=%d page=%u ensure=0x%02X %s\n",
               ok ? "ok" : "fail", request.worker_id, request.page_id, ensure,
               result_text);
        return;
    }

}

static void reset_face_feature_fusion(FaceRecognizeCache &cache)
{
    cache.fusion_valid = false;
    cache.fusion_worker_id = 0;
    cache.fusion_label.clear();
    cache.fusion_count = 0;
    cache.fusion_next = 0;
}

static void add_face_feature_sample(FaceRecognizeCache &cache,
                                    const FaceEntry *match,
                                    const float *feat)
{
    if (!match || !feat) {
        reset_face_feature_fusion(cache);
        return;
    }

    std::string label = build_face_match_label(match);
    if (!cache.fusion_valid ||
        cache.fusion_worker_id != match->worker_id ||
        cache.fusion_label != label) {
        reset_face_feature_fusion(cache);
        cache.fusion_valid = true;
        cache.fusion_worker_id = match->worker_id;
        cache.fusion_label = label;
    }

    for (int i = 0; i < FACENET_FEAT_DIM; ++i) {
        cache.fusion_feats[cache.fusion_next][i] = feat[i];
    }
    cache.fusion_next = (cache.fusion_next + 1) % FACE_FEATURE_FUSION_MAX_FRAMES;
    if (cache.fusion_count < FACE_FEATURE_FUSION_MAX_FRAMES) {
        cache.fusion_count++;
    }
}

static bool get_fused_face_feature(const FaceRecognizeCache &cache, float *out_feat)
{
    if (!out_feat ||
        !cache.fusion_valid ||
        cache.fusion_count < FACE_FEATURE_FUSION_MIN_FRAMES) {
        return false;
    }

    for (int i = 0; i < FACENET_FEAT_DIM; ++i) out_feat[i] = 0.f;
    for (int sample = 0; sample < cache.fusion_count; ++sample) {
        for (int i = 0; i < FACENET_FEAT_DIM; ++i) {
            out_feat[i] += cache.fusion_feats[sample][i];
        }
    }
    const float inv_count = 1.f / (float)cache.fusion_count;
    for (int i = 0; i < FACENET_FEAT_DIM; ++i) out_feat[i] *= inv_count;
    l2_normalize(out_feat);
    return true;
}

static bool face_landmarks_plausible(const object_detect_result *det,
                                     const image_transform_t &transform,
                                     int face_w,
                                     int face_h)
{
    if (!det || face_w <= 0 || face_h <= 0) return false;

    cv::Point2f pts[5];
    for (int i = 0; i < 5; ++i) {
        int x = det->point[i].x;
        int y = det->point[i].y;
        if (!map_coordinates(transform, &x, &y)) return false;
        pts[i] = cv::Point2f((float)x, (float)y);
    }

    const float eye_dx = pts[1].x - pts[0].x;
    const float eye_dy = pts[1].y - pts[0].y;
    const float eye_dist = std::sqrt(eye_dx * eye_dx + eye_dy * eye_dy);
    const float mouth_dx = pts[4].x - pts[3].x;
    const float mouth_dy = pts[4].y - pts[3].y;
    const float mouth_dist = std::sqrt(mouth_dx * mouth_dx + mouth_dy * mouth_dy);
    if (eye_dist < face_w * 0.18f || eye_dist > face_w * 0.85f) return false;
    if (std::fabs(eye_dy) > face_h * 0.25f) return false;
    if (mouth_dist < eye_dist * 0.30f || mouth_dist > eye_dist * 1.50f) return false;

    const float eye_mid_y = (pts[0].y + pts[1].y) * 0.5f;
    const float mouth_mid_y = (pts[3].y + pts[4].y) * 0.5f;
    if (pts[2].y <= eye_mid_y || pts[2].y >= mouth_mid_y) return false;
    return true;
}

static double estimate_face_sharpness(const cv::Mat &face_roi)
{
    if (face_roi.empty()) return 0.0;

    cv::Mat gray;
    if (face_roi.channels() == 1) {
        gray = face_roi;
    } else {
        cv::cvtColor(face_roi, gray, cv::COLOR_BGR2GRAY);
    }

    cv::Mat lap;
    cv::Laplacian(gray, lap, CV_64F);
    cv::Scalar mean;
    cv::Scalar stddev;
    cv::meanStdDev(lap, mean, stddev);
    return stddev[0] * stddev[0];
}

static bool face_quality_allows_recognition(const object_detect_result *det,
                                            const image_transform_t &transform,
                                            int face_w,
                                            int face_h,
                                            const cv::Mat &face_roi)
{
    if (!face_landmarks_plausible(det, transform, face_w, face_h)) return false;
    return estimate_face_sharpness(face_roi) >= FACE_QUALITY_MIN_SHARPNESS;
}

static bool init_face_runtime(const char *retina_path,
                              const char *facenet_path,
                              rknn_app_context_t &retina_ctx,
                              rknn_app_context_t &facenet_ctx,
                              cv::Mat &retina_input)
{
    memset(&retina_ctx, 0, sizeof(retina_ctx));
    memset(&facenet_ctx, 0, sizeof(facenet_ctx));
    retina_input.release();

    if (init_face_models(retina_path, facenet_path,
                                       &retina_ctx, &facenet_ctx) != 0) {
        printf("[Main] Face model init failed.\n");
        memset(&retina_ctx, 0, sizeof(retina_ctx));
        memset(&facenet_ctx, 0, sizeof(facenet_ctx));
        return false;
    }

    retina_input = cv::Mat(RETINA_H, RETINA_W, CV_8UC3,
                           retina_ctx.input_mems[0]->virt_addr);
    return true;
}

static void release_face_runtime(rknn_app_context_t &retina_ctx,
                                 rknn_app_context_t &facenet_ctx,
                                 cv::Mat &retina_input)
{
    retina_input.release();
    release_face_recognizer(&facenet_ctx);
    release_face_detector(&retina_ctx);
    memset(&retina_ctx, 0, sizeof(retina_ctx));
    memset(&facenet_ctx, 0, sizeof(facenet_ctx));
}

static bool init_yolo_runtime(const char *yolo_path,
                              yolov5_app_context_t &yolo_ctx)
{
    memset(&yolo_ctx, 0, sizeof(yolo_ctx));
    if (init_yolov5_model(yolo_path, &yolo_ctx) != 0) {
        printf("[Main] YOLOv5 model init failed: %s\n", yolo_path);
        release_yolov5_model(&yolo_ctx);
        memset(&yolo_ctx, 0, sizeof(yolo_ctx));
        return false;
    }
    return true;
}

static void release_yolo_runtime(yolov5_app_context_t &yolo_ctx)
{
    release_yolov5_model(&yolo_ctx);
    memset(&yolo_ctx, 0, sizeof(yolo_ctx));
}

static bool letterbox_bgr_to_yolo_input(const cv::Mat &src_bgr,
                                        yolov5_app_context_t &yolo_ctx,
                                        image_transform_t &transform,
                                        cv::Mat &resize_buf)
{
    if (src_bgr.empty() || !yolo_ctx.input_mems[0] ||
        yolo_ctx.model_width <= 0 || yolo_ctx.model_height <= 0 ||
        yolo_ctx.model_channel != 3) {
        return false;
    }

    int dst_w = yolo_ctx.model_width;
    int dst_h = yolo_ctx.model_height;
    float scale = std::min((float)dst_w / (float)src_bgr.cols,
                           (float)dst_h / (float)src_bgr.rows);
    int resized_w = std::max(1, (int)(src_bgr.cols * scale));
    int resized_h = std::max(1, (int)(src_bgr.rows * scale));
    int pad_left = (dst_w - resized_w) / 2;
    int pad_top = (dst_h - resized_h) / 2;

    cv::Mat yolo_input(dst_h, dst_w, CV_8UC3, yolo_ctx.input_mems[0]->virt_addr);
    yolo_input.setTo(cv::Scalar(114, 114, 114));

    resize_buf.create(resized_h, resized_w, CV_8UC3);
    cv::resize(src_bgr, resize_buf, cv::Size(resized_w, resized_h), 0, 0, cv::INTER_LINEAR);
    cv::Mat dst_roi = yolo_input(cv::Rect(pad_left, pad_top, resized_w, resized_h));
    cv::cvtColor(resize_buf, dst_roi, cv::COLOR_BGR2RGB);

    memset(&transform, 0, sizeof(transform));
    transform.src_w = src_bgr.cols;
    transform.src_h = src_bgr.rows;
    transform.dst_w = dst_w;
    transform.dst_h = dst_h;
    transform.resized_w = resized_w;
    transform.resized_h = resized_h;
    transform.pad_left = pad_left;
    transform.pad_top = pad_top;
    transform.scale = scale;
    return true;
}

static void draw_duty_person_result(cv::Mat &bgr,
                                    const yolov5_person_result_t &result,
                                    const image_transform_t &transform,
                                    const image_transform_t &display_transform)
{
    int left = result.box.left;
    int top = result.box.top;
    int right = result.box.right;
    int bottom = result.box.bottom;
    if (!map_coordinates(transform, &left, &top) ||
        !map_coordinates(transform, &right, &bottom)) {
        return;
    }

    left = std::max(0, std::min(left, display_transform.src_w - 1));
    top = std::max(0, std::min(top, display_transform.src_h - 1));
    right = std::max(0, std::min(right, display_transform.src_w - 1));
    bottom = std::max(0, std::min(bottom, display_transform.src_h - 1));
    if (right <= left || bottom <= top) return;

    if (!source_to_display_coordinates(display_transform, &left, &top) ||
        !source_to_display_coordinates(display_transform, &right, &bottom)) {
        return;
    }

    const int status_h = LCD_STATUS_BAR_H;
    const int panel_top = DISP_H - LCD_EVENT_PANEL_H;
    top = std::max(status_h, top);
    bottom = std::min(panel_top - 1, bottom);
    if (right <= left || bottom <= top) return;

    cv::Scalar box_color(0, 230, 255);
    int corner = std::max(6, std::min(16, std::min(right - left, bottom - top) / 4));

    cv::line(bgr, cv::Point(left, top), cv::Point(left + corner, top), box_color, 2);
    cv::line(bgr, cv::Point(left, top), cv::Point(left, top + corner), box_color, 2);
    cv::line(bgr, cv::Point(right, top), cv::Point(right - corner, top), box_color, 2);
    cv::line(bgr, cv::Point(right, top), cv::Point(right, top + corner), box_color, 2);
    cv::line(bgr, cv::Point(left, bottom), cv::Point(left + corner, bottom), box_color, 2);
    cv::line(bgr, cv::Point(left, bottom), cv::Point(left, bottom - corner), box_color, 2);
    cv::line(bgr, cv::Point(right, bottom), cv::Point(right - corner, bottom), box_color, 2);
    cv::line(bgr, cv::Point(right, bottom), cv::Point(right, bottom - corner), box_color, 2);

    char text[24];
    int score_pct = std::max(0, std::min(99, (int)(result.prop * 100.f + 0.5f)));
    snprintf(text, sizeof(text), "DUTY %02d%%", score_pct);

    const double scale = 0.32;
    int baseline = 0;
    cv::Size text_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, scale, 1, &baseline);
    int text_x = std::max(2, std::min(left, DISP_W - text_size.width - 4));
    int box_y = (top - 16 >= status_h) ? top - 16 : std::min(bottom + 3, panel_top - 14);
    cv::rectangle(bgr,
                  cv::Point(text_x - 2, box_y),
                  cv::Point(text_x + text_size.width + 3, box_y + 13),
                  cv::Scalar(0, 0, 0), cv::FILLED);
    cv::putText(bgr, text, cv::Point(text_x, box_y + 10),
                cv::FONT_HERSHEY_SIMPLEX, scale, box_color, 1, cv::LINE_AA);
}

static void update_duty_runtime(DutyRuntime &duty,
                                const yolov5_on_duty_status_t &status,
                                time_t now)
{
    if (duty.started_at == 0) duty.started_at = now;
    if (duty.last_eval_at == 0) duty.last_eval_at = now;

    int delta = (int)difftime(now, duty.last_eval_at);
    if (delta < 0) delta = 0;
    if (delta > 5) delta = 5;
    if (duty.last_on_duty) duty.on_duty_seconds += delta;
    duty.total_seconds = (int)difftime(now, duty.started_at);
    duty.last_eval_at = now;
    duty.last_on_duty = status.on_duty;
    duty.best_score = status.best_score;
}

static void refresh_duty_runtime_clock(DutyRuntime &duty, time_t now)
{
    if (duty.started_at <= 0) return;

    int total = (int)difftime(now, duty.started_at);
    if (total < 0) total = 0;
    duty.total_seconds = total;
}

static void finalize_duty_runtime(DutyRuntime &duty, time_t now)
{
    if (duty.started_at <= 0) return;

    int delta = 0;
    if (duty.last_eval_at > 0) {
        delta = (int)difftime(now, duty.last_eval_at);
        if (delta < 0) delta = 0;
        if (delta > 60) delta = 60;
    }
    if (duty.last_on_duty) duty.on_duty_seconds += delta;
    duty.total_seconds = (int)difftime(now, duty.started_at);
    if (duty.total_seconds < 0) duty.total_seconds = 0;
    duty.last_eval_at = now;
}

static std::string build_duty_status_line(const DutyRuntime &duty)
{
    char buf[96];
    snprintf(buf, sizeof(buf), "DUTY %s",
             format_duration_text(duty.on_duty_seconds).c_str());
    return std::string(buf);
}

static std::string build_duty_time_line(const DutyRuntime &duty)
{
    char buf[128];
    snprintf(buf, sizeof(buf), "TOTAL %s",
             format_duration_text(duty.total_seconds).c_str());
    return std::string(buf);
}

// =============================================================================
// 安全退出信号处理
// =============================================================================
static volatile bool g_running = true;
static void sig_handler(int) { g_running = false; }

static void print_usage(const char *program)
{
    printf("Usage: %s <detector.rknn> <recognition.rknn> <faces_dir|face_db.txt> [yolov5n.rknn]\n", program);
    printf("  faces_dir format: <name>_<worker_id>_<perm(0/1)>/*.png|*.jpg\n");
    printf("  face_db.txt format: <name> <worker_id> <perm(0/1)> <img_or_dir1> [img_or_dir2 ...]\n");
    printf("  reset db: %s --reset-db <faces_dir|face_db.txt> [fingerprint_dev|-] [baud]\n", program);
    printf("  legacy argv[4] local log path is ignored when argv[5] is present.\n");
}

static int run_reset_database_cli(int argc, char **argv)
{
    if (argc < 3 || argc > 5) {
        print_usage(argv[0]);
        return -1;
    }

    const char *db_path = argv[2];
    const char *fingerprint_dev = (argc >= 4) ? argv[3] : getenv("ATTENDANCE_FINGERPRINT_DEV");
    if (!fingerprint_dev || !fingerprint_dev[0]) fingerprint_dev = FINGERPRINT_DEV;

    bool skip_fingerprint =
        strcmp(fingerprint_dev, "-") == 0 ||
        strcasecmp(fingerprint_dev, "none") == 0 ||
        strcasecmp(fingerprint_dev, "no") == 0;

    int fingerprint_baud = FINGERPRINT_BAUDRATE;
    const char *fingerprint_baud_env = getenv("ATTENDANCE_FINGERPRINT_BAUD");
    if (fingerprint_baud_env && fingerprint_baud_env[0]) {
        int parsed_baud = atoi(fingerprint_baud_env);
        if (parsed_baud > 0) fingerprint_baud = parsed_baud;
    }
    if (argc >= 5) {
        int parsed_baud = atoi(argv[4]);
        if (parsed_baud > 0) fingerprint_baud = parsed_baud;
    }

    FingerprintRuntime fingerprint;
    if (!skip_fingerprint) {
        init_fingerprint_runtime(fingerprint,
                                 fingerprint_dev,
                                 fingerprint_baud,
                                 FINGERPRINT_DEFAULT_PAGE_NUM);
    }

    FingerprintPageMap fingerprint_map;
    fingerprint_map.path = fingerprint_map_storage_path(db_path);
    ResetDatabaseResult result =
        reset_runtime_database(fingerprint, fingerprint_map, db_path, !skip_fingerprint);

    if (fingerprint.ready) {
        fingerprint.sensor.closeDevice();
    }

    bool ok = result.face_ok && result.map_ok && result.fingerprint_ok;
    printf("[Reset] command line reset %s face=%s map=%s fingerprint=%s path=%s\n",
           ok ? "ok" : "partial",
           result.face_ok ? "ok" : "fail",
           result.map_ok ? "ok" : "fail",
           result.fingerprint_ok ? "ok" :
               (result.fingerprint_attempted ? "fail" : "skipped"),
           db_path ? db_path : "");
    return ok ? 0 : 2;
}

// =============================================================================
// main
// =============================================================================
int main(int argc, char **argv)
{
    init_app_timezone();

    if (argc >= 2 &&
        (strcmp(argv[1], "--reset-db") == 0 ||
         strcmp(argv[1], "reset-db") == 0 ||
         strcmp(argv[1], "--reset-all") == 0)) {
        return run_reset_database_cli(argc, argv);
    }

    if (argc < 4) {
        print_usage(argv[0]);
        return -1;
    }

    signal(SIGINT,  sig_handler);
    signal(SIGTERM, sig_handler);

    const char *retina_path  = argv[1];
    const char *facenet_path = argv[2];
    const char *db_path      = argv[3];
    const char *yolo_path    = getenv("ATTENDANCE_YOLO_MODEL");
    if (argc >= 6) {
        yolo_path = argv[5];
    } else if (argc >= 5 && strstr(argv[4], ".rknn") != nullptr) {
        yolo_path = argv[4];
    }
    if (!yolo_path || !yolo_path[0]) yolo_path = YOLO_MODEL_PATH;
    const char *uart_control_dev = getenv("ATTENDANCE_CONTROL_DEV");
    const char *uart_upload_dev = getenv("ATTENDANCE_UART_UPLOAD_DEV");
    const char *fingerprint_dev = getenv("ATTENDANCE_FINGERPRINT_DEV");
    if (!uart_control_dev || !uart_control_dev[0]) uart_control_dev = UART_CONTROL_DEV;
    if (!uart_upload_dev || !uart_upload_dev[0]) uart_upload_dev = UART_UPLOAD_DEV;
    if (!fingerprint_dev || !fingerprint_dev[0]) fingerprint_dev = FINGERPRINT_DEV;

    int fingerprint_baud = FINGERPRINT_BAUDRATE;
    const char *fingerprint_baud_env = getenv("ATTENDANCE_FINGERPRINT_BAUD");
    if (fingerprint_baud_env && fingerprint_baud_env[0]) {
        int parsed_baud = atoi(fingerprint_baud_env);
        if (parsed_baud > 0) fingerprint_baud = parsed_baud;
    }

    uint16_t fingerprint_page_num = FINGERPRINT_DEFAULT_PAGE_NUM;
    const char *fingerprint_page_env = getenv("ATTENDANCE_FINGERPRINT_PAGE_NUM");
    if (fingerprint_page_env && fingerprint_page_env[0]) {
        int parsed_page_num = atoi(fingerprint_page_env);
        if (parsed_page_num > 0 && parsed_page_num <= 65535) {
            fingerprint_page_num = (uint16_t)parsed_page_num;
        }
    }

    cv::setNumThreads(1);

    system("RkLunch-stop.sh");

    // -------------------------------------------------------------------------
    // 设备型号检测（仅 Pro Max 有 SPI LCD）
    // -------------------------------------------------------------------------
    bool disp_available = false;
    {
        FILE *fp = fopen("/proc/device-tree/model", "r");
        if (fp) {
            char model[64] = {};
            fgets(model, sizeof(model), fp);
            fclose(fp);
            // 去掉末尾换行
            size_t len = strlen(model);
            if (len && model[len-1] == '\n') model[len-1] = '\0';
#if APP_VERBOSE_LOG
            printf("[SYS] Device: %s\n", model);
#endif
            // Pro Max 系列均开启显示
            if (strstr(model, "Pico") != nullptr) disp_available = true;
        }
        // 额外检查 SPI 设备节点
        if (access("/dev/spidev0.0", F_OK) != 0) disp_available = false;
    }

    // -------------------------------------------------------------------------
    // LCD 初始化（2.8"  240×320）
    // -------------------------------------------------------------------------
    bool disp_on = false;
    if (disp_available) {
        disp_on = lcd_start_display();
    } else {
#if APP_VERBOSE_LOG
        printf("[LCD] Screen OFF (headless mode)\n");
#endif
    }

    // -------------------------------------------------------------------------
    // 模型初始化
    // -------------------------------------------------------------------------
    rknn_app_context_t retina_ctx, facenet_ctx;
    yolov5_app_context_t yolo_ctx;
    memset(&retina_ctx, 0, sizeof(retina_ctx));
    memset(&facenet_ctx, 0, sizeof(facenet_ctx));
    memset(&yolo_ctx, 0, sizeof(yolo_ctx));
    cv::Mat retina_input;

    bool face_model_ready = init_face_runtime(retina_path, facenet_path,
                                              retina_ctx, facenet_ctx,
                                              retina_input);
    if (!face_model_ready) return -1;
    bool yolo_model_ready = false;

    // -------------------------------------------------------------------------
    // 人脸数据库加载
    // 先加载图库，再打开摄像头，降低启动峰值内存
    // -------------------------------------------------------------------------
    int db_count = load_face_db(db_path, &retina_ctx, &facenet_ctx);
    if (db_count == 0) {
        printf("[Main] WARNING: face_db empty, all faces will be STRANGER.\n");
    }

    // -------------------------------------------------------------------------
    // 摄像头初始化
    // -------------------------------------------------------------------------
    CameraFrameWorker camera;
    cv::Mat probe;
    if (!camera.start(CAM_W, CAM_H, &probe)) {
        return -1;
    }

    bool uart_control_stdio = false;
    int uart_control_fd = uart_control_open_control_input(uart_control_dev, &uart_control_stdio);
    int uart_tx_fd      = uart_control_open_uart_port(uart_upload_dev);
    bool uart_control_ready = (uart_control_fd >= 0);
    bool uart_tx_ready      = (uart_tx_fd >= 0);

    FingerprintRuntime fingerprint;
    init_fingerprint_runtime(fingerprint,
                             fingerprint_dev,
                             fingerprint_baud,
                             fingerprint_page_num);
    FingerprintPageMap fingerprint_map;
    if (!load_fingerprint_page_map(fingerprint_map, db_path, fingerprint.page_num)) {
        printf("[AS608] WARNING: fingerprint map unavailable, fingerprint matching disabled until map saves successfully.\n");
    }

    EmployeeManagementChannel employee_channel(uart_tx_fd);

    // -------------------------------------------------------------------------
    // 工作缓冲区（全部预分配，主循环零动态分配）
    // -------------------------------------------------------------------------
    cv::Mat bgr_cam = probe;
    probe.release();
    uint64_t last_camera_seq = 0;
    cv::Mat bgr(DISP_H, DISP_W, CV_8UC3);
    LcdRefreshWorker lcd_async;
    image_transform_t retina_transform;
    memset(&retina_transform, 0, sizeof(retina_transform));
    image_transform_t yolo_transform;
    memset(&yolo_transform, 0, sizeof(yolo_transform));
    cv::Mat yolo_resize_buf;
    image_transform_t display_transform;
    memset(&display_transform, 0, sizeof(display_transform));
    cv::Mat display_resize_buf;
    ServoGimbalTracker gimbal_tracker;
    gimbal_tracker.init(CAM_W, CAM_H);

    float feat[FACENET_FEAT_DIM];
    float fused_feat[FACENET_FEAT_DIM];
    object_detect_result_list od_results;
    memset(&od_results, 0, sizeof(od_results));
    yolov5_person_result_list_t duty_results;
    memset(&duty_results, 0, sizeof(duty_results));

    // FPS 计算：按主循环间隔平滑，避免 YOLO 降频推理时显示跳变。
    FpsCounter fps_counter;
    float fps = 0.f;

    // 多帧确认状态（每个检测槽位一个）
    ConfirmState confirm[MAX_DETECT_FACES];
    FaceRecognizeCache face_cache[MAX_DETECT_FACES];
    UiBanner ui_banner;
    bool protocol_attendance_enabled = false;
    const bool idle_face_recognition_enabled =
        env_flag_enabled("ATTENDANCE_IDLE_FACE_RECOG");
    std::string startup_status = uart_control_build_uart_status_text(uart_control_ready, uart_tx_ready);
    startup_status += fingerprint.ready ? "  FP OK" : "  FP OFF";
    AttendanceService attendance_service;
    set_ui_banner(ui_banner,
                  "考勤待机",
                  "等待人脸或指纹...",
                  startup_status);
    DetectState detect_state = DETECT_STATE_FACE_ACTIVE;
    RecognitionPurpose recognition_purpose = RECOGNITION_PURPOSE_CHECKIN;
    int active_worker_id = 0;
    std::string uart_rx_buffer;
    DutyRuntime duty_runtime;
    struct timeval duty_last_infer;
    memset(&duty_last_infer, 0, sizeof(duty_last_infer));
    struct timeval lcd_last_flush;
    memset(&lcd_last_flush, 0, sizeof(lcd_last_flush));
    struct timeval face_last_infer;
    memset(&face_last_infer, 0, sizeof(face_last_infer));
    EmployeeSession employee_session;
    std::vector<cv::Mat> enroll_face_samples;
    enroll_face_samples.reserve(ENROLL_FACE_SAMPLE_TARGET);

    auto start_lcd_refresh = [&]() -> bool {
        if (!disp_available) return false;
        if (!disp_on) {
            disp_on = lcd_start_display();
            if (!disp_on) return false;
        }
        if (!lcd_async.start()) {
            printf("[LCD] async refresh start failed, display disabled.\n");
            lcd_stop_display();
            disp_on = false;
            return false;
        }
        memset(&lcd_last_flush, 0, sizeof(lcd_last_flush));
        return true;
    };

    if (disp_on && !lcd_async.start()) {
        printf("[LCD] async refresh start failed, display disabled.\n");
        lcd_stop_display();
        disp_on = false;
    }

    auto ensure_face_detection_active = [&]() -> bool {
        if (disp_available && !disp_on) {
            start_lcd_refresh();
        }

        if (!face_model_ready) {
            face_model_ready = init_face_runtime(retina_path, facenet_path,
                                                 retina_ctx, facenet_ctx,
                                                 retina_input);
            if (!face_model_ready) return false;
        }

        bool camera_ok = camera.is_running() && camera.latest(probe);
        if (!camera_ok) camera_ok = camera.start(CAM_W, CAM_H, &probe);
        if (!camera_ok) return false;

        bgr_cam = probe;
        probe.release();
        last_camera_seq = 0;
        reset_confirm_states(confirm, MAX_DETECT_FACES);
        reset_face_recognize_cache(face_cache, MAX_DETECT_FACES);
        memset(&retina_transform, 0, sizeof(retina_transform));
        memset(&face_last_infer, 0, sizeof(face_last_infer));
        detect_state = DETECT_STATE_FACE_ACTIVE;
        recognition_purpose = RECOGNITION_PURPOSE_CHECKIN;
        return true;
    };

    auto stop_protocol_enroll_capture = [&]() {
        employee_session.reset();
        enroll_face_samples.clear();
        reset_confirm_states(confirm, MAX_DETECT_FACES);
        reset_face_recognize_cache(face_cache, MAX_DETECT_FACES);
        recognition_purpose = RECOGNITION_PURPOSE_CHECKIN;
    };

    auto finish_protocol_checkout = [&](uint8_t seq, uint32_t employee_id) {
        const FaceEntry *entry = find_face_entry_by_worker_id((int)employee_id);
        if (!entry) {
            employee_channel.sendError(seq, EMP_PROTO_CMD_CHECKOUT_REQ,
                                       EMP_PROTO_ERROR_EMPLOYEE_NOT_FOUND);
            return;
        }

        if (active_worker_id <= 0 || active_worker_id != (int)employee_id ||
            duty_runtime.started_at <= 0) {
            employee_channel.sendError(seq, EMP_PROTO_CMD_CHECKOUT_REQ,
                                       EMP_PROTO_ERROR_NOT_CHECKED_IN);
            return;
        }

        time_t now = time(nullptr);
        time_t checkin_time = duty_runtime.started_at;
        uint32_t duration = (uint32_t)std::max(0, (int)difftime(now, checkin_time));

        AttendanceRecord attendance;
        if (!attendance_service.record_explicit(entry, false, &attendance)) {
            employee_channel.sendError(seq, EMP_PROTO_CMD_CHECKOUT_REQ,
                                       EMP_PROTO_ERROR_NOT_CHECKED_IN);
            return;
        }
        finalize_duty_runtime(duty_runtime, now);
        employee_channel.sendCheckoutResult(
            seq, employee_id, (uint32_t)checkin_time, (uint32_t)now, duration);

        active_worker_id = 0;
        recognition_purpose = RECOGNITION_PURPOSE_CHECKIN;
        protocol_attendance_enabled = false;
        employee_session.reset();

        if (yolo_model_ready) {
            release_yolo_runtime(yolo_ctx);
            yolo_model_ready = false;
        }
        yolo_resize_buf.release();
        memset(&duty_results, 0, sizeof(duty_results));
        memset(&duty_last_infer, 0, sizeof(duty_last_infer));
        duty_runtime = DutyRuntime();
        detect_state = DETECT_STATE_FACE_ACTIVE;

        set_ui_banner(ui_banner,
                      "考勤待机",
                      "等待上位机指令...",
                      uart_control_build_uart_status_text(uart_control_ready, uart_tx_ready));
    };

    auto handle_employee_request = [&](const EmployeeRequest &request) {
        if (request.type == EmployeeRequestType::ENROLL) {
            if (employee_session.busy()) {
                employee_channel.sendError(
                    request.seq, EMP_PROTO_CMD_ENROLL_REQ, EMP_PROTO_ERROR_TIMEOUT);
                return;
            }
            if (find_face_entry_by_worker_id((int)request.employee_id)) {
                employee_channel.sendEnrollResult(
                    request.seq, request.employee_id, EMP_PROTO_ENROLL_STATUS_EXISTS);
                return;
            }
            if (!fingerprint.ready) {
                employee_channel.sendEnrollResult(
                    request.seq, request.employee_id, EMP_PROTO_ENROLL_STATUS_FINGER_FAIL);
                return;
            }
            uint16_t next_page = 0;
            if (!fingerprint_page_map_allocate(fingerprint_map,
                                               fingerprint.page_num,
                                               &next_page)) {
                employee_channel.sendEnrollResult(
                    request.seq, request.employee_id, EMP_PROTO_ENROLL_STATUS_STORAGE);
                return;
            }
            if (!ensure_face_detection_active()) {
                employee_channel.sendEnrollResult(
                    request.seq, request.employee_id, EMP_PROTO_ENROLL_STATUS_FACE_FAIL);
                return;
            }

            employee_session.reset();
            employee_session.task = EmployeeTask::ENROLL;
            employee_session.seq = request.seq;
            employee_session.employee_id = request.employee_id;
            employee_session.name = request.name;
            employee_session.started_at = time(nullptr);
            enroll_face_samples.clear();
            char capture_line[48];
            snprintf(capture_line, sizeof(capture_line),
                     "CAPTURE %d FACE SNAPSHOTS", ENROLL_FACE_SAMPLE_TARGET);
            set_ui_banner(ui_banner,
                          "协议录入请求",
                          compact_employee_label(request.name, request.employee_id),
                          capture_line);
            return;
        }

        if (request.type == EmployeeRequestType::CHECKIN) {
            if (employee_session.busy()) {
                employee_channel.sendCheckinFail(request.seq, EMP_PROTO_CHECKIN_FAIL_BUSY);
                return;
            }
            if (active_worker_id > 0) {
                employee_channel.sendCheckinFail(request.seq, EMP_PROTO_CHECKIN_FAIL_ALREADY);
                return;
            }
            if (!ensure_face_detection_active()) {
                employee_channel.sendCheckinFail(request.seq, EMP_PROTO_CHECKIN_FAIL_FACE_TIMEOUT);
                return;
            }

            employee_session.reset();
            employee_session.task = EmployeeTask::CHECKIN;
            employee_session.seq = request.seq;
            employee_session.started_at = time(nullptr);
            protocol_attendance_enabled = true;
            recognition_purpose = RECOGNITION_PURPOSE_CHECKIN;
            set_ui_banner(ui_banner,
                          "协议签到请求",
                          "FACE OR FINGER",
                          "WAITING");
            return;
        }

        if (request.type == EmployeeRequestType::CHECKOUT) {
            if (employee_session.busy()) {
                employee_channel.sendError(
                    request.seq, EMP_PROTO_CMD_CHECKOUT_REQ, EMP_PROTO_ERROR_TIMEOUT);
                return;
            }
            const FaceEntry *entry = find_face_entry_by_worker_id((int)request.employee_id);
            if (!entry) {
                employee_channel.sendError(
                    request.seq, EMP_PROTO_CMD_CHECKOUT_REQ, EMP_PROTO_ERROR_EMPLOYEE_NOT_FOUND);
                return;
            }
            if (active_worker_id <= 0 || active_worker_id != (int)request.employee_id ||
                duty_runtime.started_at <= 0) {
                employee_channel.sendError(
                    request.seq, EMP_PROTO_CMD_CHECKOUT_REQ, EMP_PROTO_ERROR_NOT_CHECKED_IN);
                return;
            }
            if (!ensure_face_detection_active()) {
                employee_channel.sendError(
                    request.seq, EMP_PROTO_CMD_CHECKOUT_REQ, EMP_PROTO_ERROR_TIMEOUT);
                return;
            }

            employee_session.reset();
            employee_session.task = EmployeeTask::CHECKOUT;
            employee_session.seq = request.seq;
            employee_session.employee_id = request.employee_id;
            employee_session.name = entry->name;
            employee_session.started_at = time(nullptr);
            protocol_attendance_enabled = true;
            recognition_purpose = RECOGNITION_PURPOSE_CHECKOUT;
            set_ui_banner(ui_banner,
                          "协议签退请求",
                          compact_employee_label(entry->name, request.employee_id),
                          "FACE OR FINGER");
            return;
        }

        if (request.type == EmployeeRequestType::DELETE_EMPLOYEE) {
            if (employee_session.busy()) {
                employee_channel.sendError(
                    request.seq, EMP_PROTO_CMD_DELETE_EMPLOYEE_REQ, EMP_PROTO_ERROR_TIMEOUT);
                return;
            }

            DeleteEmployeeResult delete_result =
                delete_employee_records(request.employee_id, fingerprint,
                                        fingerprint_map, db_path);
            if (!delete_result.found) {
                employee_channel.sendError(
                    request.seq, EMP_PROTO_CMD_DELETE_EMPLOYEE_REQ,
                    EMP_PROTO_ERROR_EMPLOYEE_NOT_FOUND);
                return;
            }

            reset_confirm_states(confirm, MAX_DETECT_FACES);
            reset_face_recognize_cache(face_cache, MAX_DETECT_FACES);

            if (active_worker_id == (int)request.employee_id) {
                active_worker_id = 0;
                protocol_attendance_enabled = false;
                recognition_purpose = RECOGNITION_PURPOSE_CHECKIN;
                finalize_duty_runtime(duty_runtime, time(nullptr));
                if (yolo_model_ready) {
                    release_yolo_runtime(yolo_ctx);
                    yolo_model_ready = false;
                }
                yolo_resize_buf.release();
                memset(&duty_results, 0, sizeof(duty_results));
                memset(&duty_last_infer, 0, sizeof(duty_last_infer));
                duty_runtime = DutyRuntime();
                detect_state = DETECT_STATE_FACE_ACTIVE;
            }

            char line[96];
            snprintf(line, sizeof(line), "ID:%u %s",
                     request.employee_id, delete_result.name.c_str());
            set_ui_banner(ui_banner,
                          (delete_result.face_ok &&
                           delete_result.fingerprint_ok &&
                           delete_result.map_ok)
                              ? "员工删除成功"
                              : "员工删除部分失败",
                          line,
                          (delete_result.fingerprint_ok && delete_result.map_ok)
                              ? "FACE DB UPDATED"
                              : "FINGERPRINT/MAP DELETE FAILED");
            printf("[EMP-PROTO] DELETE_EMPLOYEE done employee=%u face=%s fingerprint=%s map=%s\n",
                   request.employee_id, delete_result.face_ok ? "ok" : "fail",
                   delete_result.fingerprint_ok ? "ok" : "fail",
                   delete_result.map_ok ? "ok" : "fail");
            return;
        }
    };

    auto process_employee_requests = [&]() {
        EmployeeRequest request;
        while (employee_channel.popRequest(&request)) {
            handle_employee_request(request);
        }
    };

    auto check_employee_protocol_timeout = [&]() {
        if (!employee_session.busy()) return;
        time_t now = time(nullptr);
        if (employee_session.task == EmployeeTask::ENROLL &&
            difftime(now, employee_session.started_at) >= EMPLOYEE_ENROLL_TIMEOUT_S) {
            employee_channel.sendEnrollResult(
                employee_session.seq, employee_session.employee_id, EMP_PROTO_ENROLL_STATUS_FACE_FAIL);
            stop_protocol_enroll_capture();
            return;
        }
        if (employee_session.task == EmployeeTask::CHECKIN &&
            difftime(now, employee_session.started_at) >= EMPLOYEE_CHECKIN_TIMEOUT_S) {
            uint8_t reason = employee_session.face_employee_id == 0
                ? EMP_PROTO_CHECKIN_FAIL_FACE_TIMEOUT
                : EMP_PROTO_CHECKIN_FAIL_FINGER_TIMEOUT;
            employee_channel.sendCheckinFail(employee_session.seq, reason);
            protocol_attendance_enabled = false;
            recognition_purpose = RECOGNITION_PURPOSE_CHECKIN;
            employee_session.reset();
        }
        if (employee_session.task == EmployeeTask::CHECKOUT &&
            difftime(now, employee_session.started_at) >= EMPLOYEE_CHECKIN_TIMEOUT_S) {
            employee_channel.sendError(
                employee_session.seq, EMP_PROTO_CMD_CHECKOUT_REQ, EMP_PROTO_ERROR_TIMEOUT);
            protocol_attendance_enabled = false;
            recognition_purpose = RECOGNITION_PURPOSE_CHECKIN;
            employee_session.reset();
        }
    };

    auto handle_attendance_match = [&](const FaceEntry *match,
                                       const char *source,
                                       time_t now) -> bool {
        if (!match) return false;
        if (!protocol_attendance_enabled) return false;
        if (employee_session.task != EmployeeTask::CHECKIN &&
            employee_session.task != EmployeeTask::CHECKOUT &&
            recognition_purpose != RECOGNITION_PURPOSE_CHECKOUT) {
            return false;
        }

        bool desired_checkin = (recognition_purpose == RECOGNITION_PURPOSE_CHECKIN);
        bool protocol_checkin_commit = false;
        bool protocol_checkout_commit = false;
        if (employee_session.task == EmployeeTask::CHECKIN ||
            employee_session.task == EmployeeTask::CHECKOUT) {
            bool from_fingerprint = source && strcmp(source, "指纹") == 0;
            if (from_fingerprint) {
                employee_session.fingerprint_employee_id = (uint32_t)match->worker_id;
            } else {
                employee_session.face_employee_id = (uint32_t)match->worker_id;
            }

            if (employee_session.task == EmployeeTask::CHECKIN) {
                protocol_checkin_commit = true;
                desired_checkin = true;
                recognition_purpose = RECOGNITION_PURPOSE_CHECKIN;
            } else {
                if (employee_session.employee_id == 0 ||
                    match->worker_id != employee_session.employee_id) {
                    employee_channel.sendError(employee_session.seq,
                                               EMP_PROTO_CMD_CHECKOUT_REQ,
                                               EMP_PROTO_ERROR_NOT_CHECKED_IN);
                    set_ui_banner(ui_banner,
                                  "协议签退失败",
                                  "人脸与指纹不匹配",
                                  "请重新发起签退");
                    protocol_attendance_enabled = false;
                    recognition_purpose = RECOGNITION_PURPOSE_CHECKIN;
                    employee_session.reset();
                    return false;
                }
                if ((employee_session.face_employee_id != 0 &&
                     employee_session.face_employee_id != employee_session.employee_id) ||
                    (employee_session.fingerprint_employee_id != 0 &&
                     employee_session.fingerprint_employee_id != employee_session.employee_id)) {
                    employee_channel.sendError(employee_session.seq,
                                               EMP_PROTO_CMD_CHECKOUT_REQ,
                                               EMP_PROTO_ERROR_NOT_CHECKED_IN);
                    set_ui_banner(ui_banner,
                                  "协议签退失败",
                                  "人脸与指纹不匹配",
                                  "请重新发起签退");
                    protocol_attendance_enabled = false;
                    recognition_purpose = RECOGNITION_PURPOSE_CHECKIN;
                    employee_session.reset();
                    return false;
                }

                protocol_checkout_commit = true;
                desired_checkin = false;
                recognition_purpose = RECOGNITION_PURPOSE_CHECKOUT;
            }
        }

        auto reject_attendance_record = [&](const std::string &detail) -> bool {
            if (employee_session.task == EmployeeTask::CHECKIN) {
                employee_channel.sendCheckinFail(employee_session.seq, EMP_PROTO_CHECKIN_FAIL_ALREADY);
                protocol_attendance_enabled = false;
                recognition_purpose = RECOGNITION_PURPOSE_CHECKIN;
                employee_session.reset();
            }
            set_ui_banner(ui_banner,
                          "考勤状态未更新",
                          compact_employee_label(match->name, (uint32_t)match->worker_id),
                          detail);
            printf("[ATTEND] %s考勤状态未更新 | 姓名=%s | 工号=%d | 动作=%s | 原因=%s\n",
                   source ? source : "识别",
                   match->name.c_str(),
                   match->worker_id,
                   desired_checkin ? "签入" : "签退",
                   detail.c_str());
            return false;
        };

        if (recognition_purpose == RECOGNITION_PURPOSE_CHECKOUT &&
            active_worker_id > 0 &&
            match->worker_id != active_worker_id) {
            printf("[ATTEND] 签退身份不匹配 | 当前在岗工号=%d | 识别工号=%d | 来源=%s\n",
                   active_worker_id, match->worker_id, source ? source : "识别");
            set_ui_banner(ui_banner,
                          "签退身份不匹配",
                          "请当前在岗人员签退",
                          "等待人脸或指纹...");
            return false;
        }

        AttendanceRecord attendance;
        if (attendance_service.record_explicit(match, desired_checkin, &attendance)) {
            if (disp_available && !disp_on) start_lcd_refresh();

            std::string ts = format_time_text(now);
            const char *attendance_type = attendance.is_checkin ? "签入" : "签退";
            char info_line[128];
            snprintf(info_line, sizeof(info_line), "%s",
                     compact_employee_label(match->name, (uint32_t)match->worker_id).c_str());
            char detail_line[160];
            snprintf(detail_line, sizeof(detail_line),
                     "%s:%s %s", attendance_type, ts.c_str(),
                     attendance.is_checkin ? "进入在岗检测" : "返回待机");
            set_ui_banner(ui_banner,
                          attendance.is_checkin ? "考勤签入成功" : "考勤签退成功",
                          info_line, detail_line);

            printf("[ATTEND] %s考勤%s成功 | 姓名=%s | 工号=%d | 时间=%s | 状态=%s\n",
                   source ? source : "识别", attendance_type,
                   match->name.c_str(), match->worker_id, ts.c_str(),
                   attendance.is_checkin ? "进入在岗检测" : "返回待机");

            if (!attendance.is_checkin) {
                finalize_duty_runtime(duty_runtime, now);
            }
            if (protocol_checkin_commit && attendance.is_checkin) {
                employee_channel.sendCheckinOk(
                    employee_session.seq,
                    (uint32_t)match->worker_id,
                    (uint32_t)attendance.timestamp);
                protocol_attendance_enabled = false;
                employee_session.reset();
            }
            if (protocol_checkout_commit && !attendance.is_checkin) {
                employee_channel.sendCheckoutResult(
                    employee_session.seq,
                    (uint32_t)match->worker_id,
                    (uint32_t)duty_runtime.started_at,
                    (uint32_t)attendance.timestamp,
                    (uint32_t)std::max(0, (int)difftime(attendance.timestamp, duty_runtime.started_at)));
                employee_session.reset();
            }

            reset_confirm_states(confirm, MAX_DETECT_FACES);
            reset_face_recognize_cache(face_cache, MAX_DETECT_FACES);

            if (attendance.is_checkin) {
                active_worker_id = match->worker_id;
                recognition_purpose = RECOGNITION_PURPOSE_CHECKOUT;
            } else {
                active_worker_id = 0;
                recognition_purpose = RECOGNITION_PURPOSE_CHECKIN;
                protocol_attendance_enabled = false;
            }

            if (attendance.is_checkin && face_model_ready) {
                release_face_runtime(retina_ctx, facenet_ctx, retina_input);
                face_model_ready = false;
            }

            if (!attendance.is_checkin) {
                if (yolo_model_ready) {
                    release_yolo_runtime(yolo_ctx);
                    yolo_model_ready = false;
                }
                yolo_resize_buf.release();
                memset(&duty_results, 0, sizeof(duty_results));
                memset(&duty_last_infer, 0, sizeof(duty_last_infer));
                duty_runtime = DutyRuntime();
                detect_state = DETECT_STATE_FACE_ACTIVE;
                set_ui_banner(ui_banner,
                              "考勤待机",
                              "等待上位机指令...",
                              uart_control_build_uart_status_text(uart_control_ready,
                                                                  uart_tx_ready));
                return true;
            }

            bool camera_ok = camera.is_running() && camera.latest(probe);
            if (!camera_ok) camera_ok = camera.start(CAM_W, CAM_H, &probe);
            if (!camera_ok) {
                set_ui_banner(ui_banner,
                              "在岗检测启动失败",
                              "摄像头打开失败",
                              "可重新打卡或检查 sensor/ISP");
                detect_state = DETECT_STATE_SLEEP;
                return true;
            }
            bgr_cam = probe;
            probe.release();
            last_camera_seq = 0;

            if (yolo_model_ready) {
                release_yolo_runtime(yolo_ctx);
                yolo_model_ready = false;
            }
            memset(&duty_results, 0, sizeof(duty_results));
            memset(&duty_last_infer, 0, sizeof(duty_last_infer));
            duty_runtime = DutyRuntime();
            duty_runtime.started_at = now;
            duty_runtime.last_eval_at = now;

            yolo_model_ready = init_yolo_runtime(yolo_path, yolo_ctx);
            detect_state = DETECT_STATE_DUTY_ACTIVE;
            if (yolo_model_ready) {
                set_ui_banner(ui_banner,
                              "进入在岗检测",
                              "YOLOv5 person 模型运行中",
                              "等待上位机签退请求");
            } else {
                set_ui_banner(ui_banner,
                              "在岗模型加载失败",
                              "已释放人脸模型降低内存",
                              "等待上位机签退请求");
            }
            return true;
        }

        return reject_attendance_record(desired_checkin ? "CHECKIN NOT RECORDED" : "CHECKOUT NOT RECORDED");
    };

    auto poll_fingerprint_attendance = [&]() -> bool {
        if (!protocol_attendance_enabled) {
            return false;
        }
        if (employee_session.task != EmployeeTask::CHECKIN &&
            recognition_purpose != RECOGNITION_PURPOSE_CHECKOUT) {
            return false;
        }
        if (!fingerprint.ready || !fingerprint_poll_due(fingerprint)) return false;

        AS608::SearchResult fp_result;
        uint8_t ensure = AS608::kTimeout;
        if (!fingerprint.sensor.identify(&fp_result, 0, fingerprint.page_num, &ensure)) {
            if (ensure != AS608::kNoFinger &&
                ensure != AS608::kNotFound &&
                ensure != AS608::kTimeout) {
                printf("[AS608] identify failed: 0x%02X %s\n",
                       ensure, fingerprint.sensor.ensureMessage(ensure));
            }
            return false;
        }

        const FaceEntry *match = find_face_entry_by_fingerprint_page(fingerprint_map,
                                                                     fp_result.page_id);
        if (!match) {
            printf("[AS608] matched unmapped page=%u score=%u, ignored\n",
                   fp_result.page_id, fp_result.match_score);
            set_ui_banner(ui_banner,
                          "未知指纹",
                          "指纹页号未绑定工号",
                          "请先录入到人脸库工号");
            return false;
        }

        printf("[AS608] matched page=%u score=%u worker=%d\n",
               fp_result.page_id, fp_result.match_score, match->worker_id);
        return handle_attendance_match(match, "指纹", time(nullptr));
    };

#if APP_VERBOSE_LOG
    printf("[Main] Loop start. Press Ctrl+C to exit.\n");
#endif

    // -------------------------------------------------------------------------
    // 主循环
    // -------------------------------------------------------------------------
    while (g_running) {
        fps = update_loop_fps(fps_counter);

        employee_channel.pollInput();
        process_employee_requests();
        check_employee_protocol_timeout();
        uart_control_poll_input(uart_control_fd, uart_rx_buffer);
        process_fingerprint_manage_request(fingerprint, fingerprint_map, ui_banner);

        if (detect_state == DETECT_STATE_SLEEP) {
            if (poll_fingerprint_attendance()) {
                continue;
            }
            usleep(100000);
            continue;
        }

        if (detect_state == DETECT_STATE_DUTY_ACTIVE) {
            time_t duty_now = time(nullptr);

            if (camera.latest(bgr_cam)) {
                resize_cover_to_mat(bgr_cam, bgr, &display_transform, display_resize_buf);

                bool should_infer = false;
                if (duty_last_infer.tv_sec == 0 && duty_last_infer.tv_usec == 0) {
                    should_infer = true;
                } else {
                    struct timeval tv_now;
                    gettimeofday(&tv_now, nullptr);
                    should_infer = elapsed_ms(duty_last_infer, tv_now) >= DUTY_INFER_INTERVAL_MS;
                }

                if (should_infer && yolo_model_ready) {
                    gettimeofday(&duty_last_infer, nullptr);
                    memset(&duty_results, 0, sizeof(duty_results));
                    if (letterbox_bgr_to_yolo_input(bgr_cam, yolo_ctx,
                                                    yolo_transform, yolo_resize_buf) &&
                        inference_yolov5_model(&yolo_ctx, &duty_results) == 0) {
                        yolov5_on_duty_status_t duty_status =
                            evaluate_on_duty_status(&duty_results,
                                                    yolo_ctx.model_width,
                                                    yolo_ctx.model_height,
                                                    YOLOV5_ON_DUTY_MIN_SCORE,
                                                    YOLOV5_ON_DUTY_MIN_AREA);
                        update_duty_runtime(duty_runtime, duty_status, duty_now);
                        set_ui_banner(ui_banner,
                                      duty_runtime.last_on_duty ? "在岗检测中" : "未检测到在岗",
                                      build_duty_status_line(duty_runtime),
                                      build_duty_time_line(duty_runtime));
                    } else {
                        set_ui_banner(ui_banner,
                                      "在岗检测异常",
                                      "YOLOv5 推理失败",
                                      "等待上位机签退请求");
                    }
                }

                for (int i = 0; i < duty_results.count; ++i) {
                    draw_duty_person_result(bgr, duty_results.results[i],
                                            yolo_transform, display_transform);
                }
            } else {
                bgr.setTo(cv::Scalar(0, 0, 0));
            }

            refresh_duty_runtime_clock(duty_runtime, duty_now);
            if (disp_on && should_flush_lcd(lcd_last_flush)) {
                draw_status_bar(bgr, fps, lcd_async.lcd_fps());
                draw_event_panel(bgr, ui_banner, uart_tx_ready);
                lcd_async.submit(bgr);
            }

            usleep(20000);
            continue;
        }

        if (detect_state == DETECT_STATE_FACE_ACTIVE && poll_fingerprint_attendance()) {
            continue;
        }

        if (!face_model_ready) {
            set_ui_banner(ui_banner,
                          "人脸模型未就绪",
                          "等待协议请求或重启",
                          uart_control_build_uart_status_text(uart_control_ready, uart_tx_ready));
            usleep(100000);
            continue;
        }

        time_t frame_now = time(nullptr);

        // 1. 采集原生帧
        uint64_t camera_seq = 0;
        if (!camera.latest(bgr_cam, &camera_seq)) { usleep(5000); continue; }
        if (camera_seq == last_camera_seq) { usleep(3000); continue; }
        last_camera_seq = camera_seq;

	        // 2. 缩放到LCD尺寸（等比例填满，用同一变换绘制检测框）
	        resize_cover_to_mat(bgr_cam, bgr, &display_transform, display_resize_buf);

        struct timeval face_infer_now;
        gettimeofday(&face_infer_now, nullptr);
        bool face_detect_due =
            timeval_is_zero(face_last_infer) ||
            elapsed_ms(face_last_infer, face_infer_now) >= FACE_DETECT_INTERVAL_MS;
        bool face_detect_ran = false;

        if (face_detect_due) {
            // 3. 缩放到人脸检测输入（零拷贝写入NPU内存）
            letterbox_to_mat(bgr_cam, retina_input, &retina_transform);

            // 4. 人脸检测推理：降频运行，显示帧复用最近一次检测结果。
            if (run_face_detector(&retina_ctx, &od_results) != 0) {
                printf("[Main] face detector inference error\n");
                memset(&od_results, 0, sizeof(od_results));
            } else {
                face_last_infer = face_infer_now;
                face_detect_ran = true;
            }
        }

        // 5. 单工位只处理主脸；检测降频更新框，显示帧复用短缓存。
        int processed_faces = 0;
        bool gimbal_tracked_face = false;
        struct timeval recog_now;
        gettimeofday(&recog_now, nullptr);
        for (int i = 0; i < od_results.count; i++) {
            if (processed_faces >= FACE_PROCESS_MAX_FACES) break;

            object_detect_result *det = &od_results.results[i];
            if (det->prop < MIN_FACE_DETECT_SCORE) continue;

            object_detect_result draw_det = *det;

            // 坐标先从 Retina letterbox 空间还原回原始相机空间
            int cam_left   = det->box.left;
            int cam_top    = det->box.top;
            int cam_right  = det->box.right;
            int cam_bottom = det->box.bottom;
            if (!map_coordinates(retina_transform, &cam_left, &cam_top) ||
                !map_coordinates(retina_transform, &cam_right, &cam_bottom)) {
                continue;
            }

            // 边界保护
            cam_left   = std::max(0, cam_left);
            cam_top    = std::max(0, cam_top);
            cam_right  = std::min(bgr_cam.cols - 1, cam_right);
            cam_bottom = std::min(bgr_cam.rows - 1, cam_bottom);

            int bw = cam_right  - cam_left;
            int bh = cam_bottom - cam_top;
            if (bw < MIN_FACE_ROI_SIZE || bh < MIN_FACE_ROI_SIZE) continue;

            if (face_detect_ran && !gimbal_tracked_face) {
                gimbal_tracker.updateFaceBox(cam_left, cam_top, cam_right, cam_bottom, det->prop);
                gimbal_tracked_face = true;
            }

            int slot = processed_faces;
            FaceRecognizeCache &cache = face_cache[slot];
            bool enroll_needs_face = employee_session.task == EmployeeTask::ENROLL;
            bool attendance_needs_face =
                (protocol_attendance_enabled &&
                 (employee_session.task == EmployeeTask::CHECKIN ||
                  employee_session.task == EmployeeTask::CHECKOUT)) ||
                (recognition_purpose == RECOGNITION_PURPOSE_CHECKOUT &&
                 active_worker_id > 0);
            bool recognition_needed =
                enroll_needs_face || attendance_needs_face || idle_face_recognition_enabled;
            bool recog_due = face_detect_ran &&
                              (!cache.valid ||
                              timeval_is_zero(cache.last_run) ||
                              elapsed_ms(cache.last_run, recog_now) >= FACE_RECOG_INTERVAL_MS);
            bool cache_fresh = cache.valid &&
                               !timeval_is_zero(cache.last_run) &&
                               elapsed_ms(cache.last_run, recog_now) <= FACE_RECOG_CACHE_TTL_MS;
            bool recog_ran = false;

            if (!recognition_needed) {
                if (!cache.valid || cache.label != "FACE") {
                    set_face_detect_only_cache(cache, recog_now);
                }
            } else if (face_detect_ran && (recog_due || !cache_fresh || enroll_needs_face)) {
                cv::Mat face_roi;
                if (!align_face_from_detection(bgr_cam, det, &retina_transform, face_roi)) {
                    cv::Rect roi(cam_left, cam_top, bw, bh);
                    face_roi = bgr_cam(roi).clone();
                }

                if (!face_quality_allows_recognition(det, retina_transform, bw, bh, face_roi)) {
                    recog_ran = false;
                    if (!cache_fresh) {
                        set_face_detect_only_cache(cache, recog_now);
                    }
                } else {
                    if (employee_session.task == EmployeeTask::ENROLL) {
                        if (enroll_face_samples.size() < ENROLL_FACE_SAMPLE_TARGET) {
                            enroll_face_samples.push_back(face_roi.clone());
                            char sample_line[128];
                            snprintf(sample_line, sizeof(sample_line), "SNAPSHOT %zu/%d",
                                     enroll_face_samples.size(), ENROLL_FACE_SAMPLE_TARGET);
                            set_ui_banner(ui_banner,
                                          "协议人脸采集中",
                                          compact_employee_label(employee_session.name,
                                                                 employee_session.employee_id),
                                          sample_line);
                            if (enroll_face_samples.size() < ENROLL_FACE_SAMPLE_TARGET) {
                                break;
                            }
                        }

                        if (!fingerprint.ready) {
                            employee_channel.sendEnrollResult(
                                employee_session.seq,
                                employee_session.employee_id,
                                EMP_PROTO_ENROLL_STATUS_FINGER_FAIL);
                            stop_protocol_enroll_capture();
                            break;
                        }
                        uint16_t page_id = 0;
                        if (!fingerprint_page_map_allocate(fingerprint_map,
                                                           fingerprint.page_num,
                                                           &page_id)) {
                            employee_channel.sendEnrollResult(
                                employee_session.seq,
                                employee_session.employee_id,
                                EMP_PROTO_ENROLL_STATUS_STORAGE);
                            stop_protocol_enroll_capture();
                            break;
                        }

                        char enroll_line[128];
                        snprintf(enroll_line, sizeof(enroll_line), "ID %u PAGE %u FACE %zu/%d",
                                 employee_session.employee_id, page_id,
                                 enroll_face_samples.size(), ENROLL_FACE_SAMPLE_TARGET);
                        set_ui_banner(ui_banner,
                                      "协议指纹录入中",
                                      enroll_line,
                                      "PRESS SENSOR TWICE");

                        uint8_t ensure = AS608::kTimeout;
                        bool fp_ok = fingerprint.sensor.enroll(page_id, &ensure);
                        if (!fp_ok) {
                            printf("[EMP-PROTO] enroll fingerprint failed employee=%u page=%u ensure=0x%02X %s\n",
                                   employee_session.employee_id, page_id, ensure,
                                   fingerprint.sensor.ensureMessage(ensure));
                            employee_channel.sendEnrollResult(
                                employee_session.seq,
                                employee_session.employee_id,
                                EMP_PROTO_ENROLL_STATUS_FINGER_FAIL);
                            stop_protocol_enroll_capture();
                            break;
                        }

                        bool face_ok = register_face_samples(employee_session.name,
                                                             (int)employee_session.employee_id,
                                                             true,
                                                             &facenet_ctx,
                                                             enroll_face_samples,
                                                             db_path);
                        if (!face_ok) {
                            fingerprint.sensor.remove(page_id, 1, nullptr);
                            uint8_t status = find_face_entry_by_worker_id(
                                (int)employee_session.employee_id)
                                ? EMP_PROTO_ENROLL_STATUS_EXISTS
                                : EMP_PROTO_ENROLL_STATUS_STORAGE;
                            employee_channel.sendEnrollResult(
                                employee_session.seq, employee_session.employee_id, status);
                            stop_protocol_enroll_capture();
                            break;
                        }

                        if (!fingerprint_page_map_set(fingerprint_map,
                                                      (int)employee_session.employee_id,
                                                      page_id,
                                                      fingerprint.page_num) ||
                            !save_fingerprint_page_map(fingerprint_map)) {
                            fingerprint_page_map_remove_worker(
                                fingerprint_map, (int)employee_session.employee_id);
                            fingerprint.sensor.remove(page_id, 1, nullptr);
                            remove_face_employee((int)employee_session.employee_id, db_path);
                            employee_channel.sendEnrollResult(
                                employee_session.seq,
                                employee_session.employee_id,
                                EMP_PROTO_ENROLL_STATUS_STORAGE);
                            stop_protocol_enroll_capture();
                            break;
                        }

                        employee_channel.sendEnrollResult(
                            employee_session.seq, employee_session.employee_id, EMP_PROTO_ENROLL_STATUS_OK);
                        set_ui_banner(ui_banner,
                                      "协议录入成功",
                                      compact_employee_label(employee_session.name,
                                                             employee_session.employee_id),
                                      "READY FOR CHECKIN");
                        stop_protocol_enroll_capture();
                        break;
                    }
                    // 5. 人脸特征提取（零拷贝）。降频执行，避免每帧阻塞 UI。
                    extract_face_embedding(&facenet_ctx, face_roi, feat);

                    // 6. 单帧先按原阈值匹配进入连续确认；多帧均值只在同一工号稳定时增强结果。
                    float best_dist = get_face_db().empty() ? -1.f : 9999.f;
                    const FaceEntry *match = match_face(feat, &best_dist);

                    const FaceEntry *accepted_match = nullptr;
                    float accepted_dist = best_dist;
                    if (match) {
                        accepted_match = match;
                        add_face_feature_sample(cache, match, feat);
                        if (get_fused_face_feature(cache, fused_feat)) {
                            float fused_dist = 9999.f;
                            const FaceEntry *fused_match = match_face(fused_feat, &fused_dist);
                            if (fused_match &&
                                fused_match->worker_id == match->worker_id &&
                                build_face_match_label(fused_match) == cache.fusion_label) {
                                accepted_match = fused_match;
                                accepted_dist = fused_dist;
                            }
                        }
                    } else {
                        reset_face_feature_fusion(cache);
                        recog_ran = true;
                    }

                    cache.valid = true;
                    cache.last_run = recog_now;
                    cache.match = accepted_match;
                    cache.dist = accepted_match ? accepted_dist : best_dist;
                    cache.is_known = (accepted_match != nullptr);
                    cache.has_perm = accepted_match && accepted_match->has_permission;
                    cache.compared = true;
                    if (accepted_match) {
                        cache.label = build_face_match_label(accepted_match);
                        recog_ran = true;
                    } else if (match) {
                        cache.label = cache.fusion_label.empty() ? build_face_match_label(match)
                                                                 : cache.fusion_label;
                    } else {
                        cache.label = "STRANGER";
                    }
                }
            }

            const char *label = cache.valid ? cache.label.c_str() : "STRANGER";
            bool is_known = cache.valid && cache.is_known;
            bool has_perm = cache.valid && cache.has_perm;
            const FaceEntry *match = cache.valid ? cache.match : nullptr;
            float best_dist = cache.valid ? cache.dist : -1.f;
            FaceBoxVisualState visual_state = FACE_BOX_DETECT_ONLY;
            if (cache.valid && cache.compared) {
                visual_state = (is_known && has_perm)
                                   ? FACE_BOX_ACCEPTED
                                   : FACE_BOX_REJECTED;
            }

            // 7. 多帧确认防抖
            if (slot < MAX_DETECT_FACES && recog_ran) {
                if (confirm[slot].pending_name == label) {
                    confirm[slot].confirm_count++;
                } else {
                    confirm[slot].pending_name  = label;
                    confirm[slot].confirm_count = 1;
                }

                bool attendance_allowed =
                    protocol_attendance_enabled &&
                    (employee_session.task == EmployeeTask::CHECKIN ||
                     recognition_purpose == RECOGNITION_PURPOSE_CHECKOUT);
                if (is_known && has_perm && attendance_allowed &&
                    confirm[slot].confirm_count >= CONFIRM_FRAMES) {
                    (void)handle_attendance_match(match, "人脸", frame_now);
                    confirm[slot].confirm_count = 0;
                } else if ((!is_known || !has_perm) &&
                           confirm[slot].confirm_count >= CONFIRM_FRAMES) {
                    if (employee_session.task == EmployeeTask::CHECKIN) {
                        const char *reject_reason = is_known ? "未授权人员" : "未注册人员";
                        employee_channel.sendCheckinFail(
                            employee_session.seq,
                            EMP_PROTO_CHECKIN_FAIL_UNREGISTERED);
                        set_ui_banner(ui_banner,
                                      "协议签到失败",
                                      reject_reason,
                                      "请重新发起签到");
                        protocol_attendance_enabled = false;
                        recognition_purpose = RECOGNITION_PURPOSE_CHECKIN;
                        employee_session.reset();
                        confirm[slot].confirm_count = 0;
                        continue;
                    }
	                    confirm[slot].confirm_count = 0;
	                }
	            }

            // 9. 绘制到帧缓冲
            draw_det.box.left   = cam_left;
            draw_det.box.top    = cam_top;
            draw_det.box.right  = cam_right;
            draw_det.box.bottom = cam_bottom;
	            source_to_display_coordinates(display_transform, &draw_det.box.left,  &draw_det.box.top);
	            source_to_display_coordinates(display_transform, &draw_det.box.right, &draw_det.box.bottom);
            draw_face_result(bgr, &draw_det, label, visual_state, is_known, has_perm, best_dist);
            processed_faces++;

            if (detect_state == DETECT_STATE_DUTY_ACTIVE) break;
        }

        for (int i = processed_faces; i < MAX_DETECT_FACES && detect_state == DETECT_STATE_FACE_ACTIVE; ++i) {
            confirm[i].pending_name.clear();
            confirm[i].confirm_count = 0;
            face_cache[i] = FaceRecognizeCache();
        }
        if (face_detect_ran && !gimbal_tracked_face) {
            gimbal_tracker.onFrameNoFace();
        }

        // 10. 状态栏 + LCD 刷新
        if (disp_on && should_flush_lcd(lcd_last_flush)) {
            draw_status_bar(bgr, fps, lcd_async.lcd_fps());
            draw_event_panel(bgr, ui_banner, uart_tx_ready);
            lcd_async.submit(bgr);
        }

    }

    // -------------------------------------------------------------------------
    // 清理
    // -------------------------------------------------------------------------
    gimbal_tracker.shutdown();
    camera.stop();
    if (yolo_model_ready) release_yolo_runtime(yolo_ctx);
    if (face_model_ready) release_face_runtime(retina_ctx, facenet_ctx, retina_input);
    fingerprint.sensor.closeDevice();
    if (uart_control_fd >= 0 && !uart_control_stdio) close(uart_control_fd);
    if (uart_tx_fd >= 0) close(uart_tx_fd);
    lcd_async.stop();
    if (disp_on && env_flag_enabled("ATTENDANCE_EXIT_SCREEN")) {
        lcd_show_exit_screen(bgr);
        usleep(150000);
    }
    if (disp_on) {
        DEV_ModuleExit();
        LCD_BL_0;
    }

#if APP_VERBOSE_LOG
    printf("[Main] Clean exit.\n");
#endif
    return 0;
}
