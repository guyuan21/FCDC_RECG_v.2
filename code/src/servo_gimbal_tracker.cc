#include "servo_gimbal_tracker.h"

#include <errno.h>
#include <fcntl.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <string>

namespace {

static bool env_flag_disabled(const char *name)
{
    const char *value = getenv(name);
    if (!value || !value[0]) return false;
    return strcmp(value, "0") == 0 ||
           strcasecmp(value, "false") == 0 ||
           strcasecmp(value, "no") == 0 ||
           strcasecmp(value, "off") == 0;
}

static int env_int_clamped(const char *name, int fallback, int lo, int hi)
{
    const char *value = getenv(name);
    if (!value || !value[0]) return fallback;

    char *end = nullptr;
    long parsed = strtol(value, &end, 10);
    if (end == value || *end != '\0') return fallback;
    return std::max(lo, std::min((int)parsed, hi));
}

static float env_float_clamped(const char *name, float fallback, float lo, float hi)
{
    const char *value = getenv(name);
    if (!value || !value[0]) return fallback;

    char *end = nullptr;
    float parsed = strtof(value, &end);
    if (end == value || *end != '\0') return fallback;
    return std::max(lo, std::min(parsed, hi));
}

static bool path_exists(const std::string &path)
{
    struct stat st;
    return stat(path.c_str(), &st) == 0;
}

static bool write_text(const std::string &path, const std::string &text)
{
    int fd = open(path.c_str(), O_WRONLY);
    if (fd < 0) {
        printf("[Gimbal] open %s failed: %s\n", path.c_str(), strerror(errno));
        return false;
    }

    const char *data = text.c_str();
    size_t left = text.size();
    while (left > 0) {
        ssize_t n = write(fd, data, left);
        if (n < 0) {
            printf("[Gimbal] write %s failed: %s\n", path.c_str(), strerror(errno));
            close(fd);
            return false;
        }
        data += n;
        left -= (size_t)n;
    }

    close(fd);
    return true;
}

static bool write_int64(const std::string &path, long long value)
{
    char buf[64];
    snprintf(buf, sizeof(buf), "%lld", value);
    return write_text(path, buf);
}

static int clamp_int(int value, int lo, int hi)
{
    return std::max(lo, std::min(value, hi));
}

static int step_towards(int current, int target, int max_step)
{
    int delta = target - current;
    if (delta > max_step) return current + max_step;
    if (delta < -max_step) return current - max_step;
    return target;
}

} // namespace

ServoGimbalTracker::ServoGimbalTracker()
    : frame_width_(0),
      frame_height_(0),
      update_every_(1),
      update_count_(0),
      missed_count_(0),
      pan_gain_us_(220),
      tilt_gain_us_(150),
      pan_max_step_us_(120),
      tilt_max_step_us_(90),
      min_write_delta_us_(4),
      deadzone_x_(0.10f),
      deadzone_y_(0.12f),
      release_deadzone_x_(0.18f),
      release_deadzone_y_(0.20f),
      smooth_alpha_(0.35f),
      smooth_x_(0.f),
      smooth_y_(0.f),
      has_smooth_face_(false),
      pan_locked_(true),
      tilt_locked_(true),
      enabled_(false),
      pan_{"/sys/class/pwm/pwmchip6", 0, 500, 2400, 1600, 1600, false},
      tilt_{"/sys/class/pwm/pwmchip2", 0, 500, 1400, 1000, 1000, false}
{
}

bool ServoGimbalTracker::init(int frame_width, int frame_height)
{
    frame_width_ = frame_width;
    frame_height_ = frame_height;
    loadEnvOptions();

    if (env_flag_disabled("ATTENDANCE_GIMBAL_TRACK")) {
        printf("[Gimbal] disabled by ATTENDANCE_GIMBAL_TRACK\n");
        return false;
    }

    bool pan_ok = initChannel(pan_);
    bool tilt_ok = initChannel(tilt_);
    enabled_ = pan_ok || tilt_ok;
    if (enabled_) {
        printf("[Gimbal] tracker ready pan=%dus range=%d..%d tilt=%dus range=%d..%d update_every=%d deadzone=%.2f/%.2f release=%.2f/%.2f\n",
               pan_.pulse_us, pan_.min_us, pan_.max_us,
               tilt_.pulse_us, tilt_.min_us, tilt_.max_us,
               update_every_,
               deadzone_x_, deadzone_y_,
               release_deadzone_x_, release_deadzone_y_);
    } else {
        printf("[Gimbal] tracker unavailable, face recognition continues without servo tracking\n");
    }
    return enabled_;
}

void ServoGimbalTracker::loadEnvOptions()
{
    update_every_ = env_int_clamped("ATTENDANCE_GIMBAL_UPDATE_EVERY", update_every_, 1, 60);
    pan_gain_us_ = env_int_clamped("ATTENDANCE_GIMBAL_PAN_GAIN_US", pan_gain_us_, 5, 300);
    tilt_gain_us_ = env_int_clamped("ATTENDANCE_GIMBAL_TILT_GAIN_US", tilt_gain_us_, 5, 300);
    pan_max_step_us_ = env_int_clamped("ATTENDANCE_GIMBAL_PAN_MAX_STEP_US", pan_max_step_us_, 5, 300);
    tilt_max_step_us_ = env_int_clamped("ATTENDANCE_GIMBAL_TILT_MAX_STEP_US", tilt_max_step_us_, 5, 300);
    min_write_delta_us_ = env_int_clamped("ATTENDANCE_GIMBAL_MIN_WRITE_DELTA_US", min_write_delta_us_, 1, 50);
    deadzone_x_ = env_float_clamped("ATTENDANCE_GIMBAL_DEADZONE_X", deadzone_x_, 0.01f, 0.40f);
    deadzone_y_ = env_float_clamped("ATTENDANCE_GIMBAL_DEADZONE_Y", deadzone_y_, 0.01f, 0.40f);
    release_deadzone_x_ = env_float_clamped("ATTENDANCE_GIMBAL_RELEASE_DEADZONE_X",
                                            release_deadzone_x_, deadzone_x_, 0.60f);
    release_deadzone_y_ = env_float_clamped("ATTENDANCE_GIMBAL_RELEASE_DEADZONE_Y",
                                            release_deadzone_y_, deadzone_y_, 0.60f);
    smooth_alpha_ = env_float_clamped("ATTENDANCE_GIMBAL_SMOOTH_ALPHA", smooth_alpha_, 0.05f, 1.0f);
}

bool ServoGimbalTracker::initChannel(PwmChannel &pwm)
{
    if (!path_exists(pwm.chip_path)) {
        printf("[Gimbal] PWM chip not found: %s\n", pwm.chip_path);
        return false;
    }

    const std::string path = pwmPath(pwm);
    if (!path_exists(path)) {
        if (!write_int64(std::string(pwm.chip_path) + "/export", pwm.channel)) {
            return false;
        }
        for (int i = 0; i < 20 && !path_exists(path); ++i) {
            usleep(10000);
        }
    }
    if (!path_exists(path)) {
        printf("[Gimbal] PWM path did not appear after export: %s\n", path.c_str());
        return false;
    }

    write_int64(path + "/enable", 0);
    write_text(path + "/polarity", "normal");
    if (!write_int64(path + "/period", 20000000LL)) return false;

    pwm.pulse_us = clamp_int(pwm.home_us, pwm.min_us, pwm.max_us);
    if (!write_int64(path + "/duty_cycle", (long long)pwm.pulse_us * 1000LL)) return false;
    if (!write_int64(path + "/enable", 1)) return false;

    pwm.ready = true;
    return true;
}

bool ServoGimbalTracker::writePulse(PwmChannel &pwm, int pulse_us)
{
    if (!pwm.ready) return false;

    pulse_us = clamp_int(pulse_us, pwm.min_us, pwm.max_us);
    if (abs(pulse_us - pwm.pulse_us) < min_write_delta_us_) {
        return true;
    }

    const std::string path = pwmPath(pwm);
    if (!write_int64(path + "/duty_cycle", (long long)pulse_us * 1000LL)) {
        return false;
    }
    pwm.pulse_us = pulse_us;
    return true;
}

std::string ServoGimbalTracker::pwmPath(const PwmChannel &pwm) const
{
    char suffix[32];
    snprintf(suffix, sizeof(suffix), "/pwm%d", pwm.channel);
    return std::string(pwm.chip_path) + suffix;
}

void ServoGimbalTracker::updateFaceBox(int left, int top, int right, int bottom, float score)
{
    (void)score;
    if (!enabled_ || frame_width_ <= 0 || frame_height_ <= 0) return;
    if (right <= left || bottom <= top) return;

    ++update_count_;
    if (update_count_ % update_every_ != 0) return;

    const float raw_x = ((float)left + (float)right) * 0.5f;
    const float raw_y = ((float)top + (float)bottom) * 0.5f;
    const float half_w = (float)frame_width_ * 0.5f;
    const float half_h = (float)frame_height_ * 0.5f;
    float raw_error_x = (raw_x - half_w) / half_w;
    float raw_error_y = (raw_y - half_h) / half_h;
    raw_error_x = std::max(-1.0f, std::min(raw_error_x, 1.0f));
    raw_error_y = std::max(-1.0f, std::min(raw_error_y, 1.0f));

    if (!has_smooth_face_) {
        smooth_x_ = raw_x;
        smooth_y_ = raw_y;
        has_smooth_face_ = true;
    } else {
        smooth_x_ = smooth_x_ * (1.0f - smooth_alpha_) + raw_x * smooth_alpha_;
        smooth_y_ = smooth_y_ * (1.0f - smooth_alpha_) + raw_y * smooth_alpha_;
    }

    float error_x = (smooth_x_ - half_w) / half_w;
    float error_y = (smooth_y_ - half_h) / half_h;
    error_x = std::max(-1.0f, std::min(error_x, 1.0f));
    error_y = std::max(-1.0f, std::min(error_y, 1.0f));

    int next_pan = pan_.pulse_us;
    int next_tilt = tilt_.pulse_us;

    if (fabsf(raw_error_x) <= deadzone_x_) {
        pan_locked_ = true;
        smooth_x_ = raw_x;
    } else if (fabsf(raw_error_x) >= release_deadzone_x_) {
        pan_locked_ = false;
    }
    if (fabsf(raw_error_y) <= deadzone_y_) {
        tilt_locked_ = true;
        smooth_y_ = raw_y;
    } else if (fabsf(raw_error_y) >= release_deadzone_y_) {
        tilt_locked_ = false;
    }

    if (!pan_locked_) {
        int target = pan_.pulse_us - (int)(error_x * (float)pan_gain_us_ + (error_x >= 0.f ? 0.5f : -0.5f));
        next_pan = step_towards(pan_.pulse_us, target, pan_max_step_us_);
    }
    if (!tilt_locked_) {
        int target = tilt_.pulse_us + (int)(error_y * (float)tilt_gain_us_ + (error_y >= 0.f ? 0.5f : -0.5f));
        next_tilt = step_towards(tilt_.pulse_us, target, tilt_max_step_us_);
    }

    writePulse(pan_, next_pan);
    writePulse(tilt_, next_tilt);
    missed_count_ = 0;
}

void ServoGimbalTracker::onFrameNoFace()
{
    if (!enabled_) return;
    ++missed_count_;
    if (missed_count_ >= 8) {
        has_smooth_face_ = false;
        pan_locked_ = true;
        tilt_locked_ = true;
    }
}

void ServoGimbalTracker::shutdown()
{
    if (!enabled_) return;

    if (pan_.ready) {
        write_int64(pwmPath(pan_) + "/enable", 0);
        pan_.ready = false;
    }
    if (tilt_.ready) {
        write_int64(pwmPath(tilt_) + "/enable", 0);
        tilt_.ready = false;
    }
    enabled_ = false;
}
