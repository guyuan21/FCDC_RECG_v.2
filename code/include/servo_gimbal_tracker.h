#pragma once

#include <string>

class ServoGimbalTracker {
public:
    ServoGimbalTracker();

    bool init(int frame_width, int frame_height);
    void updateFaceBox(int left, int top, int right, int bottom, float score);
    void onFrameNoFace();
    void shutdown();

    bool enabled() const { return enabled_; }

private:
    struct PwmChannel {
        const char *chip_path;
        int channel;
        int min_us;
        int max_us;
        int home_us;
        int pulse_us;
        bool ready;
    };

    bool initChannel(PwmChannel &pwm);
    bool writePulse(PwmChannel &pwm, int pulse_us);
    std::string pwmPath(const PwmChannel &pwm) const;
    void loadEnvOptions();

    int frame_width_;
    int frame_height_;
    int update_every_;
    int update_count_;
    int missed_count_;
    int pan_gain_us_;
    int tilt_gain_us_;
    int pan_max_step_us_;
    int tilt_max_step_us_;
    int min_write_delta_us_;
    float deadzone_x_;
    float deadzone_y_;
    float release_deadzone_x_;
    float release_deadzone_y_;
    float smooth_alpha_;
    float smooth_x_;
    float smooth_y_;
    bool has_smooth_face_;
    bool pan_locked_;
    bool tilt_locked_;
    bool enabled_;
    PwmChannel pan_;
    PwmChannel tilt_;
};
