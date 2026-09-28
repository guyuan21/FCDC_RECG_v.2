#pragma once

#include <time.h>

#include <unordered_map>

#include "face_models.h"

class AttendanceService {
public:
    explicit AttendanceService(int cooldown_sec = ATTENDANCE_COOLDOWN_S);

    bool record(const FaceEntry *entry, AttendanceRecord *out_record = nullptr);
    bool record_explicit(const FaceEntry *entry, bool is_checkin,
                         AttendanceRecord *out_record = nullptr);
    bool is_recent_duplicate(int worker_id, bool is_checkin, time_t now) const;

private:
    int cooldown_sec_;
    std::unordered_map<int, time_t> last_attendance_;
    std::unordered_map<int, bool> last_checkin_state_;
};
