#include "attendance_service.h"

AttendanceService::AttendanceService(int cooldown_sec)
    : cooldown_sec_(cooldown_sec)
{
}

bool AttendanceService::record(const FaceEntry *entry, AttendanceRecord *out_record)
{
    if (!entry) return false;
    time_t now = time(nullptr);

    auto it = last_attendance_.find(entry->worker_id);
    if (it != last_attendance_.end() &&
        difftime(now, it->second) < cooldown_sec_) {
        return false;
    }

    bool is_checkin = true;
    auto state_it = last_checkin_state_.find(entry->worker_id);
    if (state_it != last_checkin_state_.end()) {
        is_checkin = !state_it->second;
    }

    AttendanceRecord rec;
    rec.name = entry->name;
    rec.worker_id = entry->worker_id;
    rec.timestamp = now;
    rec.is_checkin = is_checkin;

    last_attendance_[entry->worker_id] = now;
    last_checkin_state_[entry->worker_id] = is_checkin;

    if (out_record) *out_record = rec;
    return true;
}

bool AttendanceService::record_explicit(const FaceEntry *entry, bool is_checkin,
                                        AttendanceRecord *out_record)
{
    if (!entry) return false;
    time_t now = time(nullptr);
    if (is_recent_duplicate(entry->worker_id, is_checkin, now)) {
        return false;
    }

    AttendanceRecord rec;
    rec.name = entry->name;
    rec.worker_id = entry->worker_id;
    rec.timestamp = now;
    rec.is_checkin = is_checkin;

    last_attendance_[entry->worker_id] = rec.timestamp;
    last_checkin_state_[entry->worker_id] = is_checkin;

    if (out_record) *out_record = rec;
    return true;
}

bool AttendanceService::is_recent_duplicate(int worker_id,
                                            bool is_checkin,
                                            time_t now) const
{
    if (cooldown_sec_ <= 0) return false;

    auto time_it = last_attendance_.find(worker_id);
    auto state_it = last_checkin_state_.find(worker_id);
    if (time_it == last_attendance_.end() ||
        state_it == last_checkin_state_.end()) {
        return false;
    }

    if (state_it->second != is_checkin) return false;

    double delta = difftime(now, time_it->second);
    if (delta < 0) return false;
    return delta < cooldown_sec_;
}
