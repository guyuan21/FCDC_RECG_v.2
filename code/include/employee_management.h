#ifndef EMPLOYEE_MANAGEMENT_H
#define EMPLOYEE_MANAGEMENT_H

#include <stdint.h>
#include <time.h>

#include <deque>
#include <string>

#include "employee_protocol.h"

enum class EmployeeRequestType {
    ENROLL,
    CHECKIN,
    CHECKOUT,
    DELETE_EMPLOYEE,
};

struct EmployeeRequest {
    EmployeeRequestType type = EmployeeRequestType::CHECKIN;
    uint8_t seq = 0;
    uint32_t employee_id = 0;
    std::string name;
};

enum class EmployeeTask {
    NONE,
    ENROLL,
    CHECKIN,
    CHECKOUT,
};

struct EmployeeSession {
    EmployeeTask task = EmployeeTask::NONE;
    uint8_t seq = 0;
    uint32_t employee_id = 0;
    std::string name;
    time_t started_at = 0;
    uint32_t face_employee_id = 0;
    uint32_t fingerprint_employee_id = 0;

    bool busy() const { return task != EmployeeTask::NONE; }
    void reset() { *this = EmployeeSession(); }
};

class EmployeeManagementChannel {
public:
    explicit EmployeeManagementChannel(int fd = -1);

    void setFd(int fd);
    bool ready() const;
    void pollInput();
    bool popRequest(EmployeeRequest *request);

    int sendEnrollResult(uint8_t seq, uint32_t employee_id, uint8_t status);
    int sendCheckinOk(uint8_t seq, uint32_t employee_id, uint32_t checkin_time);
    int sendCheckinFail(uint8_t seq, uint8_t reason);
    int sendCheckoutResult(uint8_t seq,
                           uint32_t employee_id,
                           uint32_t checkin_time,
                           uint32_t checkout_time,
                           uint32_t duration_sec);
    int sendError(uint8_t seq, uint8_t orig_cmd, uint8_t err_code);

private:
    void handleFrame(const EmployeeProtocolFrame &frame);
    bool enqueueRequest(const EmployeeRequest &request, uint8_t orig_cmd);
    int sendPackedFrame(int frame_len, const uint8_t *data, const char *tag);
    int writeAll(const uint8_t *data, uint16_t len);

    int fd_ = -1;
    EmployeeProtocolParser parser_;
    std::deque<EmployeeRequest> pending_;
};

#endif /* EMPLOYEE_MANAGEMENT_H */
