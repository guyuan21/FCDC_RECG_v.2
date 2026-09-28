#include "employee_management.h"

#include <errno.h>
#include <poll.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <termios.h>
#include <time.h>
#include <unistd.h>

namespace {

constexpr uint16_t kMaxFrameLength = EMP_PROTO_FRAME_OVERHEAD + EMP_PROTO_MAX_DATA_LEN;
constexpr size_t kMaxPendingRequests = 8;

std::string sanitize_name(std::string name)
{
    for (char &ch : name) {
        unsigned char value = (unsigned char)ch;
        if (value <= 0x20u || value == 0x7fu) ch = '_';
    }
    return name;
}

const char *cmd_name(uint8_t cmd)
{
    switch (cmd) {
    case EMP_PROTO_CMD_ENROLL_REQ: return "ENROLL_REQ";
    case EMP_PROTO_CMD_CHECKIN_REQ: return "CHECKIN_REQ";
    case EMP_PROTO_CMD_CHECKIN_OK: return "CHECKIN_OK";
    case EMP_PROTO_CMD_CHECKIN_FAIL: return "CHECKIN_FAIL";
    case EMP_PROTO_CMD_CHECKOUT_REQ: return "CHECKOUT_REQ";
    case EMP_PROTO_CMD_CHECKOUT_RSP: return "CHECKOUT_RSP";
    case EMP_PROTO_CMD_DELETE_EMPLOYEE_REQ: return "DELETE_EMPLOYEE_REQ";
    case EMP_PROTO_CMD_ERROR: return "ERROR";
    default: return "UNKNOWN";
    }
}

const char *error_name(uint8_t err_code)
{
    switch (err_code) {
    case EMP_PROTO_ERROR_FRAME: return "FRAME";
    case EMP_PROTO_ERROR_HEAD: return "HEAD";
    case EMP_PROTO_ERROR_CRC: return "CRC";
    case EMP_PROTO_ERROR_UNKNOWN_CMD: return "UNKNOWN_CMD";
    case EMP_PROTO_ERROR_LENGTH: return "LENGTH";
    case EMP_PROTO_ERROR_EMPLOYEE_NOT_FOUND: return "EMPLOYEE_NOT_FOUND";
    case EMP_PROTO_ERROR_NOT_CHECKED_IN: return "NOT_CHECKED_IN";
    case EMP_PROTO_ERROR_TIMEOUT: return "TIMEOUT";
    default: return "UNKNOWN_ERROR";
    }
}

const char *enroll_status_name(uint8_t status)
{
    switch (status) {
    case EMP_PROTO_ENROLL_STATUS_OK: return "OK";
    case EMP_PROTO_ENROLL_STATUS_FACE_FAIL: return "FACE_FAIL";
    case EMP_PROTO_ENROLL_STATUS_FINGER_FAIL: return "FINGER_FAIL";
    case EMP_PROTO_ENROLL_STATUS_EXISTS: return "EXISTS";
    case EMP_PROTO_ENROLL_STATUS_STORAGE: return "STORAGE";
    default: return "UNKNOWN";
    }
}

bool protocol_log_enabled()
{
    static int enabled = -1;
    if (enabled < 0) {
        const char *value = getenv("ATTENDANCE_PROTOCOL_LOG");
        enabled = (value && strcmp(value, "1") == 0) ? 1 : 0;
    }
    return enabled != 0;
}

void format_local_time(uint32_t timestamp, char *buf, size_t buf_size)
{
    if (!buf || buf_size == 0) return;
    time_t ts = (time_t)timestamp;
    struct tm tm_info;
    if (localtime_r(&ts, &tm_info) &&
        strftime(buf, buf_size, "%Y-%m-%d %H:%M:%S %z", &tm_info) > 0) {
        return;
    }
    snprintf(buf, buf_size, "N/A");
}

} // namespace

EmployeeManagementChannel::EmployeeManagementChannel(int fd)
    : fd_(fd)
{
    employee_protocol_parser_init(&parser_);
}

void EmployeeManagementChannel::setFd(int fd)
{
    fd_ = fd;
    pending_.clear();
    employee_protocol_parser_init(&parser_);
}

bool EmployeeManagementChannel::ready() const
{
    return fd_ >= 0;
}

void EmployeeManagementChannel::pollInput()
{
    if (fd_ < 0) return;

    uint8_t buffer[128];
    while (true) {
        struct pollfd pfd;
        memset(&pfd, 0, sizeof(pfd));
        pfd.fd = fd_;
        pfd.events = POLLIN;

        int poll_ret = poll(&pfd, 1, 0);
        if (poll_ret <= 0) {
            if (poll_ret < 0 && errno != EINTR) {
                printf("[EMP-PROTO] poll failed: %s\n", strerror(errno));
            }
            return;
        }
        if ((pfd.revents & (POLLERR | POLLHUP | POLLNVAL)) != 0) {
            printf("[EMP-PROTO] UART disconnected, revents=0x%x\n", pfd.revents);
            return;
        }
        if ((pfd.revents & POLLIN) == 0) return;

        ssize_t count = read(fd_, buffer, sizeof(buffer));
        if (count <= 0) {
            if (count < 0 && errno != EAGAIN && errno != EWOULDBLOCK) {
                printf("[EMP-PROTO] read failed: %s\n", strerror(errno));
            }
            return;
        }

        for (ssize_t i = 0; i < count; ++i) {
            EmployeeProtocolFrame frame;
            int ret = employee_protocol_parse_byte_ex(&parser_, buffer[i], &frame);
            if (ret == 1) {
                handleFrame(frame);
            } else if (ret == EMP_PROTO_ERR_CHECKSUM) {
                sendError(frame.seq, frame.cmd, EMP_PROTO_ERROR_CRC);
            } else if (ret == EMP_PROTO_ERR_LENGTH) {
                sendError(frame.seq, frame.cmd, EMP_PROTO_ERROR_LENGTH);
            }
        }
    }
}

bool EmployeeManagementChannel::popRequest(EmployeeRequest *request)
{
    if (!request || pending_.empty()) return false;
    *request = pending_.front();
    pending_.pop_front();
    return true;
}

bool EmployeeManagementChannel::enqueueRequest(const EmployeeRequest &request,
                                               uint8_t orig_cmd)
{
    if (pending_.size() >= kMaxPendingRequests) {
        printf("[EMP-PROTO] request queue full, drop cmd=0x%02X(%s)\n",
               orig_cmd, cmd_name(orig_cmd));
        if (orig_cmd == EMP_PROTO_CMD_CHECKIN_REQ) {
            sendCheckinFail(request.seq, EMP_PROTO_CHECKIN_FAIL_BUSY);
        } else {
            sendError(request.seq, orig_cmd, EMP_PROTO_ERROR_TIMEOUT);
        }
        return false;
    }

    pending_.push_back(request);
    return true;
}

void EmployeeManagementChannel::handleFrame(const EmployeeProtocolFrame &frame)
{
    EmployeeRequest request;

    if (frame.cmd == EMP_PROTO_CMD_ENROLL_REQ) {
        if (!employee_protocol_parse_enroll_req(
                frame, &request.employee_id, &request.name) ||
            request.employee_id == 0u || request.name.empty()) {
            sendError(frame.seq, frame.cmd, EMP_PROTO_ERROR_LENGTH);
            return;
        }
        request.type = EmployeeRequestType::ENROLL;
        request.seq = frame.seq;
        request.name = sanitize_name(request.name);
        enqueueRequest(request, frame.cmd);
        return;
    }

    if (frame.cmd == EMP_PROTO_CMD_CHECKIN_REQ) {
        if (!employee_protocol_parse_checkin_req(frame)) {
            sendError(frame.seq, frame.cmd, EMP_PROTO_ERROR_LENGTH);
            return;
        }
        request.type = EmployeeRequestType::CHECKIN;
        request.seq = frame.seq;
        enqueueRequest(request, frame.cmd);
        return;
    }

    if (frame.cmd == EMP_PROTO_CMD_CHECKOUT_REQ) {
        if (!employee_protocol_parse_checkout_req(frame, &request.employee_id) ||
            request.employee_id == 0u) {
            sendError(frame.seq, frame.cmd, EMP_PROTO_ERROR_LENGTH);
            return;
        }
        request.type = EmployeeRequestType::CHECKOUT;
        request.seq = frame.seq;
        enqueueRequest(request, frame.cmd);
        return;
    }

    if (frame.cmd == EMP_PROTO_CMD_DELETE_EMPLOYEE_REQ) {
        if (!employee_protocol_parse_delete_employee_req(frame, &request.employee_id) ||
            request.employee_id == 0u) {
            sendError(frame.seq, frame.cmd, EMP_PROTO_ERROR_LENGTH);
            return;
        }
        request.type = EmployeeRequestType::DELETE_EMPLOYEE;
        request.seq = frame.seq;
        enqueueRequest(request, frame.cmd);
        return;
    }

    sendError(frame.seq, frame.cmd, EMP_PROTO_ERROR_UNKNOWN_CMD);
}

int EmployeeManagementChannel::writeAll(const uint8_t *data, uint16_t len)
{
    if (fd_ < 0 || !data) return -1;

    uint16_t sent = 0;
    while (sent < len) {
        ssize_t written = write(fd_, data + sent, len - sent);
        if (written > 0) {
            sent = (uint16_t)(sent + written);
            continue;
        }

        if (written < 0 && (errno == EAGAIN || errno == EWOULDBLOCK)) {
            struct pollfd pfd;
            memset(&pfd, 0, sizeof(pfd));
            pfd.fd = fd_;
            pfd.events = POLLOUT;
            int poll_ret = poll(&pfd, 1, 100);
            if (poll_ret > 0 && (pfd.revents & POLLOUT)) continue;
        }

        printf("[EMP-PROTO] write failed: %s\n", strerror(errno));
        return -1;
    }

    tcdrain(fd_);
    return (int)sent;
}

int EmployeeManagementChannel::sendPackedFrame(int frame_len,
                                               const uint8_t *data,
                                               const char *tag)
{
    if (frame_len < 0 || !data) return frame_len;
    int ret = writeAll(data, (uint16_t)frame_len);
    if (protocol_log_enabled()) {
        printf("[EMP-PROTO] TX %s ret=%d len=%d\n",
               tag ? tag : "FRAME", ret, frame_len);
    }
    return ret;
}

int EmployeeManagementChannel::sendEnrollResult(uint8_t seq,
                                                uint32_t employee_id,
                                                uint8_t status)
{
    uint8_t frame[kMaxFrameLength];
    int len = employee_protocol_pack_enroll_rsp(
        seq, employee_id, status, frame, sizeof(frame));
    int ret = sendPackedFrame(len, frame, "ENROLL_RSP");
    if (status != EMP_PROTO_ENROLL_STATUS_OK || protocol_log_enabled()) {
        printf("[EMP-PROTO] ENROLL_RSP employee=%u status=0x%02X %s\n",
               employee_id, status, enroll_status_name(status));
    }
    return ret;
}

int EmployeeManagementChannel::sendCheckinOk(uint8_t seq,
                                             uint32_t employee_id,
                                             uint32_t checkin_time)
{
    uint8_t frame[kMaxFrameLength];
    int len = employee_protocol_pack_checkin_ok(
        seq, employee_id, checkin_time, frame, sizeof(frame));
    int ret = sendPackedFrame(len, frame, "CHECKIN_OK");
    if (protocol_log_enabled()) {
        char checkin_text[40];
        format_local_time(checkin_time, checkin_text, sizeof(checkin_text));
        printf("[EMP-PROTO] CHECKIN_OK employee=%u checkin_time=%u local=%s\n",
               employee_id, checkin_time, checkin_text);
    }
    return ret;
}

int EmployeeManagementChannel::sendCheckinFail(uint8_t seq, uint8_t reason)
{
    uint8_t frame[kMaxFrameLength];
    int len = employee_protocol_pack_checkin_fail(seq, reason, frame, sizeof(frame));
    return sendPackedFrame(len, frame, "CHECKIN_FAIL");
}

int EmployeeManagementChannel::sendCheckoutResult(uint8_t seq,
                                                  uint32_t employee_id,
                                                  uint32_t checkin_time,
                                                  uint32_t checkout_time,
                                                  uint32_t duration_sec)
{
    uint8_t frame[kMaxFrameLength];
    int len = employee_protocol_pack_checkout_rsp(
        seq, employee_id, checkin_time, checkout_time, duration_sec,
        frame, sizeof(frame));
    int ret = sendPackedFrame(len, frame, "CHECKOUT_RSP");
    if (protocol_log_enabled()) {
        char checkin_text[40];
        char checkout_text[40];
        format_local_time(checkin_time, checkin_text, sizeof(checkin_text));
        format_local_time(checkout_time, checkout_text, sizeof(checkout_text));
        printf("[EMP-PROTO] CHECKOUT_RSP employee=%u checkin_time=%u local=%s checkout_time=%u local=%s duration=%u\n",
               employee_id, checkin_time, checkin_text,
               checkout_time, checkout_text, duration_sec);
    }
    return ret;
}

int EmployeeManagementChannel::sendError(uint8_t seq, uint8_t orig_cmd, uint8_t err_code)
{
    uint8_t frame[kMaxFrameLength];
    int len = employee_protocol_pack_error(
        seq, orig_cmd, err_code, frame, sizeof(frame));
    printf("[EMP-PROTO] ERROR detail seq=%u orig_cmd=0x%02X(%s) err=0x%02X(%s)\n",
           seq, orig_cmd, cmd_name(orig_cmd), err_code, error_name(err_code));
    return sendPackedFrame(len, frame, "ERROR");
}
