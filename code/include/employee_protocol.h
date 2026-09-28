#ifndef EMPLOYEE_PROTOCOL_H
#define EMPLOYEE_PROTOCOL_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
#include <string>
#include <vector>
#endif

typedef struct EmployeeProtocolFrame {
    uint8_t cmd;
    uint8_t seq;
    uint16_t len;
#ifdef __cplusplus
    std::vector<uint8_t> payload;
#endif
} EmployeeProtocolFrame;

#ifdef __cplusplus
struct EmployeeProtocolParser {
    enum State {
        WAIT_HEAD1 = 0,
        WAIT_HEAD2,
        READ_CMD,
        READ_SEQ,
        READ_LEN,
        READ_PAYLOAD,
        READ_CRC_H,
        READ_CRC_L,
        READ_TAIL
    } state = WAIT_HEAD1;

    uint8_t cmd = 0;
    uint8_t seq = 0;
    uint16_t len = 0;
    uint16_t payload_index = 0;
    uint16_t crc_recv = 0;
    std::vector<uint8_t> payload;
};
#endif

enum {
    EMP_PROTO_OK = 0,
    EMP_PROTO_ERR_CHECKSUM = -1,
    EMP_PROTO_ERR_LENGTH = -2,
    EMP_PROTO_ERR_BUFFER = -3,
    EMP_PROTO_ERR_PARAM = -4,
    EMP_PROTO_ERR_UNSUPPORTED = -5
};

static constexpr uint8_t EMP_PROTO_HEAD1 = 0xAAu;
static constexpr uint8_t EMP_PROTO_HEAD2 = 0x55u;
static constexpr uint8_t EMP_PROTO_TAIL = 0xFFu;
static constexpr uint16_t EMP_PROTO_MAX_DATA_LEN = 255u;
static constexpr uint16_t EMP_PROTO_FRAME_OVERHEAD = 8u;

static constexpr uint8_t EMP_PROTO_CMD_ENROLL_REQ = 0x01u;
static constexpr uint8_t EMP_PROTO_CMD_ENROLL_RSP = 0x02u;
static constexpr uint8_t EMP_PROTO_CMD_CHECKIN_REQ = 0x10u;
static constexpr uint8_t EMP_PROTO_CMD_CHECKIN_OK = 0x11u;
static constexpr uint8_t EMP_PROTO_CMD_CHECKIN_FAIL = 0x12u;
static constexpr uint8_t EMP_PROTO_CMD_CHECKOUT_REQ = 0x20u;
static constexpr uint8_t EMP_PROTO_CMD_CHECKOUT_RSP = 0x21u;
static constexpr uint8_t EMP_PROTO_CMD_DELETE_EMPLOYEE_REQ = 0x30u;
static constexpr uint8_t EMP_PROTO_CMD_ERROR = 0xFFu;

static constexpr uint16_t EMP_PROTO_ENROLL_REQ_MIN_LEN = 5u;
static constexpr uint16_t EMP_PROTO_ENROLL_REQ_MAX_LEN = 36u;
static constexpr uint16_t EMP_PROTO_CHECKIN_REQ_LEN = 1u;
static constexpr uint16_t EMP_PROTO_ENROLL_RSP_LEN = 5u;
static constexpr uint16_t EMP_PROTO_CHECKIN_OK_LEN = 8u;
static constexpr uint16_t EMP_PROTO_CHECKIN_FAIL_LEN = 1u;
static constexpr uint16_t EMP_PROTO_CHECKOUT_REQ_LEN = 4u;
static constexpr uint16_t EMP_PROTO_CHECKOUT_RSP_LEN = 16u;
static constexpr uint16_t EMP_PROTO_DELETE_EMPLOYEE_REQ_LEN = 4u;
static constexpr uint16_t EMP_PROTO_ERROR_FRAME_DATA_LEN = 2u;

static constexpr uint8_t EMP_PROTO_ENROLL_STATUS_OK = 0x00u;
static constexpr uint8_t EMP_PROTO_ENROLL_STATUS_FACE_FAIL = 0x01u;
static constexpr uint8_t EMP_PROTO_ENROLL_STATUS_FINGER_FAIL = 0x02u;
static constexpr uint8_t EMP_PROTO_ENROLL_STATUS_EXISTS = 0x03u;
static constexpr uint8_t EMP_PROTO_ENROLL_STATUS_STORAGE = 0x04u;

static constexpr uint8_t EMP_PROTO_CHECKIN_FAIL_FACE_TIMEOUT = 0x01u;
static constexpr uint8_t EMP_PROTO_CHECKIN_FAIL_FINGER_TIMEOUT = 0x02u;
static constexpr uint8_t EMP_PROTO_CHECKIN_FAIL_MISMATCH = 0x03u;
static constexpr uint8_t EMP_PROTO_CHECKIN_FAIL_UNREGISTERED = 0x04u;
static constexpr uint8_t EMP_PROTO_CHECKIN_FAIL_BUSY = 0x05u;
static constexpr uint8_t EMP_PROTO_CHECKIN_FAIL_ALREADY = 0x06u;

static constexpr uint8_t EMP_PROTO_ERROR_FRAME = 0xFFu;
static constexpr uint8_t EMP_PROTO_ERROR_HEAD = 0x01u;
static constexpr uint8_t EMP_PROTO_ERROR_CRC = 0x02u;
static constexpr uint8_t EMP_PROTO_ERROR_UNKNOWN_CMD = 0x03u;
static constexpr uint8_t EMP_PROTO_ERROR_LENGTH = 0x04u;
static constexpr uint8_t EMP_PROTO_ERROR_EMPLOYEE_NOT_FOUND = 0x05u;
static constexpr uint8_t EMP_PROTO_ERROR_NOT_CHECKED_IN = 0x06u;
static constexpr uint8_t EMP_PROTO_ERROR_TIMEOUT = 0x07u;

#ifdef __cplusplus
void employee_protocol_parser_init(EmployeeProtocolParser *parser);
uint16_t employee_protocol_crc16_modbus(const uint8_t *data, uint16_t len);
int employee_protocol_pack(uint8_t cmd,
                           uint8_t seq,
                           const uint8_t *payload,
                           uint16_t len,
                           uint8_t *out_buf,
                           uint16_t out_size);
int employee_protocol_pack_enroll_rsp(uint8_t seq,
                                      uint32_t employee_id,
                                      uint8_t status,
                                      uint8_t *out_buf,
                                      uint16_t out_size);
int employee_protocol_pack_checkin_ok(uint8_t seq,
                                      uint32_t employee_id,
                                      uint32_t checkin_time,
                                      uint8_t *out_buf,
                                      uint16_t out_size);
int employee_protocol_pack_checkin_fail(uint8_t seq,
                                        uint8_t reason,
                                        uint8_t *out_buf,
                                        uint16_t out_size);
int employee_protocol_pack_checkout_rsp(uint8_t seq,
                                        uint32_t employee_id,
                                        uint32_t checkin_time,
                                        uint32_t checkout_time,
                                        uint32_t duration_sec,
                                        uint8_t *out_buf,
                                        uint16_t out_size);
int employee_protocol_pack_error(uint8_t seq,
                                 uint8_t orig_cmd,
                                 uint8_t err_code,
                                 uint8_t *out_buf,
                                 uint16_t out_size);
int employee_protocol_parse_byte_ex(EmployeeProtocolParser *parser,
                                    uint8_t byte,
                                    EmployeeProtocolFrame *frame);

bool employee_protocol_parse_enroll_req(const EmployeeProtocolFrame &frame,
                                        uint32_t *employee_id,
                                        std::string *name);
bool employee_protocol_parse_checkin_req(const EmployeeProtocolFrame &frame);
bool employee_protocol_parse_checkout_req(const EmployeeProtocolFrame &frame,
                                          uint32_t *employee_id);
bool employee_protocol_parse_delete_employee_req(const EmployeeProtocolFrame &frame,
                                                 uint32_t *employee_id);

void employee_protocol_write_u16_be(uint8_t *buf, uint16_t value);
void employee_protocol_write_u32_be(uint8_t *buf, uint32_t value);
uint16_t employee_protocol_read_u16_be(const uint8_t *buf);
uint32_t employee_protocol_read_u32_be(const uint8_t *buf);
#endif

#endif /* EMPLOYEE_PROTOCOL_H */
