#include "employee_protocol.h"

#ifdef __cplusplus
#include <vector>
#endif

#include <string.h>

#ifdef __cplusplus
namespace {

static void employee_protocol_reset_parser(EmployeeProtocolParser *parser)
{
    if (!parser) return;
    parser->state = EmployeeProtocolParser::WAIT_HEAD1;
    parser->cmd = 0;
    parser->seq = 0;
    parser->len = 0;
    parser->payload_index = 0;
    parser->crc_recv = 0;
    parser->payload.clear();
}

static uint16_t employee_protocol_calc_crc(const EmployeeProtocolFrame &frame)
{
    std::vector<uint8_t> buffer;
    buffer.reserve((size_t)3u + frame.len);
    buffer.push_back(frame.cmd);
    buffer.push_back(frame.seq);
    buffer.push_back((uint8_t)(frame.len & 0xFFu));
    buffer.insert(buffer.end(), frame.payload.begin(), frame.payload.end());
    if (buffer.empty()) return 0u;
    return employee_protocol_crc16_modbus(buffer.data(), (uint16_t)buffer.size());
}

static int employee_protocol_pack_raw(uint8_t cmd,
                                      uint8_t seq,
                                      const uint8_t *payload,
                                      uint16_t len,
                                      uint8_t *out_buf,
                                      uint16_t out_size)
{
    uint16_t frame_len;
    uint16_t crc;

    if (!out_buf) return EMP_PROTO_ERR_PARAM;
    if (len > EMP_PROTO_MAX_DATA_LEN) return EMP_PROTO_ERR_LENGTH;
    if (len > 0 && !payload) return EMP_PROTO_ERR_PARAM;

    frame_len = (uint16_t)(EMP_PROTO_FRAME_OVERHEAD + len);
    if (out_size < frame_len) return EMP_PROTO_ERR_BUFFER;

    out_buf[0] = EMP_PROTO_HEAD1;
    out_buf[1] = EMP_PROTO_HEAD2;
    out_buf[2] = cmd;
    out_buf[3] = seq;
    out_buf[4] = (uint8_t)len;
    if (len > 0) memcpy(&out_buf[5], payload, len);

    crc = employee_protocol_crc16_modbus(&out_buf[2], (uint16_t)(3u + len));
    employee_protocol_write_u16_be(&out_buf[5 + len], crc);
    out_buf[7 + len] = EMP_PROTO_TAIL;
    return (int)frame_len;
}

} // namespace
#endif

void employee_protocol_parser_init(EmployeeProtocolParser *parser)
{
#ifdef __cplusplus
    employee_protocol_reset_parser(parser);
#else
    (void)parser;
#endif
}

uint16_t employee_protocol_crc16_modbus(const uint8_t *data, uint16_t len)
{
    uint16_t crc = 0xFFFFu;
    if (!data) return crc;

    for (uint16_t i = 0; i < len; ++i) {
        crc ^= data[i];
        for (int j = 0; j < 8; ++j) {
            if (crc & 0x0001u) {
                crc = (uint16_t)((crc >> 1) ^ 0xA001u);
            } else {
                crc >>= 1;
            }
        }
    }
    return crc;
}

void employee_protocol_write_u16_be(uint8_t *buf, uint16_t value)
{
    if (!buf) return;
    buf[0] = (uint8_t)((value >> 8) & 0xFFu);
    buf[1] = (uint8_t)(value & 0xFFu);
}

void employee_protocol_write_u32_be(uint8_t *buf, uint32_t value)
{
    if (!buf) return;
    buf[0] = (uint8_t)((value >> 24) & 0xFFu);
    buf[1] = (uint8_t)((value >> 16) & 0xFFu);
    buf[2] = (uint8_t)((value >> 8) & 0xFFu);
    buf[3] = (uint8_t)(value & 0xFFu);
}

uint16_t employee_protocol_read_u16_be(const uint8_t *buf)
{
    if (!buf) return 0u;
    return (uint16_t)(((uint16_t)buf[0] << 8) | (uint16_t)buf[1]);
}

uint32_t employee_protocol_read_u32_be(const uint8_t *buf)
{
    if (!buf) return 0u;
    return ((uint32_t)buf[0] << 24) |
           ((uint32_t)buf[1] << 16) |
           ((uint32_t)buf[2] << 8) |
           ((uint32_t)buf[3]);
}

int employee_protocol_pack(uint8_t cmd,
                           uint8_t seq,
                           const uint8_t *payload,
                           uint16_t len,
                           uint8_t *out_buf,
                           uint16_t out_size)
{
    uint16_t expected_len = 0u;

    if (cmd == EMP_PROTO_CMD_CHECKIN_REQ) expected_len = EMP_PROTO_CHECKIN_REQ_LEN;
    else if (cmd == EMP_PROTO_CMD_ENROLL_RSP) expected_len = EMP_PROTO_ENROLL_RSP_LEN;
    else if (cmd == EMP_PROTO_CMD_CHECKIN_OK) expected_len = EMP_PROTO_CHECKIN_OK_LEN;
    else if (cmd == EMP_PROTO_CMD_CHECKIN_FAIL) expected_len = EMP_PROTO_CHECKIN_FAIL_LEN;
    else if (cmd == EMP_PROTO_CMD_CHECKOUT_REQ) expected_len = EMP_PROTO_CHECKOUT_REQ_LEN;
    else if (cmd == EMP_PROTO_CMD_CHECKOUT_RSP) expected_len = EMP_PROTO_CHECKOUT_RSP_LEN;
    else if (cmd == EMP_PROTO_CMD_DELETE_EMPLOYEE_REQ) expected_len = EMP_PROTO_DELETE_EMPLOYEE_REQ_LEN;
    else if (cmd == EMP_PROTO_CMD_ERROR) expected_len = EMP_PROTO_ERROR_FRAME_DATA_LEN;

    if (expected_len != 0u && len != expected_len) return EMP_PROTO_ERR_LENGTH;
    return employee_protocol_pack_raw(cmd, seq, payload, len, out_buf, out_size);
}

int employee_protocol_pack_enroll_rsp(uint8_t seq,
                                      uint32_t employee_id,
                                      uint8_t status,
                                      uint8_t *out_buf,
                                      uint16_t out_size)
{
    uint8_t payload[5];
    employee_protocol_write_u32_be(payload, employee_id);
    payload[4] = status;
    return employee_protocol_pack_raw(EMP_PROTO_CMD_ENROLL_RSP, seq, payload, sizeof(payload), out_buf, out_size);
}

int employee_protocol_pack_checkin_ok(uint8_t seq,
                                      uint32_t employee_id,
                                      uint32_t checkin_time,
                                      uint8_t *out_buf,
                                      uint16_t out_size)
{
    uint8_t payload[8];
    employee_protocol_write_u32_be(&payload[0], employee_id);
    employee_protocol_write_u32_be(&payload[4], checkin_time);
    return employee_protocol_pack_raw(EMP_PROTO_CMD_CHECKIN_OK, seq, payload, sizeof(payload), out_buf, out_size);
}

int employee_protocol_pack_checkin_fail(uint8_t seq,
                                        uint8_t reason,
                                        uint8_t *out_buf,
                                        uint16_t out_size)
{
    uint8_t payload[1] = {reason};
    return employee_protocol_pack_raw(EMP_PROTO_CMD_CHECKIN_FAIL, seq, payload, sizeof(payload), out_buf, out_size);
}

int employee_protocol_pack_checkout_rsp(uint8_t seq,
                                        uint32_t employee_id,
                                        uint32_t checkin_time,
                                        uint32_t checkout_time,
                                        uint32_t duration_sec,
                                        uint8_t *out_buf,
                                        uint16_t out_size)
{
    uint8_t payload[16];
    employee_protocol_write_u32_be(&payload[0], employee_id);
    employee_protocol_write_u32_be(&payload[4], checkin_time);
    employee_protocol_write_u32_be(&payload[8], checkout_time);
    employee_protocol_write_u32_be(&payload[12], duration_sec);
    return employee_protocol_pack_raw(EMP_PROTO_CMD_CHECKOUT_RSP, seq, payload, sizeof(payload), out_buf, out_size);
}

int employee_protocol_pack_error(uint8_t seq,
                                 uint8_t orig_cmd,
                                 uint8_t err_code,
                                 uint8_t *out_buf,
                                 uint16_t out_size)
{
    uint8_t payload[2] = {orig_cmd, err_code};
    return employee_protocol_pack_raw(EMP_PROTO_CMD_ERROR, seq, payload, sizeof(payload), out_buf, out_size);
}

int employee_protocol_parse_byte_ex(EmployeeProtocolParser *parser,
                                    uint8_t byte,
                                    EmployeeProtocolFrame *frame)
{
    uint16_t crc;

    if (!parser || !frame) return EMP_PROTO_ERR_PARAM;

    switch (parser->state) {
    case EmployeeProtocolParser::WAIT_HEAD1:
        if (byte == EMP_PROTO_HEAD1) parser->state = EmployeeProtocolParser::WAIT_HEAD2;
        return 0;
    case EmployeeProtocolParser::WAIT_HEAD2:
        if (byte == EMP_PROTO_HEAD2) {
            parser->state = EmployeeProtocolParser::READ_CMD;
        } else if (byte == EMP_PROTO_HEAD1) {
            parser->state = EmployeeProtocolParser::WAIT_HEAD2;
        } else {
            parser->state = EmployeeProtocolParser::WAIT_HEAD1;
        }
        return 0;
    case EmployeeProtocolParser::READ_CMD:
        parser->cmd = byte;
        parser->state = EmployeeProtocolParser::READ_SEQ;
        return 0;
    case EmployeeProtocolParser::READ_SEQ:
        parser->seq = byte;
        parser->state = EmployeeProtocolParser::READ_LEN;
        return 0;
    case EmployeeProtocolParser::READ_LEN:
        parser->len = byte;
        if (parser->len > EMP_PROTO_MAX_DATA_LEN) {
            frame->cmd = parser->cmd;
            frame->seq = parser->seq;
            frame->len = parser->len;
            frame->payload.clear();
            employee_protocol_reset_parser(parser);
            return EMP_PROTO_ERR_LENGTH;
        }
        parser->payload_index = 0;
        parser->payload.assign(parser->len, 0u);
        parser->state = (parser->len == 0u) ? EmployeeProtocolParser::READ_CRC_H
                                            : EmployeeProtocolParser::READ_PAYLOAD;
        return 0;
    case EmployeeProtocolParser::READ_PAYLOAD:
        if (parser->payload_index < parser->payload.size()) {
            parser->payload[parser->payload_index++] = byte;
        }
        if (parser->payload_index >= parser->payload.size()) {
            parser->state = EmployeeProtocolParser::READ_CRC_H;
        }
        return 0;
    case EmployeeProtocolParser::READ_CRC_H:
        parser->crc_recv = (uint16_t)byte << 8;
        parser->state = EmployeeProtocolParser::READ_CRC_L;
        return 0;
    case EmployeeProtocolParser::READ_CRC_L:
        parser->crc_recv |= byte;
        parser->state = EmployeeProtocolParser::READ_TAIL;
        return 0;
    case EmployeeProtocolParser::READ_TAIL:
        frame->cmd = parser->cmd;
        frame->seq = parser->seq;
        frame->len = parser->len;
        frame->payload = parser->payload;
        if (byte != EMP_PROTO_TAIL) {
            employee_protocol_reset_parser(parser);
            return EMP_PROTO_ERR_LENGTH;
        }
        crc = employee_protocol_calc_crc(*frame);
        if (crc != parser->crc_recv) {
            employee_protocol_reset_parser(parser);
            return EMP_PROTO_ERR_CHECKSUM;
        }
        employee_protocol_reset_parser(parser);
        return 1;
    default:
        employee_protocol_reset_parser(parser);
        return 0;
    }
}

bool employee_protocol_parse_enroll_req(const EmployeeProtocolFrame &frame,
                                        uint32_t *employee_id,
                                        std::string *name)
{
    if (frame.cmd != EMP_PROTO_CMD_ENROLL_REQ) return false;
    if (frame.len < EMP_PROTO_ENROLL_REQ_MIN_LEN ||
        frame.len > EMP_PROTO_ENROLL_REQ_MAX_LEN) return false;
    if (frame.payload.size() != frame.len) return false;
    if (employee_id) *employee_id = employee_protocol_read_u32_be(&frame.payload[0]);
    if (name) {
        name->assign((const char *)&frame.payload[4], (size_t)frame.len - 4u);
    }
    return true;
}

bool employee_protocol_parse_checkin_req(const EmployeeProtocolFrame &frame)
{
    return frame.cmd == EMP_PROTO_CMD_CHECKIN_REQ &&
           frame.len == EMP_PROTO_CHECKIN_REQ_LEN &&
           frame.payload.size() == frame.len &&
           frame.payload[0] == 0x00u;
}

bool employee_protocol_parse_checkout_req(const EmployeeProtocolFrame &frame,
                                          uint32_t *employee_id)
{
    if (frame.cmd != EMP_PROTO_CMD_CHECKOUT_REQ) return false;
    if (frame.len != EMP_PROTO_CHECKOUT_REQ_LEN || frame.payload.size() != frame.len) return false;
    if (employee_id) *employee_id = employee_protocol_read_u32_be(&frame.payload[0]);
    return true;
}

bool employee_protocol_parse_delete_employee_req(const EmployeeProtocolFrame &frame,
                                                 uint32_t *employee_id)
{
    if (frame.cmd != EMP_PROTO_CMD_DELETE_EMPLOYEE_REQ) return false;
    if (frame.len != EMP_PROTO_DELETE_EMPLOYEE_REQ_LEN || frame.payload.size() != frame.len) return false;
    if (employee_id) *employee_id = employee_protocol_read_u32_be(&frame.payload[0]);
    return true;
}
