#include "AS608.h"

#include <errno.h>
#include <fcntl.h>
#include <poll.h>
#include <stdio.h>
#include <string.h>
#include <termios.h>
#include <unistd.h>

#include <algorithm>

namespace {

const uint8_t kPacketCommand = 0x01;
const uint8_t kPacketAck = 0x07;
const uint8_t kCharBuffer1 = 0x01;
const uint8_t kCharBuffer2 = 0x02;

speed_t baudToTermios(int baudrate)
{
    switch (baudrate) {
    case 9600: return B9600;
    case 19200: return B19200;
    case 38400: return B38400;
    case 57600: return B57600;
    case 115200: return B115200;
    default: return B57600;
    }
}

void pushU16(std::vector<uint8_t> &buf, uint16_t value)
{
    buf.push_back((uint8_t)(value >> 8));
    buf.push_back((uint8_t)(value & 0xff));
}

uint16_t readU16(const std::vector<uint8_t> &buf, size_t offset)
{
    if (offset + 1 >= buf.size()) return 0;
    return (uint16_t)(((uint16_t)buf[offset] << 8) | (uint16_t)buf[offset + 1]);
}

uint32_t readU32(const std::vector<uint8_t> &buf, size_t offset)
{
    if (offset + 3 >= buf.size()) return 0;
    return ((uint32_t)buf[offset] << 24) |
           ((uint32_t)buf[offset + 1] << 16) |
           ((uint32_t)buf[offset + 2] << 8) |
           (uint32_t)buf[offset + 3];
}

} // namespace

AS608::AS608() = default;

AS608::AS608(const std::string &device, int baudrate)
{
    openDevice(device, baudrate);
}

AS608::~AS608()
{
    closeDevice();
}

bool AS608::openDevice(const std::string &device, int baudrate)
{
    closeDevice();
    device_path_ = device.empty() ? kDefaultDevice : device;
    baudrate_ = baudrate > 0 ? baudrate : kDefaultBaudrate;
    capacity_ = kDefaultCapacity;

    fd_ = open(device_path_.c_str(), O_RDWR | O_NOCTTY | O_NONBLOCK);
    if (fd_ < 0) {
        printf("[AS608] open %s failed: %s\n", device_path_.c_str(), strerror(errno));
        return false;
    }

    struct termios tty;
    memset(&tty, 0, sizeof(tty));
    if (tcgetattr(fd_, &tty) != 0) {
        printf("[AS608] tcgetattr failed: %s\n", strerror(errno));
        closeDevice();
        return false;
    }

    speed_t speed = baudToTermios(baudrate_);
    cfsetospeed(&tty, speed);
    cfsetispeed(&tty, speed);
    tty.c_cflag = (tty.c_cflag & ~CSIZE) | CS8;
    tty.c_iflag &= ~(IGNBRK | IXON | IXOFF | IXANY);
    tty.c_lflag = 0;
    tty.c_oflag = 0;
    tty.c_cc[VMIN] = 0;
    tty.c_cc[VTIME] = 1;
    tty.c_cflag |= (CLOCAL | CREAD);
    tty.c_cflag &= ~(PARENB | PARODD);
    tty.c_cflag &= ~CSTOPB;
    tty.c_cflag &= ~CRTSCTS;

    if (tcsetattr(fd_, TCSANOW, &tty) != 0) {
        printf("[AS608] tcsetattr failed: %s\n", strerror(errno));
        closeDevice();
        return false;
    }

    tcflush(fd_, TCIOFLUSH);
    return true;
}

void AS608::closeDevice()
{
    if (fd_ >= 0) {
        close(fd_);
        fd_ = -1;
    }
}

bool AS608::isOpen() const
{
    return fd_ >= 0;
}

const std::string &AS608::devicePath() const
{
    return device_path_;
}

int AS608::baudrate() const
{
    return baudrate_;
}

uint16_t AS608::capacity() const
{
    return capacity_;
}

bool AS608::writePacket(uint8_t pid, const std::vector<uint8_t> &payload)
{
    if (fd_ < 0) return false;

    uint16_t length = (uint16_t)(payload.size() + 2);
    uint16_t checksum = (uint16_t)pid + length;
    std::vector<uint8_t> packet;
    packet.reserve(payload.size() + 11);

    packet.push_back(0xef);
    packet.push_back(0x01);
    packet.push_back((uint8_t)(address_ >> 24));
    packet.push_back((uint8_t)(address_ >> 16));
    packet.push_back((uint8_t)(address_ >> 8));
    packet.push_back((uint8_t)address_);
    packet.push_back(pid);
    packet.push_back((uint8_t)(length >> 8));
    packet.push_back((uint8_t)length);

    for (uint8_t byte : payload) {
        packet.push_back(byte);
        checksum = (uint16_t)(checksum + byte);
    }
    packet.push_back((uint8_t)(checksum >> 8));
    packet.push_back((uint8_t)checksum);

    size_t sent = 0;
    while (sent < packet.size()) {
        ssize_t n = write(fd_, packet.data() + sent, packet.size() - sent);
        if (n > 0) {
            sent += (size_t)n;
            continue;
        }
        if (n < 0 && errno == EINTR) {
            continue;
        }
        if (n < 0 && (errno == EAGAIN || errno == EWOULDBLOCK)) {
            struct pollfd pfd;
            memset(&pfd, 0, sizeof(pfd));
            pfd.fd = fd_;
            pfd.events = POLLOUT;
            if (poll(&pfd, 1, 100) > 0 && (pfd.revents & POLLOUT)) continue;
        }
        return false;
    }

    tcdrain(fd_);
    return true;
}

bool AS608::readByte(uint8_t *byte, int timeout_ms)
{
    if (!byte || fd_ < 0) return false;

    struct pollfd pfd;
    memset(&pfd, 0, sizeof(pfd));
    pfd.fd = fd_;
    pfd.events = POLLIN;

    int ret;
    do {
        ret = poll(&pfd, 1, timeout_ms);
    } while (ret < 0 && errno == EINTR);
    if (ret <= 0 || (pfd.revents & POLLIN) == 0) return false;

    ssize_t n = read(fd_, byte, 1);
    if (n < 0 && errno == EINTR) return readByte(byte, timeout_ms);
    return n == 1;
}

bool AS608::readPacket(Packet *packet, int timeout_ms)
{
    if (!packet || fd_ < 0) return false;

    uint8_t byte = 0;
    int matched = 0;
    const uint8_t header[2] = {0xef, 0x01};

    while (matched < 2) {
        if (!readByte(&byte, timeout_ms)) return false;
        if (byte == header[matched]) {
            matched++;
        } else {
            matched = (byte == header[0]) ? 1 : 0;
        }
    }

    uint8_t fixed[7];
    for (size_t i = 0; i < sizeof(fixed); ++i) {
        if (!readByte(&fixed[i], timeout_ms)) return false;
    }

    uint32_t addr = ((uint32_t)fixed[0] << 24) |
                    ((uint32_t)fixed[1] << 16) |
                    ((uint32_t)fixed[2] << 8) |
                    (uint32_t)fixed[3];
    (void)addr;

    uint8_t pid = fixed[4];
    uint16_t length = (uint16_t)(((uint16_t)fixed[5] << 8) | fixed[6]);
    if (length < 2 || length > 256) return false;

    std::vector<uint8_t> data(length);
    for (uint16_t i = 0; i < length; ++i) {
        if (!readByte(&data[i], timeout_ms)) return false;
    }

    uint16_t checksum = (uint16_t)pid + length;
    for (uint16_t i = 0; i + 2 < length; ++i) {
        checksum = (uint16_t)(checksum + data[i]);
    }
    uint16_t received = (uint16_t)(((uint16_t)data[length - 2] << 8) | data[length - 1]);
    if (checksum != received) return false;

    packet->pid = pid;
    packet->payload.assign(data.begin(), data.end() - 2);
    return true;
}

uint8_t AS608::command(uint8_t instruction,
                       const std::vector<uint8_t> &params,
                       Packet *reply,
                       int timeout_ms)
{
    if (fd_ < 0) return kTimeout;

    std::vector<uint8_t> payload;
    payload.reserve(params.size() + 1);
    payload.push_back(instruction);
    payload.insert(payload.end(), params.begin(), params.end());

    tcflush(fd_, TCIFLUSH);
    if (!writePacket(kPacketCommand, payload)) return kTimeout;

    Packet ack;
    if (!readPacket(&ack, timeout_ms) || ack.pid != kPacketAck || ack.payload.empty()) {
        return kTimeout;
    }

    if (reply) *reply = ack;
    return ack.payload[0];
}

uint8_t AS608::handShake(uint32_t *module_addr)
{
    Packet reply;
    uint8_t ensure = command(0x17, {}, &reply, 1000);
    if (ensure == kOk && module_addr) {
        *module_addr = address_;
    }
    return ensure;
}

uint8_t AS608::getImage()
{
    return command(0x01, {}, nullptr, 1000);
}

uint8_t AS608::genChar(uint8_t buffer_id)
{
    return command(0x02, {buffer_id}, nullptr, 1000);
}

uint8_t AS608::match()
{
    return command(0x03, {}, nullptr, 1000);
}

uint8_t AS608::search(uint8_t buffer_id, uint16_t start_page, uint16_t page_num,
                      SearchResult *result)
{
    std::vector<uint8_t> params;
    params.push_back(buffer_id);
    pushU16(params, start_page);
    pushU16(params, page_num);

    Packet reply;
    uint8_t ensure = command(0x04, params, &reply, 2000);
    if (ensure == kOk && result && reply.payload.size() >= 5) {
        result->page_id = readU16(reply.payload, 1);
        result->match_score = readU16(reply.payload, 3);
    }
    return ensure;
}

uint8_t AS608::highSpeedSearch(uint8_t buffer_id, uint16_t start_page, uint16_t page_num,
                               SearchResult *result)
{
    std::vector<uint8_t> params;
    params.push_back(buffer_id);
    pushU16(params, start_page);
    pushU16(params, page_num);

    Packet reply;
    uint8_t ensure = command(0x1b, params, &reply, 2000);
    if (ensure == kOk && result && reply.payload.size() >= 5) {
        result->page_id = readU16(reply.payload, 1);
        result->match_score = readU16(reply.payload, 3);
    }
    return ensure;
}

uint8_t AS608::regModel()
{
    return command(0x05, {}, nullptr, 1000);
}

uint8_t AS608::storeChar(uint8_t buffer_id, uint16_t page_id)
{
    std::vector<uint8_t> params;
    params.push_back(buffer_id);
    pushU16(params, page_id);
    return command(0x06, params, nullptr, 1000);
}

uint8_t AS608::deleteChar(uint16_t page_id, uint16_t count)
{
    std::vector<uint8_t> params;
    pushU16(params, page_id);
    pushU16(params, count);
    return command(0x0c, params, nullptr, 1000);
}

uint8_t AS608::empty()
{
    return command(0x0d, {}, nullptr, 2000);
}

uint8_t AS608::readSysParam(SysParam *param)
{
    Packet reply;
    uint8_t ensure = command(0x0f, {}, &reply, 1000);
    if (ensure == kOk && param && reply.payload.size() >= 17) {
        param->max_templates = readU16(reply.payload, 5);
        param->security_level = reply.payload[8];
        param->address = readU32(reply.payload, 9);
        param->packet_size = reply.payload[14];
        param->baud_rate = reply.payload[16];
        if (param->max_templates > 0) capacity_ = param->max_templates;
    }
    return ensure;
}

uint8_t AS608::validTemplateNum(uint16_t *valid_count)
{
    Packet reply;
    uint8_t ensure = command(0x1d, {}, &reply, 1000);
    if (ensure == kOk && valid_count && reply.payload.size() >= 3) {
        *valid_count = readU16(reply.payload, 1);
    }
    return ensure;
}

bool AS608::identify(SearchResult *result, uint16_t start_page, uint16_t page_num,
                     uint8_t *ensure)
{
    uint8_t code = getImage();
    if (code != kOk) {
        if (ensure) *ensure = code;
        return false;
    }

    code = genChar(kCharBuffer1);
    if (code != kOk) {
        if (ensure) *ensure = code;
        return false;
    }

    code = highSpeedSearch(kCharBuffer1, start_page, page_num, result);
    if (ensure) *ensure = code;
    return code == kOk;
}

bool AS608::waitNoFinger(int timeout_ms)
{
    const int step_ms = 100;
    int elapsed = 0;
    while (elapsed < timeout_ms) {
        uint8_t code = getImage();
        if (code == kNoFinger) return true;
        usleep(step_ms * 1000);
        elapsed += step_ms;
    }
    return false;
}

bool AS608::enroll(uint16_t page_id, uint8_t *ensure)
{
    uint8_t code = kTimeout;

    for (int i = 0; i < 80; ++i) {
        code = getImage();
        if (code == kOk) break;
        if (code != kNoFinger) {
            if (ensure) *ensure = code;
            return false;
        }
        usleep(100000);
    }
    if (code != kOk) {
        if (ensure) *ensure = code;
        return false;
    }

    code = genChar(kCharBuffer1);
    if (code != kOk) {
        if (ensure) *ensure = code;
        return false;
    }

    if (!waitNoFinger(5000)) {
        if (ensure) *ensure = kTimeout;
        return false;
    }

    for (int i = 0; i < 80; ++i) {
        code = getImage();
        if (code == kOk) break;
        if (code != kNoFinger) {
            if (ensure) *ensure = code;
            return false;
        }
        usleep(100000);
    }
    if (code != kOk) {
        if (ensure) *ensure = code;
        return false;
    }

    code = genChar(kCharBuffer2);
    if (code != kOk) {
        if (ensure) *ensure = code;
        return false;
    }

    code = regModel();
    if (code != kOk) {
        if (ensure) *ensure = code;
        return false;
    }

    code = storeChar(kCharBuffer1, page_id);
    if (ensure) *ensure = code;
    return code == kOk;
}

bool AS608::remove(uint16_t page_id, uint16_t count, uint8_t *ensure)
{
    uint8_t code = deleteChar(page_id, count);
    if (ensure) *ensure = code;
    return code == kOk;
}

const char *AS608::ensureMessage(uint8_t ensure) const
{
    switch (ensure) {
    case 0x00: return "ok";
    case 0x01: return "packet error";
    case 0x02: return "no finger";
    case 0x03: return "image failed";
    case 0x06: return "image too messy";
    case 0x07: return "feature too few";
    case 0x08: return "finger mismatch";
    case 0x09: return "not found";
    case 0x0a: return "merge failed";
    case 0x0b: return "page id invalid";
    case 0x0c: return "template read failed";
    case 0x0d: return "template upload failed";
    case 0x0e: return "data upload failed";
    case 0x0f: return "data download failed";
    case 0x10: return "template download failed";
    case 0x11: return "library upload failed";
    case 0x12: return "library download failed";
    case 0x13: return "notepad upload failed";
    case 0x14: return "delete failed";
    case 0x15: return "empty failed";
    case 0x18: return "flash write failed";
    case 0x19: return "address error";
    case 0x1a: return "password error";
    case 0xff: return "timeout";
    default: return "unknown";
    }
}
