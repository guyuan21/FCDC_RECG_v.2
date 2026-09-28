#include "uart_control.h"

#include <ctype.h>
#include <errno.h>
#include <fcntl.h>
#include <poll.h>
#include <stdio.h>
#include <string.h>
#include <termios.h>
#include <unistd.h>

namespace {

constexpr char kControlDev[] = "/dev/ttyS2";
constexpr speed_t kUartBaudrate = B115200;

FingerprintManageRequest g_pending_fingerprint_request;

std::string trim_uart_command(const std::string &text)
{
    size_t start = 0;
    while (start < text.size() && isspace((unsigned char)text[start])) start++;

    size_t end = text.size();
    while (end > start && isspace((unsigned char)text[end - 1])) end--;

    return text.substr(start, end - start);
}

std::string normalize_uart_command(const std::string &text)
{
    std::string normalized = trim_uart_command(text);
    for (size_t i = 0; i < normalized.size(); ++i) {
        normalized[i] = (char)toupper((unsigned char)normalized[i]);
    }
    return normalized;
}

bool parse_uint16_text(const std::string &text, uint16_t *value)
{
    if (!value || text.empty()) return false;

    char *end = nullptr;
    errno = 0;
    unsigned long parsed = strtoul(text.c_str(), &end, 10);
    if (errno != 0 || !end || *end != '\0' || parsed > 65535ul) return false;

    *value = (uint16_t)parsed;
    return true;
}

bool parse_uint16_field_after(const std::string &text,
                              const std::string &key,
                              uint16_t *value)
{
    if (!value || key.empty()) return false;

    size_t pos = text.find(key);
    if (pos == std::string::npos) return false;
    pos += key.size();

    size_t end = pos;
    while (end < text.size() && isdigit((unsigned char)text[end])) end++;
    if (end == pos) return false;

    return parse_uint16_text(text.substr(pos, end - pos), value);
}

bool parse_fingerprint_command(const std::string &cmd,
                               FingerprintManageRequest *request)
{
    if (!request) return false;

    if (cmd.find("CMD|ACTION=FP_ENROLL") != std::string::npos ||
        cmd.find("CMD|ACTION=FP_ADD") != std::string::npos ||
        cmd.find("CMD|ACTION=FINGER_ENROLL") != std::string::npos) {
        uint16_t id = 0;
        if (!parse_uint16_field_after(cmd, "ID=", &id) &&
            !parse_uint16_field_after(cmd, "WORKER_ID=", &id)) {
            return false;
        }
        request->type = FingerprintManageType::ENROLL;
        request->worker_id = (int)id;
        request->page_id = 0;
        request->count = 1;
        return true;
    }

    if (cmd.find("CMD|ACTION=FP_DELETE") != std::string::npos ||
        cmd.find("CMD|ACTION=FP_DEL") != std::string::npos ||
        cmd.find("CMD|ACTION=FINGER_DELETE") != std::string::npos) {
        uint16_t id = 0;
        if (!parse_uint16_field_after(cmd, "ID=", &id) &&
            !parse_uint16_field_after(cmd, "WORKER_ID=", &id)) {
            return false;
        }
        request->type = FingerprintManageType::DELETE_ONE;
        request->worker_id = (int)id;
        request->page_id = 0;
        request->count = 1;
        return true;
    }

    const char *enroll_prefixes[] = {
        "FP_ENROLL:", "FP_ADD:", "FINGER_ENROLL:", "FINGER_ADD:", "ENROLL "
    };
    for (const char *prefix : enroll_prefixes) {
        size_t len = strlen(prefix);
        if (cmd.compare(0, len, prefix) == 0) {
            uint16_t id = 0;
            if (!parse_uint16_text(trim_uart_command(cmd.substr(len)), &id)) return false;
            request->type = FingerprintManageType::ENROLL;
            request->worker_id = (int)id;
            request->page_id = 0;
            request->count = 1;
            return true;
        }
    }

    const char *delete_prefixes[] = {
        "FP_DELETE:", "FP_DEL:", "FINGER_DELETE:", "DELETE "
    };
    for (const char *prefix : delete_prefixes) {
        size_t len = strlen(prefix);
        if (cmd.compare(0, len, prefix) == 0) {
            uint16_t id = 0;
            if (!parse_uint16_text(trim_uart_command(cmd.substr(len)), &id)) return false;
            request->type = FingerprintManageType::DELETE_ONE;
            request->worker_id = (int)id;
            request->page_id = 0;
            request->count = 1;
            return true;
        }
    }

    return false;
}

void append_uart_command_byte(std::string &rx_buffer, unsigned char byte)
{
    if (byte == '\r' || byte == '\n') {
        std::string line = normalize_uart_command(rx_buffer);
        rx_buffer.clear();
        if (line.empty()) return;

        FingerprintManageRequest fp_request;
        if (parse_fingerprint_command(line, &fp_request)) {
            uart_control_queue_fingerprint_request(fp_request);
            printf("[AS608] queued command: %s\n", line.c_str());
        }
        return;
    }

    if (isprint(byte) || byte == '\t') {
        rx_buffer.push_back((char)byte);
        if (rx_buffer.size() > 256) {
            rx_buffer.erase(0, rx_buffer.size() - 128);
        }
    }
}

} // namespace

int uart_control_open_uart_port(const char *dev_path)
{
    int fd = open(dev_path, O_RDWR | O_NOCTTY | O_NONBLOCK);
    if (fd < 0) {
        printf("[UART] open %s failed: %s\n", dev_path, strerror(errno));
        return -1;
    }

    struct termios tty;
    memset(&tty, 0, sizeof(tty));
    if (tcgetattr(fd, &tty) != 0) {
        printf("[UART] tcgetattr failed: %s\n", strerror(errno));
        close(fd);
        return -1;
    }

    cfsetospeed(&tty, kUartBaudrate);
    cfsetispeed(&tty, kUartBaudrate);
    tty.c_cflag = (tty.c_cflag & ~CSIZE) | CS8;
    tty.c_iflag &= ~(IGNBRK | IXON | IXOFF | IXANY);
    tty.c_lflag = 0;
    tty.c_oflag = 0;
    tty.c_cc[VMIN]  = 0;
    tty.c_cc[VTIME] = 1;
    tty.c_cflag |= (CLOCAL | CREAD);
    tty.c_cflag &= ~(PARENB | PARODD);
    tty.c_cflag &= ~CSTOPB;
    tty.c_cflag &= ~CRTSCTS;

    if (tcsetattr(fd, TCSANOW, &tty) != 0) {
        printf("[UART] tcsetattr failed: %s\n", strerror(errno));
        close(fd);
        return -1;
    }

    tcflush(fd, TCIOFLUSH);
    return fd;
}

int uart_control_open_control_input(const char *dev_path, bool *using_stdin_fallback)
{
    if (using_stdin_fallback) *using_stdin_fallback = false;

    if (strcmp(dev_path, kControlDev) == 0) {
        if (isatty(STDIN_FILENO)) {
            if (using_stdin_fallback) *using_stdin_fallback = true;
            return STDIN_FILENO;
        }
    }

    return uart_control_open_uart_port(dev_path);
}

std::string uart_control_build_uart_status_text(bool control_ready, bool upload_ready)
{
    char buf[64];
    snprintf(buf, sizeof(buf),
             "CTRL:%s  TX:%s",
             control_ready ? "UART2 OK" : "UART2 OFF",
             upload_ready ? "UART4 OK" : "UART4 OFF");
    return std::string(buf);
}

void uart_control_queue_fingerprint_request(const FingerprintManageRequest &request)
{
    g_pending_fingerprint_request = request;
}

bool uart_control_consume_fingerprint_request(FingerprintManageRequest *request)
{
    if (!request || g_pending_fingerprint_request.type == FingerprintManageType::NONE) {
        return false;
    }

    *request = g_pending_fingerprint_request;
    g_pending_fingerprint_request = FingerprintManageRequest();
    return true;
}

void uart_control_poll_input(int fd, std::string &rx_buffer)
{
    if (fd < 0) return;

    char temp[128];

    while (true) {
        struct pollfd pfd;
        memset(&pfd, 0, sizeof(pfd));
        pfd.fd = fd;
        pfd.events = POLLIN;

        int poll_ret = poll(&pfd, 1, 0);
        if (poll_ret <= 0) {
            if (poll_ret < 0 && errno != EINTR) {
                printf("[UART] poll failed: %s\n", strerror(errno));
            }
            break;
        }
        if ((pfd.revents & (POLLERR | POLLHUP | POLLNVAL)) != 0) {
            printf("[UART] control input disconnected or invalid, revents=0x%x\n", pfd.revents);
            break;
        }
        if ((pfd.revents & POLLIN) == 0) break;

        ssize_t n = read(fd, temp, sizeof(temp));
        if (n <= 0) {
            if (n < 0 && errno != EAGAIN && errno != EWOULDBLOCK) {
                printf("[UART] read failed: %s\n", strerror(errno));
            }
            break;
        }

        for (ssize_t i = 0; i < n; ++i) {
            append_uart_command_byte(rx_buffer, (unsigned char)temp[i]);
        }
    }
}
