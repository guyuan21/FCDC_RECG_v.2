#ifndef UART_CONTROL_H
#define UART_CONTROL_H

#include <stdint.h>

#include <string>

enum class FingerprintManageType {
    NONE,
    ENROLL,
    DELETE_ONE,
};

struct FingerprintManageRequest {
    FingerprintManageType type = FingerprintManageType::NONE;
    int worker_id = 0;
    uint16_t page_id = 0;
    uint16_t count = 1;
};

int uart_control_open_uart_port(const char *dev_path);
int uart_control_open_control_input(const char *dev_path, bool *using_stdin_fallback);
std::string uart_control_build_uart_status_text(bool control_ready, bool upload_ready);

void uart_control_queue_fingerprint_request(const FingerprintManageRequest &request);
bool uart_control_consume_fingerprint_request(FingerprintManageRequest *request);

void uart_control_poll_input(int fd, std::string &rx_buffer);

#endif /* UART_CONTROL_H */
