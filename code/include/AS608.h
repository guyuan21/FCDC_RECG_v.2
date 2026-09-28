#pragma once

#include <stdint.h>

#include <string>
#include <vector>

class AS608 {
public:
    static constexpr const char *kDefaultDevice = "/dev/ttyS1";
    static constexpr int kDefaultBaudrate = 57600;
    static constexpr uint16_t kDefaultCapacity = 300;

    struct SearchResult {
        uint16_t page_id = 0;
        uint16_t match_score = 0;
    };

    struct SysParam {
        uint16_t max_templates = 0;
        uint8_t security_level = 0;
        uint32_t address = 0;
        uint8_t packet_size = 0;
        uint8_t baud_rate = 0;
    };

    static const uint8_t kOk = 0x00;
    static const uint8_t kNoFinger = 0x02;
    static const uint8_t kNotFound = 0x09;
    static const uint8_t kTimeout = 0xff;

    AS608();
    AS608(const std::string &device, int baudrate = kDefaultBaudrate);
    ~AS608();

    bool openDevice(const std::string &device = kDefaultDevice,
                    int baudrate = kDefaultBaudrate);
    void closeDevice();
    bool isOpen() const;
    const std::string &devicePath() const;
    int baudrate() const;
    uint16_t capacity() const;

    uint8_t handShake(uint32_t *module_addr = nullptr);
    uint8_t getImage();
    uint8_t genChar(uint8_t buffer_id);
    uint8_t match();
    uint8_t search(uint8_t buffer_id, uint16_t start_page, uint16_t page_num,
                   SearchResult *result);
    uint8_t highSpeedSearch(uint8_t buffer_id, uint16_t start_page, uint16_t page_num,
                            SearchResult *result);
    uint8_t regModel();
    uint8_t storeChar(uint8_t buffer_id, uint16_t page_id);
    uint8_t deleteChar(uint16_t page_id, uint16_t count);
    uint8_t empty();
    uint8_t readSysParam(SysParam *param);
    uint8_t validTemplateNum(uint16_t *valid_count);

    bool identify(SearchResult *result, uint16_t start_page, uint16_t page_num,
                  uint8_t *ensure = nullptr);
    bool enroll(uint16_t page_id, uint8_t *ensure = nullptr);
    bool remove(uint16_t page_id, uint16_t count = 1, uint8_t *ensure = nullptr);

    const char *ensureMessage(uint8_t ensure) const;

private:
    struct Packet {
        uint8_t pid = 0;
        std::vector<uint8_t> payload;
    };

    uint8_t command(uint8_t instruction,
                    const std::vector<uint8_t> &params,
                    Packet *reply,
                    int timeout_ms);
    bool writePacket(uint8_t pid, const std::vector<uint8_t> &payload);
    bool readPacket(Packet *packet, int timeout_ms);
    bool readByte(uint8_t *byte, int timeout_ms);
    bool waitNoFinger(int timeout_ms);

    int fd_ = -1;
    uint32_t address_ = 0xffffffffu;
    std::string device_path_ = kDefaultDevice;
    int baudrate_ = kDefaultBaudrate;
    uint16_t capacity_ = kDefaultCapacity;
};
