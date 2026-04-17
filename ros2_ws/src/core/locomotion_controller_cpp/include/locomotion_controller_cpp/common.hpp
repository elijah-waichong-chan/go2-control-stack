#pragma once

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <vector>

#include "hq_pcot_msgs/msg/loop_status.hpp"
#include "unitree_go/msg/low_cmd.hpp"

namespace locomotion_controller_cpp
{

constexpr std::uint8_t kLowLevel = 0xFF;
constexpr std::uint8_t kHead0 = 0xFE;
constexpr std::uint8_t kHead1 = 0xEF;
constexpr float kPosStop = 2.146e9F;
constexpr float kVelStop = 16000.0F;

template <typename T>
inline T Clamp(const T value, const T lo, const T hi)
{
  return std::max(lo, std::min(hi, value));
}

inline hq_pcot_msgs::msg::LoopStatus MakeLoopStatus(
  const int status_code,
  const float avg_loop_ms = -1.0F,
  const float p99_loop_ms = -1.0F,
  const float max_loop_ms = -1.0F,
  const float budget_ms = -1.0F,
  const int deadline_miss_count = -1,
  const int sample_count = -1)
{
  hq_pcot_msgs::msg::LoopStatus msg;
  msg.status = status_code;
  msg.avg_loop_ms = avg_loop_ms;
  msg.p99_loop_ms = p99_loop_ms;
  msg.max_loop_ms = max_loop_ms;
  msg.budget_ms = budget_ms;
  msg.deadline_miss_count = deadline_miss_count;
  msg.sample_count = sample_count;
  return msg;
}

struct MotorCmdRaw
{
  std::uint8_t mode;
  float q;
  float dq;
  float tau;
  float kp;
  float kd;
  std::uint32_t reserve[3];
};

struct BmsCmdRaw
{
  std::uint8_t off;
  std::uint8_t reserve[3];
};

struct LowCmdRaw
{
  std::uint8_t head[2];
  std::uint8_t level_flag;
  std::uint8_t frame_reserve;
  std::uint32_t sn[2];
  std::uint32_t version[2];
  std::uint16_t bandwidth;
  MotorCmdRaw motor_cmd[20];
  BmsCmdRaw bms;
  std::uint8_t wireless_remote[40];
  std::uint8_t led[12];
  std::uint8_t fan[2];
  std::uint8_t gpio;
  std::uint32_t reserve;
  std::uint32_t crc;
};

inline std::uint32_t Crc32Core(const std::vector<std::uint32_t>& words)
{
  constexpr std::uint32_t poly = 0x04C11DB7U;
  std::uint32_t crc32 = 0xFFFFFFFFU;

  for (const std::uint32_t data_word : words)
  {
    std::uint32_t xbit = 1U << 31;
    for (int i = 0; i < 32; ++i)
    {
      if ((crc32 & 0x80000000U) != 0U)
      {
        crc32 = ((crc32 << 1U) & 0xFFFFFFFFU) ^ poly;
      }
      else
      {
        crc32 = (crc32 << 1U) & 0xFFFFFFFFU;
      }
      if ((data_word & xbit) != 0U)
      {
        crc32 ^= poly;
      }
      xbit >>= 1U;
    }
  }

  return crc32 & 0xFFFFFFFFU;
}

inline std::uint32_t GetCrc(const unitree_go::msg::LowCmd& msg)
{
  LowCmdRaw raw{};
  raw.head[0] = static_cast<std::uint8_t>(msg.head[0]);
  raw.head[1] = static_cast<std::uint8_t>(msg.head[1]);
  raw.level_flag = static_cast<std::uint8_t>(msg.level_flag);
  raw.frame_reserve = static_cast<std::uint8_t>(msg.frame_reserve);
  raw.sn[0] = static_cast<std::uint32_t>(msg.sn[0]);
  raw.sn[1] = static_cast<std::uint32_t>(msg.sn[1]);
  raw.version[0] = static_cast<std::uint32_t>(msg.version[0]);
  raw.version[1] = static_cast<std::uint32_t>(msg.version[1]);
  raw.bandwidth = static_cast<std::uint16_t>(msg.bandwidth);

  for (std::size_t i = 0; i < 20; ++i)
  {
    const auto& src = msg.motor_cmd[i];
    auto& dst = raw.motor_cmd[i];
    dst.mode = static_cast<std::uint8_t>(src.mode);
    dst.q = static_cast<float>(src.q);
    dst.dq = static_cast<float>(src.dq);
    dst.tau = static_cast<float>(src.tau);
    dst.kp = static_cast<float>(src.kp);
    dst.kd = static_cast<float>(src.kd);
    for (std::size_t j = 0; j < 3; ++j)
    {
      dst.reserve[j] = static_cast<std::uint32_t>(src.reserve[j]);
    }
  }

  raw.bms.off = static_cast<std::uint8_t>(msg.bms_cmd.off);
  for (std::size_t i = 0; i < 3; ++i)
  {
    raw.bms.reserve[i] = static_cast<std::uint8_t>(msg.bms_cmd.reserve[i]);
  }
  for (std::size_t i = 0; i < 40; ++i)
  {
    raw.wireless_remote[i] = static_cast<std::uint8_t>(msg.wireless_remote[i]);
  }
  for (std::size_t i = 0; i < 12; ++i)
  {
    raw.led[i] = static_cast<std::uint8_t>(msg.led[i]);
  }
  for (std::size_t i = 0; i < 2; ++i)
  {
    raw.fan[i] = static_cast<std::uint8_t>(msg.fan[i]);
  }
  raw.gpio = static_cast<std::uint8_t>(msg.gpio);
  raw.reserve = static_cast<std::uint32_t>(msg.reserve);

  constexpr std::size_t kSizeWithoutCrc = sizeof(LowCmdRaw) - sizeof(std::uint32_t);
  std::vector<std::uint8_t> payload(kSizeWithoutCrc);
  std::memcpy(payload.data(), &raw, kSizeWithoutCrc);

  std::vector<std::uint32_t> words(payload.size() / sizeof(std::uint32_t), 0U);
  std::memcpy(words.data(), payload.data(), payload.size());
  return Crc32Core(words);
}

}  // namespace locomotion_controller_cpp
