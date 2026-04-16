#include <chrono>
#include <cmath>
#include <functional>
#include <optional>

#include "go2_msgs/msg/locomotion_cmd.hpp"
#include "go2_msgs/msg/push_event.hpp"
#include "locomotion_controller_cpp/common.hpp"
#include "rclcpp/rclcpp.hpp"
#include "unitree_go/msg/wireless_controller.hpp"

namespace locomotion_controller_cpp
{

class WirelessCmdBridgeNode : public rclcpp::Node
{
public:
  WirelessCmdBridgeNode()
  : Node("wireless_cmd_bridge")
  {
    this->declare_parameter("push_event_hz", 10.0);
    this->declare_parameter("publish_hz", 50.0);
    this->declare_parameter("cmd_timeout_s", 0.5);
    this->declare_parameter("deadzone", 0.05);
    this->declare_parameter("scale_x", 0.6);
    this->declare_parameter("scale_y", -0.4);
    this->declare_parameter("scale_yaw", -1.2);
    this->declare_parameter("z_pos", 0.27);

    push_event_hz_ = this->get_parameter("push_event_hz").as_double();
    publish_hz_ = this->get_parameter("publish_hz").as_double();
    cmd_timeout_s_ = this->get_parameter("cmd_timeout_s").as_double();
    deadzone_ = this->get_parameter("deadzone").as_double();
    scale_x_ = this->get_parameter("scale_x").as_double();
    scale_y_ = this->get_parameter("scale_y").as_double();
    scale_yaw_ = this->get_parameter("scale_yaw").as_double();
    z_pos_ = this->get_parameter("z_pos").as_double();

    auto qos = rclcpp::QoS(rclcpp::KeepLast(10)).best_effort();
    sub_wireless_ = this->create_subscription<unitree_go::msg::WirelessController>(
      "/wirelesscontroller", qos,
      std::bind(&WirelessCmdBridgeNode::OnWireless, this, std::placeholders::_1));
    pub_cmd_ = this->create_publisher<go2_msgs::msg::LocomotionCmd>("/locomotion_cmd", qos);
    pub_push_event_ = this->create_publisher<go2_msgs::msg::PushEvent>("/data/push_event", qos);

    timer_ = this->create_wall_timer(
      std::chrono::duration<double>(1.0 / std::max(1.0, publish_hz_)),
      std::bind(&WirelessCmdBridgeNode::OnTimer, this));
    push_event_timer_ = this->create_wall_timer(
      std::chrono::duration<double>(1.0 / std::max(1.0, push_event_hz_)),
      std::bind(&WirelessCmdBridgeNode::OnPushEventTimer, this));

    RCLCPP_INFO(
      this->get_logger(),
      "wireless_cmd_bridge running: /wirelesscontroller -> /locomotion_cmd, push events -> /data/push_event");
  }

private:
  static constexpr int kAMask = 1 << 8;
  static constexpr int kYMask = 1 << 11;
  static constexpr int kDpadUpMask = 1 << 12;
  static constexpr int kDpadRightMask = 1 << 13;
  static constexpr int kDpadDownMask = 1 << 14;
  static constexpr int kDpadLeftMask = 1 << 15;

  double ApplyDeadzone(const double value) const
  {
    return std::abs(value) < deadzone_ ? 0.0 : value;
  }

  bool IsStale() const
  {
    if (!last_wireless_time_.has_value())
    {
      return true;
    }
    const auto age_s =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - *last_wireless_time_).count();
    return age_s > cmd_timeout_s_;
  }

  std::uint8_t LabelFromKeys(const int keys) const
  {
    if ((keys & kYMask) != 0)
    {
      return go2_msgs::msg::PushEvent::UP;
    }
    if ((keys & kAMask) != 0)
    {
      return go2_msgs::msg::PushEvent::DOWN;
    }
    if ((keys & kDpadUpMask) != 0)
    {
      return go2_msgs::msg::PushEvent::FWD;
    }
    if ((keys & kDpadDownMask) != 0)
    {
      return go2_msgs::msg::PushEvent::BACK;
    }
    if ((keys & kDpadLeftMask) != 0)
    {
      return go2_msgs::msg::PushEvent::LEFT;
    }
    if ((keys & kDpadRightMask) != 0)
    {
      return go2_msgs::msg::PushEvent::RIGHT;
    }
    return go2_msgs::msg::PushEvent::NO_PUSH;
  }

  void PublishPushEvent(const std::uint8_t label)
  {
    go2_msgs::msg::PushEvent msg;
    msg.header.stamp = this->now();
    msg.header.frame_id = "wireless_cmd_bridge";
    msg.label = label;
    pub_push_event_->publish(msg);
  }

  void OnWireless(const unitree_go::msg::WirelessController::SharedPtr msg)
  {
    latest_keys_ = static_cast<int>(msg->keys) & 0xFFFF;
    lx_ = ApplyDeadzone(static_cast<double>(msg->lx));
    ly_ = ApplyDeadzone(static_cast<double>(msg->ly));
    rx_ = ApplyDeadzone(static_cast<double>(msg->rx));
    last_wireless_time_ = std::chrono::steady_clock::now();
  }

  void OnPushEventTimer()
  {
    if (IsStale())
    {
      PublishPushEvent(go2_msgs::msg::PushEvent::NO_PUSH);
      return;
    }
    PublishPushEvent(LabelFromKeys(latest_keys_));
  }

  void OnTimer()
  {
    if (IsStale())
    {
      return;
    }

    go2_msgs::msg::LocomotionCmd msg;
    msg.stamp = this->now();
    msg.x_vel = Clamp(ly_ * scale_x_, -std::abs(scale_x_), std::abs(scale_x_));
    msg.y_vel = Clamp(lx_ * scale_y_, -std::abs(scale_y_), std::abs(scale_y_));
    msg.yaw_rate = Clamp(rx_ * scale_yaw_, -std::abs(scale_yaw_), std::abs(scale_yaw_));
    msg.z_pos = z_pos_;
    pub_cmd_->publish(msg);
  }

  rclcpp::Subscription<unitree_go::msg::WirelessController>::SharedPtr sub_wireless_;
  rclcpp::Publisher<go2_msgs::msg::LocomotionCmd>::SharedPtr pub_cmd_;
  rclcpp::Publisher<go2_msgs::msg::PushEvent>::SharedPtr pub_push_event_;
  rclcpp::TimerBase::SharedPtr timer_;
  rclcpp::TimerBase::SharedPtr push_event_timer_;

  std::optional<std::chrono::steady_clock::time_point> last_wireless_time_;
  double lx_{0.0};
  double ly_{0.0};
  double rx_{0.0};
  int latest_keys_{0};

  double push_event_hz_{10.0};
  double publish_hz_{50.0};
  double cmd_timeout_s_{0.5};
  double deadzone_{0.05};
  double scale_x_{0.6};
  double scale_y_{-0.4};
  double scale_yaw_{-1.2};
  double z_pos_{0.27};
};

}  // namespace locomotion_controller_cpp

int main(int argc, char* argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<locomotion_controller_cpp::WirelessCmdBridgeNode>());
  if (rclcpp::ok())
  {
    rclcpp::shutdown();
  }
  return 0;
}
