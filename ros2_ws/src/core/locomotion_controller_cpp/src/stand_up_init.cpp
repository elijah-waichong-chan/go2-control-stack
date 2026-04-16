#include <array>
#include <functional>
#include <optional>
#include <string>

#include "go2_msgs/msg/loop_status.hpp"
#include "locomotion_controller_cpp/common.hpp"
#include "rclcpp/rclcpp.hpp"
#include "unitree_go/msg/low_cmd.hpp"
#include "unitree_go/msg/low_state.hpp"

namespace locomotion_controller_cpp
{

namespace
{

constexpr std::array<double, 12> kDefaultCrouchPos = {
  0.0, 1.36, -2.65,
  0.0, 1.36, -2.65,
  0.0, 1.36, -2.65,
  0.0, 1.36, -2.65,
};

constexpr std::array<double, 12> kDefaultTargetPos = {
  0.0, 0.9, -1.8,
  0.0, 0.9, -1.8,
  0.0, 0.9, -1.8,
  0.0, 0.9, -1.8,
};

constexpr std::array<double, 12> kFallbackTargetPos = {
  0.0, 0.67, -1.3,
  0.0, 0.67, -1.3,
  0.0, 0.67, -1.3,
  0.0, 0.67, -1.3,
};

}  // namespace

class StandUpInitNode : public rclcpp::Node
{
public:
  StandUpInitNode()
  : Node("stand_up_init")
  {
    this->declare_parameter("kp", 60.0);
    this->declare_parameter("kd", 5.0);
    this->declare_parameter("crouch_time_s", 1.0);
    this->declare_parameter("crouch_hold_s", 0.2);
    this->declare_parameter("ramp_time_s", 2.0);
    this->declare_parameter("start_delay_s", 0.5);
    this->declare_parameter("command_hz", 500.0);
    this->declare_parameter("status_hz", 10.0);
    this->declare_parameter("exit_after_standing_s", 2.0);
    this->declare_parameter("crouch_pos", std::vector<double>(kDefaultCrouchPos.begin(), kDefaultCrouchPos.end()));
    this->declare_parameter("target_pos", std::vector<double>(kDefaultTargetPos.begin(), kDefaultTargetPos.end()));

    kp_ = this->get_parameter("kp").as_double();
    kd_ = this->get_parameter("kd").as_double();
    crouch_time_s_ = std::max(0.01, this->get_parameter("crouch_time_s").as_double());
    crouch_hold_s_ = std::max(0.0, this->get_parameter("crouch_hold_s").as_double());
    ramp_time_s_ = std::max(0.01, this->get_parameter("ramp_time_s").as_double());
    start_delay_s_ = std::max(0.0, this->get_parameter("start_delay_s").as_double());
    command_hz_ = std::max(1.0, this->get_parameter("command_hz").as_double());
    status_hz_ = std::max(1.0, this->get_parameter("status_hz").as_double());
    exit_after_standing_s_ =
      std::max(0.0, this->get_parameter("exit_after_standing_s").as_double());

    crouch_pos_ = this->get_parameter("crouch_pos").as_double_array();
    if (crouch_pos_.size() != 12U)
    {
      RCLCPP_WARN(
        this->get_logger(),
        "crouch_pos must have 12 elements (FR,FL,RR,RL). Using defaults.");
      crouch_pos_.assign(kDefaultCrouchPos.begin(), kDefaultCrouchPos.end());
    }

    target_pos_ = this->get_parameter("target_pos").as_double_array();
    if (target_pos_.size() != 12U)
    {
      RCLCPP_WARN(
        this->get_logger(),
        "target_pos must have 12 elements (FR,FL,RR,RL). Using defaults.");
      target_pos_.assign(kFallbackTargetPos.begin(), kFallbackTargetPos.end());
    }

    auto qos = rclcpp::QoS(rclcpp::KeepLast(10)).reliable();
    auto status_qos = rclcpp::QoS(rclcpp::KeepLast(1)).reliable().transient_local();

    lowcmd_pub_ = this->create_publisher<unitree_go::msg::LowCmd>("/lowcmd", qos);
    lowstate_sub_ = this->create_subscription<unitree_go::msg::LowState>(
      "/lowstate", qos, std::bind(&StandUpInitNode::OnLowState, this, std::placeholders::_1));
    status_pub_ =
      this->create_publisher<go2_msgs::msg::LoopStatus>("/status/standing_init", status_qos);
    ctrl_status_sub_ = this->create_subscription<go2_msgs::msg::LoopStatus>(
      "/status/loco_ctrl", status_qos,
      std::bind(&StandUpInitNode::OnCtrlStatus, this, std::placeholders::_1));

    timer_ = this->create_wall_timer(
      std::chrono::duration<double>(1.0 / command_hz_),
      std::bind(&StandUpInitNode::OnTimer, this));
    status_timer_ = this->create_wall_timer(
      std::chrono::duration<double>(1.0 / status_hz_),
      std::bind(&StandUpInitNode::OnStatusTimer, this));

    RCLCPP_INFO(this->get_logger(), "stand_up_init running");
  }

private:
  static constexpr int kStatusRunning = 1;
  static constexpr int kStatusWaitingForLowstate = 2;
  static constexpr int kStatusComplete = 3;

  void SetStatus(const int status_code)
  {
    status_code_ = status_code;
  }

  void OnStatusTimer()
  {
    status_pub_->publish(MakeLoopStatus(status_code_));
  }

  void RequestShutdown(const std::string& reason)
  {
    if (shutdown_requested_)
    {
      return;
    }
    shutdown_requested_ = true;
    RCLCPP_INFO(this->get_logger(), "%s", reason.c_str());
    if (timer_)
    {
      timer_->cancel();
    }
    if (status_timer_)
    {
      status_timer_->cancel();
    }
    rclcpp::shutdown();
  }

  void OnLowState(const unitree_go::msg::LowState::SharedPtr msg)
  {
    last_state_ = msg;
    have_state_ = true;
  }

  void OnCtrlStatus(const go2_msgs::msg::LoopStatus::SharedPtr msg)
  {
    if (static_cast<int>(msg->status) != 1 || ctrl_running_)
    {
      return;
    }
    ctrl_running_ = true;
    RequestShutdown("locomotion_controller_cpp is running; stopping stand_up_init.");
  }

  void OnTimer()
  {
    if (!have_state_ || !last_state_)
    {
      SetStatus(kStatusWaitingForLowstate);
      return;
    }

    if (!started_)
    {
      for (std::size_t i = 0; i < 12; ++i)
      {
        start_pos_[i] = static_cast<double>(last_state_->motor_state[i].q);
      }
      start_time_ = this->now();
      started_ = true;
    }

    if (status_sent_)
    {
      SetStatus(kStatusComplete);
    }
    else
    {
      SetStatus(kStatusRunning);
    }

    const double t = (this->now() - start_time_).seconds() - start_delay_s_;
    if (t < 0.0)
    {
      return;
    }

    const double t_crouch_end = crouch_time_s_;
    const double t_hold_end = t_crouch_end + crouch_hold_s_;
    const double t_stand_end = t_hold_end + ramp_time_s_;

    if (t >= t_stand_end && !status_sent_)
    {
      SetStatus(kStatusComplete);
      status_sent_ = true;
      standing_done_time_ns_ = this->now().nanoseconds();
    }

    if (
      status_sent_ && !ctrl_running_ && standing_done_time_ns_.has_value() &&
      (this->now().nanoseconds() - *standing_done_time_ns_) * 1e-9 > exit_after_standing_s_)
    {
      RequestShutdown(
        "standing_init done; RL running status not received in time, stopping stand_up_init.");
      return;
    }

    unitree_go::msg::LowCmd cmd;
    cmd.head[0] = kHead0;
    cmd.head[1] = kHead1;
    cmd.level_flag = kLowLevel;
    cmd.gpio = 0;

    for (std::size_t i = 0; i < 20; ++i)
    {
      cmd.motor_cmd[i].mode = 0x01;
      cmd.motor_cmd[i].q = kPosStop;
      cmd.motor_cmd[i].dq = kVelStop;
      cmd.motor_cmd[i].kp = 0.0;
      cmd.motor_cmd[i].kd = 0.0;
      cmd.motor_cmd[i].tau = 0.0;
    }

    for (std::size_t i = 0; i < 12; ++i)
    {
      double q_des = target_pos_[i];
      if (t < t_crouch_end)
      {
        const double a = Clamp(t / crouch_time_s_, 0.0, 1.0);
        q_des = (1.0 - a) * start_pos_[i] + a * crouch_pos_[i];
      }
      else if (t < t_hold_end)
      {
        q_des = crouch_pos_[i];
      }
      else if (t < t_stand_end)
      {
        const double a = Clamp((t - t_hold_end) / ramp_time_s_, 0.0, 1.0);
        q_des = (1.0 - a) * crouch_pos_[i] + a * target_pos_[i];
      }

      cmd.motor_cmd[i].q = static_cast<float>(q_des);
      cmd.motor_cmd[i].dq = 0.0;
      cmd.motor_cmd[i].kp = static_cast<float>(kp_);
      cmd.motor_cmd[i].kd = static_cast<float>(kd_);
      cmd.motor_cmd[i].tau = 0.0;
      cmd.motor_cmd[i].mode = 0x01;
    }

    cmd.crc = GetCrc(cmd);
    lowcmd_pub_->publish(cmd);
  }

  rclcpp::Publisher<unitree_go::msg::LowCmd>::SharedPtr lowcmd_pub_;
  rclcpp::Publisher<go2_msgs::msg::LoopStatus>::SharedPtr status_pub_;
  rclcpp::Subscription<unitree_go::msg::LowState>::SharedPtr lowstate_sub_;
  rclcpp::Subscription<go2_msgs::msg::LoopStatus>::SharedPtr ctrl_status_sub_;
  rclcpp::TimerBase::SharedPtr timer_;
  rclcpp::TimerBase::SharedPtr status_timer_;

  unitree_go::msg::LowState::SharedPtr last_state_;
  bool have_state_{false};
  bool started_{false};
  bool status_sent_{false};
  bool ctrl_running_{false};
  bool shutdown_requested_{false};
  int status_code_{kStatusWaitingForLowstate};
  rclcpp::Time start_time_{0, 0, RCL_ROS_TIME};
  std::array<double, 12> start_pos_{};
  std::vector<double> crouch_pos_;
  std::vector<double> target_pos_;
  std::optional<std::int64_t> standing_done_time_ns_;

  double kp_{60.0};
  double kd_{5.0};
  double crouch_time_s_{1.0};
  double crouch_hold_s_{0.2};
  double ramp_time_s_{2.0};
  double start_delay_s_{0.5};
  double command_hz_{500.0};
  double status_hz_{10.0};
  double exit_after_standing_s_{2.0};
};

}  // namespace locomotion_controller_cpp

int main(int argc, char* argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<locomotion_controller_cpp::StandUpInitNode>());
  if (rclcpp::ok())
  {
    rclcpp::shutdown();
  }
  return 0;
}
