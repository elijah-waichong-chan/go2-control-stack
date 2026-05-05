#include <array>
#include <cassert>
#include <cmath>
#include <functional>
#include <string>
#include <vector>

#include "icon_lab_d1_ros2/msg/servo_feedback.hpp"
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/imu.hpp"
#include "sensor_msgs/msg/joint_state.hpp"
#include "unitree_go/msg/low_state.hpp"

class StateConverterNode : public rclcpp::Node
{
public:
  StateConverterNode()
  : Node("state_converter")
  , leg_nq_(12)
  , arm_nq_(6)
  , leg_joint_names_({
      "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
      "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
      "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
      "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
    })
  , arm_joint_names_({
      "d1_Joint1", "d1_Joint2", "d1_Joint3",
      "d1_Joint4", "d1_Joint5", "d1_Joint6",
    })
  , urdf_to_sdk_index_({
      3, 4, 5,
      0, 1, 2,
      9, 10, 11,
      6, 7, 8,
    })
  {
    jointstate_publisher_ = this->create_publisher<sensor_msgs::msg::JointState>("joint_states", 10);
    imu_publisher_ = this->create_publisher<sensor_msgs::msg::Imu>("imu", 10);

    lowstate_subscription_ = this->create_subscription<unitree_go::msg::LowState>(
      "lowstate", 10, std::bind(&StateConverterNode::state_callback, this, std::placeholders::_1));
    servo_feedback_subscription_ =
      this->create_subscription<icon_lab_d1_ros2::msg::ServoFeedback>(
      "/arm/servo_feedback", 10,
      std::bind(&StateConverterNode::servo_feedback_callback, this, std::placeholders::_1));

    assert(leg_joint_names_.size() == leg_nq_);
    assert(arm_joint_names_.size() == arm_nq_);
    assert(urdf_to_sdk_index_.size() == leg_nq_);
    jointstate_msg_.name = leg_joint_names_;
    jointstate_msg_.name.insert(
      jointstate_msg_.name.end(), arm_joint_names_.begin(), arm_joint_names_.end());
    jointstate_msg_.position.assign(leg_nq_ + arm_nq_, 0.0);
    jointstate_msg_.velocity.assign(leg_nq_ + arm_nq_, 0.0);
    jointstate_msg_.effort.assign(leg_nq_ + arm_nq_, 0.0);

    imu_msg_.header.frame_id = "imu";
    imu_msg_.orientation_covariance = {3e-10, 0, 0, 0, 3e-10, 0, 0, 0, 3e-4};
    imu_msg_.angular_velocity_covariance = {1e-6, 0, 0, 0, 1e-6, 0, 0, 0, 1e-6};
    imu_msg_.linear_acceleration_covariance = {6e-2, 0, 0, 0, 6e-2, 0, 0, 0, 6e-2};
  }

private:
  void publish_joint_state()
  {
    jointstate_msg_.header.stamp = this->get_clock()->now();
    jointstate_publisher_->publish(jointstate_msg_);
  }

  void servo_feedback_callback(const icon_lab_d1_ros2::msg::ServoFeedback::SharedPtr msg)
  {
    for (size_t index_arm = 0; index_arm < arm_nq_; ++index_arm)
    {
      double angle_rad = static_cast<double>(msg->angle_deg[index_arm]) * kDegToRad;
      if (index_arm == 0 || index_arm == 3)
      {
        angle_rad = -angle_rad;
      }
      jointstate_msg_.position[leg_nq_ + index_arm] = angle_rad;
    }

    if (have_lowstate_)
    {
      publish_joint_state();
    }
  }

  void state_callback(const unitree_go::msg::LowState::SharedPtr msg)
  {
    for (size_t index_urdf = 0; index_urdf < leg_nq_; ++index_urdf)
    {
      const size_t index_sdk = urdf_to_sdk_index_[index_urdf];
      jointstate_msg_.position[index_urdf] = msg->motor_state[index_sdk].q;
      jointstate_msg_.velocity[index_urdf] = msg->motor_state[index_sdk].dq;
      jointstate_msg_.effort[index_urdf] = msg->motor_state[index_sdk].tau_est;
    }
    have_lowstate_ = true;
    publish_joint_state();

    imu_msg_.header.stamp = this->get_clock()->now();
    imu_msg_.orientation.w = msg->imu_state.quaternion[0];
    imu_msg_.orientation.x = msg->imu_state.quaternion[1];
    imu_msg_.orientation.y = msg->imu_state.quaternion[2];
    imu_msg_.orientation.z = msg->imu_state.quaternion[3];
    imu_msg_.angular_velocity.x = msg->imu_state.gyroscope[0];
    imu_msg_.angular_velocity.y = msg->imu_state.gyroscope[1];
    imu_msg_.angular_velocity.z = msg->imu_state.gyroscope[2];
    imu_msg_.linear_acceleration.x = msg->imu_state.accelerometer[0];
    imu_msg_.linear_acceleration.y = msg->imu_state.accelerometer[1];
    imu_msg_.linear_acceleration.z = msg->imu_state.accelerometer[2];
    imu_publisher_->publish(imu_msg_);
  }

  static constexpr double kDegToRad = M_PI / 180.0;

  const size_t leg_nq_;
  const size_t arm_nq_;
  const std::vector<std::string> leg_joint_names_;
  const std::vector<std::string> arm_joint_names_;
  const std::vector<size_t> urdf_to_sdk_index_;

  sensor_msgs::msg::Imu imu_msg_;
  sensor_msgs::msg::JointState jointstate_msg_;
  bool have_lowstate_{false};
  rclcpp::Publisher<sensor_msgs::msg::Imu>::SharedPtr imu_publisher_;
  rclcpp::Publisher<sensor_msgs::msg::JointState>::SharedPtr jointstate_publisher_;
  rclcpp::Subscription<unitree_go::msg::LowState>::SharedPtr lowstate_subscription_;
  rclcpp::Subscription<icon_lab_d1_ros2::msg::ServoFeedback>::SharedPtr
    servo_feedback_subscription_;
};

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<StateConverterNode>());
  rclcpp::shutdown();
  return 0;
}
