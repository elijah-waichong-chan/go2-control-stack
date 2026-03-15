#include <functional>
#include <string>

#include "rclcpp/rclcpp.hpp"
#include "rosgraph_msgs/msg/clock.hpp"
#include "sensor_msgs/msg/imu.hpp"
#include "sensor_msgs/msg/joint_state.hpp"
#include "unitree_go/msg/low_state.hpp"

class StateConverterNode : public rclcpp::Node
{
public:
  StateConverterNode()
  : Node("state_converter")
  , nq(12)
  // clang-format off
  , urdf_joint_names_({
        "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
        "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
        "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
        "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
    })
    , urdf_to_sdk_index_({
        3,  4,  5,
        0,  1,  2,
        9, 10, 11,
        6,  7,  8,
    })
  // clang-format on
  {
    // Create useful subscribers/publishers
    jointstate_publisher_ = this->create_publisher<sensor_msgs::msg::JointState>("joint_states", 10);
    imu_publisher_ = this->create_publisher<sensor_msgs::msg::Imu>("imu", 10);

    lowstate_subscription_ = this->create_subscription<unitree_go::msg::LowState>(
      "lowstate", 10, std::bind(&StateConverterNode::state_callback, this, std::placeholders::_1));

    // Pre-fill joint state messages
    assert(urdf_joint_names_.size() == nq);
    assert(urdf_to_sdk_index_.size() == nq);
    jointstate_msg_.name = urdf_joint_names_;
    jointstate_msg_.position.resize(nq);
    jointstate_msg_.velocity.resize(nq);
    jointstate_msg_.effort.resize(nq);

    imu_msg_.header.frame_id = "imu";
    // See datasheet https://www.wit-motion.com/proztcgd/7.html
    imu_msg_.orientation_covariance = {3e-10, 0, 0, 0, 3e-10, 0, 0, 0, 3e-4};       // accuracy = 0.001° (XY), 1° (Z)
    imu_msg_.angular_velocity_covariance = {1e-6, 0, 0, 0, 1e-6, 0, 0, 0, 1e-6};    // accuracy = 0.07 °/s
    imu_msg_.linear_acceleration_covariance = {6e-2, 0, 0, 0, 6e-2, 0, 0, 0, 6e-2}; // accuracy = 25 mG
  }

protected:
  const size_t nq;                                  // Robot DoF
  const std::vector<std::string> urdf_joint_names_; // Robot joint names
  const std::vector<size_t> urdf_to_sdk_index_;     // Joint indexes in ros msgs (in the urdf order)

private:
  void state_callback(const unitree_go::msg::LowState::SharedPtr msg);

  rosgraph_msgs::msg::Clock clock_msg_;
  sensor_msgs::msg::Imu imu_msg_;
  sensor_msgs::msg::JointState jointstate_msg_;
  rclcpp::Publisher<sensor_msgs::msg::Imu>::SharedPtr imu_publisher_;
  rclcpp::Publisher<sensor_msgs::msg::JointState>::SharedPtr jointstate_publisher_;
  rclcpp::Subscription<unitree_go::msg::LowState>::SharedPtr lowstate_subscription_;
};

// Extract and re-order joint sensor measurements
void StateConverterNode::state_callback(const unitree_go::msg::LowState::SharedPtr msg)
{
  jointstate_msg_.header.stamp = this->get_clock()->now();
  for (size_t index_urdf = 0; index_urdf < nq; index_urdf++)
  {
    const size_t index_sdk = urdf_to_sdk_index_[index_urdf];
    jointstate_msg_.position[index_urdf] = msg->motor_state[index_sdk].q;
    jointstate_msg_.velocity[index_urdf] = msg->motor_state[index_sdk].dq;
    jointstate_msg_.effort[index_urdf] = msg->motor_state[index_sdk].tau_est;
  }
  jointstate_publisher_->publish(jointstate_msg_);

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

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<StateConverterNode>());
  rclcpp::shutdown();
  return 0;
}
