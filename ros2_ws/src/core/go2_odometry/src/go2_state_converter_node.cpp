#include <array>
#include <functional>
#include <string>

#include "go2_msgs/msg/arm_angles.hpp"
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
  , nq_legs(12)
  , nq_arm_msg(7)
  , nq_arm_pub(8)
  // clang-format off
  , leg_joint_names_({
        "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
        "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
        "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
        "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
    })
    , arm_joint_names_({
        "d1_Joint1", "d1_Joint2", "d1_Joint3", "d1_Joint4",
        "d1_Joint5", "d1_Joint6", "d1_Joint7_1", "d1_Joint7_2",
    })
    , leg_to_sdk_index_({
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
    arm_angles_subscription_ = this->create_subscription<go2_msgs::msg::ArmAngles>(
      "arm_angles", 10, std::bind(&StateConverterNode::arm_angles_callback, this, std::placeholders::_1));

    // Pre-fill joint state messages
    assert(leg_joint_names_.size() == nq_legs);
    assert(arm_joint_names_.size() == nq_arm_pub);
    assert(leg_to_sdk_index_.size() == nq_legs);
    jointstate_msg_.name = leg_joint_names_;
    jointstate_msg_.name.insert(
      jointstate_msg_.name.end(), arm_joint_names_.begin(), arm_joint_names_.end());
    jointstate_msg_.position.resize(nq_legs + nq_arm_pub, 0.0);
    jointstate_msg_.velocity.resize(nq_legs + nq_arm_pub, 0.0);
    jointstate_msg_.effort.resize(nq_legs + nq_arm_pub, 0.0);
    arm_joint_positions_.fill(0.0);
    arm_joint_velocities_.fill(0.0);
    arm_joint_efforts_.fill(0.0);

    imu_msg_.header.frame_id = "imu";
    // See datasheet https://www.wit-motion.com/proztcgd/7.html
    imu_msg_.orientation_covariance = {3e-10, 0, 0, 0, 3e-10, 0, 0, 0, 3e-4};       // accuracy = 0.001° (XY), 1° (Z)
    imu_msg_.angular_velocity_covariance = {1e-6, 0, 0, 0, 1e-6, 0, 0, 0, 1e-6};    // accuracy = 0.07 °/s
    imu_msg_.linear_acceleration_covariance = {6e-2, 0, 0, 0, 6e-2, 0, 0, 0, 6e-2}; // accuracy = 25 mG
  }

protected:
  const size_t nq_legs;                              // Go2 leg DoF
  const size_t nq_arm_msg;                           // Arm angles received on /arm_angles
  const size_t nq_arm_pub;                           // Arm joints exposed on /joint_states
  const std::vector<std::string> leg_joint_names_;   // Go2 joint names
  const std::vector<std::string> arm_joint_names_;   // Arm joint names for telemetry
  const std::vector<size_t> leg_to_sdk_index_;       // Joint indexes in ros msgs (in the urdf order)

private:
  void arm_angles_callback(const go2_msgs::msg::ArmAngles::SharedPtr msg);
  void state_callback(const unitree_go::msg::LowState::SharedPtr msg);

  rosgraph_msgs::msg::Clock clock_msg_;
  sensor_msgs::msg::Imu imu_msg_;
  sensor_msgs::msg::JointState jointstate_msg_;
  rclcpp::Publisher<sensor_msgs::msg::Imu>::SharedPtr imu_publisher_;
  rclcpp::Publisher<sensor_msgs::msg::JointState>::SharedPtr jointstate_publisher_;
  rclcpp::Subscription<go2_msgs::msg::ArmAngles>::SharedPtr arm_angles_subscription_;
  rclcpp::Subscription<unitree_go::msg::LowState>::SharedPtr lowstate_subscription_;
  std::array<double, 8> arm_joint_positions_;
  std::array<double, 8> arm_joint_velocities_;
  std::array<double, 8> arm_joint_efforts_;

  static constexpr double kDegToRad = 3.14159265358979323846 / 180.0;
};

void StateConverterNode::arm_angles_callback(const go2_msgs::msg::ArmAngles::SharedPtr msg)
{
  for (size_t i = 0; i < nq_arm_msg; ++i)
  {
    const double angle_rad = static_cast<double>(msg->angle_deg[i]) * kDegToRad;
    arm_joint_positions_[i] = (i == 0) ? -angle_rad : angle_rad;
  }

  // The arm feedback exposes one finger angle. Mirror it into the opposing prismatic joint
  // so downstream consumers can see the full gripper state if they care to.
  arm_joint_positions_[6] = static_cast<double>(msg->angle_deg[6]) * kDegToRad;
  arm_joint_positions_[7] = -arm_joint_positions_[6];
}

// Extract and re-order joint sensor measurements
void StateConverterNode::state_callback(const unitree_go::msg::LowState::SharedPtr msg)
{
  jointstate_msg_.header.stamp = this->get_clock()->now();
  for (size_t index_urdf = 0; index_urdf < nq_legs; index_urdf++)
  {
    const size_t index_sdk = leg_to_sdk_index_[index_urdf];
    jointstate_msg_.position[index_urdf] = msg->motor_state[index_sdk].q;
    jointstate_msg_.velocity[index_urdf] = msg->motor_state[index_sdk].dq;
    jointstate_msg_.effort[index_urdf] = msg->motor_state[index_sdk].tau_est;
  }

  for (size_t i = 0; i < nq_arm_pub; ++i)
  {
    const size_t index_joint_state = nq_legs + i;
    jointstate_msg_.position[index_joint_state] = arm_joint_positions_[i];
    jointstate_msg_.velocity[index_joint_state] = arm_joint_velocities_[i];
    jointstate_msg_.effort[index_joint_state] = arm_joint_efforts_[i];
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
