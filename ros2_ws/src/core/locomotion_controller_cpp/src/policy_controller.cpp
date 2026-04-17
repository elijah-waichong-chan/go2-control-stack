#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <deque>
#include <functional>
#include <limits>
#include <memory>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "ament_index_cpp/get_package_share_directory.hpp"
#include "hq_pcot_msgs/msg/locomotion_cmd.hpp"
#include "hq_pcot_msgs/msg/loop_status.hpp"
#include "locomotion_controller_cpp/common.hpp"
#include "onnxruntime_cxx_api.h"
#include "rclcpp/rclcpp.hpp"
#include "unitree_go/msg/low_cmd.hpp"
#include "unitree_go/msg/low_state.hpp"
#include "yaml-cpp/yaml.h"

namespace locomotion_controller_cpp
{

namespace
{

std::array<float, 9> QuatToRotWb(
  const float qw,
  const float qx,
  const float qy,
  const float qz)
{
  return {
    1.0F - 2.0F * (qy * qy + qz * qz),
    2.0F * (qx * qy - qz * qw),
    2.0F * (qx * qz + qy * qw),
    2.0F * (qx * qy + qz * qw),
    1.0F - 2.0F * (qx * qx + qz * qz),
    2.0F * (qy * qz - qx * qw),
    2.0F * (qx * qz - qy * qw),
    2.0F * (qy * qz + qx * qw),
    1.0F - 2.0F * (qx * qx + qy * qy),
  };
}

std::vector<float> LoadFloatVector(const YAML::Node& node)
{
  std::vector<float> out;
  if (!node || node.IsNull())
  {
    return out;
  }
  out.reserve(node.size());
  for (const auto& value : node)
  {
    out.push_back(value.as<float>());
  }
  return out;
}

std::vector<int> LoadIntVector(const YAML::Node& node)
{
  std::vector<int> out;
  if (!node || node.IsNull())
  {
    return out;
  }
  out.reserve(node.size());
  for (const auto& value : node)
  {
    out.push_back(value.as<int>());
  }
  return out;
}

float Percentile99(const std::deque<float>& samples)
{
  if (samples.empty())
  {
    return -1.0F;
  }
  std::vector<float> ordered(samples.begin(), samples.end());
  std::sort(ordered.begin(), ordered.end());
  const std::size_t idx = static_cast<std::size_t>(
    std::ceil(0.99 * static_cast<double>(ordered.size())) - 1.0);
  return ordered[std::min(idx, ordered.size() - 1)];
}

std::vector<float> ZeroForShape(const std::vector<int64_t>& shape)
{
  std::size_t size = 1U;
  for (const auto dim : shape)
  {
    size *= static_cast<std::size_t>(dim > 0 ? dim : 1);
  }
  return std::vector<float>(size, 0.0F);
}

}  // namespace

struct ObservationTerm
{
  std::string name;
  std::vector<float> scale;
  std::optional<std::pair<float, float>> clip;
  int history_length{1};
  std::deque<std::vector<float>> buffer;
};

class PolicyControllerNode : public rclcpp::Node
{
public:
  PolicyControllerNode()
  : Node("policy_controller"),
    env_(ORT_LOGGING_LEVEL_WARNING, "locomotion_controller_cpp")
  {
    const auto share_dir = ament_index_cpp::get_package_share_directory("locomotion_controller_cpp");
    policy_dir_ = share_dir + "/config/policy_dir";
    lowstate_topic_ = "/lowstate";
    locomotion_cmd_topic_ = "/locomotion_cmd";
    lowcmd_topic_ = "/lowcmd";
    status_topic_ = "/status/loco_ctrl";

    control_hz_ = 50.0;
    control_period_s_ = 1.0 / std::max(control_hz_, 1.0);
    loop_budget_ms_ = control_period_s_ * 1000.0;
    status_hz_ = 10.0;
    cmd_timeout_s_ = 0.5;
    lowstate_timeout_s_ = 0.1;
    loop_stats_window_ = std::max(100, static_cast<int>(control_hz_ * 10.0));

    LoadDeployConfig();
    CreateOnnxSession(policy_dir_ + "/exported/policy.onnx");

    last_raw_action_.assign(action_dim_, 0.0F);

    auto sensor_qos = rclcpp::QoS(rclcpp::KeepLast(10)).best_effort();
    auto command_qos = rclcpp::QoS(rclcpp::KeepLast(10)).reliable();
    auto status_qos = rclcpp::QoS(rclcpp::KeepLast(1)).reliable().transient_local();

    pub_lowcmd_ = this->create_publisher<unitree_go::msg::LowCmd>(lowcmd_topic_, command_qos);
    pub_status_ = this->create_publisher<hq_pcot_msgs::msg::LoopStatus>(status_topic_, status_qos);
    SetStatus(kStatusIdle);

    sub_lowstate_ = this->create_subscription<unitree_go::msg::LowState>(
      lowstate_topic_, sensor_qos,
      std::bind(&PolicyControllerNode::OnLowstate, this, std::placeholders::_1));
    sub_cmd_ = this->create_subscription<hq_pcot_msgs::msg::LocomotionCmd>(
      locomotion_cmd_topic_, sensor_qos,
      std::bind(&PolicyControllerNode::OnLocomotionCmd, this, std::placeholders::_1));
    sub_standing_ = this->create_subscription<hq_pcot_msgs::msg::LoopStatus>(
      "/status/standing_init", status_qos,
      std::bind(&PolicyControllerNode::OnStandingStatus, this, std::placeholders::_1));

    timer_ = this->create_wall_timer(
      std::chrono::duration<double>(control_period_s_),
      std::bind(&PolicyControllerNode::OnTimer, this));
    status_timer_ = this->create_wall_timer(
      std::chrono::duration<double>(1.0 / std::max(status_hz_, 1.0)),
      std::bind(&PolicyControllerNode::OnStatusTimer, this));
  }

private:
  static constexpr int kStatusIdle = 0;
  static constexpr int kStatusRunning = 1;
  static constexpr int kStatusWaitingForLowstate = 2;
  static constexpr int kStatusWaitingForStandingInit = 3;

  void LoadDeployConfig()
  {
    const std::string deploy_path = policy_dir_ + "/params/deploy.yaml";
    YAML::Node deploy_cfg = YAML::LoadFile(deploy_path);
    if (!deploy_cfg.IsMap())
    {
      throw std::runtime_error("deploy.yaml must be a YAML mapping.");
    }

    joint_ids_map_ = LoadIntVector(deploy_cfg["joint_ids_map"]);
    default_joint_pos_ = LoadFloatVector(deploy_cfg["default_joint_pos"]);
    joint_stiffness_ = LoadFloatVector(deploy_cfg["stiffness"]);
    joint_damping_ = LoadFloatVector(deploy_cfg["damping"]);
    action_dim_ = joint_ids_map_.size();
    if (default_joint_pos_.size() != action_dim_ ||
      joint_stiffness_.size() != action_dim_ ||
      joint_damping_.size() != action_dim_)
    {
      throw std::runtime_error("default_joint_pos/stiffness/damping must match joint_ids_map length.");
    }

    YAML::Node action_cfg = deploy_cfg["actions"]["JointPositionAction"];
    if (!action_cfg || action_cfg.IsNull())
    {
      throw std::runtime_error("deploy.yaml actions must contain JointPositionAction.");
    }
    action_scale_ = LoadFloatVector(action_cfg["scale"]);
    action_offset_ = LoadFloatVector(action_cfg["offset"]);
    if (action_scale_.empty())
    {
      action_scale_.assign(action_dim_, 1.0F);
    }
    if (action_offset_.empty())
    {
      action_offset_.assign(action_dim_, 0.0F);
    }
    if (action_scale_.size() != action_dim_ || action_offset_.size() != action_dim_)
    {
      throw std::runtime_error("Action scale/offset size must match joint_ids_map length.");
    }
    if (action_cfg["clip"] && !action_cfg["clip"].IsNull())
    {
      action_clip_ = std::make_pair(action_cfg["clip"][0].as<float>(), action_cfg["clip"][1].as<float>());
    }

    YAML::Node ranges = deploy_cfg["commands"]["base_velocity"]["ranges"];
    if (ranges && ranges["lin_vel_x"])
    {
      cmd_lin_x_ = {ranges["lin_vel_x"][0].as<float>(), ranges["lin_vel_x"][1].as<float>()};
    }
    if (ranges && ranges["lin_vel_y"])
    {
      cmd_lin_y_ = {ranges["lin_vel_y"][0].as<float>(), ranges["lin_vel_y"][1].as<float>()};
    }
    if (ranges && ranges["ang_vel_z"])
    {
      cmd_ang_z_ = {ranges["ang_vel_z"][0].as<float>(), ranges["ang_vel_z"][1].as<float>()};
    }

    YAML::Node observations = deploy_cfg["observations"];
    if (!observations || !observations.IsMap())
    {
      throw std::runtime_error("deploy.yaml observations must be a mapping.");
    }
    for (const auto& it : observations)
    {
      ObservationTerm term;
      term.name = it.first.as<std::string>();
      const YAML::Node cfg = it.second;
      term.scale = LoadFloatVector(cfg["scale"]);
      if (cfg["clip"] && !cfg["clip"].IsNull())
      {
        term.clip = std::make_pair(cfg["clip"][0].as<float>(), cfg["clip"][1].as<float>());
      }
      term.history_length = std::max(1, cfg["history_length"] ? cfg["history_length"].as<int>() : 1);
      obs_terms_.push_back(term);
    }
  }

  void CreateOnnxSession(const std::string& model_path)
  {
    RCLCPP_INFO(this->get_logger(), "Loading ONNX policy from %s", model_path.c_str());
    Ort::SessionOptions options;
    options.SetIntraOpNumThreads(1);
    options.SetInterOpNumThreads(1);
    options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
    session_ = std::make_unique<Ort::Session>(env_, model_path.c_str(), options);

    Ort::AllocatorWithDefaultOptions allocator;

    const std::size_t input_count = session_->GetInputCount();
    for (std::size_t i = 0; i < input_count; ++i)
    {
      auto input_name = session_->GetInputNameAllocated(i, allocator);
      std::string name(input_name.get());
      auto shape = session_->GetInputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape();
      for (auto& dim : shape)
      {
        if (dim <= 0)
        {
          dim = 1;
        }
      }

      if (i == 0U)
      {
        input_name_ = name;
      }
      else
      {
        recurrent_shapes_[name] = shape;
        recurrent_inputs_[name] = ZeroForShape(shape);
      }
    }

    const std::size_t output_count = session_->GetOutputCount();
    output_names_.reserve(output_count);
    output_name_ptrs_.reserve(output_count);
    for (std::size_t i = 0; i < output_count; ++i)
    {
      auto output_name = session_->GetOutputNameAllocated(i, allocator);
      output_names_.emplace_back(output_name.get());
    }
    for (const auto& name : output_names_)
    {
      output_name_ptrs_.push_back(name.c_str());
    }
  }

  void PublishStatus()
  {
    float avg = -1.0F;
    float p99 = -1.0F;
    float max_val = -1.0F;
    int miss_count = -1;
    int sample_count = -1;

    if (!loop_times_ms_.empty())
    {
      sample_count = static_cast<int>(loop_times_ms_.size());
      max_val = *std::max_element(loop_times_ms_.begin(), loop_times_ms_.end());
      p99 = Percentile99(loop_times_ms_);
      const float sum = std::accumulate(loop_times_ms_.begin(), loop_times_ms_.end(), 0.0F);
      avg = sum / static_cast<float>(loop_times_ms_.size());
      miss_count = static_cast<int>(std::count(deadline_flags_.begin(), deadline_flags_.end(), 1));
    }

    pub_status_->publish(
      MakeLoopStatus(status_code_, avg, p99, max_val, static_cast<float>(loop_budget_ms_), miss_count, sample_count));
  }

  void SetStatus(const int status_code)
  {
    if (status_code_ == status_code)
    {
      return;
    }
    status_code_ = status_code;
    PublishStatus();
  }

  void OnStatusTimer()
  {
    PublishStatus();
  }

  void RecordLoopTimeMs(const float loop_time_ms)
  {
    loop_times_ms_.push_back(loop_time_ms);
    deadline_flags_.push_back(loop_time_ms > loop_budget_ms_ ? 1 : 0);
    while (static_cast<int>(loop_times_ms_.size()) > loop_stats_window_)
    {
      loop_times_ms_.pop_front();
    }
    while (static_cast<int>(deadline_flags_.size()) > loop_stats_window_)
    {
      deadline_flags_.pop_front();
    }
  }

  void OnLowstate(const unitree_go::msg::LowState::SharedPtr msg)
  {
    last_lowstate_ = msg;
    last_lowstate_time_ns_ = this->get_clock()->now().nanoseconds();
    lowstate_stale_logged_ = false;
  }

  void OnLocomotionCmd(const hq_pcot_msgs::msg::LocomotionCmd::SharedPtr msg)
  {
    last_locomotion_cmd_ = msg;
    last_cmd_time_ns_ = this->get_clock()->now().nanoseconds();
  }

  void OnStandingStatus(const hq_pcot_msgs::msg::LoopStatus::SharedPtr msg)
  {
    standing_ready_ = static_cast<int>(msg->status) == 3;
  }

  std::vector<float> ComputeTerm(
    const std::string& name,
    const unitree_go::msg::LowState& lowstate)
  {
    if (name == "base_ang_vel")
    {
      return {
        static_cast<float>(lowstate.imu_state.gyroscope[0]),
        static_cast<float>(lowstate.imu_state.gyroscope[1]),
        static_cast<float>(lowstate.imu_state.gyroscope[2]),
      };
    }

    if (name == "projected_gravity")
    {
      const auto q = lowstate.imu_state.quaternion;
      const auto rot = QuatToRotWb(
        static_cast<float>(q[0]),
        static_cast<float>(q[1]),
        static_cast<float>(q[2]),
        static_cast<float>(q[3]));
      const std::array<float, 3> gravity = {0.0F, 0.0F, -1.0F};
      return {
        rot[0] * gravity[0] + rot[3] * gravity[1] + rot[6] * gravity[2],
        rot[1] * gravity[0] + rot[4] * gravity[1] + rot[7] * gravity[2],
        rot[2] * gravity[0] + rot[5] * gravity[1] + rot[8] * gravity[2],
      };
    }

    if (name == "velocity_commands")
    {
      const auto now_ns = this->get_clock()->now().nanoseconds();
      const bool stale =
        !last_locomotion_cmd_ || !last_cmd_time_ns_.has_value() ||
        (now_ns - *last_cmd_time_ns_) * 1e-9 > cmd_timeout_s_;
      if (stale)
      {
        return {0.0F, 0.0F, 0.0F};
      }
      return {
        Clamp(
          static_cast<float>(last_locomotion_cmd_->x_vel),
          cmd_lin_x_.first,
          cmd_lin_x_.second),
        Clamp(
          static_cast<float>(last_locomotion_cmd_->y_vel),
          cmd_lin_y_.first,
          cmd_lin_y_.second),
        Clamp(
          static_cast<float>(last_locomotion_cmd_->yaw_rate),
          cmd_ang_z_.first,
          cmd_ang_z_.second),
      };
    }

    if (name == "joint_pos_rel")
    {
      std::vector<float> out(action_dim_, 0.0F);
      for (std::size_t i = 0; i < action_dim_; ++i)
      {
        out[i] = static_cast<float>(lowstate.motor_state[joint_ids_map_[i]].q) - default_joint_pos_[i];
      }
      return out;
    }

    if (name == "joint_vel_rel")
    {
      std::vector<float> out(action_dim_, 0.0F);
      for (std::size_t i = 0; i < action_dim_; ++i)
      {
        out[i] = static_cast<float>(lowstate.motor_state[joint_ids_map_[i]].dq);
      }
      return out;
    }

    if (name == "last_action")
    {
      return last_raw_action_;
    }

    throw std::runtime_error("Unsupported observation term in deploy.yaml: " + name);
  }

  std::vector<float> ApplyTermPost(
    const std::vector<float>& term_val,
    const ObservationTerm& term) const
  {
    std::vector<float> out = term_val;
    if (term.clip.has_value())
    {
      for (auto& value : out)
      {
        value = Clamp(value, term.clip->first, term.clip->second);
      }
    }
    if (!term.scale.empty())
    {
      if (term.scale.size() != out.size())
      {
        throw std::runtime_error("Observation scale size mismatch for term " + term.name);
      }
      for (std::size_t i = 0; i < out.size(); ++i)
      {
        out[i] *= term.scale[i];
      }
    }
    return out;
  }

  std::vector<float> BuildObservation(const unitree_go::msg::LowState& lowstate)
  {
    std::vector<float> obs;
    for (auto& term : obs_terms_)
    {
      auto value = ApplyTermPost(ComputeTerm(term.name, lowstate), term);
      if (term.buffer.empty())
      {
        for (int i = 0; i < term.history_length; ++i)
        {
          term.buffer.push_back(value);
        }
      }
      else
      {
        term.buffer.push_back(value);
        while (static_cast<int>(term.buffer.size()) > term.history_length)
        {
          term.buffer.pop_front();
        }
      }

      for (const auto& history_value : term.buffer)
      {
        obs.insert(obs.end(), history_value.begin(), history_value.end());
      }
    }
    return obs;
  }

  std::vector<float> InferPolicy(const std::vector<float>& obs)
  {
    std::vector<float> obs_batch = obs;
    const std::array<int64_t, 2> obs_shape = {1, static_cast<int64_t>(obs.size())};
    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);

    std::vector<const char*> input_names;
    std::vector<Ort::Value> input_tensors;
    input_names.reserve(1U + recurrent_inputs_.size());
    input_tensors.reserve(1U + recurrent_inputs_.size());
    input_names.push_back(input_name_.c_str());
    input_tensors.emplace_back(
      Ort::Value::CreateTensor<float>(
        memory_info,
        obs_batch.data(),
        obs_batch.size(),
        obs_shape.data(),
        obs_shape.size()));

    for (auto& entry : recurrent_inputs_)
    {
      input_names.push_back(entry.first.c_str());
      const auto& shape = recurrent_shapes_.at(entry.first);
      input_tensors.emplace_back(
        Ort::Value::CreateTensor<float>(
          memory_info,
          entry.second.data(),
          entry.second.size(),
          shape.data(),
          shape.size()));
    }

    auto output_tensors = session_->Run(
      Ort::RunOptions{nullptr},
      input_names.data(),
      input_tensors.data(),
      input_tensors.size(),
      output_name_ptrs_.data(),
      output_name_ptrs_.size());

    for (std::size_t i = 0; i < output_names_.size(); ++i)
    {
      const std::string& output_name = output_names_[i];
      if (output_name.size() > 4 && output_name.substr(output_name.size() - 4) == "_out")
      {
        const std::string mapped = output_name.substr(0, output_name.size() - 4) + "_in";
        auto it = recurrent_inputs_.find(mapped);
        if (it != recurrent_inputs_.end())
        {
          const auto info = output_tensors[i].GetTensorTypeAndShapeInfo();
          const auto shape = info.GetShape();
          const auto count = info.GetElementCount();
          const float* data = output_tensors[i].GetTensorData<float>();
          recurrent_shapes_[mapped] = shape;
          it->second.assign(data, data + count);
        }
      }
    }

    auto action_info = output_tensors.front().GetTensorTypeAndShapeInfo();
    const auto action_count = action_info.GetElementCount();
    const float* action_data = output_tensors.front().GetTensorData<float>();
    std::vector<float> raw_action(action_data, action_data + action_count);

    if (raw_action.size() != action_dim_)
    {
      RCLCPP_WARN(
        this->get_logger(),
        "Policy output dim %zu != expected %zu; trunc/pad applied.",
        raw_action.size(),
        action_dim_);
      std::vector<float> fixed(action_dim_, 0.0F);
      const std::size_t count = std::min(action_dim_, raw_action.size());
      std::copy(raw_action.begin(), raw_action.begin() + count, fixed.begin());
      raw_action = fixed;
    }

    return raw_action;
  }

  std::vector<float> ProcessAction(const std::vector<float>& raw_action) const
  {
    std::vector<float> out(action_dim_, 0.0F);
    for (std::size_t i = 0; i < action_dim_; ++i)
    {
      out[i] = raw_action[i] * action_scale_[i] + action_offset_[i];
      if (action_clip_.has_value())
      {
        out[i] = Clamp(out[i], action_clip_->first, action_clip_->second);
      }
    }
    return out;
  }

  unitree_go::msg::LowCmd BuildLowCmd(const std::vector<float>& processed_action) const
  {
    unitree_go::msg::LowCmd msg;
    msg.head[0] = kHead0;
    msg.head[1] = kHead1;
    msg.level_flag = kLowLevel;
    msg.gpio = 0;

    for (std::size_t i = 0; i < 20; ++i)
    {
      msg.motor_cmd[i].mode = 0x01;
      msg.motor_cmd[i].q = kPosStop;
      msg.motor_cmd[i].dq = kVelStop;
      msg.motor_cmd[i].kp = 0.0;
      msg.motor_cmd[i].kd = 0.0;
      msg.motor_cmd[i].tau = 0.0;
    }

    for (std::size_t i = 0; i < action_dim_; ++i)
    {
      const int sdk_idx = joint_ids_map_[i];
      msg.motor_cmd[sdk_idx].mode = 0x01;
      msg.motor_cmd[sdk_idx].q = processed_action[i];
      msg.motor_cmd[sdk_idx].dq = 0.0;
      msg.motor_cmd[sdk_idx].kp = joint_stiffness_[i];
      msg.motor_cmd[sdk_idx].kd = joint_damping_[i];
      msg.motor_cmd[sdk_idx].tau = 0.0;
    }

    msg.crc = GetCrc(msg);
    return msg;
  }

  void FinishLoop(const std::chrono::steady_clock::time_point& start_time)
  {
    const auto end_time = std::chrono::steady_clock::now();
    const float loop_time_ms =
      static_cast<float>(std::chrono::duration<double, std::milli>(end_time - start_time).count());
    RecordLoopTimeMs(loop_time_ms);
  }

  void OnTimer()
  {
    const auto start_time = std::chrono::steady_clock::now();
    const auto now_ns = this->get_clock()->now().nanoseconds();

    if (!standing_ready_)
    {
      if (!wait_logged_)
      {
        RCLCPP_INFO(this->get_logger(), "policy_controller waiting for /status/standing_init...");
        wait_logged_ = true;
      }
      SetStatus(kStatusWaitingForStandingInit);
      FinishLoop(start_time);
      return;
    }

    if (!last_lowstate_)
    {
      if (!lowstate_wait_logged_)
      {
        RCLCPP_INFO(this->get_logger(), "policy_controller waiting for /lowstate...");
        lowstate_wait_logged_ = true;
      }
      SetStatus(kStatusWaitingForLowstate);
      FinishLoop(start_time);
      return;
    }

    if (
      !last_lowstate_time_ns_.has_value() ||
      (lowstate_timeout_s_ > 0.0 &&
      (now_ns - *last_lowstate_time_ns_) * 1e-9 > lowstate_timeout_s_))
    {
      if (!lowstate_stale_logged_)
      {
        const double age_s = !last_lowstate_time_ns_.has_value()
          ? std::numeric_limits<double>::infinity()
          : (now_ns - *last_lowstate_time_ns_) * 1e-9;
        RCLCPP_WARN(
          this->get_logger(),
          "policy_controller waiting for fresh /lowstate (age=%.3fs, timeout=%.3fs).",
          age_s,
          lowstate_timeout_s_);
        lowstate_stale_logged_ = true;
      }
      SetStatus(kStatusWaitingForLowstate);
      FinishLoop(start_time);
      return;
    }

    if (last_lowstate_->motor_state.size() <= *std::max_element(joint_ids_map_.begin(), joint_ids_map_.end()))
    {
      if (!motor_state_wait_logged_)
      {
        RCLCPP_WARN(this->get_logger(), "policy_controller waiting for full /lowstate motor_state.");
        motor_state_wait_logged_ = true;
      }
      SetStatus(kStatusWaitingForLowstate);
      FinishLoop(start_time);
      return;
    }

    try
    {
      auto obs = BuildObservation(*last_lowstate_);
      auto raw_action = InferPolicy(obs);
      auto processed_action = ProcessAction(raw_action);
      auto cmd_msg = BuildLowCmd(processed_action);
      pub_lowcmd_->publish(cmd_msg);

      last_raw_action_ = raw_action;
      if (!ready_sent_)
      {
        if (!running_logged_)
        {
          RCLCPP_INFO(this->get_logger(), "policy_controller running");
          running_logged_ = true;
        }
        ready_sent_ = true;
      }
      SetStatus(kStatusRunning);
    }
    catch (const std::exception& exc)
    {
      RCLCPP_ERROR(this->get_logger(), "policy_controller error: %s", exc.what());
    }

    FinishLoop(start_time);
  }

  rclcpp::Publisher<unitree_go::msg::LowCmd>::SharedPtr pub_lowcmd_;
  rclcpp::Publisher<hq_pcot_msgs::msg::LoopStatus>::SharedPtr pub_status_;
  rclcpp::Subscription<unitree_go::msg::LowState>::SharedPtr sub_lowstate_;
  rclcpp::Subscription<hq_pcot_msgs::msg::LocomotionCmd>::SharedPtr sub_cmd_;
  rclcpp::Subscription<hq_pcot_msgs::msg::LoopStatus>::SharedPtr sub_standing_;
  rclcpp::TimerBase::SharedPtr timer_;
  rclcpp::TimerBase::SharedPtr status_timer_;

  Ort::Env env_;
  std::unique_ptr<Ort::Session> session_;
  std::string input_name_;
  std::vector<std::string> output_names_;
  std::vector<const char*> output_name_ptrs_;
  std::unordered_map<std::string, std::vector<float>> recurrent_inputs_;
  std::unordered_map<std::string, std::vector<int64_t>> recurrent_shapes_;

  std::string policy_dir_;
  std::string lowstate_topic_;
  std::string locomotion_cmd_topic_;
  std::string lowcmd_topic_;
  std::string status_topic_;

  double control_hz_{50.0};
  double control_period_s_{0.02};
  double loop_budget_ms_{20.0};
  double status_hz_{10.0};
  double cmd_timeout_s_{0.5};
  double lowstate_timeout_s_{0.1};
  int loop_stats_window_{500};

  std::vector<int> joint_ids_map_;
  std::vector<float> action_scale_;
  std::vector<float> action_offset_;
  std::optional<std::pair<float, float>> action_clip_;
  std::vector<float> default_joint_pos_;
  std::vector<float> joint_stiffness_;
  std::vector<float> joint_damping_;
  std::vector<float> last_raw_action_;
  std::size_t action_dim_{0};
  std::pair<float, float> cmd_lin_x_{-0.5F, 1.0F};
  std::pair<float, float> cmd_lin_y_{-0.5F, 0.5F};
  std::pair<float, float> cmd_ang_z_{-1.0F, 1.0F};
  std::vector<ObservationTerm> obs_terms_;

  unitree_go::msg::LowState::SharedPtr last_lowstate_;
  hq_pcot_msgs::msg::LocomotionCmd::SharedPtr last_locomotion_cmd_;
  std::optional<std::int64_t> last_lowstate_time_ns_;
  std::optional<std::int64_t> last_cmd_time_ns_;
  bool standing_ready_{false};
  bool wait_logged_{false};
  bool lowstate_wait_logged_{false};
  bool lowstate_stale_logged_{false};
  bool ready_sent_{false};
  bool running_logged_{false};
  bool motor_state_wait_logged_{false};
  int status_code_{-1};
  std::deque<float> loop_times_ms_;
  std::deque<int> deadline_flags_;
};

}  // namespace locomotion_controller_cpp

int main(int argc, char* argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<locomotion_controller_cpp::PolicyControllerNode>());
  if (rclcpp::ok())
  {
    rclcpp::shutdown();
  }
  return 0;
}
