#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <string>

#include <pinocchio/algorithm/joint-configuration.hpp>
#include <pinocchio/algorithm/kinematics.hpp>
#include <pinocchio/parsers/urdf.hpp>
#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/jacobian.hpp>
#include <pinocchio/algorithm/kinematics.hpp>
#include <pinocchio/algorithm/rnea.hpp>
#include <pinocchio/multibody/fwd.hpp>
#include <pinocchio/spatial/fwd.hpp>

#include <fmt/format.h>

struct Parameters
{
    std::string urdf_path;
    std::string tip_frame;
};

class JointStateListener : public rclcpp::Node
{
public:
    JointStateListener()
    : Node("joint_state_listener")
    {
        // Declare and retrieve the 'robot_description' parameter
        this->declare_parameter<std::string>("urdf_path", "");
        this->get_parameter("urdf_path", m_parameters.urdf_path);
        RCLCPP_INFO(this->get_logger(), "urdf_path parameter loaded: %s", m_parameters.urdf_path);

        this->declare_parameter<std::string>("tip_frame", "");
        this->get_parameter("tip_frame", m_parameters.tip_frame);
        RCLCPP_INFO(this->get_logger(), "tip_frame parameter loaded: %s", m_parameters.tip_frame);

        // set up robot model
        pinocchio::urdf::buildModel(m_parameters.urdf_path, m_robot_model, false);
        m_robot_data = pinocchio::Data(m_robot_model);

        // Subscription to joint_states
        m_subscription = this->create_subscription<sensor_msgs::msg::JointState>(
            "/joint_states", 10,
            std::bind(&JointStateListener::jointStateCallback, this, std::placeholders::_1));
    }

private:
    void jointStateCallback(const sensor_msgs::msg::JointState::SharedPtr msg)
    {
        RCLCPP_INFO(this->get_logger(), "Received joint state:");

        Eigen::VectorXd q = Eigen::VectorXd::Zero(m_robot_model.nq);
        Eigen::VectorXd v = Eigen::VectorXd::Zero(m_robot_model.nv);
        Eigen::VectorXd a = Eigen::VectorXd::Zero(
        m_robot_model.nv);  // acceleration is of size nv according to pinocchio docs

        // APPLY JOINT INFORMATION TO PINOCCHIO FOR KINEMATICS AND DYNAMICS CALCULATIONS
        for (size_t i = 0; i < msg->name.size(); ++i) {
            RCLCPP_INFO(this->get_logger(), "  %s: %.2f", msg->name[i].c_str(), msg->position[i]);

            const auto& joint_name = msg->name.at(i);
            const auto& joint_position = msg->position.at(i);
            // const auto& joint_velocity = msg->velocity.at(i);

            pinocchio::JointIndex joint_idx = m_robot_model.getJointId(joint_name);

            // if can't find name
            if (joint_idx == m_robot_model.joints.size()) {
                throw std::runtime_error(
                fmt::format("couldn't find joint name: {} in model\n", joint_name));
            }

            // if number of params per joint is unexpected
            if (m_robot_model.joints[joint_idx].nq() != 1) {
                RCLCPP_INFO_STREAM(
                get_logger(),
                fmt::format(
                    "Joint {} has {} DoFs, skipping\n", joint_name, m_robot_model.joints[joint_idx].nq()));
                continue;
            }

            // fill up q and v
            auto joint_model = m_robot_model.joints[joint_idx];
            q[joint_model.idx_q()] = joint_position;
            // v[joint_model.idx_v()] = joint_velocity;
        }

    
        pinocchio::forwardKinematics(m_robot_model, m_robot_data, q, v);
        pinocchio::updateFramePlacements(m_robot_model, m_robot_data);
        pinocchio::computeJointJacobians(m_robot_model, m_robot_data, q);

        // GET TIP FRAME INFORMATION AND JACOBIAN
        auto frame_id = m_robot_model.getFrameId(m_parameters.tip_frame);
        const pinocchio::SE3 & tip_frame = m_robot_data.oMf[frame_id];
        auto tip_jacobian = pinocchio::getFrameJacobian(
        m_robot_model, m_robot_data, frame_id, pinocchio::ReferenceFrame::LOCAL_WORLD_ALIGNED);

        Eigen::IOFormat clean_format(3, 0, ", ", "\n", "[", "]", "[", "]");

        std::ostringstream oss;
        oss.setf(std::ios::fixed);         // force fixed-point format
        oss.precision(3);                  // set number of decimals
        oss << tip_jacobian.format(clean_format);
        std::string jac_str = oss.str();

        RCLCPP_INFO(this->get_logger(), "jacobian: \n%s", jac_str.c_str());

    }

    rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr m_subscription;
    pinocchio::Model m_robot_model;
    pinocchio::Data m_robot_data;
    Parameters m_parameters;
};

int main(int argc, char * argv[])
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<JointStateListener>());
    rclcpp::shutdown();
    return 0;
}
