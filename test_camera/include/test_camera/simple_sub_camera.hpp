#pragma once

#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "cv_bridge/cv_bridge.h"

#include <opencv2/highgui.hpp>
#include "opencv2/xfeatures2d.hpp"
#include "opencv2/features2d.hpp"
#include <opencv2/calib3d.hpp>


using std::placeholders::_1;

class SimpleSubCamera : public rclcpp::Node
{
public:
  SimpleSubCamera()
  : Node("simple_sub_camera"), 
  m_rgbImagePtr(nullptr), 
  m_irImagePtr(nullptr)
  {
    m_RGBSubscriber = create_subscription<sensor_msgs::msg::Image>
    ("/camera/color/image_raw", 30, std::bind(&SimpleSubCamera::rgb_callback, this, _1));

    m_IRSubscriber = create_subscription<sensor_msgs::msg::Image>
    ("/camera/infra1/image_rect_raw", 30, std::bind(&SimpleSubCamera::ir_callback, this, _1));
  }

  ~SimpleSubCamera();

private:
  void applyNDVI(cv::Mat &rgb_im, cv::Mat &ir_im, cv::Mat &ndvi_im);
  void alignImages(cv::Mat &ir_image, cv::Mat &rgb_image, cv::Mat &im_registered);


  void generateNDVIimage();

  inline void ir_callback(const sensor_msgs::msg::Image::SharedPtr msg)
  {
    try
    {
      m_irImagePtr = cv_bridge::toCvCopy(msg, msg->encoding);
    }
    catch (cv_bridge::Exception &e)
    {
      RCLCPP_ERROR(get_logger(), "cv_bridge exception %s", e.what());
    }

    if((m_irImagePtr != nullptr) && (m_rgbImagePtr != nullptr)){
      generateNDVIimage();
    }
  }

  inline void rgb_callback(const sensor_msgs::msg::Image::SharedPtr msg)
  {
    try
    {
      m_rgbImagePtr = cv_bridge::toCvCopy(msg, msg->encoding);
    }
    catch (cv_bridge::Exception &e)
    {
      RCLCPP_ERROR(get_logger(), "cv_bridge exception %s", e.what());
    }

    if((m_irImagePtr != nullptr) && (m_rgbImagePtr != nullptr)){
      generateNDVIimage();
    }
  }

  cv_bridge::CvImagePtr m_rgbImagePtr;
  cv_bridge::CvImagePtr m_irImagePtr;

  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr m_RGBSubscriber;
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr m_IRSubscriber;

  static const std::string RGB_WINDOW;
  static const std::string IR_WINDOW;
  static const std::string NDVI_WINDOW;

  static const int32_t MAX_FEATURES;

  static const float GOOD_MATCH_PERCENT;
};