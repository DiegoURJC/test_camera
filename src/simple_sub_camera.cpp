#include "test_camera/simple_sub_camera.hpp"


const std::string SimpleSubCamera::RGB_WINDOW = "RGB";
const std::string SimpleSubCamera::IR_WINDOW = "IR";
const std::string SimpleSubCamera::NDVI_WINDOW = "NDVI";
const int32_t SimpleSubCamera::MAX_FEATURES = 7000;
const float SimpleSubCamera::GOOD_MATCH_PERCENT = 0.10f;



SimpleSubCamera::~SimpleSubCamera()
{
  cv::destroyWindow(RGB_WINDOW);
  cv::destroyWindow(IR_WINDOW);
  cv::destroyWindow(NDVI_WINDOW);
}

void SimpleSubCamera::applyNDVI(cv::Mat &rgb_im, cv::Mat &ir_im, cv::Mat &ndvi_im)
{
  std::vector<cv::Mat> rgb_channels;
  std::vector<cv::Mat> ndvi_channels;

  cv::split(rgb_im, rgb_channels);
  cv::split(ndvi_im, ndvi_channels);

  double NIR, RED, NDVI;

  for(int i = 0; i < rgb_im.rows; ++i){
    for(int j = 0; j < rgb_im.cols; ++j){
      NIR = ir_im.at<uchar>(i, j);
      RED = rgb_channels[2].at<uchar>(i, j);

      NDVI = (NIR - RED) / (NIR + RED);

      if(NDVI <= 0){
        ndvi_channels[0].at<uchar>(i,j) = 0;
        ndvi_channels[1].at<uchar>(i, j) = 0;
        ndvi_channels[2].at<uchar>(i, j) = 255;
      }
      else if((NDVI > 0) && (NDVI <= 0.25)){
        ndvi_channels[0].at<uchar>(i,j) = 0;
        ndvi_channels[1].at<uchar>(i, j) = 128;
        ndvi_channels[2].at<uchar>(i, j) = 255;
      }
      else if((NDVI > 0.25) && (NDVI <= 0.5)){
        ndvi_channels[0].at<uchar>(i,j) = 0;
        ndvi_channels[1].at<uchar>(i, j) = 255;
        ndvi_channels[2].at<uchar>(i, j) = 255;
      }
      else if((NDVI > 0.5) && (NDVI <= 0.75)){
        ndvi_channels[0].at<uchar>(i,j) = 0;
        ndvi_channels[1].at<uchar>(i, j) = 255;
        ndvi_channels[2].at<uchar>(i, j) = 0;
      }
      else{
        ndvi_channels[0].at<uchar>(i,j) = 0;
        ndvi_channels[1].at<uchar>(i, j) = 128;
        ndvi_channels[2].at<uchar>(i, j) = 0;
      }
    }
  }

  cv::merge(ndvi_channels, ndvi_im);
}

void SimpleSubCamera::applyNDVIGrayScale(cv::Mat &rgb_im, cv::Mat &ir_im, cv::Mat &ndvi_im)
{
  std::vector<cv::Mat> rgb_channels;
  std::vector<cv::Mat> ndvi_channels;

  cv::split(rgb_im, rgb_channels);
  cv::split(ndvi_im, ndvi_channels);

  double NIR, RED, NDVI;

  for(int i = 0; i < rgb_im.rows; ++i){
    for(int j = 0; j < rgb_im.cols; ++j){
      NIR = ir_im.at<uchar>(i, j);
      RED = rgb_channels[2].at<uchar>(i, j);

      NDVI = (NIR - RED) / (NIR + RED);

      if(NDVI <= 0)
      {
        ndvi_channels[0].at<uchar>(i,j) = 0;
      }
      else{
        ndvi_channels[0].at<uchar>(i,j) = NDVI * 255;
      }
    }
  }

  cv::merge(ndvi_channels, ndvi_im);
}


void SimpleSubCamera::alignImages(cv::Mat &ir_image, cv::Mat &rgb_image, cv::Mat &im_registered)
{

  cv::Mat rgb_image_gray;
  cv::Mat h;

//   cvtColor(im1, im1Gray, cv::COLOR_BGR2GRAY);
  cvtColor(rgb_image, rgb_image_gray, cv::COLOR_BGR2GRAY);
 
  // Variables to store keypoints and descriptors
  std::vector<cv::KeyPoint> keypoints1, keypoints2;
  cv::Mat descriptors1, descriptors2;
 
  // Detect ORB features and compute descriptors.
  cv::Ptr<cv::Feature2D> orb = cv::ORB::create(MAX_FEATURES);
  orb->detectAndCompute(ir_image, cv::Mat(), keypoints1, descriptors1);
  orb->detectAndCompute(rgb_image_gray, cv::Mat(), keypoints2, descriptors2);
 
 
  // Match features.
  std::vector<cv::DMatch> matches;
  cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create("BruteForce-Hamming");
  matcher->match(descriptors1, descriptors2, matches, cv::Mat());
 
  // Sort matches by score
  std::sort(matches.begin(), matches.end());
 
  // Remove not so good matches
  const int numGoodMatches = matches.size() * GOOD_MATCH_PERCENT;
  matches.erase(matches.begin()+numGoodMatches, matches.end());
 
  // Draw top matches
  cv::Mat imMatches;
  drawMatches(ir_image, keypoints1, rgb_image, keypoints2, matches, imMatches);
 
  // Extract location of good matches
  std::vector<cv::Point2f> points1, points2;
 
  for( size_t i = 0; i < matches.size(); i++ )
  {
    points1.push_back( keypoints1[ matches[i].queryIdx ].pt );
    points2.push_back( keypoints2[ matches[i].trainIdx ].pt );
  }
 
  // Find homography
  h = cv::findHomography( points1, points2, cv::RANSAC );
 
  // Use homography to warp image
  cv::warpPerspective(ir_image, im_registered, h, rgb_image.size());
}
 

void SimpleSubCamera::generateNDVIimage()
{
  cv::namedWindow(RGB_WINDOW);
  cv::namedWindow(IR_WINDOW);
  cv::namedWindow(NDVI_WINDOW);

  cv::Mat rgb_image = m_rgbImagePtr->image;
  cv::Mat ir_image = m_irImagePtr->image;
  cv::Mat ir_registered;

  m_calculating = true;

  alignImages(ir_image, rgb_image, ir_registered);

  cv::Mat ndvi_image(rgb_image.size(), CV_8UC3, cv::Scalar(0,0,0));
  // cv::Mat ndvi_grayscale(rgb_image.size(), CV_8UC1, cv::Scalar(0));

  applyNDVI(rgb_image, ir_registered, ndvi_image);
  // applyNDVIGrayScale(rgb_image, ir_registered, ndvi_grayscale);

  cv::cvtColor(rgb_image, rgb_image, cv::COLOR_BGR2RGB);

  cv::imshow(RGB_WINDOW, rgb_image);
  cv::imshow(IR_WINDOW, ir_registered);
  cv::imshow(NDVI_WINDOW, ndvi_image);
  // cv::imshow("NDVI GRAYSCALE", ndvi_grayscale);

  cv::waitKey(3);

  m_rgbImagePtr = nullptr;
  m_irImagePtr = nullptr;
  m_calculating = false;
}


int main(int argc, char *argv[])
{
  rclcpp::init(argc, argv);

  rclcpp::spin(std::make_shared<SimpleSubCamera>());

  rclcpp::shutdown();

  return 0;
}
