cmake_minimum_required(VERSION 2.8) 
PROJECT (traffic_sign_classifier)

find_package(OpenCV REQUIRED)

add_executable(traffic_sign_classifier main.cpp)
target_link_libraries(traffic_sign_classifier ${OpenCV_LIBS})