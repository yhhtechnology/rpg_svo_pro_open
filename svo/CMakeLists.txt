project(svo)
cmake_minimum_required(VERSION 2.8.3)

find_package(catkin_simple)
catkin_simple()

include(SvoSetup)

add_subdirectory(Thirdparty/g2o)
list(APPEND SVO_SOURCEFILES
  src/frame_handler_mono.cpp
  src/frame_handler_stereo.cpp
  src/frame_handler_array.cpp
  src/frame_handler_base.cpp
  src/map.cpp
  src/pose_optimizer.cpp
  src/initialization.cpp
  src/reprojector.cpp
  src/imu_handler.cpp
  src/stereo_triangulation.cpp
  src/io.cpp
  # # src/G2oTypes.cc
  src/g2o_ba.cpp
)

list(APPEND SVO_HEADERFILES
  include/svo/abstract_bundle_adjustment.h
  include/svo/frame_handler_base.h
  include/svo/frame_handler_mono.h
  include/svo/frame_handler_stereo.h
  include/svo/frame_handler_array.h
  include/svo/global.h
  include/svo/imu_handler.h
  include/svo/initialization.h
  include/svo/map.h
  include/svo/pose_optimizer.h
  include/svo/reprojector.h
  include/svo/stereo_triangulation.h
  include/svo/io.h
  include/svo/svo.h
  # # include/svo/G2oTypes.h
  include/svo/g2o_ba.h
  )



################################################################################
# Create svo library
include_directories(include ${INCLUDE_DIRS})
cs_add_library(svo SHARED ${SVO_SOURCEFILES} ${SVO_HEADERFILES})
target_link_libraries(svo ${LINK_LIBS}
${PROJECT_SOURCE_DIR}/Thirdparty/g2o/lib/libg2o.so
)

################################################################################
###
### GTEST
###

catkin_add_gtest(${PROJECT_NAME}-test-frame test/test_frame.cpp)
target_link_libraries(${PROJECT_NAME}-test-frame ${PROJECT_NAME})
# target_link_libraries(${PROJECT_NAME}-test-frame ${PROJECT_NAME})

################################################################################
cs_install()
cs_export()
