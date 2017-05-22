#将之前生成的libseeta_facedet_lib,libseeta_fa_lib复制到build文件夹中（1）
#`src/test/CMakeLists.txt`修改之后如下
aux_source_directory (. SRC_LIST)
message(${SRC_LIST})
link_directories(${PROJECT_BINARY_DIR})
#link_directories("~/SeetaFaceEngine-master/FaceIdentification/build")
#link_directories("~/facerecognize")
# add external libraries
#find_package(OpenCV REQUIRED)

#enable_testing ()
#foreach (f ${SRC_LIST})
  #add_executable(demo demo.cpp)
  #add_executable(test_face_verification test_face_verification.cpp)
  #ADD_LIBRARY (face_rec SHARED face_rec.cpp)
  #target_link_libraries(test_face_verification viplnet ${OpenCV_LIBS} facedet_lib fa_lib) 
  #target_link_libraries(face_rec viplnet faced_lib facea_lib) 
  #target_link_libraries(demo viplnet ${OpenCV_LIBS} face_rec facedet_lib fa_lib) 
#endforeach ()
