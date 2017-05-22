#include<iostream>

#ifdef MEIYOU   //_WIN32
#pragma once
#include <opencv2/core/version.hpp>
#include <opencv/cv.h>
#include <opencv/highgui.h>
#define CV_VERSION_ID CVAUX_STR(CV_MAJOR_VERSION) CVAUX_STR(CV_MINOR_VERSION) \
  CVAUX_STR(CV_SUBMINOR_VERSION)
#ifdef _DEBUG
#define cvLIB(name) "opencv_" name CV_VERSION_ID "d"
#else
#define cvLIB(name) "opencv_" name CV_VERSION_ID
#endif //_DEBUG
#pragma comment( lib, cvLIB("core") )
#pragma comment( lib, cvLIB("imgproc") )
#pragma comment( lib, cvLIB("highgui") )
#endif //_WIN32

#if defined(__unix__) || defined(__APPLE__)
#ifndef fopen_s
#define fopen_s(pFile,filename,mode) ((*(pFile))=fopen((filename),(mode)))==NULL
#endif //fopen_s
#endif //__unix

#include "face_identification.h"
#include "recognizer.h"
#include "face_detection.h"
#include "face_alignment.h"
#include "math_functions.h"

using namespace seeta;

void Face_Rec_Init(char *path = NULL);		//不添加路径时将bin文件和生成的可执行文件放在一块, QT下建议填写bin文件所在的绝对路径
int Face_Rec_Extract(ImageData img_data_color,ImageData img_data_gray,float* img_fea);			//返回-1 表示没有找到人脸
float Face_Rec_Compare(float * img1_fea,float * img2_fea);
void Face_Rec_Deinit();