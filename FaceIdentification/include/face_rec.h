
#include <iostream>
#include "face_identification.h"
#include "recognizer.h"
#include "face_detection.h"
#include "face_alignment.h"
#include "math_functions.h"

using namespace seeta;


#define Face_Rec_Pthread_MAX_NUM    100

typedef void(*Face_Rec_Extract_cb_t)(int state,int FaceNum,float* img_fea);
typedef void(*Face_Rec_Detect_cb_t)(int state,int FaceNum,std::vector<seeta::FaceInfo> face_info);


int Face_Rec_Init(int ChannelNum,char *path = NULL);
int Face_Rec_Detect(int ChannelID,ImageData img_data_color,ImageData img_data_gray,Face_Rec_Detect_cb_t callback_function);			
int Face_Rec_Extract(int ChannelID,ImageData img_data_color,ImageData img_data_gray,float* img_fea,Face_Rec_Extract_cb_t callback_function);			
float Face_Rec_Compare(float * img1_fea,float * img2_fea);
int Face_Rec_Deinit();
// return value  -2:id error    -1: other error

