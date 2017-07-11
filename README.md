FaceDAI Library API Usage
Definition of details can be referred at face_rec.h


int Face_Rec_Init(int ChannelNum,char *path)
//Function: Initialize the face detection/recognize module
//Param : 
//  ChannelNum: the max of thread
//  path: ANN binary path (can be omitted)
//Return Value:
//  0: Noraml -1: Thread Create Failed, -2: Thread Number exceed the max of thread, -3: Double Initiate



int Face_Rec_Extract(int ChannelID,ImageData img_data_color,ImageData img_data_gray,float* img_fea,Face_Rec_Extract_cb_t callback_function)
//Function: Recognize face from picture
//Param : 
//  ChannelID: ID of the thread,
//  img_data_color: Original Image,
//  img_data_gray: Gray Image,
//  callback_function: Callback when complete detect

//Return Value:
//  0: Noraml, -1: Module Busy, -2: Thread Number exceed the max of thread, -3: Face Not Detected, -4: Input Paramater Null



int Face_Rec_Detect(int ChannelID,ImageData img_data_color,ImageData img_data_gray,void * res_faces, Face_Rec_Detect_cb_t callback_function)
//Function: Detect face from picture
//Param : 
//  ChannelID: ID of the thread,
//  img_data_color: Original Image,
//  img_data_gray: Gray Image,
//  callback_function: Callback when complete detect
//Return Value:
//  0: Noraml, -1: Module Busy, -2: Thread Number exceed the max of thread, -3: Face Not Detected, -4: Input Param is Null


float Face_Rec_Compare(float * img1_fea,float * img2_fea)
//Function: Compare two face feature value
//Param : 
//  img1_fea: image 1 feature value
//  img2_fea: image 2 feature value
//Return Value:
//  simularity of two faces


int Face_Rec_Deinit()
//Function: Deinitialize the face detection/recognize module
//Param : 
//Return Value:
//  0: Noraml
