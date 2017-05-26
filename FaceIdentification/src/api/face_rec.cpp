/*
*
* This file is part of the open-source SeetaFace engine, which includes three modules:
* SeetaFace Detection, SeetaFace Alignment, and SeetaFace Identification.
*
* This file is part of the SeetaFace Identification module, containing codes implementing the
* face identification method described in the following paper:
*
*
*   VIPLFaceNet: An Open Source Deep Face Recognition SDK,
*   Xin Liu, Meina Kan, Wanglong Wu, Shiguang Shan, Xilin Chen.
*   In Frontiers of Computer Science.
*
*
* Copyright (C) 2016, Visual Information Processing and Learning (VIPL) group,
* Institute of Computing Technology, Chinese Academy of Sciences, Beijing, China.
*
* The codes are mainly developed by Jie Zhang(a Ph.D supervised by Prof. Shiguang Shan)
*
* As an open-source face recognition engine: you can redistribute SeetaFace source codes
* and/or modify it under the terms of the BSD 2-Clause License.
*
* You should have received a copy of the BSD 2-Clause License along with the software.
* If not, see < https://opensource.org/licenses/BSD-2-Clause>.
*
* Contact Info: you can send an email to SeetaFace@vipl.ict.ac.cn for any problems.
*
* Note: the above information must be kept whenever or wherever the codes are used.
*
*/
#include "face_rec.h"
#include <stdlib.h>
#include <pthread.h>
#include <time.h>
#include <sys/time.h>
#include <errno.h>
#include <stdio.h>
#include <unistd.h>


#define TEST(major, minor) major##_##minor##_Tester()
#define EXPECT_NE(a, b) if ((a) == (b)) std::cout << "ERROR: "
#define EXPECT_EQ(a, b) if ((a) != (b)) std::cout << "ERROR: "

#ifdef _WIN32
std::string DATA_DIR = "../../data/";
std::string MODEL_DIR = "../../model/";
#else
std::string DATA_DIR = "../data/";
std::string MODEL_DIR = "../model/";
#endif
using namespace std;
using namespace seeta;

static pthread_t thread;
static int thread_run = 0;
static pthread_mutex_t mutex;
static pthread_cond_t cond;

typedef enum { /* bitmapped status flags */
    FACE_REC_STEP_INIT,
    FACE_REC_STEP_EXTR,
    FACE_REC_STEP_DECT,    
    FACE_REC_STEP_COMP,
    FACE_REC_STEP_OVER
} Face_Rec_Step_EM;

struct Face_Rec_Imp_ST{
	int ChannelNum;
	ImageData img_data_color;
	ImageData img_data_gray;
	float* img_fea_result;
	float* img_fea_para1;
	float* img_fea_para2;
	Face_Rec_Extract_cb_t callback_function1;
	Face_Rec_Detect_cb_t callback_function2;
	Face_Rec_Step_EM Step;
	Face_Rec_Imp_ST()
	{
		ChannelNum=0;
		img_fea_result=NULL;
		img_fea_para1=NULL;
		img_fea_para2=NULL;
		callback_function1=NULL;
        callback_function2=NULL;	
		Step=FACE_REC_STEP_INIT;	
	}  
};

static FaceDetection *detector=NULL;
static FaceAlignment *point_detector=NULL;
static FaceIdentification *face_recognizer=NULL;
//std::string test_dir = DATA_DIR + "test_face_recognizer/";	
static Face_Rec_Imp_ST MAIN_ST[Face_Rec_Pthread_MAX_NUM];
static int Face_Rec_ACT_NUM=0;
static int Limit_Count=0;



int Set_Face_Rec_Flag(int ChanNum,int opera);

static float simd(const float* x, const float* y, const long& len) {

  float inner_prod = 0.0f;
  //#ifdef _WIN32
  //float op[4] = {0, 0, 0, 0};  
  __m128 X, Y; // 128-bit values
  __m128 acc = _mm_setzero_ps(); // set to (0, 0, 0, 0)
 // __m128 acc = _mm_loadu_ps(op);  
  float temp[4];

  long i;
  for (i = 0; i + 4 < len; i += 4) {
      X = _mm_loadu_ps(x + i); // load chunk of 4 floats
      Y = _mm_loadu_ps(y + i);
      acc = _mm_add_ps(acc, _mm_mul_ps(X, Y));
  }
  _mm_storeu_ps(&temp[0], acc); // store acc into an array
  inner_prod = temp[0] + temp[1] + temp[2] + temp[3];

  // add the remaining values
  for (; i < len; ++i) {
      inner_prod += x[i] * y[i];
  }
   // #endif
  return inner_prod;

}


static void *timer_thread(void *arg)
{
//  PRINT_INFO("%s - is called enter .....\n",__FUNCTION__);
    Face_Rec_Step_EM steps=FACE_REC_STEP_INIT; 
    int state=0;
    ImageData imgdata_color;
    ImageData imgdata_gray;
    float* imgfea_result=NULL;
    float* imgfea_para1=NULL;
    float* imgfea_para2=NULL;
    Face_Rec_Extract_cb_t callback_func1=NULL;
    Face_Rec_Detect_cb_t callback_func2=NULL;
    int Face_Rec_Imp_Count=0;

    while(thread_run) {
        pthread_mutex_lock(&mutex);
        if(MAIN_ST[Face_Rec_Imp_Count].Step==FACE_REC_STEP_EXTR) 
        {
            steps=MAIN_ST[Face_Rec_Imp_Count].Step;	
            imgdata_color=MAIN_ST[Face_Rec_Imp_Count].img_data_color;
            imgdata_gray=MAIN_ST[Face_Rec_Imp_Count].img_data_gray;
            imgfea_result=MAIN_ST[Face_Rec_Imp_Count].img_fea_result;
            callback_func1=MAIN_ST[Face_Rec_Imp_Count].callback_function1;
        }
        else if(MAIN_ST[Face_Rec_Imp_Count].Step==FACE_REC_STEP_DECT) 
        {
            steps=MAIN_ST[Face_Rec_Imp_Count].Step;
            imgdata_color=MAIN_ST[Face_Rec_Imp_Count].img_data_color;
            imgdata_gray=MAIN_ST[Face_Rec_Imp_Count].img_data_gray;
            callback_func2=MAIN_ST[Face_Rec_Imp_Count].callback_function2;
        }
        pthread_mutex_unlock(&mutex);
        	
        if((steps==FACE_REC_STEP_EXTR)||(steps==FACE_REC_STEP_DECT))
        {
            std::vector<seeta::FaceInfo> gallery_faces = detector->Detect(imgdata_gray);
            int32_t gallery_face_num = static_cast<int32_t>(gallery_faces.size());
            if (gallery_face_num == 0)
            {
            	//std::cout << "Faces are not detected."<<endl;
            	state=-1;
            }	

            if(steps==FACE_REC_STEP_EXTR)
            {
                // Detect 5 facial landmarks
                seeta::FacialLandmark gallery_points[5];
                point_detector->PointDetectLandmarks(imgdata_gray, gallery_faces[0], gallery_points);	

                face_recognizer->ExtractFeatureWithCrop(imgdata_color, gallery_points, imgfea_result);	
            }
			
			pthread_mutex_lock(&mutex);
			MAIN_ST[Face_Rec_Imp_Count].Step=FACE_REC_STEP_OVER;
			pthread_mutex_unlock(&mutex);

            if((callback_func1!=NULL)&&(steps==FACE_REC_STEP_EXTR))
            {
                callback_func1(state,gallery_face_num,imgfea_result);
            }			
            else if((callback_func2!=NULL)&&(steps==FACE_REC_STEP_DECT))
            {
                callback_func2(state,gallery_face_num,gallery_faces);
            }
            steps=FACE_REC_STEP_OVER;
        }

        pthread_mutex_lock(&mutex);
        if(Face_Rec_Imp_Count>=Face_Rec_ACT_NUM)
            Face_Rec_Imp_Count=0;
        else
            Face_Rec_Imp_Count++;
        pthread_mutex_unlock(&mutex);
        
        sleep(1);
    }
    return NULL;
}


    
int Face_Rec_Init(int ChannelNum,char *path)
{
    if(thread_run == 0) {
        pthread_mutex_init(&mutex, NULL);
        pthread_cond_init(&cond, NULL);
        thread_run = 1; /* must set to 1 before thread creation */

    	string alignment_path;
    	string detector_path;
    	string recognizer_path;		
    	if(path!=NULL)
    	{
    		alignment_path=path;
    		detector_path=path;
    		recognizer_path=path;
    		alignment_path+="fa.bin";
    		detector_path+="fd.bin";
    		recognizer_path+="fr.bin";	
    	}
    	else
    	{
    		alignment_path="fa.bin";
    		detector_path="fd.bin";
    		recognizer_path="fr.bin";		
    	}		

        if(point_detector==NULL)
        {
    	    point_detector=new FaceAlignment((char *)alignment_path.c_str());
        }
        if(detector==NULL)
        {
    	    detector=new FaceDetection((char *)detector_path.c_str());
        	detector->SetMinFaceSize(40);
        	detector->SetScoreThresh(2.f);
        	detector->SetImagePyramidScaleFactor(0.8f);
        	detector->SetWindowStep(4, 4);
        }
        if(face_recognizer==NULL)
        {
            face_recognizer=new FaceIdentification((char *)recognizer_path.c_str());
        }
       
        if(pthread_create(&thread, NULL, timer_thread, NULL) != 0) {
          std::cout << "Thread creation failed"<<endl;
          thread_run = 0;
          return -1;
        }
    }    
	
	
    pthread_mutex_lock(&mutex);
    memset(MAIN_ST,0,sizeof(MAIN_ST));
    if(ChannelNum>Face_Rec_Pthread_MAX_NUM)
    {
        pthread_mutex_unlock(&mutex);
        return -2;
    }
    Face_Rec_ACT_NUM=ChannelNum;
    pthread_mutex_unlock(&mutex);
    return 0;
}

int Face_Rec_Extract(int ChannelID,ImageData img_data_color,ImageData img_data_gray,float* img_fea,Face_Rec_Extract_cb_t callback_function)
{
    int ret=0;
	if(Limit_Count>100)
		return -1;
	else
		Limit_Count++;	
	
    pthread_mutex_lock(&mutex);
    if(ChannelID>=Face_Rec_ACT_NUM)
        ret=-2;
    if((MAIN_ST[ChannelID].Step==FACE_REC_STEP_EXTR)||(MAIN_ST[ChannelID].Step==FACE_REC_STEP_DECT))
        ret=-1;
    if(ret==0)
    {
        MAIN_ST[ChannelID].Step=FACE_REC_STEP_EXTR;
        MAIN_ST[ChannelID].img_data_color=img_data_color;
        MAIN_ST[ChannelID].img_data_gray=img_data_gray;
        MAIN_ST[ChannelID].img_fea_result=img_fea;
        MAIN_ST[ChannelID].callback_function1=callback_function;
    }
    pthread_mutex_unlock(&mutex);
    return ret;
}


int Face_Rec_Detect(int ChannelID,ImageData img_data_color,ImageData img_data_gray,Face_Rec_Detect_cb_t callback_function)
{
    int ret=0;
    pthread_mutex_lock(&mutex);
    if(ChannelID>=Face_Rec_ACT_NUM)
        ret=-2;
    if((MAIN_ST[ChannelID].Step==FACE_REC_STEP_EXTR)||(MAIN_ST[ChannelID].Step==FACE_REC_STEP_DECT))
        ret=-1;
    if(ret==0)
    {    
        MAIN_ST[ChannelID].Step=FACE_REC_STEP_DECT;
        MAIN_ST[ChannelID].img_data_color=img_data_color;
        MAIN_ST[ChannelID].img_data_gray=img_data_gray;
        MAIN_ST[ChannelID].callback_function2=callback_function;
    }
    pthread_mutex_unlock(&mutex);
    return ret;
}



float Face_Rec_Compare(float * img1_fea,float * img2_fea)
{
    long dim=2048;
    return simd(img1_fea, img2_fea, dim)	  
    /(sqrt(simd(img1_fea, img1_fea, dim))      
    * sqrt(simd(img2_fea, img2_fea, dim))); 
}

int Face_Rec_Deinit()
{
    pthread_mutex_lock(&mutex);
    memset(MAIN_ST,0,sizeof(MAIN_ST));
    Face_Rec_ACT_NUM=0;
    pthread_mutex_unlock(&mutex);

    
    thread_run = 0;
    pthread_join(thread, NULL);
	delete point_detector;
	point_detector=NULL;
	delete detector;
	detector=NULL;
	delete face_recognizer;
	face_recognizer=NULL;
}

