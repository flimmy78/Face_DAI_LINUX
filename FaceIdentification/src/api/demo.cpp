#include "face_rec.h"
#include <iostream>
#include <fstream>
#include <sstream>

#include "opencv2/core/version.hpp"
#include "opencv/cv.h"
#include "opencv/highgui.h"

using namespace std;
using namespace seeta;
using namespace cv;

float* gallery_fea = NULL;
float* probe_fea = NULL;

ImageData gallery_src_data_color;
ImageData gallery_src_data_gray;
ImageData gallery_dst_data_color;
ImageData gallery_dst_data_gray;


static void Face_Rec_Extract_callback1(int state,int FaceNum,float* img_fea);
static void Face_Rec_Extract_callback2(int state,int FaceNum,float* img_fea);
static void Face_Rec_Extract_callback3(int state,int FaceNum,std::vector<seeta::FaceInfo> face);
static void Face_Rec_Extract_callback4(int state,int FaceNum,std::vector<seeta::FaceInfo> face);

void Face_Rec_Extract_callback1(int state,int FaceNum,float* img_fea)
{
	if((state==0)&&(img_fea==gallery_fea))
	{
		std::cout << "picture 1 face num is:"<<FaceNum <<endl;
	}
}
void Face_Rec_Extract_callback2(int state,int FaceNum,float* img_fea)
{
	if((state==0)&&(img_fea==probe_fea))
	{
		std::cout << "picture 2 face num is:"<<FaceNum <<endl;
	}
}
void Face_Rec_Extract_callback3(int state,int FaceNum,std::vector<seeta::FaceInfo> face)
{
	if(state==0)
	{
		vector<seeta::FaceInfo>::iterator it=face.begin();
		std::cout << "picture 1 detct score is: "<<(*it).score<<endl;
	}
}
void Face_Rec_Extract_callback4(int state,int FaceNum,std::vector<seeta::FaceInfo> face)
{
	if(state==0)
	{
		vector<seeta::FaceInfo>::iterator it=face.begin();
		std::cout << "picture 2 detect score is:"<<(*it).score<<endl;		
	}	
}
int main(int argc, char *argv[]) 
{
	char* srcImg=argv[1];
	char* dstImg=argv[2];

	Face_Rec_Init(4);

	cv::Mat gallery_img_color = cv::imread(srcImg, 1);
	cv::Mat gallery_img_gray;
	cv::cvtColor(gallery_img_color, gallery_img_gray, CV_BGR2GRAY);

	gallery_src_data_color.data = gallery_img_color.data;
    gallery_src_data_color.width = gallery_img_color.cols;
    gallery_src_data_color.height = gallery_img_color.rows;
    gallery_src_data_color.num_channels = gallery_img_color.channels();

	gallery_src_data_gray.data = gallery_img_gray.data;
    gallery_src_data_gray.width = gallery_img_gray.cols;
    gallery_src_data_gray.height = gallery_img_gray.rows;
    gallery_src_data_gray.num_channels = gallery_img_gray.channels();

	cv::Mat probe_img_color = cv::imread(dstImg, 1);
	cv::Mat probe_img_gray;
	cv::cvtColor(probe_img_color, probe_img_gray, CV_BGR2GRAY);

	gallery_dst_data_color.data = probe_img_color.data;
    gallery_dst_data_color.width = probe_img_color.cols;
    gallery_dst_data_color.height = probe_img_color.rows;
    gallery_dst_data_color.num_channels = probe_img_color.channels();

	gallery_dst_data_gray.data = probe_img_gray.data;
    gallery_dst_data_gray.width = probe_img_gray.cols;
    gallery_dst_data_gray.height = probe_img_gray.rows;
    gallery_dst_data_gray.num_channels = probe_img_gray.channels();

	//detect
	gallery_fea = new float[2048];           
	probe_fea = new float[2048]; 

	Face_Rec_Extract(1,gallery_src_data_color,gallery_src_data_gray,gallery_fea,Face_Rec_Extract_callback1);
	Face_Rec_Extract(2,gallery_dst_data_color,gallery_dst_data_gray,probe_fea,Face_Rec_Extract_callback2);	
	Face_Rec_Detect(3,gallery_src_data_color,gallery_src_data_gray,Face_Rec_Extract_callback3);
	Face_Rec_Detect(4,gallery_dst_data_color,gallery_dst_data_gray,Face_Rec_Extract_callback4);

	while(1)
	{
		if((Face_Rec_Current_Step(1)==FACE_REC_STEP_IDLE)
		&&(Face_Rec_Current_Step(2)==FACE_REC_STEP_IDLE)
		&&(Face_Rec_Current_Step(3)==FACE_REC_STEP_IDLE)
		&&(Face_Rec_Current_Step(4)==FACE_REC_STEP_IDLE))
		{
			break;
		}
	}
	
	//Caculate Sim
	float sim = Face_Rec_Compare(gallery_fea,probe_fea);
	std::cout <<"sim of two face is:"<< sim <<endl;
	delete(gallery_fea);
	delete(probe_fea);
	gallery_fea=NULL;
	probe_fea=NULL;	

	Face_Rec_Deinit();	
	std::cout << "two picture detect successfully"<<endl;
	std::cout << "demo is over!!!"<<endl;
	return 0;
}