#include "face_rec.h"
#include <iostream>
#include <fstream>
#include <sstream>

#include <opencv2/core/version.hpp>
#include <opencv/cv.h>
#include <opencv/highgui.h>

using namespace std;
using namespace seeta;
using namespace cv;

int Get_Image_Data(char *img, ImageData* img_color,ImageData* img_gray);

int Get_Image_Data(char *img, ImageData* img_color,ImageData* img_gray)
{
	Mat gallery_img_color = imread(img, 1);
	Mat gallery_img_gray;
	cvtColor(gallery_img_color, gallery_img_gray, CV_BGR2GRAY);

	img_color->data = gallery_img_color.data;
    img_color->width = gallery_img_color.cols;
    img_color->height = gallery_img_color.rows;
    img_color->num_channels = gallery_img_color.channels();

	img_gray->data = gallery_img_gray.data;
    img_gray->width = gallery_img_gray.cols;
    img_gray->height = gallery_img_gray.rows;
    img_gray->num_channels = gallery_img_gray.channels();
}

int main(int argc, char *argv[]) 
{
	 char* srcImg=argv[1];
     char* dstImg=argv[2];
	 
	Face_Rec_Init();

	ImageData gallery_src_data_color;
	ImageData gallery_src_data_gray;
	ImageData gallery_dst_data_color;
	ImageData gallery_dst_data_gray;

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
	float* gallery_fea = (float*)malloc (2048*sizeof(float));
	float* probe_fea = (float*)malloc (2048*sizeof(float));


	Face_Rec_Extract(gallery_src_data_color,gallery_src_data_gray,gallery_fea);
	Face_Rec_Extract(gallery_dst_data_color,gallery_dst_data_gray,probe_fea);	 

	//calc


	float sim = Face_Rec_Compare(gallery_fea,probe_fea);
	std::cout << sim <<endl;
	free(gallery_fea);
	free(probe_fea);

	Face_Rec_Deinit();
}