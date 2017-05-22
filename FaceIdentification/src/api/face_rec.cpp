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


static FaceDetection *detector=NULL;
static FaceAlignment *point_detector=NULL;
static FaceIdentification *face_recognizer=NULL;
//std::string test_dir = DATA_DIR + "test_face_recognizer/";	
	
void Face_Rec_Init(char *path)
{
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
	
	point_detector=new FaceAlignment((char *)alignment_path.c_str());
	detector=new FaceDetection((char *)detector_path.c_str());
	face_recognizer=new FaceIdentification((char *)recognizer_path.c_str());
	
	detector->SetMinFaceSize(40);
	detector->SetScoreThresh(2.f);
	detector->SetImagePyramidScaleFactor(0.8f);
	detector->SetWindowStep(4, 4);
}


int Face_Rec_Extract(ImageData img_data_color,ImageData img_data_gray,float* img_fea)
{
	std::vector<seeta::FaceInfo> gallery_faces = detector->Detect(img_data_gray);
	int32_t gallery_face_num = static_cast<int32_t>(gallery_faces.size());
	if (gallery_face_num == 0 )
	{
		std::cout << "Faces are not detected.";
		return -1;
	}

	// Detect 5 facial landmarks
	seeta::FacialLandmark gallery_points[5];
	point_detector->PointDetectLandmarks(img_data_gray, gallery_faces[0], gallery_points);	
	
	/*
	for (int i = 0; i<5; i++)
	{
		cv::circle(gallery_img_color, cv::Point(gallery_points[i].x, gallery_points[i].y), 2,
		CV_RGB(0, 255, 0));
	}	
	*/
	//<<gallery_points[0]<<gallery_points[1]<<gallery_points[2]<<gallery_points[3]<<gallery_points[4];
	// Extract face identity feature
	face_recognizer->ExtractFeatureWithCrop(img_data_color, gallery_points, img_fea);	
	
	return 0;
}
float Face_Rec_Compare(float * img1_fea,float * img2_fea)
{
  // Caculate similarity of two faces
  float sim = face_recognizer->CalcSimilarity(img1_fea, img2_fea);
  return sim;	
}

void Face_Rec_Deinit()
{
	//ReleaseFaceDetection();
	//ReleaseFaceAlignment();
	//ReleaseFaceIdentification();
	delete point_detector;
	point_detector=NULL;
	delete detector;
	detector=NULL;
	delete face_recognizer;
	face_recognizer=NULL;	
}

