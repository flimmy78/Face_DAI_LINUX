
#include<iostream>
using namespace std;

#ifdef _WIN32
#pragma once
#include <opencv2/core/version.hpp>

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

#include"funset.hpp"
#include <iostream>
#include <string>
#include <fstream>
#include "face_detection.h"
#include "face_alignment.h"
#include "face_identification.h"
#include <opencv2/opencv.hpp>

#define MODEL_PATH "/home/jack/SeetaFaceEngine-master/FaceIdentification/build/"
#define TEST_DATA "/home/jack/Face_Test-master/testdata"

int test_detection()
{
	std::vector<std::string> images{ "1.jpg", "2.jpg", "3.jpg", "4.jpeg", "5.jpeg", "6.jpg", "7.jpg", "8.jpg", "9.jpg", "10.jpg",
		"11.jpeg", "12.jpg", "13.jpeg", "14.jpg", "15.jpeg", "16.jpg", "17.jpg", "18.jpg", "19.jpg", "20.jpg" };
	std::vector<int> count_faces{ 1, 2, 6, 0, 1, 1, 1, 2, 1, 1,
		1, 1, 1, 1, 1, 1, 1, 0, 8, 2 };

	const std::string path_images{ TEST_DATA };

	seeta::FaceDetection detector(MODEL_PATH+"seeta_fd_frontal_v1.0.bin");

	detector.SetMinFaceSize(20);
	detector.SetMaxFaceSize(200);
	detector.SetScoreThresh(2.f);
	detector.SetImagePyramidScaleFactor(0.8f);
	detector.SetWindowStep(4, 4);

	for (int i = 0; i < images.size(); i++) {
		cv::Mat src_ = cv::imread(path_images + images[i], 1);
		if (src_.empty()) {
			fprintf(stderr, "read image error: %s\n", images[i].c_str());
			continue;
		}

		cv::Mat src;
		cv::cvtColor(src_, src, CV_BGR2GRAY);

		seeta::ImageData img_data;
		img_data.data = src.data;
		img_data.width = src.cols;
		img_data.height = src.rows;
		img_data.num_channels = 1;

		std::vector<seeta::FaceInfo> faces = detector.Detect(img_data);

		fprintf(stderr, "image_name: %s, faces_num: %d\n", images[i].c_str(), faces.size());
		for (int num = 0; num < faces.size(); num++) {
			fprintf(stderr, "    score = %f\n",/*, roll = %f, pitch = %f, yaw = %f*/
				faces[num].score/*, faces[num].roll, faces[num].pitch, faces[num].yaw*/);

			cv::rectangle(src_, cv::Rect(faces[num].bbox.x, faces[num].bbox.y,
				faces[num].bbox.width, faces[num].bbox.height), cv::Scalar(0, 255, 0), 2);
		}

		std::string save_result = path_images + "_" + images[i];
		cv::imwrite(save_result, src_);
	}

	int width = 200;
	int height = 200;
	cv::Mat dst(height * 5, width * 4, CV_8UC3);
	for (int i = 0; i < images.size(); i++) {
		std::string input_image = path_images + "_" + images[i];
		cv::Mat src = cv::imread(input_image, 1);
		if (src.empty()) {
			fprintf(stderr, "read image error: %s\n", images[i].c_str());
			return -1;
		}

		cv::resize(src, src, cv::Size(width, height), 0, 0, 4);
		int x = (i * width) % (width * 4);
		int y = (i / 4) * height;
		cv::Mat part = dst(cv::Rect(x, y, width, height));
		src.copyTo(part);
	}
	std::string output_image = path_images + "result.png";
	cv::imwrite(output_image, dst);

	return 0;
}

int test_alignment()
{
	std::vector<std::string> images{ "1.jpg", "2.jpg", "3.jpg", "4.jpeg", "5.jpeg", "6.jpg", "7.jpg", "8.jpg", "9.jpg", "10.jpg",
		"11.jpeg", "12.jpg", "13.jpeg", "14.jpg", "15.jpeg", "16.jpg", "17.jpg", "18.jpg", "19.jpg", "20.jpg" };
	std::vector<int> count_faces{ 1, 2, 6, 0, 1, 1, 1, 2, 1, 1,
		1, 1, 1, 1, 1, 1, 1, 0, 8, 2 };

	const std::string path_images{ TEST_DATA };

	seeta::FaceDetection detector(MODEL_PATH"seeta_fd_frontal_v1.0.bin");

	detector.SetMinFaceSize(20);
	detector.SetMaxFaceSize(200);
	detector.SetScoreThresh(2.f);
	detector.SetImagePyramidScaleFactor(0.8f);
	detector.SetWindowStep(4, 4);

	seeta::FaceAlignment point_detector(MODEL_PATH"seeta_fa_v1.1.bin");

	for (auto name : images) {
		fprintf(stderr, "start detect image: %s\n", name.c_str());

		cv::Mat src_ = cv::imread(path_images + name, 1);
		if (src_.empty()) {
			fprintf(stderr, "read image error: %s\n", name.c_str());
			continue;
		}

		cv::Mat src;
		cv::cvtColor(src_, src, CV_BGR2GRAY);

		seeta::ImageData img_data;
		img_data.data = src.data;
		img_data.width = src.cols;
		img_data.height = src.rows;
		img_data.num_channels = 1;

		std::vector<seeta::FaceInfo> faces = detector.Detect(img_data);

		for (auto face : faces) {
			// Detect 5 facial landmarks: two eye centers, nose tip and two mouth corners
			seeta::FacialLandmark points[5];
			point_detector.PointDetectLandmarks(img_data, face, points);

			cv::rectangle(src_, cv::Rect(face.bbox.x, face.bbox.y,
				face.bbox.width, face.bbox.height), cv::Scalar(0, 255, 0), 2);

			for (auto point : points) {
				cv::circle(src_, cv::Point(point.x, point.y), 2, cv::Scalar(0, 0, 255), 2);
			}
		}

		std::string save_result = path_images + "_" + name;
		cv::imwrite(save_result, src_);
	}

	int width = 200;
	int height = 200;
	cv::Mat dst(height * 5, width * 4, CV_8UC3);
	for (int i = 0; i < images.size(); i++) {
		std::string input_image = path_images + "_" + images[i];
		cv::Mat src = cv::imread(input_image, 1);
		if (src.empty()) {
			fprintf(stderr, "read image error: %s\n", images[i].c_str());
			return -1;
		}

		cv::resize(src, src, cv::Size(width, height), 0, 0, 4);
		int x = (i * width) % (width * 4);
		int y = (i / 4) * height;
		cv::Mat part = dst(cv::Rect(x, y, width, height));
		src.copyTo(part);
	}
	std::string output_image = path_images + "result.png";
	cv::imwrite(output_image, dst);

	return 0;
}

int test_recognize()
{
	const std::string path_images{ TEST_DATA };
	seeta::FaceDetection detector(MODEL_PATH"seeta_fd_frontal_v1.0.bin");
	seeta::FaceAlignment alignment(MODEL_PATH"seeta_fa_v1.1.bin");
	seeta::FaceIdentification face_recognizer(MODEL_PATH"seeta_fr_v1.0.bin");

	detector.SetMinFaceSize(20);
	detector.SetMaxFaceSize(200);
	detector.SetScoreThresh(2.f);
	detector.SetImagePyramidScaleFactor(0.8f);
	detector.SetWindowStep(4, 4);

	std::vector<std::vector<seeta::FacialLandmark>> landmards;

	// detect and alignment
	for (int i = 0; i < 20; i++) {
		std::string image = path_images + std::to_string(i) + ".jpg";
		//fprintf(stderr, "start process image: %s\n", image.c_str());

		cv::Mat src_ = cv::imread(image, 1);
		if (src_.empty()) {
			fprintf(stderr, "read image error: %s\n", image.c_str());
			continue;
		}

		cv::Mat src;
		cv::cvtColor(src_, src, CV_BGR2GRAY);

		seeta::ImageData img_data;
		img_data.data = src.data;
		img_data.width = src.cols;
		img_data.height = src.rows;
		img_data.num_channels = 1;

		std::vector<seeta::FaceInfo> faces = detector.Detect(img_data);
		if (faces.size() == 0) {
			fprintf(stderr, "%s don't detect face\n", image.c_str());
			continue;
		}

		// Detect 5 facial landmarks: two eye centers, nose tip and two mouth corners
		std::vector<seeta::FacialLandmark> landmard(5);
		alignment.PointDetectLandmarks(img_data, faces[0], &landmard[0]);

		landmards.push_back(landmard);

		cv::rectangle(src_, cv::Rect(faces[0].bbox.x, faces[0].bbox.y,
			faces[0].bbox.width, faces[0].bbox.height), cv::Scalar(0, 255, 0), 2);

		for (auto point : landmard) {
			cv::circle(src_, cv::Point(point.x, point.y), 2, cv::Scalar(0, 0, 255), 2);
		}

		std::string save_result = path_images + "_" + std::to_string(i) + ".jpg";
		cv::imwrite(save_result, src_);
	}

	int width = 200;
	int height = 200;
	cv::Mat dst(height * 5, width * 4, CV_8UC3);
	for (int i = 0; i < 20; i++) {
		std::string input_image = path_images + "_" + std::to_string(i) + ".jpg";
		cv::Mat src = cv::imread(input_image, 1);
		if (src.empty()) {
			fprintf(stderr, "read image error: %s\n", input_image.c_str());
			return -1;
		}

		cv::resize(src, src, cv::Size(width, height), 0, 0, 4);
		int x = (i * width) % (width * 4);
		int y = (i / 4) * height;
		cv::Mat part = dst(cv::Rect(x, y, width, height));
		src.copyTo(part);
	}
	std::string output_image = path_images + "result_alignment.png";
	cv::imwrite(output_image, dst);

	// crop image
	for (int i = 0; i < 20; i++) {
		std::string image = path_images + std::to_string(i) + ".jpg";
		//fprintf(stderr, "start process image: %s\n", image.c_str());

		cv::Mat src_img = cv::imread(image, 1);
		if (src_img.data == nullptr) {
			fprintf(stderr, "Load image error: %s\n", image.c_str());
			return -1;
		}

		if (face_recognizer.crop_channels() != src_img.channels()) {
			fprintf(stderr, "channels dismatch: %d, %d\n", face_recognizer.crop_channels(), src_img.channels());
			return -1;
		}

		// ImageData store data of an image without memory alignment.
		seeta::ImageData src_img_data(src_img.cols, src_img.rows, src_img.channels());
		src_img_data.data = src_img.data;

		// Create a image to store crop face.
		cv::Mat dst_img(face_recognizer.crop_height(), face_recognizer.crop_width(), CV_8UC(face_recognizer.crop_channels()));
		seeta::ImageData dst_img_data(dst_img.cols, dst_img.rows, dst_img.channels());
		dst_img_data.data = dst_img.data;
		// Crop Face
		face_recognizer.CropFace(src_img_data, &landmards[i][0], dst_img_data);

		std::string save_image_name = path_images + "crop_" + std::to_string(i) + ".jpg";
		cv::imwrite(save_image_name, dst_img);
	}

	dst = cv::Mat(height * 5, width * 4, CV_8UC3);
	for (int i = 0; i < 20; i++) {
		std::string input_image = path_images + "crop_" + std::to_string(i) + ".jpg";
		cv::Mat src_img = cv::imread(input_image, 1);
		if (src_img.empty()) {
			fprintf(stderr, "read image error: %s\n", input_image.c_str());
			return -1;
		}

		cv::resize(src_img, src_img, cv::Size(width, height), 0, 0, 4);
		int x = (i * width) % (width * 4);
		int y = (i / 4) * height;
		cv::Mat part = dst(cv::Rect(x, y, width, height));
		src_img.copyTo(part);
	}
	output_image = path_images + "result_crop.png";
	cv::imwrite(output_image, dst);

	// extract feature
	int feat_size = face_recognizer.feature_size();
	if (feat_size != 2048) {
		fprintf(stderr, "feature size mismatch: %d\n", feat_size);
		return -1;
	}

	float* feat_sdk = new float[feat_size * 20];

	for (int i = 0; i < 20; i++) {
		std::string input_image = path_images + "crop_" + std::to_string(i) + ".jpg";
		cv::Mat src_img = cv::imread(input_image, 1);
		if (src_img.empty()) {
			fprintf(stderr, "read image error: %s\n", input_image.c_str());
			return -1;
		}

		cv::resize(src_img, src_img, cv::Size(face_recognizer.crop_height(), face_recognizer.crop_width()));

		// ImageData store data of an image without memory alignment.
		seeta::ImageData src_img_data(src_img.cols, src_img.rows, src_img.channels());
		src_img_data.data = src_img.data;

		// Extract feature
		face_recognizer.ExtractFeature(src_img_data, feat_sdk + i * feat_size);
	}

	float* feat1 = feat_sdk;
	// varify(recognize)
	for (int i = 1; i < 20; i++) {
		std::string image = std::to_string(i) + ".jpg";
		float* feat_other = feat_sdk + i * feat_size;

		// Caculate similarity
		float sim = face_recognizer.CalcSimilarity(feat1, feat_other);
		fprintf(stdout, "0.jpg -- %s similarity: %f\n", image.c_str(), sim);
	}

	delete[] feat_sdk;

	return 0;
}

int test_identification_CropFace()
{
	seeta::FaceIdentification face_recognizer(nullptr);
	std::string test_dir = MODEL_PATH;
	std::string save_dir = TEST_DATA;

	// data initialize
	std::ifstream ifs;
	ifs.open(test_dir + "test_file_list.txt", std::ifstream::in);
	if (!ifs.is_open()) {
		fprintf(stderr, "open test file list fail\n");
		return -1;
	}

	std::string img_name;
	seeta::FacialLandmark pt5[5];

	while (ifs >> img_name) {
		fprintf(stderr, " start process image: %s\n", img_name.c_str());

		cv::Mat src_img = cv::imread(test_dir + img_name, 1);
		if (src_img.data == nullptr) {
			fprintf(stderr, "Load image error: %s\n", img_name.c_str());
			return -1;
		}

		// ImageData store data of an image without memory alignment.
		seeta::ImageData src_img_data(src_img.cols, src_img.rows, src_img.channels());
		src_img_data.data = src_img.data;

		// 5 located landmark points (left eye, right eye, nose, left and right corner of mouse).
		for (int i = 0; i < 5; ++i) {
			ifs >> pt5[i].x >> pt5[i].y;
		}

		// Create a image to store crop face.
		cv::Mat dst_img(face_recognizer.crop_height(), face_recognizer.crop_width(), CV_8UC(face_recognizer.crop_channels()));
		seeta::ImageData dst_img_data(dst_img.cols, dst_img.rows, dst_img.channels());
		dst_img_data.data = dst_img.data;
		// Crop Face
		face_recognizer.CropFace(src_img_data, pt5, dst_img_data);

		std::string name = get_image_name(img_name);
		
		std::string save_image_name = save_dir + name;
		cv::imwrite(save_image_name, dst_img);
	}

	ifs.close();

	return 0;
}

int test_identification_ExtractFeature()
{
	seeta::FaceIdentification face_recognizer(MODEL_PATH"seeta_fr_v1.0.bin");
	std::string test_dir = MODEL_PATH;

	int feat_size = face_recognizer.feature_size();
	if (feat_size != 2048) {
		fprintf(stderr, "feature size mismatch: %d\n", feat_size);
		return -1;
	}

	FILE* feat_file{nullptr};
	// Load features extract from caffe
	int ret = fopen_s(&feat_file, (test_dir + "feats.dat").c_str(), "rb");
	if (ret != 0) {
		fprintf(stderr, "open feature file fail: %d\n", ret);
		return -1;
	}

	int n, c, h, w;
	fread(&n, sizeof(int), 1, feat_file);
	fread(&c, sizeof(int), 1, feat_file);
	fread(&h, sizeof(int), 1, feat_file);
	fread(&w, sizeof(int), 1, feat_file);

	float* feat_caffe = new float[n * c * h * w];
	float* feat_sdk = new float[n * c * h * w];
	fread(feat_caffe, sizeof(float), n * c * h * w, feat_file);

	if (feat_size != c * h * w) {
		fprintf(stderr, "feature length mismatch: %d != %d\n", feat_size, c * h * w);
		return -1;
	}

	// Data initialize
	std::ifstream ifs(test_dir + "crop_file_list.txt");
	std::string img_name;

	int img_num = 0, lb;
	double average_sim = 0.0;
	while (ifs >> img_name >> lb) {
		// read image
		cv::Mat src_img = cv::imread(test_dir + img_name, 1);
		if (src_img.data == nullptr) {
			fprintf(stderr, "Load image error: %s\n", img_name.c_str());
			return -1;
		}
		cv::resize(src_img, src_img, cv::Size(face_recognizer.crop_height(), face_recognizer.crop_width()));

		// ImageData store data of an image without memory alignment.
		seeta::ImageData src_img_data(src_img.cols, src_img.rows, src_img.channels());
		src_img_data.data = src_img.data;

		// Extract feature
		face_recognizer.ExtractFeature(src_img_data, feat_sdk + img_num * feat_size);

		// Caculate similarity
		float* feat1 = feat_caffe + img_num * feat_size;
		float* feat2 = feat_sdk + img_num * feat_size;
		float sim = face_recognizer.CalcSimilarity(feat1, feat2);
		average_sim += sim;
		img_num++;
	}

	average_sim /= img_num;
	if (1.0 - average_sim > 0.01)
		fprintf(stderr, "average similarity: %f\n", average_sim);
	else
		fprintf(stderr, "Test successful!\n");

	ifs.close();
	delete[] feat_caffe;
	delete[] feat_sdk;

	return 0;
}

std::string get_image_name(std::string name)
{
	std::string name_ = name;
	int pos = name_.find_last_of("/");
	return name_.erase(0, pos+1);
}

int main()
{
	int ret = test_recognize();

	if (ret == 0) fprintf(stderr, "test success\n");
	else fprintf(stderr, "test fail\n");

	return 0;
}
