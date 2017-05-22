/*
 * Copyright (c) 2011. Philipp Wagner <bytefish[at]gmx[dot]de>.
 * Released to public domain under terms of the BSD Simplified license.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above copyright
 *     notice, this list of conditions and the following disclaimer in the
 *     documentation and/or other materials provided with the distribution.
 *   * Neither the name of the organization nor the names of its contributors
 *     may be used to endorse or promote products derived from this software
 *     without specific prior written permission.
 *
 *   See <http://www.opensource.org/licenses/bsd-license>
 */

#include "opencv2/core/core.hpp"
#include "opencv2/contrib/contrib.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/objdetect/objdetect.hpp"

#include <iostream>
#include <fstream>
#include <sstream>

#include <opencv/cv.h>
#include <opencv/highgui.h>
#include "face_identification.h"
#include "recognizer.h"
#include "face_detection.h"
#include "face_alignment.h"
#include "math_functions.h"

using namespace cv;
using namespace std;
using namespace seeta;

std::string DATA_DIR = "../data/";
std::string MODEL_DIR = "../model/";

// Initialize face detection model
seeta::FaceDetection detector("fd.bin");
// Initialize face alignment model 
seeta::FaceAlignment point_detector("fa.bin");
// Initialize face Identification model 
FaceIdentification face_recognizer("fr.bin");

void InitFaceDetectionModel()
{
    detector.SetMinFaceSize(40);
    detector.SetScoreThresh(2.f);
    detector.SetImagePyramidScaleFactor(0.8f);
    detector.SetWindowStep(4, 4);
}

void HandleInputImgData(Mat& gallery_img_color , float * gallery_fea)
{
    cv::Mat gallery_img_gray;
    //cvt image
    cv::cvtColor(gallery_img_color, gallery_img_gray, CV_BGR2GRAY);
    //gallery image
    ImageData gallery_img_data_color(gallery_img_color.cols, gallery_img_color.rows, gallery_img_color.channels());
  gallery_img_data_color.data = gallery_img_color.data;
  
    ImageData gallery_img_data_gray(gallery_img_gray.cols, gallery_img_gray.rows, gallery_img_gray.channels());
  gallery_img_data_gray.data = gallery_img_gray.data;
  
    // Detect faces
    std::vector<seeta::FaceInfo> gallery_faces = detector.Detect(gallery_img_data_gray);
    int32_t gallery_face_num = static_cast<int32_t>(gallery_faces.size());
    if (gallery_face_num == 0 )
    {
        //std::cout << "Faces are not detected."<<endl;
        return ;
    }
  
   // Detect 5 facial landmarks
    seeta::FacialLandmark gallery_points[5];
    point_detector.PointDetectLandmarks(gallery_img_data_gray, gallery_faces[0], gallery_points);
    for (int i = 0; i<5; i++)
    {
        cv::circle(gallery_img_color, cv::Point(gallery_points[i].x, gallery_points[i].y), 2,
        CV_RGB(0, 255, 0));
    }

    // Extract face identity feature
    face_recognizer.ExtractFeatureWithCrop(gallery_img_data_color, gallery_points,gallery_fea);
}

static void help()
{
    cout << "\nThis program is face recognize process.\n"
            "Usage:\n"
            "./facerecognize\n" 
            "   [ </path/haar_cascade> this is the Haar Cascade for face detection.]\n"
            "   [ </path/face_img> this is the you need recognition image file.]\n"
            "   [ </dev/device id> this is the webcam device id to grab frames from, make sure plug in device.]\n"
            "Example:\n"
            "./facerecognize /path/haarcascade_frontalface_alt.xml /path/pic.jpg /dev/video0\n\n"
            "During execution:\n\tHit 'q' key to quit.\n"
            "\tUsing OpenCV version " << CV_VERSION << "\n" << endl;
}

int main(int argc, const char *argv[]) {
    // Check for valid command line arguments, print usage
    // if no arguments were given.
    if (argc != 4)
    {
        help();
        return 0;
    }
    // Get the path to your CSV:
    string fn_haar = string(argv[1]);
    string srcImg = string(argv[2]);
    int deviceId = atoi(argv[3]);
    InitFaceDetectionModel();
    float gallery_fea[2048];
    float probe_fea[2048];
    float setfacepredition=0.5;
    //load image
    cv::Mat gallery_img_color = cv::imread(srcImg.c_str(), 1);
    HandleInputImgData(gallery_img_color,gallery_fea);
  
    CascadeClassifier haar_cascade;
    haar_cascade.load(fn_haar);
    // Get a handle to the Video device:
    VideoCapture cap(deviceId);
    // Check if we can use this device at all:
    if(!cap.isOpened()) {
        cerr << "Capture Device ID " << deviceId << "cannot be opened." << endl;
        return -1;
    }
    // Holds the current frame from the Video device:
    Mat frame;
    for(;;) {
        cap >> frame;
        // Clone the current frame:
        Mat original = frame.clone();
        // Convert the current frame to grayscale:
        Mat gray;
        cvtColor(original, gray, CV_BGR2GRAY);
        // Find the faces in the frame:
        vector< Rect_<int> > faces;
        haar_cascade.detectMultiScale(gray, faces);
       
        for(int i = 0; i < faces.size(); i++) {
            // Process face by face:
            Rect_<int> face_i = faces[i];
            
             HandleInputImgData(original,probe_fea);
              //printf("size:%d\n",sizeof(probe_fea));
             float prediction = face_recognizer.CalcSimilarity(gallery_fea, probe_fea);
             if (prediction >= setfacepredition)
             {
                // And finally write all we've found out to the original image!
                // First of all draw a green rectangle around the detected face:
                rectangle(original, face_i, CV_RGB(0, 255,0), 1);
                // Create the text we will annotate the box with:
                string box_text = format("Prediction = %0.3f", prediction);
                
                // Calculate the position for annotated text (make sure we don't
                // put illegal values in there):
                int pos_x = std::max(face_i.tl().x - 10, 0);
                int pos_y = std::max(face_i.tl().y - 10, 0);
                // And now put it into the image:
                putText(original, box_text, Point(pos_x, pos_y), FONT_HERSHEY_PLAIN, 1.0, CV_RGB(0,255,0), 2.0);
            }
        }
 
        // Show the result:
        imshow("face_recognizer", original);
        // And display it:
        char key = (char) waitKey(10);
        // Exit this loop on escape:
        if(key == 'q')
        {
            cout<<"\nquit the program."<<endl;
            break;
        }    
    }
    return 0;
}
