#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include <iostream>
#include <vector>
#include <cmath>
#include <ctype.h>
#include "features.h"

using namespace cv;
using namespace std;

void showsift(Mat& img,Mat& out,vector<KeyPoint> &pts)
{
    SiftFeatureDetector detector(2000);
    out=img.clone();
    detector.detect(img, pts);
    drawKeypoints(img, pts, out);
}


void showorb(Mat& img,Mat& out,vector<KeyPoint> &pts)
{
    OrbFeatureDetector detector(2000);//The input parameter specifies the number of features to compute
    //OrbFeatureDetector detector;
    out=img.clone();
    detector.detect(img, pts);
    drawKeypoints(img, pts, out);
}

void showsurf(Mat& img,Mat& out,vector<KeyPoint> &pts)
{
    SurfFeatureDetector detector;
    out=img.clone();
    detector.detect(img, pts);
    drawKeypoints(img, pts, out);
}

void showfast(Mat& img,Mat& out,vector<KeyPoint> &pts)
{
    FastFeatureDetector detector;
    out=img.clone();
    detector.detect(img, pts);
    drawKeypoints(img, pts, out);
}