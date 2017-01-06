/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/* 
 * File:   features.h
 * Author: KSH
 *
 * Created on December 30, 2016, 12:24 AM
 */

#ifndef FEATURES_H
#define FEATURES_H

using namespace cv;
using namespace std;

#ifdef __cplusplus
extern "C" {
#endif
    
void showsift(Mat& img,
              Mat& out,
              vector<KeyPoint> &pts);

void showorb(Mat& img,
              Mat& out,
              vector<KeyPoint> &pts);

void showsurf(Mat& img,
              Mat& out,
              vector<KeyPoint> &pts);

void showfast(Mat& img,
              Mat& out,
              vector<KeyPoint> &pts);

void showhog(Mat& img,
              Mat& out,
              vector<float> &pts);

int comparehog(vector<float> &pts);



#ifdef __cplusplus
}
#endif



#endif /* FEATURES_H */

