/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/* 
 * File:   morph.h
 * Author: KSH
 *
 * Created on January 2, 2017, 9:24 AM
 */

#ifndef MORPH_H
#define MORPH_H

using namespace cv;
using namespace std;

#ifdef __cplusplus
extern "C" {
#endif
    
    void imgerode(Mat& img,
                  Mat& out,
                  int erosion_size);
    
    void imgdilate(Mat& img,
                   Mat& out,
                   int erosion_size);
    
    void extractwindows(Mat& img,Mat& out,vector<Point2f> *points);
    
    void FindBlobs(const Mat &binary,Mat &labels, vector < vector<Point2i> > &blobs);
    
    void extractwindowsrefined(Mat& img,Mat& blob, Mat& out, vector<Point2f> *points, vector < vector<Point2i> > &blobs);
    
#ifdef __cplusplus
}
#endif



#endif /* MORPH_H */

