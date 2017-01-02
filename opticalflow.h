/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/* 
 * File:   opticalflow.h
 * Author: KSH
 *
 * Created on December 29, 2016, 2:18 AM
 */

#ifndef OPTICALFLOW_H
#define OPTICALFLOW_H

using namespace cv;
using namespace std;

#ifdef __cplusplus
extern "C" {
#endif

void drawOptFlowMap (const Mat& flow, Mat& cflowmap, int step, const Scalar& color);
void drawoptflowsparse(Mat& prv,Mat& next,Mat& imsparse,vector<Point2f> *points);



#ifdef __cplusplus
}
#endif

#endif /* OPTICALFLOW_H */

