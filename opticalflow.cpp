#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <cmath>
#include "opticalflow.h"

using namespace cv;
using namespace std;

void drawOptFlowMap (const Mat& flow, Mat& cflowmap, int step, const Scalar& color) {
    for(int y = 0; y < cflowmap.rows; y += step)
        for(int x = 0; x < cflowmap.cols; x += step)
        {
            const Point2f& fxy = flow.at< Point2f>(y, x);
            line(cflowmap, Point(x,y), Point(cvRound(x+fxy.x), cvRound(y+fxy.y)),
                 color);
            circle(cflowmap, Point(cvRound(x+fxy.x), cvRound(y+fxy.y)), 1, color, -1);
        }
    }

/*The below function will draw the sparse optical flow points*/
void drawoptflowsparse(Mat& prv,Mat& next,Mat& imsparse,vector<Point2f> *points)
{
    //cout<<points[0].size()<<points[1].size()<<endl;
    imsparse=next.clone();
    for(int y=0;y<points[0].size();y++)
    {
        const Point2f& p1=points[0][y];
        const Point2f& p2=points[1][y];
        line(imsparse,Point(cvRound(p1.x),cvRound(p1.y)),Point(cvRound(p2.x),cvRound(p2.y)),
                CV_RGB(0, 255, 0));
        circle(imsparse, Point(cvRound(p2.x), cvRound(p2.y)), 1, CV_RGB(0, 255, 0), -1);
    }
}


