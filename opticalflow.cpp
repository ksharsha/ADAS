#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <cmath>
#include "opticalflow.h"
#include "morph.h"
#include "features.h"


using namespace cv;
using namespace std;

#define oferr 10

Rect rect1(45, 50, 125, 220);
Rect rect2(175, 50, 300, 220);
Rect rect3(480, 50, 125, 220);
Rect rect4(685, 10, 125, 220);
Rect rect5(815, 10, 300, 260);
Rect rect6(1120, 10, 125, 240);
Rect rect7(45, 410, 120, 200);
Rect rect8(175, 410, 300, 210);
Rect rect9(480, 410, 125, 200);
Rect rect10(685, 370, 120, 260);
Rect rect11(815, 370, 300, 250);
Rect rect12(1120, 370, 125, 220);

extern int rectstat[12];


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
void drawoptflowsparse(Mat& prv,Mat& next,Mat& imsparse,vector<Point2f> *points,vector<uchar> &status,vector<float> err)
{
    //cout<<points[0].size()<<points[1].size()<<endl;
    imsparse=next.clone();
    for(size_t y=0;y<points[0].size();y++)
    {
        //cout<<int(status[y])<<" "<<status.size()<<" "<<points[0].size()<<endl;
        if(int(status[y]) && err[y]<oferr)
        {
            //cout<<"The error in this track is "<<err[y]<<endl;
            const Point2f& p1=points[0][y];
            const Point2f& p2=points[1][y];
            line(imsparse,Point(cvRound(p1.x),cvRound(p1.y)),Point(cvRound(p2.x),cvRound(p2.y)),
                    CV_RGB(0, 255, 0));
            circle(imsparse, Point(cvRound(p2.x), cvRound(p2.y)), 1, CV_RGB(0, 255, 0), -1);
        }
    }
}

void drawrect(Mat& imsparse,Rect rect,vector<Point2f> *points,vector<uchar> &status,vector<float> err,int winnum)
{
    double flowx=0,flowy=0,flow=0;
    for(int y=0;y<points[0].size();y++)
    {
        const Point2f& p1=points[0][y];
        const Point2f& p2=points[1][y];
        if(rect.x < p1.x && p1.x < rect.x+rect.width)
        {
            if(rect.y < p1.y && p1.y < rect.y+rect.height)
            {
                if(int(status[y]) && err[y]<oferr)
                {
                    flowx=flowx+(p2.x-p1.x);
                    flowy=flowy+(p2.y-p1.y);
                    //flow=flow + (p2.x-p1.x)*(p2.x-p1.x) + (p2.y-p1.y)*(p2.y-p1.y);
                }
            }
        }
    }
    flow=flowx*flowx+flowy*flowy;
    if(flow>2000)
    {
        rectstat[winnum]=1;
        rectangle(imsparse, rect, Scalar(0,0,255), 1, 8, 0);
    }
    else
    {
        rectstat[winnum]=0;
        rectangle(imsparse, rect, Scalar(255,0,0), 1, 8, 0);
    }
}

void findobst(Mat& prv, Mat& next,Mat& imsparse,vector<Point2f> *points,vector<uchar> &status,vector<float> err)
{
    drawrect(imsparse,rect1,points,status,err,1);
    drawrect(imsparse,rect2,points,status,err,2);
    drawrect(imsparse,rect3,points,status,err,3);
    drawrect(imsparse,rect4,points,status,err,4);
    drawrect(imsparse,rect5,points,status,err,5);
    drawrect(imsparse,rect6,points,status,err,6);
    drawrect(imsparse,rect7,points,status,err,7);
    drawrect(imsparse,rect8,points,status,err,8);
    drawrect(imsparse,rect9,points,status,err,9);
    drawrect(imsparse,rect10,points,status,err,10);
    drawrect(imsparse,rect11,points,status,err,11);
    drawrect(imsparse,rect12,points,status,err,12);
    
}
/*
 Finds all the static objects in the frame
 */
void findstaticobst(Mat& prv, Mat& next,Mat& imsparse,vector<Point2f> *points,vector<uchar> &status,vector<float> err)
{
    vector<Point2f> pts[2];
    vector < vector<Point2i> > blobs;
    Mat binimg,cc,grady,gradx,grad;
    Mat abs_grad_x, abs_grad_y;
    Mat colnext=next.clone();
    cvtColor(next, next, CV_BGR2GRAY);
    Mat diff = abs(next-prv); //Difference of images as an indication of motion
    /*imshow("Previous Image",prv);
    imshow("Next Image",next);
    imshow("Difference Image",diff);*/
    //and an approximation of optical flow
    Sobel(next, gradx, CV_64F, 1, 0, 3);
    Sobel(next, grady, CV_64F, 0, 1, 3);
    convertScaleAbs( gradx, abs_grad_x );
    convertScaleAbs( grady, abs_grad_y );
    addWeighted( abs_grad_x, 0.5, abs_grad_y, 0.5, 0, grad );
    threshold(grad, binimg, 100, 1, THRESH_BINARY);
    imgerode(binimg, binimg, 1.2);
    imgdilate(binimg, binimg, 2);
    
    FindBlobs(binimg,cc, blobs);  
    //cout<<"The number of blobs detected are"<<blobs.size()<<endl;
    for(int y=0;y<points[0].size();y++)
    {
        if(int(status[y]) && err[y]<oferr)
        {
            const Point2f& p1=points[0][y];
            const Point2f& p2=points[1][y];
            if(abs(p1.x-p2.x)<0.05 && abs(p1.y-p2.y)<0.05)
            {
                circle(imsparse, Point(cvRound(p2.x), cvRound(p2.y)), 1, CV_RGB(0, 255, 0), -1);
                pts[0].push_back(p2);
            }
        }       
    }
    if(pts[0].size()>0)
    {
        extractstaticwindowsrefined(colnext,diff,cc,imsparse, pts,blobs);
        //extractwindows(next,imsparse, pts);
    }
}
/*
 Finds the static obstacles in the region of interests and plots it in green
 if found*/
void findstatobst(Mat& diff,Mat& next)
{
    double sumpixels=0;
    Mat featimg,binimg,blobimg,winimg;
    vector < vector<Point2i > > blobs;
    vector<KeyPoint> feats;
    vector<Point2f> points[2];
    Mat roi=next(rect2).clone();
    Mat diffroi=diff(rect2).clone();
    showorb(roi, featimg,feats);
    imshow("ORBIMAGE",featimg);
    points[0].clear();
    //cout<<"Found ORB features are "<<feats.size()<<endl;
    for(int i=0;i<feats.size();i++)
    {
        if(diffroi.at<double>(round(feats[i].pt.y),round(feats[i].pt.x)) == 0)
            points[0].push_back(Point2f(feats[i].pt.x,feats[i].pt.y));
    }
    /*
     75 Appears to be a good threshold
     */
    //cout<<"Refined the points to consider only the static ones "<<points[0].size()<<endl;
    threshold(next, binimg, 75, 1, THRESH_BINARY);
    //cout<<"About to find the blobs "<<binimg.size()<<endl;
    FindBlobs(binimg,blobimg, blobs);
    //cout<<"Found the blobs"<<endl;
    if(points[0].size()>0)
    {
        //cout<<roi.size()<<" "<<blobimg.size()<<"Blobs detected "<<blobs.size()<<endl;
        extractwindowsclose(next,blobimg,winimg,points,blobs);
        //imshow("BinaryImage",binimg);
        imshow("WindowsImage",winimg);
    }
    /*
     The sum of all the pixels in this roi should be zero for us to treat it as
     a static window ideally
     */
}
void findobstdense(const Mat& flow, Mat& cflowmap)
{
    //Scalar(0,0,255) for red color
    //Scalar(255,0,0) for blue color
    double flowx=0,flowy=0;
    for(int i=rect1.x;i<rect1.width;i++)
    {
        for(int j=rect1.y;j<rect1.height;j++)
        {
            const Point2f& fxy = flow.at< Point2f>(j, i);
            flowx=flowx+fxy.x;
            flowy=flowy+fxy.y;
        }
    }
    cout<<flowx<<"x flow then now y flow"<<flowy<<endl;
    rectangle(cflowmap, rect1, Scalar(0,0,255), 1, 8, 0);
    rectangle(cflowmap, rect2, Scalar(255), 1, 8, 0);
    rectangle(cflowmap, rect3, Scalar(255), 1, 8, 0);
    rectangle(cflowmap, rect4, Scalar(255), 1, 8, 0);
    rectangle(cflowmap, rect5, Scalar(255), 1, 8, 0);
    rectangle(cflowmap, rect6, Scalar(255), 1, 8, 0);
    rectangle(cflowmap, rect7, Scalar(255), 1, 8, 0);
    rectangle(cflowmap, rect8, Scalar(255), 1, 8, 0);
    rectangle(cflowmap, rect9, Scalar(255), 1, 8, 0);
    rectangle(cflowmap, rect10, Scalar(255), 1, 8, 0);
    rectangle(cflowmap, rect11, Scalar(255), 1, 8, 0);
    rectangle(cflowmap, rect12, Scalar(255), 1, 8, 0);
}


