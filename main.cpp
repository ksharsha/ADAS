#include <opencv2/opencv.hpp>
#include "opencv2/video/tracking.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/video/tracking.hpp"
#include <opencv2/nonfree/features2d.hpp>

#include <iostream>
#include <ctype.h>
#include "opticalflow.h"
#include "features.h"
#include "morph.h"

using namespace cv;
using namespace std;

#define dense 0 //set this to one if we want to compute dense correspondences
#define savevideo 1//set this to one if we want to write the output to a video file
#define calcfeat 1//set this to one if we want to calculate features in every frame
vector<Mat> HogFeats;//Global variable for storing the computed HOG features of 200 road images

void init()
{
    cout<<"Started Initialization"<<endl;
    Mat src1;
    Mat gray1;
    vector<float> ders1;
    for(int i=1;i<=200;i++)
    {
        stringstream  num1;
        num1<<i;
        string str1 = string("../../../Desktop/Harsha/CMU/SurroundView/datasets/Vehicles/Roads/") +num1.str()+ ".png";
        src1 = imread(str1);
        HOGDescriptor hog(Size(96,64), Size(8,8), Size(4,4), Size(4,4), 9);
        cvtColor(src1, gray1, CV_BGR2GRAY);
        resize(gray1, gray1, Size(96, 64));
        hog.compute(gray1,ders1,Size(0,0), Size(0,0));
        Mat A(ders1.size(),1,CV_32FC1);
        memcpy(A.data,ders1.data(),ders1.size()*sizeof(float));
        HogFeats.push_back(A);
        //HogFeats[i]=A.clone();   
    }
}


int main( int argc, char** argv )
{
    const int MAX_COUNT = 5000;
    vector<Point2f> points[2];
    vector<uchar> status;
    vector<float> err;
    Mat prv,next, flow, colflow,imsparse ,colim, diff;
    Mat binimg;//Used for storing the binary images
    Mat featimg;//Used for getting the features
    Mat winimg;//img containing the rectangles
    Mat gradnext;//Gradient of a frame
    Mat blobimg;//Image with the connected components
    vector<KeyPoint> feats;
    vector < vector<Point2i > > blobs;
    int nframes=0;
    int i=0;
    /*Initializing our system first*/
    init();
    VideoCapture cap("../../../Desktop/Harsha/CMU/SurroundView/MovingObjectDetection/5.mp4");
    Size S = Size((int) cap.get(CV_CAP_PROP_FRAME_WIDTH),    
              (int) cap.get(CV_CAP_PROP_FRAME_HEIGHT));//Input Size
    VideoWriter outputVideo;
    //int ex = static_cast<int>(cap.get(CV_CAP_PROP_FOURCC));
    if(savevideo)
    {
        if(dense)
            outputVideo.open("outdense4.avi" , 1, cap.get(CV_CAP_PROP_FPS),S, true);
        else
            outputVideo.open("outsparserefinedhog.avi" , 1, cap.get(CV_CAP_PROP_FPS),S, true);
        //Reading the first frame from the video into the previous frame
        if (!outputVideo.isOpened())
        {
            cout  << "Could not open the output video for write: "  << endl;
            return -1;
        }
    }
    if(!(cap.read(prv)))
        return 0;
    cvtColor(prv, prv, CV_BGR2GRAY);
    threshold(prv, binimg, 100, 0, THRESH_BINARY);
    cout<<"The frame size is "<<prv.size()<<prv.rows<<prv.cols<<endl;
    //Computing the features to track at the beginning itself
    goodFeaturesToTrack(prv, points[0], MAX_COUNT, 0.01, 10, Mat(), 3, 0, 0.04);
    
    while(true)
    {
        if(!(cap.read(next)))
            break;
        if(next.empty())
            break;
        colim=next.clone();
        cvtColor(next, next, CV_BGR2GRAY);
        diff=abs(next-prv);
        imshow("Difference",diff);
        
        threshold(diff, binimg, 50, 1, THRESH_BINARY);//Will have very low
        //threshold because already it is difference of images
        //imshow("BinaryImage", binimg);
        Scharr(diff, gradnext, CV_64F, 0, 1, 3);
        imshow("GradientImage",gradnext);
        //threshold(gradnext, binimg, 50, 255, THRESH_BINARY);//Will have very low
        //imgerode(binimg, binimg, 1);
        imgdilate(binimg, binimg, 2);
        //imgerode(binimg, binimg, 4);
        imshow("MorphedImage",binimg);
        FindBlobs(binimg,blobimg, blobs);
        cout<<"The number of blobs detected in this frame is"<<blobs.size()<<endl;
        /*We will now calculate some features useful for identifying the obstacles*/
        if(calcfeat)
        {
            //showorb(next, featimg,feats);
            showorb(diff, featimg,feats);//Computing features on the diff of images.
            points[0].clear();
            for(i=0;i<feats.size();i++)
            {
                points[0].push_back(Point2f(feats[i].pt.x,feats[i].pt.y));
            }
            cout << points[0].size() << endl;
            /*Done calculating the features*/
        }
        //blobs.clear();
        extractwindowsrefined(next,blobimg,winimg,points,blobs);
        //extractwindows(next,winimg,points);
        imshow("WindowsImage",winimg);
#if dense==0
        calcOpticalFlowPyrLK(
        prv, next, // 2 consecutive images
        points[0], // input point positions in first im
        points[1], // output point positions in the 2nd
        status,    // tracking success
        err      // tracking error
        );
        //cout<<"Optical Flow computed for one frame"<<endl;
        drawoptflowsparse(prv,colim,imsparse,points);
        imshow("SparseFlow",imsparse);
        outputVideo.write(winimg);
        if (waitKey(5) >= 0) 
            break;
        swap(points[1], points[0]);
#endif
        
        //The dense flow calculation takes a lot of time since it matches every
        //pixel in one image to other image.
#if dense==1
        calcOpticalFlowFarneback(prv, next, flow, 0.5, 1, 5, 3, 5, 1.2, 0);
        cvtColor(prv, colflow, CV_GRAY2BGR);
        drawOptFlowMap(flow, colflow, 20, CV_RGB(0, 255, 0));
        imshow("DenseFlow",colflow);
        outputVideo.write(colflow);
        if (waitKey(5) >= 0) 
            break;
#endif
        prv = next.clone();
    }
    

    return 0;
}
