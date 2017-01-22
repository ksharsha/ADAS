#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include <iostream>
#include <vector>
#include <cmath>
#include <ctype.h>
#include "morph.h"
#include "features.h"

using namespace cv;
using namespace std;

/*height and width of the rectangular window*/
#define heightrect 64
#define widthrect 96

Rect frame1(0,0,640,300);
Rect frame2(0,360,640,300);
Rect frame3(640,0,640,300);
Rect frame4(640,360,640,300);

Rect rects1(45, 50, 125, 220);
Rect rects2(175, 50, 300, 220);
Rect rects3(480, 50, 125, 220);
Rect rects4(685, 10, 125, 220);
Rect rects5(815, 10, 300, 260);
Rect rects6(1120, 10, 125, 240);
Rect rects7(45, 410, 120, 200);
Rect rects8(175, 410, 300, 210);
Rect rects9(480, 410, 125, 200);
Rect rects10(685, 370, 120, 260);
Rect rects11(815, 370, 300, 250);
Rect rects12(1120, 370, 125, 220);

    int rectstat[12];//Will use this to denote if it has  a moving object or not

void imgerode(Mat& img,
              Mat& out,
              int erosion_size)
{
    Mat element = getStructuringElement( MORPH_ELLIPSE,
                       Size( 2*erosion_size + 1, 2*erosion_size+1 ),
                       Point( erosion_size, erosion_size ) );
    erode(img,out,element);
    
}

void imgdilate(Mat& img,
               Mat& out,
               int erosion_size)
{
    Mat element = getStructuringElement( MORPH_ELLIPSE,
                       Size( 2*erosion_size + 1, 2*erosion_size+1 ),
                       Point( erosion_size, erosion_size ) );
    dilate(img,out,element);
    
}

void extractwindows(Mat& img,Mat& out, vector<Point2f> *points)
{
    out=img.clone();
    vector<float> pts;//The HOG descriptors
    Mat refout;
    int included=0;
    cout<<"Refining the rectangles"<<endl;
    for(int i=0;i<points[0].size();i++)
    {
        const Point2f& p1=points[0][i];
        Rect windows(p1.x, p1.y, widthrect, heightrect);
        //Mat roi=img(windows).clone();
        //showhog(roi,refout,pts);
        //if(comparehog(pts))
            rectangle(out, windows, Scalar(255), 1, 8, 0);
    }
}

void extractwindowsrefined(Mat& img,Mat &blob, Mat& out, vector<Point2f> *points, vector < vector<Point2i> > &blobs)
{
    int seen[blobs.size()+1]={0};//1 if that component is already covered
    out=img.clone();
    vector<float> pts;//The HOG descriptors
    Mat refout;
    int classes=0;
    int found=0,k=0;
    int hogvalue=0;
    int count=0;
    Rect r;
    for(int i=0;i<points[0].size();i++)
    {
        //cout<<"Inside the first loop"<< i<< points[0].size()<< endl;
        count=count+1;
        const Point2f& p1=points[0][i];
        //cout<<round(p1.x)<<"  "<<round(p1.y)<<"  "<<blob.rows<<blob.cols<< " Count "<< count << endl;
        classes = blob.at<int>(round(p1.y),round(p1.x));
        //cout<<"class value "<<classes<<endl;
        if(classes>1000)
            continue;
        
        //cout<< classes << "value" <<seen[classes] << endl;
        //cout<<" Classes "<<classes<<endl;
        Rect windows(p1.x, p1.y, widthrect, heightrect);
        //cout<<" Classes "<<classes<<endl;
        if(classes>=1 && seen[classes]==0)
        {
            //cout<<"class value "<<classes<<endl;
            
            for(k=0;k<blobs[classes-1].size();k++)
            {
                
                int x = blobs[classes-1][k].x;
                int y = blobs[classes-1][k].y;
                //cout<<k<<blobs[classes].size()<<endl;
                if(seen[classes]==0)
                {              
                    Rect windows2(x, y, widthrect,heightrect);
                    if(windows2.x + windows2.width<=img.cols && windows2.y + windows2.height<=img.rows)
                        windows = windows | windows2;
                }
            }
            if(windows.x + windows.width<=img.cols && windows.y + windows.height<=img.rows)
            {
                Mat roi=img(windows).clone();
                showhog(roi,refout,pts);
                hogvalue=comparehog(pts);
                //hogvalue=1;
                //cout << "Computing HOG "<< hogvalue << endl;
                if(hogvalue)
                {
                    r = frame1 & windows;
                    if(r.area()==windows.area())
                        rectangle(out, windows, Scalar(255), 1, 8, 0);
                    r = frame2 & windows;
                    if(r.area()==windows.area())
                        rectangle(out, windows, Scalar(255), 1, 8, 0);
                    r = frame3 & windows;
                    if(r.area()==windows.area())
                        rectangle(out, windows, Scalar(255), 1, 8, 0);
                    r = frame4 & windows;
                    if(r.area()==windows.area())
                        rectangle(out, windows, Scalar(255), 1, 8, 0);
                    
                }
            }
            //rectangle(out, windows, Scalar(255), 1, 8, 0);
            seen[classes]=1;
        }
        //cout<<"Drawing Rectangle "<< blobs.size()<< endl;
        //rectangle(out, windows, Scalar(255), 1, 8, 0);
        //cout<<"Drew Rectangle "<< blobs.size()<< endl;
    }
    cout<<"Exited the function of refining the windows and extracting"<< endl;
}

/*
 We will call this function only to extract static windows from the images
 */
void extractstaticwindowsrefined(Mat& img,Mat& diff,Mat &blob, Mat& out, vector<Point2f> *points, vector < vector<Point2i> > &blobs)
{
    int seen[blobs.size()+1]={0};//1 if that component is already covered
    //out=img.clone();
    cvtColor(img, img, CV_BGR2GRAY);
    vector<float> pts;//The HOG descriptors
    Mat refout;
    int classes=0;
    int found=0,k=0;
    int hogvalue=0;
    int count=0;
    Rect r1,r2,r3,r4,r;
    for(int i=0;i<points[0].size();i++)
    {
        //cout<<"Inside the first loop"<< i<< points[0].size()<< endl;
        count=count+1;
        const Point2f& p1=points[0][i];
        //cout<<round(p1.x)<<"  "<<round(p1.y)<<"  "<<blob.rows<<blob.cols<< " Count "<< count << endl;
        classes = blob.at<int>(round(p1.y),round(p1.x));
        //cout<<"class value "<<classes<<endl;
        if(classes>1000)
            continue;
        
        //cout<< classes << "value" <<seen[classes] << endl;
        //cout<<" Classes "<<classes<<endl;
        Rect windows(p1.x, p1.y, widthrect, heightrect);
        //cout<<" Classes "<<classes<<endl;
        if(classes>=1 && seen[classes]==0)
        {
            //cout<<"class value "<<classes<<endl;
            
            for(k=0;k<blobs[classes-1].size();k++)
            {
                
                int x = blobs[classes-1][k].x;
                int y = blobs[classes-1][k].y;
                //cout<<k<<blobs[classes].size()<<endl;
                if(seen[classes]==0)
                {              
                    Rect windows2(x, y, widthrect,heightrect);
                    if(windows2.x + windows2.width<=img.cols && windows2.y + windows2.height<=img.rows)
                        windows = windows | windows2;
                }
            }
            if(windows.x + windows.width<=img.cols && windows.y + windows.height<=img.rows)
            {
                Mat roi=img(windows).clone();
                showhog(roi,refout,pts);
                hogvalue=comparehog(pts);
                Mat roidiff=diff(windows).clone();
                double motion = cv::sum( roidiff )[0];
                //hogvalue=1;
                //cout << "Computing HOG "<< hogvalue<<"Motion Value "<<motion << endl;
                if(hogvalue && motion <25000)
                {
                    r1 = frame1 & windows;
                    /*if(r1.area()==windows.area())
                    {
                        rectangle(out, windows, Scalar(255,255,0), 1, 8, 0);
                    }*/
                    r2 = frame2 & windows;
                    /*if(r2.area()==windows.area())
                    {
                        rectangle(out, windows, Scalar(255,255,0), 1, 8, 0);
                    }*/
                    r3 = frame3 & windows;
                    /*if(r3.area()==windows.area())
                    {
                        rectangle(out, windows, Scalar(255,255,0), 1, 8, 0);
                    }*/
                    r4 = frame4 & windows;
                    /*if(r4.area()==windows.area())
                    {
                        rectangle(out, windows, Scalar(255,255,0), 1, 8, 0);
                    }*/
                    r = rects1 & windows;
                    if(r.area()> 0 && r1.area()==windows.area())
                    {
                        if(rectstat[1])
                            rectangle(out, rects1, Scalar(255,255,0), 1, 8, 0);//Static plus Moving cases
                        else
                            rectangle(out, rects1, Scalar(0,255,255), 1, 8, 0);
                        rectstat[1]=0;
                    }
                    r = rects2 & windows;
                    if(r.area()> 0 && r1.area()==windows.area())
                    {
                        if(rectstat[2])
                            rectangle(out, rects2, Scalar(255,255,0), 1, 8, 0);//Static plus Moving cases
                        else
                            rectangle(out, rects2, Scalar(0,255,255), 1, 8, 0);
                        rectstat[2]=0;
                    }
                    r = rects3 & windows;
                    if(r.area()> 0 && r1.area()==windows.area())
                    {
                        if(rectstat[3])
                            rectangle(out, rects3, Scalar(255,255,0), 1, 8, 0);//Static plus Moving cases
                        else
                            rectangle(out, rects3, Scalar(0,255,255), 1, 8, 0);
                        rectstat[3]=0;
                    }
                    r = rects4 & windows;
                    if(r.area()> 0 && r2.area()==windows.area())
                    {
                        if(rectstat[4])
                            rectangle(out, rects4, Scalar(255,255,0), 1, 8, 0);//Static plus Moving cases
                        else
                            rectangle(out, rects4, Scalar(0,255,255), 1, 8, 0);
                        rectstat[4]=0;
                    }
                    r = rects5 & windows;
                    if(r.area()> 0 && r2.area()==windows.area())
                    {
                        if(rectstat[5])
                            rectangle(out, rects5, Scalar(255,255,0), 1, 8, 0);//Static plus Moving cases
                        else
                            rectangle(out, rects5, Scalar(0,255,255), 1, 8, 0);
                        rectstat[5]=0;
                    }
                    r = rects6 & windows;
                    if(r.area()> 0 && r2.area()==windows.area())
                    {
                        if(rectstat[6])
                            rectangle(out, rects6, Scalar(255,255,0), 1, 8, 0);//Static plus Moving cases
                        else
                            rectangle(out, rects6, Scalar(0,255,255), 1, 8, 0);
                        rectstat[6]=0;
                    }
                    r = rects7 & windows;
                    if(r.area()> 0 && r3.area()==windows.area())
                    {
                        if(rectstat[7])
                            rectangle(out, rects7, Scalar(255,255,0), 1, 8, 0);//Static plus Moving cases
                        else
                            rectangle(out, rects7, Scalar(0,255,255), 1, 8, 0);
                        rectstat[7]=0;
                    }
                    r = rects8 & windows;
                    if(r.area()> 0 && r3.area()==windows.area())
                    {
                        if(rectstat[8])
                            rectangle(out, rects8, Scalar(255,255,0), 1, 8, 0);//Static plus Moving cases
                        else
                            rectangle(out, rects8, Scalar(0,255,255), 1, 8, 0);
                        rectstat[8]=0;
                    }
                    r = rects9 & windows;
                    if(r.area()> 0 && r3.area()==windows.area())
                    {
                        if(rectstat[9])
                            rectangle(out, rects9, Scalar(255,255,0), 1, 8, 0);//Static plus Moving cases
                        else
                            rectangle(out, rects9, Scalar(0,255,255), 1, 8, 0);
                        rectstat[9]=0;
                    }
                    r = rects10 & windows;
                    if(r.area()> 0 && r4.area()==windows.area())
                    {
                        if(rectstat[10])
                            rectangle(out, rects10, Scalar(255,255,0), 1, 8, 0);//Static plus Moving cases
                        else
                            rectangle(out, rects10, Scalar(0,255,255), 1, 8, 0);
                        rectstat[10]=0;
                    }
                    r = rects11 & windows;
                    if(r.area()> 0 && r4.area()==windows.area())
                    {
                        if(rectstat[11])
                            rectangle(out, rects11, Scalar(255,255,0), 1, 8, 0);//Static plus Moving cases
                        else
                            rectangle(out, rects11, Scalar(0,255,255), 1, 8, 0);
                        rectstat[11]=0;
                    }
                    r = rects12 & windows;
                    if(r.area()> 0 && r4.area()==windows.area())
                    {
                        if(rectstat[12])
                            rectangle(out, rects12, Scalar(255,255,0), 1, 8, 0);//Static plus Moving cases
                        else
                            rectangle(out, rects12, Scalar(0,255,255), 1, 8, 0);
                        rectstat[12]=0;
                    }
                    
                }
            }
            //rectangle(out, windows, Scalar(255), 1, 8, 0);
            seen[classes]=1;
        }
        //cout<<"Drawing Rectangle "<< blobs.size()<< endl;
        //rectangle(out, windows, Scalar(255), 1, 8, 0);
        //cout<<"Drew Rectangle "<< blobs.size()<< endl;
    }
    //cout<<"Exited the function of refining the windows and extracting"<< endl;
}

void extractwindowsclose(Mat& img,Mat &blob, Mat& out, vector<Point2f> *points, vector < vector<Point2i> > &blobs)
{
    out=img.clone();
    const Point2f& p1=points[0][0];
    Rect windows(p1.x, p1.y, widthrect, heightrect);
    Mat refout;
    int hogvalue=0;
    vector<float> pts;//The HOG descriptors
    for(int i=1;i<points[0].size();i++)
    {
        const Point2f& p1=points[0][i];
        Rect windows2(p1.x, p1.y, widthrect, heightrect);
        if(windows2.x + windows2.width<=img.cols && windows2.y + windows2.height<=img.rows)
        {
            windows = windows | windows2; 
            Mat roi=img(windows).clone();
            showhog(roi,refout,pts);
            hogvalue=comparehog(pts);
            if(hogvalue)
                rectangle(out, windows2, Scalar(255), 1, 8, 0);
        }
    }
    //rectangle(out, windows, Scalar(255), 1, 8, 0);
}
/*Connected Components algorithm implementation taken from net*/
void FindBlobs(const Mat &binary,Mat &labels, vector < vector<Point2i> > &blobs)
{
    blobs.clear();

    // Fill the label_image with the blobs
    // 0  - background
    // 1  - unlabelled foreground
    // 2+ - labelled foreground
    labels=binary.clone();
    Mat label_image;
    binary.convertTo(label_image, CV_32SC1);

    int label_count = 2; // starts at 2 because 0,1 are used already

    for(int y=0; y < label_image.rows; y++) {
        int *row = (int*)label_image.ptr(y);
        for(int x=0; x < label_image.cols; x++) {
            if(row[x] != 1) {
                //cout<<"continuing!!"<<row[x]<<endl;
                continue;
            }

            Rect rect;
            //cout<<"Flood Filled"<<endl;
            floodFill(label_image, Point(x,y), label_count, &rect, 0, 0, 4);
            //cout<<"Flood Filled done"<<endl;

            vector <Point2i> blob;

            for(int i=rect.y; i < (rect.y+rect.height); i++) {
                int *row2 = (int*)label_image.ptr(i);
                for(int j=rect.x; j < (rect.x+rect.width); j++) {
                    if(row2[j] != label_count) {
                        continue;
                    }
                    //cout<<"assigning"<<endl;
                    labels.at<int>(i,j)=label_count;
                    blob.push_back(Point2i(j,i));
                }
            }
            //cout<<"Pushing the blobs now "<<label_count<<endl;
            blobs.push_back(blob);

            label_count++;
        }
    }
    //cout<<"Returning after finding all the blobs "<<endl;
}

