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
#define heightrect 96
#define widthrect 64

void imgerode(Mat& img,
              Mat& out,
              int erosion_size)
{
    Mat element = getStructuringElement( MORPH_RECT,
                       Size( 2*erosion_size + 1, 2*erosion_size+1 ),
                       Point( erosion_size, erosion_size ) );
    erode(img,out,element);
    
}

void imgdilate(Mat& img,
               Mat& out,
               int erosion_size)
{
    Mat element = getStructuringElement( MORPH_RECT,
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
    //cout<<"Refining the rectangles"<<endl;
    for(int i=0;i<points[0].size();i++)
    {
        const Point2f& p1=points[0][i];
        Rect windows(p1.x, p1.y, heightrect, widthrect);
        Mat roi=img(windows).clone();
        showhog(roi,refout,pts);
        if(comparehog(pts))
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
    for(int i=0;i<points[0].size();i++)
    {
        //cout<<"Inside the first loop"<< i<< points[0].size()<< endl;
        count=count+1;
        const Point2f& p1=points[0][i];
        //cout<<round(p1.x)<<"  "<<round(p1.y)<<"  "<<blob.rows<<blob.cols<< " Count "<< count << endl;
        classes = blob.at<double>(round(p1.y),round(p1.x));
        //cout<< classes << "value" <<seen[classes] << endl;
        //cout<<" Classes "<<classes<<endl;
        Rect windows(p1.x, p1.y, widthrect, heightrect);
        //cout<<" Classes "<<classes<<endl;
        if(classes>=1 && seen[classes]==0)
        {
            
            
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
                //cout << "Computing HOG "<< hogvalue << endl;
                if(hogvalue)
                    rectangle(out, windows, Scalar(255), 1, 8, 0);
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
            floodFill(label_image, Point(x,y), label_count, &rect, 0, 0, 4);

            vector <Point2i> blob;

            for(int i=rect.y; i < (rect.y+rect.height); i++) {
                int *row2 = (int*)label_image.ptr(i);
                for(int j=rect.x; j < (rect.x+rect.width); j++) {
                    if(row2[j] != label_count) {
                        continue;
                    }
                    //cout<<"assigning"<<endl;
                    labels.at<double>(i,j)=label_count;
                    blob.push_back(Point2i(j,i));
                }
            }

            blobs.push_back(blob);

            label_count++;
        }
    }
}

