#include<iostream>
#include<omp.h>
#include<cmath>
#include<opencv2/imgproc/imgproc.hpp>
#include<opencv2/highgui/highgui.hpp>
using namespace std;
using namespace cv;

int xGradient(Mat image, int x, int y)
{
return image.at<uchar>(y-1, x-1) +
            2*image.at<uchar>(y, x-1) +
             image.at<uchar>(y+1, x-1) -
              image.at<uchar>(y-1, x+1) -
               2*image.at<uchar>(y, x+1) -
                image.at<uchar>(y+1, x+1);
}
int yGradient(Mat image, int x, int y)
{
    return image.at<uchar>(y-1, x-1) +
            2*image.at<uchar>(y-1, x) +
             image.at<uchar>(y-1, x+1) -
              image.at<uchar>(y+1, x-1) -
               2*image.at<uchar>(y+1, x) -
                image.at<uchar>(y+1, x+1);
}
int main()

 {

  Mat src, grey, dst;
 double start, end;
     start = omp_get_wtime();
  //int gx, gy, sum;
src= imread("kurt.jpg");  
cvtColor(src,grey,CV_BGR2GRAY);
  dst = grey.clone();
  if( !grey.data )
  { return -1; }
 #pragma omp parallel for  
  for(int y = 0; y < grey.rows; y++)
        for(int x = 0; x < grey.cols; x++)
            dst.at<uchar>(y,x) = 0;  
 #pragma omp parallel for
   for(int y = 1; y < grey.rows - 1; y++){
        for(int x = 1; x < grey.cols - 1; x++){
            int gx = xGradient(grey, x, y);
            int gy = yGradient(grey, x, y);
            int sum = abs(gx) + abs(gy);
            sum = sum > 255 ? 255:sum;
            sum = sum < 0 ? 0 : sum;
            dst.at<uchar>(y,x) = sum;

                }
    }
    namedWindow("sobel");
    imshow("sobel", dst);
    namedWindow("grayscale");
    imshow("grayscale", grey);
    namedWindow("Original");
    imshow("Original", src);
    end = omp_get_wtime();
    cout<<"time is: "<<(end-start)<< " seconds" <<endl;
waitKey();
    return 0;
}
