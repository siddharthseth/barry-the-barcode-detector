#include "stdafx.h"
#include<iostream>
#include<omp.h>
#include<cmath>
#include <boost/random.hpp>
#include <boost/random/normal_distribution.hpp>
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

float 2d_normal_pdf(int x, int y, float standard_deviation) 
{
    static const float inverted_2pi = 0.15915494309189535;
    float exp = (x*x+y*y)/(2*standard_deviation*standard_deviation);
    return inverted_2pi / (standard_deviation * standard_deviation)
           * std::exp(-exp);
}

int blurr_filter_3 (Mat image, int x, int y, int standard_deviation)
{
    return 2d_normal_pdf(-1, -1, standard_deviation) * image.at<uchar>(y-1, x-1) +
           2d_normal_pdf(-1, 0, standard_deviation) * image.at<uchar>(y, x-1) +
           2d_normal_pdf(-1, 1, standard_deviation) * image.at<uchar>(y+1, x-1) +
           2d_normal_pdf(0, -1, standard_deviation) * image.at<uchar>(y-1, x) +
           2d_normal_pdf(0, 0, standard_deviation) * image.at<uchar>(y, x) +
           2d_normal_pdf(0, 1, standard_deviation) * image.at<uchar>(y+1, x) +
           2d_normal_pdf(1, -1, standard_deviation) * image.at<uchar>(y-1, x+1) +
           2d_normal_pdf(1, 0, standard_deviation) * image.at<uchar>(y, x+1) +
           2d_normal_pdf(1, 1, standard_deviation) * image.at<uchar>(y+1, x+1);
}
int blurr (Mat image, int x, int y, int filter_size, int standard_deviation)
{
    float sum = 0.f;
    int i, j;
    for (i = -filter_size; i <= filter_size; i++) {
        for (j = -filter_size; j <= filter_size; j++) {
          sum += 2d_normal_pdf(i, j, standard_deviation) * image.at<uchar>(y+j, x+i);
        }
    }
    return int(sum);
}

int main()
{
  Mat src, grey, dst;
  double start, end;
  start = omp_get_wtime();
  int gx, gy, sum;
  src= imread("E:/image.jpg");  
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
