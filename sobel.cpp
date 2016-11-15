#include <iostream>
#include <omp.h>
#include <cmath>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
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

/**
  * Calculates the 2d zero-mean gaussian with the given x, y and standard deviation.
 **/
float normal_pdf_2d(int x, int y, float standard_deviation) 
{
    static const float inverted_2pi = 0.15915494309189535;
    float exp = (x*x+y*y)/(2*standard_deviation*standard_deviation);
    return inverted_2pi / (standard_deviation * standard_deviation)
           * std::exp(-exp);
}

/**
  * Applies a hard coded blurr with filter size 3 at the given x and y in IMAGE.
  * Returns the blurred value to place at x, y im IMAGE.
 **/
int blurr_filter_3 (Mat image, int x, int y, int standard_deviation)
{
    return normal_pdf_2d(-1, -1, standard_deviation) * image.at<uchar>(y-1, x-1) +
           normal_pdf_2d(-1, 0, standard_deviation) * image.at<uchar>(y, x-1) +
           normal_pdf_2d(-1, 1, standard_deviation) * image.at<uchar>(y+1, x-1) +
           normal_pdf_2d(0, -1, standard_deviation) * image.at<uchar>(y-1, x) +
           normal_pdf_2d(0, 0, standard_deviation) * image.at<uchar>(y, x) +
           normal_pdf_2d(0, 1, standard_deviation) * image.at<uchar>(y+1, x) +
           normal_pdf_2d(1, -1, standard_deviation) * image.at<uchar>(y-1, x+1) +
           normal_pdf_2d(1, 0, standard_deviation) * image.at<uchar>(y, x+1) +
           normal_pdf_2d(1, 1, standard_deviation) * image.at<uchar>(y+1, x+1);
}

/**
  * Generalized blurr with the given filter size and standard deviation.
 **/
int blurr (Mat image, int x, int y, int filter_size, int standard_deviation)
{
    float sum = 0.f;
    int i, j;
    for (i = -filter_size; i <= filter_size; i++) {
        for (j = -filter_size; j <= filter_size; j++) {
          sum += normal_pdf_2d(i, j, standard_deviation) * image.at<uchar>(y+j, x+i);
        }
    }
    return int(sum);
}

// 1 for success, 0 for fail
int grayscale(Mat &src, Mat &grey, Mat &dst) {
  double start, end;
  start = omp_get_wtime();
  int gx, gy, sum;
  // TODO: replace this so it reads the specific image rather than a hardcoded one
  src= imread("kurt.jpg");  
  cvtColor(src,grey,CV_BGR2GRAY);
  dst = grey.clone();
  if( !grey.data )
  { 
    return 0; 
  }
  return 1;
}

/** 
 *  Applies a Sobel filter to a greyscale image in GREY and writes the result values
 *  to DST.
 **/
int sobel(Mat &grey, Mat &dst) {
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
}

// Grayscales and runs sober operation
int main()
{
  Mat src, grey, dst;

  double start, end;
  start = omp_get_wtime();

  if (!grayscale(src, grey, dst)) {
    return -1;
  }
  sobel(grey, dst);

  // For debugging purposes, can get rid of this once we're done with it
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
