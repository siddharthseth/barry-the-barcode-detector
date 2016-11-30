#include <iostream>
#include <omp.h>
#include <cmath>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <sstream>
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
int blurr_filter_3 (Mat image, int x, int y, float standard_deviation)
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
int blurr (Mat image, int x, int y, int filter_size, float standard_deviation)
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

int blurr_image (Mat &image, Mat &dst, int filter_size, int margin)
{
    #pragma omp parallel for
    for (int y = 0; y < dst.rows; y++)
        for (int x = 0; x < dst.cols; x++)
            dst.at<uchar>(y, x) = 0;
    #pragma omp parallel for
    for (int y = filter_size + margin; y < image.rows - filter_size - margin; y++) 
        for (int x = filter_size + margin; x < image.cols - filter_size - margin; x++)
        {
            int blurred = blurr (image, x, y, filter_size, 1.0f);
            blurred = blurred > 255 ? 255 : blurred;
            blurred = blurred < 0 ? 0 : blurred;
            dst.at<uchar>(y, x) = blurred;
        }
}

// 1 for success, 0 for fail
int grayscale(Mat &src, Mat &grey, Mat &dst) {
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
      int sum = abs(gx) - abs(gy);
      sum = sum > 255 ? 255:sum;
      sum = sum < 0 ? 0 : sum;
      dst.at<uchar>(y,x) = sum;
    }
  }
}

int sobel_add(Mat &grey, Mat &dst) {
  #pragma omp parallel for  
  for(int y = 0; y < grey.rows; y++)
    for(int x = 0; x < grey.cols; x++)
      dst.at<uchar>(y,x) = 0;  
  #pragma omp parallel for
  for(int y = 1; y < grey.rows - 1; y++){
    for(int x = 1; x < grey.cols - 1; x++){
      int gx = xGradient(grey, x, y);
      int gy = yGradient(grey, x, y);
      int sum = abs(gy)*abs(gy) + abs(gx)*abs(gx);
      sum = sum > 255 ? 255:sum;
      sum = sum < 0 ? 0 : sum;
      dst.at<uchar>(y,x) = sum;
    }
  }
}

// Takes greyscaled picture, thresholds it and puts it dst
// In the python file, they use 225 for threshold_val and 255 for max_val
int threshold(Mat grey, Mat &dst, int threshold_val, int max_val) {
  #pragma omp parallel for  
  for(int y = 0; y < grey.rows; y++)
    for(int x = 0; x < grey.cols; x++)
      dst.at<uchar>(y,x) = 0;  
  #pragma omp parallel for
  for (int y = 0; y < grey.rows-1; y++) {
    for (int x = 0; x < grey.cols-1; x++) {
      // cout << "Thresholding at: " << x << ", " << y << "\n";
      if (grey.at<uchar>(y, x) >= threshold_val) {
        dst.at<uchar>(y, x) = max_val;
      } else {
        dst.at<uchar>(y, x) = 0;
      }
    }
  }
}

// Grayscales and runs sobel operation
/**  @function Erosion
  *  @Mat src -> source image Matrix
  *  @Mat dst -> pointer to destination Matrix
  *  @int erosion_elem -> 0 = rect, 1 = cross, 2 = ellipse
  *  @int erosion_size -> represents size of erosion around anchor point
  */
void Erosion(Mat& src, Mat& dst, int erosion_elem, int erosion_size)
{
  int erosion_type;
  if( erosion_elem == 0 ){ erosion_type = MORPH_RECT; }
  else if( erosion_elem == 1 ){ erosion_type = MORPH_CROSS; }
  else if( erosion_elem == 2) { erosion_type = MORPH_ELLIPSE; }

  Mat element = getStructuringElement( erosion_type,
                                       Size( 2*erosion_size + 1, 2*erosion_size+1 ),
                                       Point( erosion_size, erosion_size ) );

  /// Apply the erosion operation
  erode( src, dst, element );
}

/** @function Dilation 
  * @Mat src -> source image Matrix
  * @Mat dst -> pointer to destination Matrix
  * @int dilation_elem -> 0 = rect, 1 = cross, 2 = ellipse
  * @int dilation_size -> represents size of dilation around anchor point
  */
void Dilation(Mat& src, Mat& dst, int dilation_elem, int dilation_size)
{
  int dilation_type;
  if( dilation_elem == 0 ){ dilation_type = MORPH_RECT; }
  else if( dilation_elem == 1 ){ dilation_type = MORPH_CROSS; }
  else if( dilation_elem == 2) { dilation_type = MORPH_ELLIPSE; }

  Mat element = getStructuringElement( dilation_type,
                                       Size( 2*dilation_size + 1, 2*dilation_size+1 ),
                                       Point( dilation_size, dilation_size ) );
  /// Apply the dilation operation
  dilate( src, dst, element );
}

void multByElem(Mat& in1, Mat& in2, Mat& result){
  #pragma omp for
  for(int y = 0; y < in1.rows; y++){
    for(int x = 0; x < in1.cols; x++){
      result.at<uchar>(y, x) = in1.at<uchar>(y, x) * in2.at<uchar>(y, x);
    }
  }
}

void argMaxX(Mat in, int (&results)[2], int threshold) 
{
  results[0] = 0;
  results[1] = 0;
  int sums[in.rows];
  int maxIndex = 0;
  int secondMaxIndex = 0;
  #pragma omp for
  for (int y = 0; y < in.rows; y++)
    sums[y] = 0;
  #pragma omp for
  for (int y = 0; y < in.rows; y++)
        for (int x = 0; x < in.cols; x++)
            if (in.at<uchar>(y, x) > threshold) {
              sums[y] += 1;
            }
            // sums[y] += in.at<uchar>(y, x);
            // sums[y] += 1;

  #pragma omp for
  for (int z = 0; z < in.rows; z++) {
    if (abs(z - maxIndex) <= 5 || abs(z - secondMaxIndex) <= 5)
      continue;
    if (sums[z] > results[1]) {
      #pragma omp critical
      if (sums[z] > results[0]) {
        results[1] = results[0];
        results[0] = sums[z];
        secondMaxIndex = maxIndex;
        maxIndex = z;
      } else {
        results[1] = sums[z];
        secondMaxIndex = z;
      }
    }
  }
  results[0] = maxIndex;
  results[1] = secondMaxIndex;
}

void argMaxY(Mat in, int (&results)[2], int threshold)
{
  results[0] = 0;
  results[1] = 0;
  int sums[in.cols];
  int maxIndex = 0;
  int secondMaxIndex = 0;
  #pragma omp for
  for (int x = 0; x < in.cols; x++)
    sums[x] = 0;
  #pragma omp for
  for (int x = 0; x < in.cols; x++)
    for (int y = 0; y < in.rows; y++){
      if (in.at<uchar>(y, x) > threshold) {
        sums[x] += 1;
      }
      // sums[x] += in.at<uchar>(y, x);
      // cout << (int)in.at<uchar>(y, x) << "\n";
      // sums[x] += 1;
    }

  #pragma omp for
  for (int z = 0; z < in.cols; z++) {
    if (abs(z - maxIndex) <= 5 || abs(z - secondMaxIndex) <= 5)
      continue;
    if (sums[z] > results[1]) {
      #pragma omp critical
      if (sums[z] > results[0]) {
        results[1] = results[0];
        results[0] = sums[z];
        secondMaxIndex = maxIndex;
        maxIndex = z;
      } else {
        results[1] = sums[z];
        secondMaxIndex = z;
      }
    }
  }
  results[0] = maxIndex;
  results[1] = secondMaxIndex;
}

void drawLines(Mat in, Mat &out, int * xCoords, int * yCoords) {
  #pragma omp for
  for (int y = 0; y < in.rows; y++) {
    for (int x = 0; x < in.cols; x++) {
      if (x == xCoords[0] || x == xCoords[1] || y == yCoords[0] || y == yCoords[1]){
        // out.at<uchar>(y, x) = 0;
        out.at<Vec3b>(y, x)[0] = 0;
        out.at<Vec3b>(y, x)[1] = 0;
        out.at<Vec3b>(y, x)[2] = 0;
      // } else {
        // out.at<uchar>(y, x) = 0;
      }
    }
  }
}

void rotateMatrix(Mat src, Mat &dst, int degree){
  Point2f pc(src.cols/2., src.rows/2.);
  Mat r = getRotationMatrix2D(pc, degree, 1.0);
  warpAffine(src, dst, r, src.size());
}

Mat runAlgo(Mat src, Mat grey, Mat dst){
  Mat blur1 = grey.clone(); 
  Mat blur2 = grey.clone(); 
  Mat sobel1 = grey.clone();
  Mat sobel2 = grey.clone();
  Mat temp = grey.clone();
  Mat dilated = grey.clone();
  Mat dilated2 = grey.clone();

  blurr_image(grey, blur1, 3, 0);
  sobel(blur1, temp);
  threshold(temp, sobel1, 225, 255);

  blurr_image(blur1, blur2, 3, 3);
  sobel(blur2, temp);
  threshold(temp, sobel2, 225, 255);

  blurr_image(sobel2, temp, 3, 6);
  threshold(temp, sobel2, 150, 255);

  Dilation(sobel2, dilated, 0, 10);
  Dilation(dilated, dilated2, 0, 10);

  sobel_add(dilated2, temp);
  blurr_image(temp, dst, 5, 9);

  int horizontal[2];
  int vertical[2];

  argMaxX(dst, horizontal, 100);
  argMaxY(dst, vertical, 100);

  temp = src.clone();
  drawLines(dst, temp, vertical, horizontal);
  return temp;
}

// Grayscales and runs sobel operation
int main(int argc, char** argv)
{
  if(argc < 2){
    cout << "Requires 1 input: of 'filename', invalid input given. Given " << argc-1 << " input(s)." << endl;
    return -1;
  }
  if(argc == 3){
      omp_set_num_threads(atoi(argv[2]));
  }

  Mat src, grey, dst;
  src= imread(argv[1]);  

  double start, end;
  start = omp_get_wtime();

  namedWindow("Original");
  imshow("Original", src);

  if(!grayscale(src, grey, dst)){
    return -1;
  }
  Mat final = runAlgo(src, grey, dst);

  /*
  #pragma omp for
  for(int i = 0; i <= 90; i += 10){
    Mat rotated = src.clone();
    if(i != 0) rotateMatrix(src, rotated, i);

    if (!grayscale(rotated, grey, dst)) continue;

    rotated = runAlgo(rotated, grey, dst);
    ostringstream os;
    os << i;
    string name = "Rotation by: " + os.str();
    namedWindow(name);
    imshow(name, rotated);
  }*/
  
  end = omp_get_wtime();
  cout<<"time to run 1 algo is: "<<(end-start)<< " seconds" <<endl;

  namedWindow("Output");
  imshow("Output", final);

  start = omp_get_wtime();
  Mat rotated = src.clone();
  rotateMatrix(src, rotated, 45);
  end = omp_get_wtime();
  cout<<"time to rotate once with opencv is: " << (end-start)<<" seconds"<<endl;


  waitKey();
  return 0;
}
