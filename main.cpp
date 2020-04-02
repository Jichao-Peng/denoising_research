#include <iostream>

#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include "noise.h"
#include "denoise.h"

using namespace std;
using namespace cv;

int main() {
    Mat src = imread("house.jpg", 2);
    namedWindow("src",1);
    imshow("src", src);

    /*
     * 添加噪声
     */
    Noise noise;

    //椒盐噪声
//    Mat saltNoise = noise.CreateSaltNoise(src, 5000);
//    namedWindow("saltNoise",1);
//    imshow("saltNoise", saltNoise);

    //高斯噪声
    Mat gaussianNoise = noise.CreateGaussianNoise(src, 0,0.5);
    namedWindow("gaussianNoise",1);
    imshow("gaussianNoise", gaussianNoise);

    /*
     * 消除噪声
     */
    Denoise denoise;

//    //中值滤波
//    Mat medeanDenoise = denoise.MedeanFilter(saltNoise, 3);
//    namedWindow("medeanDenoise",1);
//    imshow("medeanDenoise", medeanDenoise);

//    //均值滤波
//    Mat meanDenoise = denoise.MeanFilter(gaussianNoise, 3);
//    namedWindow("meanDenoise",1);
//    imshow("meanDenoise", meanDenoise);
//
//    //高斯滤波
//    Mat gaussianDenoise = denoise.GaussianFilter(gaussianNoise, 3, 0.5);
//    namedWindow("gaussianDenoise",1);
//    imshow("gaussianDenoise", gaussianDenoise);
//
//    //双边滤波
//    Mat bilateralDenoise = denoise.BilateralFilter(gaussianNoise, 3, 0.5, 0.5);
//    namedWindow("bilateralDenoise",1);
//    imshow("bilateralDenoise", gaussianDenoise);


//    //OpenCV自带的双边滤波
//    Mat cvBilateralDenoise;
//    bilateralFilter(gaussianNoise, cvBilateralDenoise, 5, 0.2, 0.2);
//    namedWindow("cvBilateralDenoise",1);
//    imshow("cvBilateralDenoise", cvBilateralDenoise);


//    //OpenCV自带NLM降噪
//    Mat cvNmlDenoise;
//    double time1 = static_cast<double>( getTickCount());
//    fastNlMeansDenoising(gaussianNoise, cvNmlDenoise, 20, 7, 21);
//    double time2 = (static_cast<double>( getTickCount()) - time1)/getTickFrequency();
//    cout<<"cost time："<< time2 <<"seconds"<<endl;
//    namedWindow("cvNmlDenoise",1);
//    imshow("cvNmlDenoise", cvNmlDenoise);

    //高斯低通滤波（频域）
//    Mat gaussianLowPassDenoise = denoise.GaussianLowPassFilter(gaussianNoise, 80);
//    namedWindow("gaussianLowPassDenoise",1);
//    imshow("gaussianLowPassDenoise", gaussianLowPassDenoise);

    //维纳滤波
//    Mat ref = imread("lena.bmp",2);
//    Mat WienerDenoise = denoise.WienerFilter(gaussianNoise, ref ,20);
//    namedWindow("WienerDenoise",1);
//    imshow("WienerDenoise", WienerDenoise);

    //小波变换
//    Mat WaveletDenoise = denoise.WaveletFilter(gaussianNoise, 3 , 10);
//    namedWindow("WaveletDenoise",1);
//    imshow("WaveletDenoise", WaveletDenoise);

    //非局部均值滤波
    double time1 = static_cast<double>( getTickCount());
    Mat NonLocalMeansFilter = denoise.NonLocalMeansFilter
            (gaussianNoise, 21 , 7, 20, 0.3);
    double time2 = (static_cast<double>( getTickCount()) - time1)/getTickFrequency();
    cout<<"cost time："<< time2 <<"seconds"<<endl;
    namedWindow("NonLocalMeansFilter",1);
    imshow("NonLocalMeansFilter", NonLocalMeansFilter);
//    cost time：135.034seconds
//    cost time：29.6425seconds;
//    cost time：0.512942seconds
    waitKey();
    return 0;
}
