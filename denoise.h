//
// Created by leo on 2020/3/16.
//

#ifndef NOISE_DENOISE_H
#define NOISE_DENOISE_H

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <iostream>
#include "wavelet2d.h"

using namespace std;
using namespace cv;

class Denoise
{
public:
    Mat MedeanFilter(const Mat& src, int size);
    Mat MeanFilter(const Mat& src, int size);
    Mat GaussianFilter(const Mat& src, int size, double sigma);
    Mat BilateralFilter(const Mat& src, int size, double sigmaD, double sigmaR);
    Mat GaussianLowPassFilter(const Mat& src, double sigma);
    Mat WienerFilter(const Mat& src, const Mat& ref, int stddev);
    Mat WaveletFilter(const Mat& src, int num, int percentage);
    Mat NonLocalMeansFilter(const Mat& src, int searchWindowSize, int templateWindowSize, double sigma, double h);
    Mat NonLocalMeansFilter2(const Mat& src, int searchWindowSize, int templateWindowSize, double sigma, double h);
private:
    //for GaussianFlter
    vector<vector<double>> GaussianTemplate(int size, double sigma);
    //for GaussianLowPassFilter
    Mat ShiftQuadrant(const Mat& src);
    //for WienerFileter
    Mat GetSpectrum(const Mat& src);
};


#endif //NOISE_DENOISE_H
