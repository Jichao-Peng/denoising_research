//
// Created by leo on 2020/3/16.
//

#ifndef NOISE_NOISE_H
#define NOISE_NOISE_H

#include <opencv2/opencv.hpp>
#include <iostream>

using namespace std;
using namespace cv;

class Noise
{
public:
    Mat CreateSaltNoise(const Mat& src, int n);
    Mat CreateGaussianNoise(const Mat& src, double m, double sigma);
};


#endif //NOISE_NOISE_H
