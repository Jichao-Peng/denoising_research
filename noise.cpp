//
// Created by leo on 2020/3/16.
//

#include "noise.h"

Mat Noise::CreateSaltNoise(const Mat &src, int n)
{
    Mat dst = src.clone();
    for(int k = 0; k<n; k++)
    {
        int i = rand() % dst.rows;
        int j = rand() % dst.cols;

        if(dst.channels() == 1)
        {
            dst.at<uchar>(i,j) = 255;
        }
        else
        {
            dst.at<Vec3b>(i,j)[0] = 255;
            dst.at<Vec3b>(i,j)[1] = 255;
            dst.at<Vec3b>(i,j)[2] = 255;
        }
    }

    for(int k = 0; k<n; k++)
    {
        int i = rand() % dst.rows;
        int j = rand() % dst.cols;

        if(dst.channels() == 1)
        {
            dst.at<uchar>(i,j) = 0;
        }
        else
        {
            dst.at<Vec3b>(i,j)[0] = 0;
            dst.at<Vec3b>(i,j)[1] = 0;
            dst.at<Vec3b>(i,j)[2] = 0;
        }
    }
    return dst;
}

Mat Noise::CreateGaussianNoise(const Mat &src, double mu, double sigma)
{
    Mat dst = src.clone();
    int row = dst.rows;
    int col = dst.cols;
    for(int i = 0; i<row; i++)
    {
        for(int j = 0; j<col; j++)
        {
            if(dst.channels() == 1)
            {
                //构建高斯噪声
                double u1, u2;
                do
                {
                    u1 = rand() * (1.0 / RAND_MAX);
                    u2 = rand() * (1.0 / RAND_MAX);
                } while (u1 <= numeric_limits<double>::min());//u1不能为0

                double z = sigma * sqrt(-2.0 * log(u1)) * cos(2 * CV_PI * u2) + mu;
                //double z = sigma * sqrt(-2.0 * log(u1)) * sin(2 * CV_PI * u2) + mu;

                int val = dst.at<uchar>(i,j) + z * 32;
                val = (val<0)?0:val;
                val = (val>255)?255:val;

                dst.at<uchar>(i,j) = (uchar)val;
            }
            else
            {
                for(int k = 0; k<dst.channels(); k++)
                {
                    //构建高斯噪声
                    double u1, u2;
                    do
                    {
                        u1 = rand() * (1.0 / RAND_MAX);
                        u2 = rand() * (1.0 / RAND_MAX);
                    } while (u1 <= numeric_limits<double>::min());//u1不能为0

                    double z = sigma * sqrt(-2.0 * log(u1)) * cos(2 * CV_PI * u2) + mu;
                    //double z = sigma * sqrt(-2.0 * log(u1)) * sin(2 * CV_PI * u2) + mu;

                    int val = dst.at<Vec3b>(i,j)[k] + z * 32;
                    int test = dst.at<Vec3b>(i,j)[k];
                    val = (val<0)?0:val;
                    val = (val>255)?255:val;
                    dst.at<Vec3b>(i,j)[k] = (uchar)val;
                }
            }
        }
    }
    return dst;
}