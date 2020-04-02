//
// Created by leo on 2020/3/16.
//

#include "denoise.h"

//中值滤波
Mat Denoise::MedeanFilter(const Mat &src, int size)
{
    Mat dst = src.clone();
    int start = size/2;
    for(int i = start; i < dst.rows-start; i++)
    {
        for(int j = start; j < dst.cols-start; j++)
        {
            vector<uchar> model;
            for(int m = i-start; m <= i+start; m++)
            {
                for(int n = j-start; n <= j+start; n++)
                {
                    model.push_back(src.at<uchar>(m,n));
                }
            }
            sort(model.begin(), model.end());
            dst.at<uchar>(i,j) = model[size*size/2];
        }
    }
    return dst;
}

//均值滤波
Mat Denoise::MeanFilter(const Mat &src, int size)
{
    Mat dst = src.clone();
    int start = size/2;
    for(int i = start; i < dst.rows-start; i++)
    {
        for(int j = start; j < dst.cols-start; j++)
        {
            int sum = 0;
            for(int m = i-start; m <= i+start; m++)
            {
                for(int n = j-start; n <= j+start; n++)
                {
                   sum += src.at<uchar>(m,n);
                }
            }
            dst.at<uchar>(i,j) = (uchar)(sum/size/size);
        }
    }
    return dst;
}

//高斯滤波
Mat Denoise::GaussianFilter(const Mat &src, int size, double sigma)
{
    vector<vector<double>> gaussianTemplate = GaussianTemplate(size, sigma);
    Mat dst = src.clone();
    int start = size/2;
    for(int i = start; i < dst.rows-start; i++)
    {
        for(int j = start; j < dst.cols-start; j++)
        {
            int sum = 0;
            for(int m = i-start; m <= i+start; m++)
            {
                for(int n = j-start; n <= j+start; n++)
                {
                    sum += src.at<uchar>(m,n)*gaussianTemplate[m-i+start][n-j+start];
                }
            }
            dst.at<uchar>(i,j) = (uchar)sum;
        }
    }
    return dst;
}

vector<vector<double>> Denoise::GaussianTemplate(int size, double sigma)
{
    vector<vector<double>> temp;
    double base = 1.0 / 2.0 / CV_PI / sigma / sigma;
    for(int i = 0; i < size; i++)
    {
        vector<double> vec;
        for(int j = 0; j < size; j++)
        {
            double a = (pow(i - size/2, 2) + pow(j - size/2, 2)) / 2.0 / sigma / sigma;
            double b = base * exp(-a);
            vec.push_back(b);
        }
        temp.push_back(vec);
    }
    return temp;
}

//双边滤波
Mat Denoise::BilateralFilter(const Mat &src, int size, double sigmaD, double sigmaR)
{
    vector<vector<double>> tempD;
    vector<double> tempR;
    Mat dst = src.clone();

    //生成定义域模板
    for(int i = 0; i < size; i++)
    {
        vector<double> vec;
        for(int j = 0; j < size; j++)
        {
            double a = (pow(i - size/2, 2) + pow(j - size/2, 2)) / 2.0 / sigmaD / sigmaD;
            double b = exp(-a);
            vec.push_back(b);
        }
        tempD.push_back(vec);
    }

    //生成值域模板
    for(int i = 0; i < 256; i++)
    {
        double a = (i * i / 2.0 / sigmaR / sigmaR);
        double b = exp(-a);
        tempR.push_back(b);
    }

    int start = size/2;
    for(int i = start; i < dst.rows-start; i++)
    {
        for(int j = start; j < dst.cols-start; j++)
        {
            double sum = 0;
            double weightSum = 0;
            for(int m = i-start; m <= i+start; m++)
            {
                for(int n = j-start; n <= j+start; n++)
                {
                    double weight = tempD[m-i+start][n-j+start] * tempR[abs(src.at<uchar>(m,n)-src.at<uchar>(i,j))];
                    sum += src.at<uchar>(m,n)*weight;
                    weightSum += weight;
                }
            }
            dst.at<uchar>(i,j) = (uchar)sum/weightSum;
        }
    }
    return dst;
}

//高斯低通滤波器（频域）
Mat Denoise::GaussianLowPassFilter(const Mat &src, double sigma)
{
    //这些图片是过程中会用到的，pad是原图像0填充后的图像，cpx是双通道频域图，mag是频域幅值图，dst是滤波后的图像
    Mat pad, cpx, mag, dst;

    //获取傅里叶变化最佳图片尺寸，为2的指数
    int m = getOptimalDFTSize(src.rows);
    int n = getOptimalDFTSize(src.cols);

    //对原始图片用0进行填充获得最佳尺寸图片
    copyMakeBorder(src, pad, 0, m-src.rows, 0, n-src.cols, BORDER_CONSTANT, Scalar::all(0));

    //生成高斯模板
    Mat gaussian(pad.size(),CV_32FC2);
    for(int i = 0; i<m; i++)
    {
        float* p = gaussian.ptr<float>(i);
        for(int j = 0; j<n; j++)
        {
            double d = pow(i-m/2, 2) + pow(j-n/2, 2);
            p[2*j] = expf(-d/sigma/sigma/2.0);
            p[2*j+1] = expf(-d/sigma/sigma/2.0);
        }
    }

    //建立双通道图片，其中planes[0]填充原始图片
    Mat planes[] = {Mat_<float>(pad), Mat::zeros(pad.size(), CV_32F)};
    merge(planes, 2, cpx);

    //进行傅里叶变换
    dft(cpx, cpx);

    //分离通道并进行象限变幻
    split(cpx, planes);
    planes[0] = ShiftQuadrant(planes[0]);
    planes[1] = ShiftQuadrant(planes[1]);
//    magnitude(planes[0], planes[1], mag);
//    mag += Scalar::all(1);
//    log(mag, mag);
//    normalize(mag, mag, 0, 1, CV_MINMAX);
//    imshow("mag1", mag);

    //进行滤波
    merge(planes, 2, cpx);
    multiply(cpx, gaussian, cpx);

    //分离通道并进行象限变幻
    split(cpx, planes);
    //计算幅值，并讲幅值存储再planes[0]中
    magnitude(planes[0], planes[1], mag);
//    mag += Scalar::all(1);
//    log(mag, mag);
//    normalize(mag, mag, 0, 1, CV_MINMAX);
//    imshow("mag2", mag);
    planes[0] = ShiftQuadrant(planes[0]);
    planes[1] = ShiftQuadrant(planes[1]);

    //重新合并实部planes[0]和虚部planes[1]
    merge(planes, 2, cpx);

    //进行反傅里叶变换
    idft(cpx, dst, DFT_SCALE | DFT_REAL_OUTPUT);

    dst.convertTo(dst, CV_8UC1);
    return dst;
}

Mat Denoise::ShiftQuadrant(const Mat &src)
{
    // 交换前
    // ×××××××××××××××××××××××
    // ×    q1    ×    q2    ×
    // ×          ×          ×
    // ×××××××××××××××××××××××
    // ×    q3    ×    q4    ×
    // ×          ×          ×
    // ×××××××××××××××××××××××

    // 交换后
    // ×××××××××××××××××××××××
    // ×    q4    ×    q3    ×
    // ×          ×          ×
    // ×××××××××××××××××××××××
    // ×    q2    ×    q1    ×
    // ×          ×          ×
    // ×××××××××××××××××××××××
    Mat dst = src.clone();
    int m = src.rows, n = src.cols;

    Mat q1(dst, Rect(0,0,n/2,m/2));
    Mat q2(dst, Rect(n/2,0,n/2,m/2));
    Mat q3(dst, Rect(0,m/2,n/2,m/2));
    Mat q4(dst, Rect(n/2,m/2,n/2,m/2));

    //交换象限
    Mat temp;
    q1.copyTo(temp);
    q4.copyTo(q1);
    temp.copyTo(q4);

    q2.copyTo(temp);
    q3.copyTo(q2);
    temp.copyTo(q3);
    return dst;
}

//维纳滤波
Mat Denoise::WienerFilter(const Mat &src, const Mat &ref, int stddev)
{
    //这些图片是过程中会用到的，pad是原图像0填充后的图像，cpx是双通道频域图，mag是频域幅值图，dst是滤波后的图像
    Mat pad, cpx, dst;

    //获取傅里叶变化最佳图片尺寸，为2的指数
    int m = getOptimalDFTSize(src.rows);
    int n = getOptimalDFTSize(src.cols);

    //对原始图片用0进行填充获得最佳尺寸图片
    copyMakeBorder(src, pad, 0, m-src.rows, 0, n-src.cols, BORDER_CONSTANT, Scalar::all(0));

    //获得参考图片频谱
    Mat tmpR(pad.rows, pad.cols, CV_8U);
    resize(ref, tmpR, tmpR.size());
    Mat refSpectrum = GetSpectrum(tmpR);

    //获得噪声频谱
    Mat tmpN(pad.rows, pad.cols, CV_32F);
    randn(tmpN, Scalar::all(0), Scalar::all(stddev));
    Mat noiseSpectrum = GetSpectrum(tmpN);

    //对src进行傅里叶变换
    Mat planes[] = {Mat_<float>(pad), Mat::zeros(pad.size(), CV_32F)};
    merge(planes, 2, cpx);
    dft(cpx, cpx);
    split(cpx, planes);

    //维纳滤波因子
    Mat factor = refSpectrum / (refSpectrum + noiseSpectrum);
    multiply(planes[0], factor, planes[0]);
    multiply(planes[1], factor, planes[1]);

    //重新合并实部planes[0]和虚部planes[1]
    merge(planes, 2, cpx);

    //进行反傅里叶变换
    idft(cpx, dst, DFT_SCALE | DFT_REAL_OUTPUT);

    dst.convertTo(dst, CV_8UC1);
    return dst;
}


Mat Denoise::GetSpectrum(const Mat &src)
{
    Mat dst, cpx;
    Mat planes[] = {Mat_<float>(src), Mat::zeros(src.size(), CV_32F)};
    merge(planes, 2, cpx);
    dft(cpx, cpx);
    split(cpx, planes);
    magnitude(planes[0], planes[1], dst);
    //频谱就是频域幅度图的平方
    multiply(dst, dst, dst);
    return dst;
}



Mat Denoise::WaveletFilter(const Mat &src, int num, int percentage)
{
    //需要讲Mat数据结构转换为vector数据结构
    vector<vector<double>> dwtInput(src.rows, vector<double>(src.cols));
    for(int i = 0; i<src.rows; i++)
    {
        for(int j = 0; j<src.cols; j++)
        {
            dwtInput[i][j] = (double)src.at<uchar>(i,j);
        }
    }

    //计算小波变换，输出结果是output
    vector<int> dwtLength;
    vector<double> dwtOutput, dwtFlag;
    dwt_2d(dwtInput, num, "db2", dwtOutput, dwtFlag, dwtLength);

    //******************************显示相关******************************
//    //通过dwtLength计算dim
//    vector<int> dwtDim;
//    dwt_output_dim_sym(dwtLength, dwtDim, num);
//
//    //dim的后两位是显示尺寸，并将dwtOutput构造成display
//    int dwtRow = dwtDim[dwtDim.size()-2], dwtCol = dwtDim[dwtDim.size()-1];
//    vector<vector<double>> dwtDisplay(dwtRow, vector<double>(dwtCol));
//    dispDWT(dwtOutput, dwtDisplay, dwtLength, dwtDim, num);
//
//    //获取display中的最大值
      double m = 0;
//    for(int i = 0; i < dwtRow; i++)
//    {
//        for(int j = 0; j < dwtCol; j++)
//        {
//            if(m < dwtDisplay[i][j])
//                m = dwtDisplay[i][j];
//        }
//    }
//
//    //dst是用于显示的图像
//    Mat dst = Mat::zeros(dwtRow,dwtCol,CV_8UC1);
//    for(int i = 0; i < dwtRow; i++)
//    {
//        for(int j = 0; j < dwtCol; j++)
//        {
//            if(dwtDisplay[i][j] <= 0.0)
//                dwtDisplay[i][j] = 0.0;
//            if(i<=dwtDim[0] && j<=dwtDim[1])
//                dst.at<uchar>(i,j) = (uchar)(dwtDisplay[i][j]/m*255);
//            else
//                dst.at<uchar>(i,j) = (uchar)(dwtDisplay[i][j]);
//        }
//    }
//    imshow("dst", dst);
//    waitKey();
    //******************************************************************


    //******************************滤波相关******************************
    //开始滤波
    int filterSize = int(dwtOutput.size() / percentage);
    vector<double> filter;
    for(auto it : dwtOutput)
    {
        filter.push_back(abs(it));
    }
    sort(filter.begin(), filter.end(), greater<double>());
    double threshold = filter.at(filterSize-1);
    for(int i = 0; i<dwtOutput.size(); i++)
    {
        double tmp = abs(dwtOutput[i]);
        if(tmp < threshold)
            dwtOutput.at(i) = 0.0;
    }
    //******************************************************************


    //******************************显示相关******************************
//    dispDWT(dwtOutput, dwtDisplay, dwtLength, dwtDim, num);
//
//    //获取display中的最大值
//    m = 0;
//    for(int i = 0; i < dwtRow; i++)
//    {
//        for(int j = 0; j < dwtCol; j++)
//        {
//            if(m < dwtDisplay[i][j])
//                m = dwtDisplay[i][j];
//        }
//    }
//
//    //dst是用于显示的图像
//    Mat dst2 = Mat::zeros(dwtRow,dwtCol,CV_8UC1);
//    for(int i = 0; i < dwtRow; i++)
//    {
//        for(int j = 0; j < dwtCol; j++)
//        {
//            if(dwtDisplay[i][j] <= 0.0)
//                dwtDisplay[i][j] = 0.0;
//            if(i<=dwtDim[0] && j<=dwtDim[1])
//                dst2.at<uchar>(i,j) = (uchar)(dwtDisplay[i][j]/m*255);
//            else
//                dst2.at<uchar>(i,j) = (uchar)(dwtDisplay[i][j]);
//        }
//    }
//    imshow("dst2", dst2);
//    waitKey();
    //******************************************************************

    //下面是进行小波反变换过程
    vector<vector<double>> idwtOutput(src.rows, vector<double>(src.cols));
    idwt_2d(dwtOutput, dwtFlag, "db2", idwtOutput, dwtLength);

    int idwtRow = idwtOutput.size();
    int idwtCol = idwtOutput[0].size();

    //获取idwtOutput中的最大值
    m = 0;
    for(int i = 0; i < idwtRow; i++)
    {
        for(int j = 0; j < idwtCol; j++)
        {
            if(m < idwtOutput[i][j])
                m = idwtOutput[i][j];
        }
    }

    //显示降噪后的图像
    Mat img = Mat::zeros(idwtRow, idwtCol, CV_8UC1);
    for(int i = 0; i<idwtRow; i++)
    {
        for(int j = 0; j<idwtCol; j++)
        {
            if(idwtOutput[i][j] <= 0.0)
            {
                idwtOutput[i][j] = 0.0;
            }
            img.at<uchar>(i,j) = (uchar)(idwtOutput[i][j]/m*255);
        }
    }
    return img;
}


Mat Denoise::NonLocalMeansFilter(const Mat &src, int searchWindowSize, int templateWindowSize, double sigma, double h)
{
    Mat dst, pad;
    dst = Mat::zeros(src.rows, src.cols, CV_8UC1);

    //构建边界
    int padSize = (searchWindowSize+templateWindowSize)/2;
    copyMakeBorder(src, pad, padSize, padSize, padSize, padSize, cv::BORDER_CONSTANT);

    int tN = templateWindowSize*templateWindowSize;
    int sN = searchWindowSize*searchWindowSize;

    vector<double> gaussian(256*256, 0);
    for(int i = 0; i<256*256; i++)
    {
        double g = exp(-max(i-2.0*sigma*sigma, 0.0))/(h*h);
        gaussian[i] = g;
        if(g<0.001)
            break;
    }

    //遍历图像上每一个像素
    for(int i = 0; i<src.rows; i++)
    {
        for(int j = 0; j<src.cols; j++)
        {
            cout<<i<<" "<<j<<endl;
            //遍历搜索区域每一个像素
            int pX = i+searchWindowSize/2;
            int pY = j+searchWindowSize/2;
            vector<vector<double>> weight(searchWindowSize, vector<double>(searchWindowSize, 0));
            double weightSum = 0;
            for(int m = searchWindowSize-1; m>=0; m--)
            {
                for(int n = searchWindowSize-1; n>=0; n--)
                {
                    int qX = i+m;
                    int qY = j+n;
                    int w = 0;
                    for(int x = templateWindowSize-1; x>=0; x--)
                    {
                        for(int y = templateWindowSize-1; y>=0; y--)
                        {
                            w += pow(pad.at<uchar>(pX+x, pY+y) - pad.at<uchar>(qX+x, qY+y), 2);
                        }
                    }
                    weight[m][n] = gaussian[(int)(w/tN)];
                    weightSum += weight[m][n];
                }
            }
            dst.at<uchar>(i,j) = 0;
            double sum = 0;
            for(int m = 0; m<searchWindowSize; m++)
            {
                for(int n = 0; n<searchWindowSize; n++)
                {
                   sum += pad.at<uchar>(i+templateWindowSize/2+m, j+templateWindowSize/2+n)*weight[m][n];
                }
            }
            dst.at<uchar>(i,j) = (uchar)(sum/weightSum);
        }
    }

    return dst;
}


Mat Denoise::NonLocalMeansFilter2(const Mat &src, int searchWindowSize, int templateWindowSize, double sigma, double h)
{
    Mat dst, pad;
    dst = Mat::zeros(src.rows, src.cols, CV_8UC1);

    //构建边界
    int padSize = (searchWindowSize+templateWindowSize)/2;
    copyMakeBorder(src, pad, padSize, padSize, padSize, padSize, cv::BORDER_CONSTANT);

    int tN = templateWindowSize*templateWindowSize;
    int sN = searchWindowSize*searchWindowSize;
    int tR = templateWindowSize/2;
    int sR = searchWindowSize/2;

    vector<double> gaussian(256*256, 0);
    for(int i = 0; i<256*256; i++)
    {
        double g = exp(-max(i-2.0*sigma*sigma, 0.0))/(h*h);
        gaussian[i] = g;
        if(g<0.001)
            break;
    }

    double* pGaussian = &gaussian[0];

    const int searchWindowStep = (int)pad.step - searchWindowSize;
    const int templateWindowStep = (int)pad.step - templateWindowSize;

    for(int i = 0; i < src.rows; i++)
    {
        uchar* pDst = dst.ptr(i);
        for(int j = 0; j < src.cols; j++)
        {
            cout<<i<<" "<<j<<endl;
            int *pVariance = new int[sN];
            double *pWeight = new double[sN];
            int cnt = sN-1;
            double weightSum = 0;

            uchar* pCenter = pad.data + pad.step * (sR + i) + (sR + j);//搜索区域中心指针
            uchar* pUpLeft = pad.data + pad.step * i + j;//搜索区域左上角指针
            for(int m = searchWindowSize; m>0; m--)
            {
                uchar* pDownLeft = pUpLeft + pad.step * m;

                for(int n = searchWindowSize; n>0; n--)
                {
                    uchar* pC = pCenter;
                    uchar* pD = pDownLeft + n;

                    int w = 0;
                    for(int k = templateWindowSize; k>0; k--)
                    {
                        for(int l = templateWindowSize; l>0; l--)
                        {
                            w += (*pC - *pD)*(*pC - *pD);
                            pC++;
                            pD++;
                        }
                        pC += templateWindowStep;
                        pD += templateWindowStep;
                    }
                    w = (int)(w/tN);
                    pVariance[cnt--] = w;
                    weightSum += pGaussian[w];
                }
            }

            for(int m = 0; m<sN; m++)
            {
                pWeight[m] = pGaussian[pVariance[m]]/weightSum;
            }

            double tmp = 0.0;
            uchar* pOrigin = pad.data + pad.step * (tR + i) + (tR + j);
            for(int m = searchWindowSize, cnt = 0; m>0; m--)
            {
                for(int n = searchWindowSize; n>0; n--)
                {
                    tmp += *(pOrigin++) * pWeight[cnt++];
                }
                pOrigin += searchWindowStep;
            }
            *(pDst++) = (uchar)tmp;

            delete pWeight;
            delete pVariance;
        }
    }
    return dst;
}







