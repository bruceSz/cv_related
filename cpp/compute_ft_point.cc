#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <fstream>

using namespace std;
using namespace cv;
int main()
{
    Mat I1;
    Mat I2;
    vector<Point2f> features;
    int maxCout = 50;//定义最大个数
    double minDis = 20;//定义最小距离
    double qLevel = 0.01;//定义质量水平

    std::string file_p =  "../../data/test.png";
    std::ifstream  ifs(file_p);
    if (!ifs.good()) {
        cerr << " file not valid." << std::endl;
        return 1;
    }

    I1 = imread(file_p,0);//读取为灰度图像
    goodFeaturesToTrack(I1,features,maxCout,qLevel,minDis);
    for(int i=0;i<features.size();i++)
    {
        //将特征点画一个小圆出来--粗细为2
        circle(I1,features[i],3,Scalar(255),2);
    }
    for(auto& coor: features) {
        std::cout << "ft uv: " << coor.x << ":" << coor.y << std::endl;
    }
    imshow("features",I1);
    waitKey(0);
    return 0;
}
