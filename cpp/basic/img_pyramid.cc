// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.
//
// Simple pool/freelist for objects of the same type, typically used
// in local context.

// Author: zsyuyizhang@gmail.com
// Date: 2022-11-18


#include <vector>
#include <cstdio>
#include "opencv2/opencv.hpp"
#include "fmt/format.h"
using cv::Mat;
using cv::imread;
using cv::waitKey;



using std::vector;

void Gaussian_Pyramid(Mat &image, vector<Mat> &pyramid_images, int level) {
	Mat temp = image.clone();
	Mat dst;
	char buf[64];
	for (int i = 0; i < level; i++) {
		pyrDown(temp, dst);
		imshow(fmt::format("pyramid_up_%d", i), dst);
		sprintf(buf, "../data/result/gaussian_%d.jpg", i);
		imwrite(buf, dst);
		temp = dst.clone();
		pyramid_images.push_back(temp);
	}
}


void Laplaian_Pyramid(vector<Mat> &pyramid_images, Mat &image) {
	int num = pyramid_images.size() - 1;
	imwrite("./result/laplacian_0.jpg", pyramid_images[num]);
	imshow("laplacian_0.jpg", pyramid_images[num]);
	for (int t = num; t > -1; t--) {
		Mat dst;
		char buf[64];
		if (t - 1 < 0) {
			pyrUp(pyramid_images[t], dst, image.size());
			subtract(image, dst, dst);
			// dst = dst + Scalar(127, 127, 127);
			sprintf(buf, "./result/laplacian_%d.jpg", num - t + 1);
			imshow(buf, dst);
			imwrite(buf, dst);
		}
		else {
			pyrUp(pyramid_images[t], dst, pyramid_images[t - 1].size());
			subtract(pyramid_images[t - 1], dst, dst);
			// dst = dst + Scalar(127, 127, 127);
			sprintf(buf, "./result/laplacian_%d.jpg", num - t + 1);
			imshow(buf, dst);
			imwrite(buf, dst);
		}
	}
}

void reconstuction(int level) {
	char buf[64];
	Mat dst = imread("./result/laplacian_0.jpg");
	Mat dst2 = imread("./result/laplacian_1.jpg");
	Mat dst3 = imread("./result/laplacian_2.jpg");
	pyrUp(dst, dst);
	Mat dst4 = dst + dst2;
	pyrUp(dst4, dst4);
	Mat dst5 = dst4 + dst3;
	imshow("dst", dst5);

}

int main(int argc, char** argv) {
    Mat src = imread("../data/Curry.jpg");
	/*copyMakeBorder(src, src, 0, 3, 0, 3, BORDER_REPLICATE);*/
	imshow("01", src);
	/*resize(src, src, Size(src.cols *4, src.rows *4));*/
	/*cvtColor(src, src, COLOR_BGR2GRAY);*/
	vector<Mat> p_images;
	const int layer = 3; //金字塔层数
	Gaussian_Pyramid(src, p_images, layer - 1);
	Laplaian_Pyramid(p_images, src);
	reconstuction(layer - 1);//从拉普拉斯金字塔恢复原图
	waitKey();
	return 0;
}