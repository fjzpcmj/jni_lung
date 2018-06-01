#include <iostream>
#include <fstream>
#include <fstream>
#include <core.hpp>
#include <opencv2/opencv.hpp>
#include <highgui.hpp>  
#include <imgproc.hpp>
#include "opencv2/imgcodecs.hpp"
#include "vector"
#include "set"
#include <direct.h>
//#include <io.h>
//#include "dcmtk/config/osconfig.h"    /* make sure OS specific configuration is included first */
//#include "dcmtk/dcmdata/dctk.h"
//#include "dcmtk/dcmdata/dcpxitem.h"
//#include "dcmtk/dcmimgle/dcmimage.h"
#include<string>
#include <time.h>
#include "jni.h"


using namespace std;
//using namespace cv;

const double pi(3.14159265);
const int  WindowsCenter(-500);
const int  WindowsWidth(1500);


struct feature
{
	double Area = 0, Area_erode = 0, Area_open = 0, Area_close = 0;//面积
	double ratio_erode;
	double ratio_open;
	double ratio_close;
	double Length = 0;//周长
	//内、外及边缘灰度均值与方差
	double edgesum = 0, edgevar = 0;
	double insidesum = 0, insidevar = 0;
	double outsum = 0, outvar = 0;
	//内、外及边缘灰度梯度均值
	double Gradient_edge_avg = 0, Gradient_in_avg = 0, Gradient_out_avg = 0;
	//非均匀性
	vector<double>avg;
	vector<double>var;
	//标准化径向长度均值与方差
	double M1 = 0, M2 = 0;
	//一维熵，二维熵
	double  M3_1D = 0, M3_2D = 0;
	//横纵比、凸壳度、紧致度、光滑度、粗糙度
	double M4=-1, M5=-1, M6, M7 = 0, M8, M9;
	//gabor滤波特征，共48个数值。6个尺度，8个方向，共48个滤波器
	vector<double> gaborFeature;
	//soble滤波，共三个数值。 X方向， Y方向， XY方向
	vector<double> sobelFeature;
	//灰度共生矩阵，共16个值。四个方向，顺序为水平，垂直，45度，135度。每个方向有能量、熵、对比度、逆差矩
	vector<double> glcmFeature;

	double maxDiameter = -1;
	double maxDiameterZ = -1;
	//TIC 相关特征
	//vector<double>PE;
	//double AWS;//平均廓清率AWS
	//double AWPD;	//绝对廓清增强比
	//double WR_in;	//吸收率
	//double WR_out;	//廓清率
	//vector<double>T;	//灰度比参数

	//三维特征,表面积，体积，熵
	//double surface;
//	double volume;
//	double entropy3D;
	//double maocidu3D;//三维毛刺度
	//导管会分割进来？各个特征的意义，把tic的东西加进来，三维毛刺度
};

cv::Mat RegionGrow(cv::Mat src, cv::Point2i pt, int th);
void icvprCcaByTwoPass(const cv::Mat& _binImg, cv::Mat& _lableImg, int &maxlabel);
void FindLung(const cv::Mat& _lableImg, cv::Mat &res, int maxlabel);
int otsuThreshold(cv::Mat pic);
void resize3D(vector<cv::Mat> &src, vector<cv::Mat> &dst, double spacing_x, double spacing_y, double spacing_z);
double surface(vector<cv::Mat> &src);
double volume(vector<cv::Mat> &src, double spacing_z);
int vertexTodist(int x, int y);//find hammingDistance between i and j 
//vector<string> getAllFileNames(const string& folder_path);
//int DcmToMat(DicomImage *img, cv::Mat &output);
double  CalEntropy3D(vector<cv::Mat> &src, vector<cv::Mat> &mask);
double CalMaocidu(vector<cv::Mat> &resizedImg);
void Bin3DMorphologyErode(vector<cv::Mat> &src, vector<cv::Mat> &dst);
cv::Mat RegionGrowWithField(cv::Mat src, cv::Mat field);
void Bin3DMorphologyDialate(vector<cv::Mat> &src, vector<cv::Mat> &dst);
double MajorDiameterXY(cv::Mat img);
void feature2D(cv::Mat ROI, cv::Mat res,feature &ret);
void arrrayToMat(jint* arrray, vector<cv::Mat> &output, int roix, int roiy, int roiz);
void solidNodule(vector<cv::Mat> &maskVector, vector<cv::Mat> &imgVector, vector<cv::Mat> &dst);
void icvprCcaByTwoPass3D(const vector<cv::Mat>& _binImg, vector<cv::Mat>& _lableImg, vector<int> &useLabel);
void FindSolid(vector<cv::Mat>& _lableImg, int &maxlabel);
vector<vector<double>> getSolidConnectHuDiversityProperty(vector<cv::Mat>& connectLabel, vector<cv::Mat>& origin
	, vector<int> &useLabel, vector<double> roiXYZDim, double &outputDiversity);
vector<double> getSolidConnectVolumeProperty(vector<cv::Mat>& connectLabel, vector<int> &useLabel,
	double dxVoxel, double dyVoxel, double dzVoxel);
vector<int> getSolidVolumeHistogram(vector<double> solidConnectVolumeProperty, vector<double> xAxis);
vector<double> getLobulation(cv::Mat mask,int &output);


class Lungdetection{
private:
	//cv::Mat img;//original img
	cv::Mat res;//result 255代表结节,大小与框的大小一致
	cv::Mat ROI;//roi image
	cv::Mat LungTagMap;//肺部的mask
	//cv::Mat LungImg;//肺部图片 即LungTagMap*img
	//cv::Mat NoduleTagMap;//图像大小与img一致
	cv::Rect rec;//
	string path;
	int index;//从0开始
	int totalSlice;

public:
	//static clock_t  readtime;
	Lungdetection(){};
	bool segmented = false;//是否分割出结节，false代表无结节被分割出来
	bool insideLung = true;

	void SetImg(cv::Mat src, cv::Rect rec, string path, int idx, int totalSli);
	void NoduleSeg();
	//void LungSeg();
	void ShowResult();

	//cv::Mat getNoduleTag();
	cv::Mat getNoduleTag_Res();
};