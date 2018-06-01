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
	double Area = 0, Area_erode = 0, Area_open = 0, Area_close = 0;//���
	double ratio_erode;
	double ratio_open;
	double ratio_close;
	double Length = 0;//�ܳ�
	//�ڡ��⼰��Ե�ҶȾ�ֵ�뷽��
	double edgesum = 0, edgevar = 0;
	double insidesum = 0, insidevar = 0;
	double outsum = 0, outvar = 0;
	//�ڡ��⼰��Ե�Ҷ��ݶȾ�ֵ
	double Gradient_edge_avg = 0, Gradient_in_avg = 0, Gradient_out_avg = 0;
	//�Ǿ�����
	vector<double>avg;
	vector<double>var;
	//��׼�����򳤶Ⱦ�ֵ�뷽��
	double M1 = 0, M2 = 0;
	//һά�أ���ά��
	double  M3_1D = 0, M3_2D = 0;
	//���ݱȡ�͹�Ƕȡ����¶ȡ��⻬�ȡ��ֲڶ�
	double M4=-1, M5=-1, M6, M7 = 0, M8, M9;
	//gabor�˲���������48����ֵ��6���߶ȣ�8�����򣬹�48���˲���
	vector<double> gaborFeature;
	//soble�˲�����������ֵ�� X���� Y���� XY����
	vector<double> sobelFeature;
	//�Ҷȹ������󣬹�16��ֵ���ĸ�����˳��Ϊˮƽ����ֱ��45�ȣ�135�ȡ�ÿ���������������ء��Աȶȡ�����
	vector<double> glcmFeature;

	double maxDiameter = -1;
	double maxDiameterZ = -1;
	//TIC �������
	//vector<double>PE;
	//double AWS;//ƽ��������AWS
	//double AWPD;	//����������ǿ��
	//double WR_in;	//������
	//double WR_out;	//������
	//vector<double>T;	//�ҶȱȲ���

	//��ά����,��������������
	//double surface;
//	double volume;
//	double entropy3D;
	//double maocidu3D;//��άë�̶�
	//���ܻ�ָ�������������������壬��tic�Ķ����ӽ�������άë�̶�
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
	cv::Mat res;//result 255������,��С���Ĵ�Сһ��
	cv::Mat ROI;//roi image
	cv::Mat LungTagMap;//�β���mask
	//cv::Mat LungImg;//�β�ͼƬ ��LungTagMap*img
	//cv::Mat NoduleTagMap;//ͼ���С��imgһ��
	cv::Rect rec;//
	string path;
	int index;//��0��ʼ
	int totalSlice;

public:
	//static clock_t  readtime;
	Lungdetection(){};
	bool segmented = false;//�Ƿ�ָ����ڣ�false�����޽�ڱ��ָ����
	bool insideLung = true;

	void SetImg(cv::Mat src, cv::Rect rec, string path, int idx, int totalSli);
	void NoduleSeg();
	//void LungSeg();
	void ShowResult();

	//cv::Mat getNoduleTag();
	cv::Mat getNoduleTag_Res();
};