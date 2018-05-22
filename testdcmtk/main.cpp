
//#include<stdio.h>
#include "detection.h"
#include <time.h>


using namespace std;
using namespace cv;



void main(){
	string path = "E:/dcm/dcm_file/FANG_XIN_FU/";
	//int xMin = 356; int xMax = 374; int yMin = 223; int yMax = 243; int zMin = 55; int zMax = 64;
	int xMin = 190; int xMax = 212; int yMin = 274; int yMax = 295; int zMin = 50; int zMax = 70;

	clock_t start, end;
	
	start = clock();
	vector<string> filename;
	vector<Mat> CTImgs;
	vector<Mat> CTROIImgs;
	vector<Mat> NoduleTagMaps;
	vector<Mat> NoduleTag_Res;
	vector<bool> NoduleFlag;
	//DicomImage *DICM;
	int roiXDim = xMax - xMin + 1;
	int roiYDim = yMax - yMin + 1;
	int roiZDim = zMax - zMin + 1;
	Mat temp;
	//filename = getAllFileNames(path);
	for (int i = 0; i < filename.size(); i++){
		//cout << filename[i] << endl;
		if (i >= zMin && i <= zMax){
			//DICM = new DicomImage((path + filename[i]).c_str());
			//DcmToMat(DICM, temp);
		}		
		CTImgs.push_back(temp);
	}
	//imwrite("FANG_XIN_FU/NoduleTag/000.jpg", CTImgs[55]);
	//cout << CTImgs.size() << endl;
	//imshow("CT", CTImgs[5]);
	Rect rect = Rect(xMin, yMin, roiXDim, roiYDim);
	Lungdetection det;
	
	for (int i = zMin; i < zMax; i++){
		det.SetImg(CTImgs[i], rect, path, i, filename.size());
		//det.LungSeg();//先是识别出肺部区域
		det.NoduleSeg();//分割方框内结节
//		NoduleTagMaps.push_back(det.getNoduleTag());//这个TagMap大小与CT图像的大小一致
		NoduleTag_Res.push_back(det.getNoduleTag_Res());//这个TagMap大小与框的大小一致
		NoduleFlag.push_back(det.segmented);
		CTROIImgs.push_back(CTImgs[i](rect).clone());//框区域内的原图
	}

	//cv::waitKey();
	string rawName;
	string imgName;
	string nodulePath = path + "NoduleTag\\";
	ifstream fin0(nodulePath);
	int xDim = CTImgs[zMin].size().width;
	int yDim = CTImgs[zMin].size().height;
	if (!fin0)//文件不存在
	{
		//mkdir(nodulePath.c_str());
	}
	for (int i = 0; i < NoduleTagMaps.size(); i++){
		imgName = nodulePath + to_string(i + zMin + 1) + ".jpg";
		rawName = nodulePath + to_string(i + zMin + 1) + ".raw";
		imwrite(imgName, NoduleTagMaps[i]);

		short *tagTmp = new short[xDim*yDim];

		ifstream fin1(rawName);
		if (!fin1)//文件不存在
		{
			//这里应该初始化tagTmp
			for (int h = 0; h < yDim; h++)
			{
				for (int w = 0; w < xDim; w++)
				{
					tagTmp[h*w + w] = 0;
				}

			}
		}
		else
		{
			char noduleFileTmp[1024];
			//sprintf(noduleFileTmp, rawName.c_str());
			FILE * fidReadRawImg = fopen(noduleFileTmp, "r"); // must be with 'r'.
			fread(tagTmp, sizeof(short), xDim*yDim, fidReadRawImg);
			fclose(fidReadRawImg);
		}

		for (int yindex = 0; yindex < roiYDim; yindex++)
		{
			for (int xindex = 0; xindex < roiXDim; xindex++)
			{
				//这里是对应的roi图中的坐标点的值
				//int regionindex = zindex*roiYDim * roiXDim + yindex * roiXDim + xindex;
				int dataindex = (yindex + yMin)  *  xDim +(xindex + xMin);
				if (NoduleTag_Res[i].at<uchar>(yindex, xindex)>0)
				{
					tagTmp[dataindex] = NoduleTag_Res[i].at<uchar>(yindex, xindex)/255;//NoduleTag_Res内结节的位置用255标识的
				}

			}
		}
		char rawImgData3D[1024];
		sprintf(rawImgData3D, rawName.c_str());
		FILE * fidRawImg = fopen(rawImgData3D, "wb"); // must be with 'b'.
		fwrite(tagTmp, sizeof(short), xDim*yDim, fidRawImg);
		fclose(fidRawImg);

		delete[] tagTmp; tagTmp = NULL;
	}
	//DcmFileFormat fileFormat;
	double m_pixelSpacingX;
	double m_pixelSpacingY;
	double m_sliceThickness;
	double vol = 0;
	double sur = 0;
	vector<Mat> resizedImg;
	//OFCondition status = fileFormat.loadFile((path + filename[0]).c_str());
	//fileFormat.getDataset()->findAndGetFloat64(DCM_PixelSpacing, m_pixelSpacingX, 0);
	//fileFormat.getDataset()->findAndGetFloat64(DCM_PixelSpacing, m_pixelSpacingY, 1);
	//fileFormat.getDataset()->findAndGetFloat64(DCM_SliceThickness, m_sliceThickness);
//	resize3D(NoduleTag_Res, resizedImg, m_pixelSpacingX, m_pixelSpacingY, m_sliceThickness);
	vol = volume(resizedImg,1);
	sur = surface(resizedImg);
	cout << "volume" << vol << endl;
	cout << "surface" << sur << endl;

	int setxMin = 1000, setyMin = 1000, setzMin = 0, setxMax = 0, setyMax = 0, setzMax = 0, setMassCenterX, setMassCenterY, setMassCenterZ, setCount;
	int setLobeIndex, setMinHU, setMaxHU, setMean, setSum, setStd, setMaxHUHist, setArea = 0, setVolume = 0, setCompactness;
	int setRoundness, setMaxDiameter, setCountHistMaxPos, setHistogramNorm, setNoduleType;
	double entropy, maocidu;
	double meanHU=0, stdHU=0;
	int maxHU=-2000, minHU=2000;
//	int maxpixel = -1 , minpixel = 256;
	entropy = CalEntropy3D(CTROIImgs, NoduleTag_Res);
	maocidu = CalMaocidu(resizedImg);
	
	vector<int> countedPixel;
	for (int i = 0; i < NoduleTag_Res.size(); i++){//从中间处-2的位置开始找
		int pixel;
		if (NoduleFlag[i]){			
			for (int yindex = 0; yindex < roiYDim; yindex++)
			{
				for (int xindex = 0; xindex < roiXDim; xindex++)
				{
					if (NoduleTag_Res[i].at<uchar>(yindex, xindex)>0)
					{
						pixel = CTROIImgs[i].at<uchar>(yindex, xindex);
						if (pixel>maxHU){
							maxHU = pixel;
						}
						if (pixel<minHU){
							minHU = pixel;
						}
						countedPixel.push_back(pixel);
						meanHU = pixel + meanHU;
					}
				}
			}
		}
	}
	if (countedPixel.size() != 0){
		meanHU = meanHU / countedPixel.size();
		for (int i = 0; i < countedPixel.size(); i++){
			stdHU = stdHU + (countedPixel[i] - meanHU)*(countedPixel[i] - meanHU);
		}
		stdHU = sqrt(stdHU / countedPixel.size());
	}
	double slope = double(WindowsWidth) / 255;
	double bias = double(WindowsCenter) - double(WindowsWidth) / 2;
	maxHU = slope*maxHU + bias;
	minHU = slope*minHU + bias;
	meanHU = slope*meanHU + bias;
	stdHU = slope*stdHU;

	int zMid = (zMin + zMax) / 2 - zMin;
	//int TempMinX, TempMinY, TempMaxX, TempMaxY, TempMinZ, TempMaxZ = 0;
	for (int i = zMid-2; i < NoduleTag_Res.size(); i++){//从中间处-2的位置开始找
		if (NoduleFlag[i]){
			setzMax = i;
			for (int yindex = 0; yindex < roiYDim; yindex++)
			{
				for (int xindex = 0; xindex < roiXDim; xindex++)
				{
				
					if (NoduleTag_Res[i].at<uchar>(yindex, xindex)>0)
					{
						if (yindex>setyMax){
							setyMax = yindex;
						}
						if (yindex<setyMin){
							setyMin = yindex;
						}
						if (xindex>setxMax){
							setxMax = xindex;
						}
						if (xindex<setxMin){
							setxMin = xindex;
						}
					}

				}
			}
		}
		else if (!NoduleFlag[i] && i>zMid){
			break;
		}
	}
	for (int i = zMid + 2; i >=0; i--){//从中间处+2的位置开始找
		if (NoduleFlag[i]){
			setzMin = i;
			for (int yindex = 0; yindex < roiYDim; yindex++)
			{
				for (int xindex = 0; xindex < roiXDim; xindex++)
				{
					if (NoduleTag_Res[i].at<uchar>(yindex, xindex)>0)
					{
						if (yindex>setyMax){
							setyMax = yindex;
						}
						if (yindex<setyMin){
							setyMin = yindex;
						}
						if (xindex>setxMax){
							setxMax = xindex;
						}
						if (xindex<setxMin){
							setxMin = xindex;
						}
					}

				}
			}
		}
		else if (!NoduleFlag[i] && i<zMid){
			break;
		}
	}
	setxMin = setxMin + xMin;
	setxMax = setxMax + xMin;
	setyMin = setyMin + yMin;
	setyMax = setyMax + yMin;
	setzMax = setzMax + zMin;
	setzMin = setzMin + zMin;
	setVolume = vol;
	setArea = sur;
	
	feature NoduleFeature;
	//det.SetImg(CTImgs[(setzMin + setzMax) / 2], rect, path, (setzMin + setzMax) / 2, filename.size());
	//det.NoduleSeg();//分割方框内结节
	//det.feature2D(NoduleFeature);
	end = clock();

	cout << "exec time (sec) :" << static_cast<double>(end - start) / CLOCKS_PER_SEC << endl;
	cv::waitKey();
}
