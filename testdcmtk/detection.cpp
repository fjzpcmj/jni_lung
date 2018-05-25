#include "detection.h"

void Lungdetection::SetImg(cv::Mat src, cv::Rect re, string inpath, int idx, int totalSli){
	if (src.channels() == 3){
		cvtColor(src, src, CV_BGR2GRAY);
	}
	index = idx;
	path = inpath;
	//img = src.clone();
	ROI = src.clone();
	rec = re;
	totalSlice = totalSli;
	segmented = false;
	//LungSeg();
	//cout << src.channels()<< img.channels() << ROI.channels() << endl;
}


/***************************************************************************************
Function:  �ָ��㷨
Input:     ROI ������ԭͼ�� 
Output:    ��ڵ����ڵ����� ����ǰ�ɫ�����������Ǻ�ɫ
Description: �ô���㷨���ж�ֵ����֮��ʴ�Ͽ������αڵ����ӣ����������븽�����������������õ����λ��
Return:    Mat
Others:    NULL
***************************************************************************************/
//void Lungdetection::NoduleSeg(){
//	Mat gray;
//	Mat temp;
//	Mat binary;
//	int searchLength = 10;//���м�ı߳�ΪsearchLength�����������
//	//cvtColor(ROI, gray, CV_BGR2GRAY);
//	clock_t start = clock();
//	clock_t mid0, mid1, mid2, mid3, end;
//	LungTagMap = imread(path + "LungTag/" + to_string(totalSlice-index-1) + ".jpg", CV_LOAD_IMAGE_GRAYSCALE);
//	//cout << "read lung map time (sec) :" << static_cast<double>(clock()-start) / CLOCKS_PER_SEC << endl;
//	if (LungTagMap.size().height==0){
//		LungSeg();
//		gray = LungImg(rec).clone();
//	}
//	else{
//		gray = img.mul(LungTagMap / 255);
//		gray = gray(rec).clone();
//	}
//	
//	threshold(gray, binary, otsuThreshold(gray), 255, CV_THRESH_BINARY);
//	erode(binary, temp, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)));
//	//imshow("Nodulebinary", binary);
//	//imshow("temp", temp);
//	int rows = ROI.rows;
//	int cols = ROI.cols;
//	int x = cols / 2;
//	int y = rows / 2;
//	int searchx = x;
//	int searchy = y;
//	int DIR[4][2] = { { 0, -1 }, { 1, 0 }, { 0, 1 },{ -1, 0 } };
//	bool find = false;
//	if (temp.at<uchar>(y, x) != 255){
//		for (int i = 1; i < searchLength+1; i++){
//			if (find){
//				break;
//			}
//			for (int j = 0; j < 4; j++){
//				searchx = x + i*DIR[j][0];
//				searchy = y+i*DIR[j][1];
//				if (searchx < 0 || searchy< 0 || searchx >(cols - 1) || (searchy > rows - 1))
//					continue;
//				if (temp.at<uchar>(searchy, searchx) == 255){
//					x = searchx;
//					y = searchy;
//					find = true;
//					break;
//				}
//			}
//		}
//	}
//	else{
//		find = true;
//	}
//	if (find){
//		mid0 = clock();
//		res = RegionGrow(temp, Point2i(x, y), 2);
//		
//		dilate(res, res, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)));
//		segmented = true;
//		NoduleTagMap = Mat(img.size(), CV_8UC1, cv::Scalar::all(0));
//		for (int i = 0; i < res.rows; i++){
//			for (int j = 0; j < res.cols; j++){
//				if (res.at<uchar>(i, j) == 255){
//					NoduleTagMap.at<uchar>(i + rec.y, j + rec.x) = 255;
//				}
//			}
//		}
//	}
//	else{
//		//cout << "find no thing in this slice" << endl;
//		res = Mat(gray.size(), CV_8UC1, cv::Scalar::all(0));
//		NoduleTagMap = Mat(img.size(), CV_8UC1, cv::Scalar::all(0));
//	}
//	//cout << "seg one img time (sec) :" << static_cast<double>(clock() - start) / CLOCKS_PER_SEC << endl;
//}

/***************************************************************************************
Function:  �ָ��㷨
Input:     ROI ������ԭͼ��
Output:    ��ڵ����ڵ����� ����ǰ�ɫ(255)�����������Ǻ�ɫ
Description: �ô���㷨���ж�ֵ����֮��ʴ�Ͽ������αڵ����ӣ����������븽������ͨ���ڴ���ͨ��Ļ����Ͻ�����������ֵ���ݷ���õ��������������õ����������������(������������Ŀ�����)������
Return:    Mat
Others:    NULL
***************************************************************************************/
void Lungdetection::NoduleSeg(){
	clock_t start = clock();
	cv::Mat gray;
	cv::Mat temp;
	cv::Mat binary;
	int searchLength = 10;
	//cvtColor(ROI, gray, CV_BGR2GRAY);
	//cout << "read lung map time (sec) :" << static_cast<double>(clock() - start) / CLOCKS_PER_SEC << endl;
	LungTagMap = cv::imread(path + "LungTag/" + to_string(totalSlice - index - 1) + "-lungs.jpg", CV_LOAD_IMAGE_GRAYSCALE);
	//cout << "read lung map time (sec) :" << static_cast<double>(clock() - start) / CLOCKS_PER_SEC << endl;
	if (LungTagMap.size().height == 0){
		cout << "can not read lung Map " << endl;
		return;
		//LungSeg();
		//gray = LungImg(rec).clone();
	}
	else{
		//threshold(LungTagMap, LungTagMap, 128, 255, CV_THRESH_BINARY);
		//morphologyEx(LungTagMap, LungTagMap, cv::MORPH_CLOSE, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(10, 10)));
		//gray = img.mul(LungTagMap / 255);
		gray = ROI.mul(LungTagMap(rec).clone() / 255);
	//	string path = "E:\\PostgraduateWork\\LungDetection\\CTdata\\FANG_XIN_FU\\c++received\\";
	//	cv::imwrite(path + to_string(index) + "roi.bmp", gray);
		//imshow("lung", gray.clone());
		//gray = gray(rec).clone();
	}

	int rows = gray.rows;
	int cols = gray.cols;
	for (int i = 0; i < rows; i++){
		if (insideLung){
			for (int j = 0; j < cols; j++){
				if (gray.at<uchar>(i, j) == 0){
					insideLung = false;
					break;
				}
			}
		}
		else{
			break;
		}

	}
	int otsuTh = otsuThreshold(gray);
	//cout << "-----otsuThreshold-----" << otsuTh << " ";
	int HuThreshold = -750;
	int thresholds = 255 * (HuThreshold - (WindowsCenter - WindowsWidth / 2)) / WindowsWidth;
	threshold(gray, binary, thresholds, 255, CV_THRESH_BINARY);
	//threshold(gray, binary, otsuTh>100 ? otsuTh : 100, 255, CV_THRESH_BINARY);
	erode(binary, temp, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3)));

	//imshow("binary", binary);

	//imshow("after erode", temp.clone());
	cv::Mat   labels, img_color, stats, centroids;
	int  nccomps = connectedComponentsWithStats(temp, labels, stats, centroids,8);
	int areamax = 0, label = 0;
	cv::Point  labelcenter;
	double distance;
	double totolerance = min(rows, cols)*0.3;
	for (int i = 1; i < nccomps; i++){
		distance = sqrt(pow(centroids.at<double>(i, 0) - cols / 2.0, 2) + pow(centroids.at<double>(i, 1) - rows / 2.0, 2));
		if (stats.at<int>(i, cv::CC_STAT_AREA)>areamax && (distance <= totolerance || labels.at<int>(rows / 2, cols / 2) == i)){
			areamax = stats.at<int>(i, cv::CC_STAT_AREA);
			label = i;
			labelcenter = cv::Point(centroids.at<double>(i, 0), centroids.at<double>(i, 1));
		}

	}
	bool find = false;
	cv::Mat findImg = labels == label;
	if (label > 0){
		//double ratio = stats.at<int>(label, cv::CC_STAT_WIDTH) / (stats.at<int>(label, cv::CC_STAT_HEIGHT) + 0.1);
		vector<vector<cv::Point>>precontours;//contours������xΪ�У�yΪ��
		findContours(findImg.clone(), precontours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);
		double maxArea = 0;
		int contouridx = 0;
		for (int i = 0; i< precontours.size(); i++)
		{
			double area = contourArea(precontours[i]);
			if (area>maxArea){
				contouridx = i;
				maxArea = area;
			}
		}
		vector<vector<cv::Point>>contours;
		contours.push_back(precontours[contouridx]);
		cv::Point2f fourPoint[4];
		vector<double>line;
		//cout << "--" << precontours.size() << "--";
		for (int i = 0; i < contours.size(); i++)
		{
			cv::RotatedRect rectPoint = minAreaRect(contours[i]);
			//��rectPoint�����д洢������ֵ�ŵ� fourPoint��������
			rectPoint.points(fourPoint);
		}
		for (int i = 1; i < 4; i++)
		{
			line.push_back(sqrt(pow((fourPoint[i].y - fourPoint[i - 1].y), 2) + pow((fourPoint[i].x - fourPoint[i - 1].x), 2)));
		}
		vector<double>::iterator biggest = std::max_element(std::begin(line), std::end(line));
		vector<double>::iterator smallest = std::min_element(std::begin(line), std::end(line));
		double ratio = *smallest / (*biggest + 0.01);
		double fill = maxArea / (*smallest*(*biggest) + 0.01);
		//if (ratio < 0.38 && *biggest >10 && fill < 0.28){
		if (ratio* fill < 0.1 && *biggest >10){
			find = false;
		}
		else{
			find = true;
		}

		
	}
	//Mat   labels;
	//int  nccomps = connectedComponents(temp, labels);

	//double distance = sqrt(pow(labelcenter.x - cols / 2.0, 2) + pow(labelcenter.y - rows / 2.0, 2));
	//int x = cols / 2;
	//int y = rows / 2;
	//int searchx = x;
	//int searchy = y;
	//int DIR[4][2] = { { 0, -1 }, { 1, 0 }, { 0, 1 }, { -1, 0 } };
	//bool find = false;
	////cout << "threshold lung map time (sec) :" << static_cast<double>(clock() - start) / CLOCKS_PER_SEC << endl;
	//if (labels.at<int>(y, x) == 0){
	//	for (int i = 1; i < searchLength + 1; i++){
	//		if (find){
	//			break;
	//		}
	//		for (int j = 0; j < 4; j++){
	//			searchx = x + i*DIR[j][0];
	//			searchy = y + i*DIR[j][1];
	//			if (searchx < 0 || searchy< 0 || searchx >(cols - 1) || (searchy > rows - 1))
	//				continue;
	//			if (temp.at<uchar>(searchy, searchx) == 255){
	//				x = searchx;
	//				y = searchy;
	//				find = true;
	//				break;
	//			}
	//		}
	//	}
	//}
	//else{
	//	find = true;
	//}
	if (find){
		std::cout << to_string(totalSlice - index - 1) + "-lungs.jpg" << "---index:" << index + 1 << "otsuThreshold: " << otsuTh << " ";
		//Mat findImg = labels == labels.at<int>(y, x);
		

		//imshow("before  RegionGrow", findImg);
		//findImg = findImg / 255;
		res = RegionGrowWithField(gray, findImg);
		//imshow("after  RegionGrow", res.clone());
		cv::Mat openRes;
		morphologyEx(res, openRes, cv::MORPH_ERODE, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3)));
		morphologyEx(openRes, openRes, cv::MORPH_DILATE, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(8, 8)));
		res = res & openRes;
		res = res * 255;
		segmented = true;
		//NoduleTagMap = cv::Mat(img.size(), CV_8UC1, cv::Scalar::all(0));
		/*for (int i = 0; i < res.rows; i++){
			for (int j = 0; j < res.cols; j++){
				if (res.at<uchar>(i, j) == 255){
					NoduleTagMap.at<uchar>(i + rec.y, j + rec.x) = 255;
				}
			}
		}*/

	}
	else{
		res = cv::Mat(gray.size(), CV_8UC1, cv::Scalar::all(0));
		//NoduleTagMap = cv::Mat(img.size(), CV_8UC1, cv::Scalar::all(0));
	}
	//string path = "E:\\PostgraduateWork\\LungDetection\\CTdata\\FANG_XIN_FU\\c++received\\";
	//cv::imwrite(path + to_string(index+1) + ".bmp", res);
	//cv::imwrite(path + to_string(index + 1) + "gray.bmp", gray);

}

//cv::Mat Lungdetection::getNoduleTag(){
//	//return NoduleTagMap;
//}

cv::Mat Lungdetection::getNoduleTag_Res(){
	return res;
}

void Lungdetection::ShowResult(){
	cv::destroyWindow("res");
	imshow("res",res);
}

//void Lungdetection::LungSeg(){
//	cv::Mat binary;
//	cv::Mat labelimg;
//	cv::Mat lung;//ͨ����̬ѧ�õ��Ĵֲڵķ�ʵ������
//	//threshold(img, binary, 0, 255, CV_THRESH_OTSU | CV_THRESH_BINARY_INV);
//	threshold(img, binary, otsuThreshold(img), 255, CV_THRESH_BINARY_INV);
//	//cout << "otsuThreshold :"<< otsuThreshold(img)<<endl;
//	//erode(binary,binary,Mat());
//	//ΪʲôҪ�����㣿
//	morphologyEx(binary, binary, cv::MORPH_OPEN, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3)));
//
//	int maxlabel;
//	icvprCcaByTwoPass(binary/255, labelimg,maxlabel);
//	FindLung(labelimg, lung, maxlabel);
//	//threshold(img, binary, 40, 255,CV_THRESH_BINARY);
//	//imshow("binary", binary);
//	//imshow("beforeLung", lung);
//	
//	morphologyEx(lung, lung, cv::MORPH_CLOSE, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(20, 20)), cv::Point(-1, -1), 1);
//	vector<vector<cv::Point> > contours;
//	findContours(lung.clone(), contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE, cv::Point(0, 0));
//	std::cout << "contours.size()"<<contours.size() << endl;
//	vector<vector<cv::Point> >hull(contours.size());
//	// Int type hull
//	vector<vector<int>> hullsI(contours.size());
//	// Convexity defects
//	vector<vector<cv::Vec4i>> defects(contours.size());
//	for (size_t i = 0; i < contours.size(); i++)
//	{
//		convexHull(cv::Mat(contours[i]), hull[i], false);
//		// find int type hull
//		cv::convexHull(cv::Mat(contours[i]), hullsI[i], false);
//		// get convexity defects
//		convexityDefects(cv::Mat(contours[i]), hullsI[i], defects[i]);
//	}
//	vector<cv::Vec4i> it;
//	vector<vector<int>> rmidx;//��Ҫȥ����contours�ϵĵ��index
//	LungTagMap = cv::Mat::zeros(lung.size(), CV_8UC1);
//	for (size_t i = 0; i < contours.size(); i++)
//	{
//		vector<int> rmidx_i;
//		it = defects[i];//��i��������ȱ��
//		std::cout << "i : " << i << endl;
//		for (int j = 0; j < it.size(); j++){
//			int startidx = it[j][0];
//			cv::Point ptStart(contours[i][startidx]);
//			int endidx = it[j][1];;
//			cv::Point ptEnd(contours[i][endidx]);
//			int faridx = it[j][2];
//			cv::Point ptFar(contours[i][faridx]);
//			double thr = sqrt(pow(ptEnd.x - ptStart.x, 2) + pow(ptEnd.y - ptStart.y, 2)) / (endidx - startidx);
//			if (thr < 0.9 && (-thr * 320 + 400)>(endidx - startidx)){
//				rmidx_i.push_back(startidx);
//				rmidx_i.push_back(endidx);
//				std::cout << "startidx:" << startidx << endl;
//				std::cout << "endidx:" << endidx << endl;
//				std::cout << endl;
//				/*line(drawing, ptStart, ptFar, CV_RGB(0, 255, 0), 2);
//				line(drawing, ptEnd, ptFar, CV_RGB(0, 255, 0), 2);
//				circle(drawing, ptStart, 4, Scalar(0, 255, 0), 2);
//				circle(drawing, ptEnd, 4, Scalar(255, 0, 0), 2);
//				circle(drawing, ptFar, 4, Scalar(100, 0, 255), 2);*/
//			}
//			std::cout << "threshold:" << thr << endl;
//		}
//		rmidx.push_back(rmidx_i);
//	}
//	vector<vector<cv::Point> > newcontours;
//	for (int i = 0; i < contours.size(); i++)
//	{
//		if (rmidx[i].empty()){//��i�������Ƿ���ȱ��
//			newcontours.push_back(contours[i]);//��ȱ��
//		}
//		else{
//			vector<cv::Point> contours_i;
//			int k = 0;
//			for (int j = 0; j < contours[i].size(); j++){
//				if (j <= rmidx[i][k] || j >= rmidx[i][k+1]){
//					contours_i.push_back(contours[i][j]);
//				}
//				else{
//					j = rmidx[i][k+1];
//					j--;
//					if (rmidx[i].size()>(k + 2)){
//						k = k + 2;
//					}				
//				}
//			}
//			newcontours.push_back(contours_i);
//		}		
//	}
//	//cout << "newcontours 0:" << newcontours[0][0].x <<"  "<< newcontours[0][0].y << endl;
//	//cout << "contours 0:" << contours[0][0].x << "  " << contours[0][0].y << endl;
//	drawContours(LungTagMap, newcontours, -1, cv::Scalar::all(255), -1, 8);
//	LungImg = img.mul(LungTagMap / 255);
//	//imshow("LungImg", LungImg);
//	//imshow("lung", lung);
//}

cv::Mat RegionGrowWithField(cv::Mat src, cv::Mat field)
{
	cv::Point2i ptGrowing;                      //��������λ��  
	int nGrowLable = 0;                             //����Ƿ�������  
	double nSrcValue = 0;                              //�������Ҷ�ֵ  
	int nCurValue = 0;                              //��ǰ������Ҷ�ֵ  
	cv::Mat matDst = cv::Mat::zeros(src.size(), CV_8UC1);   //����һ���հ��������Ϊ��ɫ  
	//��������˳������  
	int DIR[8][2] = { { -1, -1 }, { 0, -1 }, { 1, -1 }, { 1, 0 }, { 1, 1 }, { 0, 1 }, { -1, 1 }, { -1, 0 } };
	vector<cv::Point2i> vcGrowPt;                     //������ջ  
	int rows = src.rows;
	int cols = src.cols;
	vector<int> countedPixel;
	for (int x = 0; x < cols; x++){
		for (int y = 0; y < rows; y++){
			if (field.at<uchar>(y, x) == 255){
				vcGrowPt.push_back(cv::Point(x, y));                         //��������ѹ��ջ��  
				matDst.at<uchar>(y, x) = 255;               //���������  
				nSrcValue = nSrcValue + src.at<uchar>(y, x);            //��¼������ĻҶ�ֵ  
				countedPixel.push_back(src.at<uchar>(y, x));
			}
		}
	}
	double th = 0;
	nSrcValue = nSrcValue / countedPixel.size();
	for (int i = 0; i < countedPixel.size(); i++){
		th = th + (countedPixel[i] - nSrcValue)*(countedPixel[i] - nSrcValue);
	}
	th = sqrt(th / countedPixel.size());

	cout << "mean: " << nSrcValue << "threshold: " << th;

	if (th<50 && th >= 10){
		th = -th + 90;
	}
	else if (th < 10){
		th = 90;
	}
	else if (th >= 50){
		th = 70;
	}
	cout << "real threshold: " << th << endl;
	cv::Point2i pt;
	while (!vcGrowPt.empty())                       //����ջ��Ϊ��������  
	{
		pt = vcGrowPt.back();                       //ȡ��һ��������  
		vcGrowPt.pop_back();

		//�ֱ�԰˸������ϵĵ��������  
		for (int i = 0; i<9; ++i)
		{
			ptGrowing.x = pt.x + DIR[i][0];
			ptGrowing.y = pt.y + DIR[i][1];
			//����Ƿ��Ǳ�Ե��  
			if (ptGrowing.x < 0 || ptGrowing.y < 0 || ptGrowing.x >(src.cols - 1) || (ptGrowing.y > src.rows - 1))
				continue;

			nGrowLable = matDst.at<uchar>(ptGrowing.y, ptGrowing.x);      //��ǰ��������ĻҶ�ֵ  

			if (nGrowLable == 0)                    //�����ǵ㻹û�б�����  
			{
				nCurValue = src.at<uchar>(ptGrowing.y, ptGrowing.x);
				if (abs(nSrcValue - nCurValue) < th && nCurValue >90)                 //����ֵ��Χ��������  
				{
					matDst.at<uchar>(ptGrowing.y, ptGrowing.x) = 255;     //���Ϊ��ɫ  
					vcGrowPt.push_back(ptGrowing);                  //����һ��������ѹ��ջ��  
				}
			}
		}
	}
	return matDst.clone();
}

//���㲡��Ķ�ά����
//ImageΪ�������ԭʼͼƬ
//binaryImageΪ��ֵͼ������ֵΪ255�Ĵ����ڣ�ֵΪ0��Ϊ������֯
//��������ret�У�ret�ڵ�������feature�������͵Ķ�������˵��
void feature2D(cv::Mat ROI, cv::Mat res,feature &ret)
{
	//if (!segmented){
	//	cout << "segmentation finds nothing" << endl;
	//	return;
	//}
	cv::Mat Image = ROI;
	cv::Mat binaryImage = res;
	cv::Mat Image_seg;
	if (binaryImage.rows == 0){
		cout << "binaryImage not found" << endl;
		return;
	}
	if (Image.rows == 0){
		cout << "ROI not found" << endl;
		return;
	}
	Image_seg = Image.mul(binaryImage / 255);
	//imshow("Image_seg", Image_seg);
	//imshow("binaryImage", binaryImage);
	//imshow("Image", Image);
	//waitKey(0);
	//struct result ret;
	//��ȡcanny��Ե
	cv::Mat cannyImage, cannyerode;
	Canny(binaryImage, cannyImage, 200, 255, 3);
	//ͨ��canny��Եͼ����ȡ��Ե���꣬��������ܳ�
	vector<vector<cv::Point>>precontours;//contours������xΪ�У�yΪ��
	findContours(cannyImage, precontours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);
	if (precontours.size() == 0){
		cout << "error in  feature" << endl;
		return;
	}
	int maxArea = 0;
	int contouridx = 0;
	for (int i = 0; i< precontours.size(); i++)
	{
		int area = contourArea(precontours[i]);
		if (area>maxArea){
			contouridx = i;
			maxArea = area;
		}
	}
	vector<vector<cv::Point>>contours;
	contours.push_back(precontours[contouridx]);

	for (int i = 0; i < (int)contours.size(); i++)
	{
		ret.Length += contours[i].size();
		//std::cout << "�ܳ�:" << contours[i].size() << endl;
	}
	int maxContoursIndex = 0;
	int maxContoursSize = 0;
	//��vector��Ե���������Сֵ�����ֵ
	vector<int> rows_peri;
	vector<int> cols_peri;
	for (int k = 0; k < contours.size(); k++){
		int peri_size = contours[k].size();
		if (peri_size > maxContoursSize){
			maxContoursIndex = k;
			maxContoursSize = peri_size;
		}
		for (int i = 0; i < peri_size; i++){
			int x = contours[k][i].x;
			int y = contours[k][i].y;
			//cout << "x: " << x << endl;
			//cout << "y: " << y << endl;
			rows_peri.push_back(y);
			cols_peri.push_back(x);
		}
	}
	sort(rows_peri.begin(), rows_peri.end());
	sort(cols_peri.begin(), cols_peri.end());
	int total_length = rows_peri.size();
	int row_min_perim = rows_peri[0];
	int row_max_perim = rows_peri[total_length - 1];
	int col_min_perim = cols_peri[0];
	int col_max_perim = cols_peri[total_length - 1];
	//cout << "����Сֵ�� " << row_min_perim << endl;
	//cout << "�����ֵ�� " << row_max_perim << endl;
	//cout << "����Сֵ�� " << col_min_perim << endl;
	//cout << "�����ֵ�� " << col_max_perim << endl;

	//����gabor��sobel��glcm����
	//Mat minImg = Image(Rect(Point(col_min_perim, row_min_perim), Point(col_max_perim, row_max_perim))).clone();
	//imshow("minImg", minImg);
	//ret.gaborFeature = gaborFeature(minImg);
	//ret.sobelFeature = sobelFeature(minImg);
	//ret.glcmFeature = glcmFeature(minImg);
	//cout << "gabor����:";
	//for (int i = 0; i < 48; i++){
	//	cout << ret.gaborFeature[i] << ",";
	//}
	//cout << endl;


	//���������
	cv::Mat Image_seg_erode, Image_seg_open, Image_seg_close;

	erode(Image_seg, Image_seg_erode, cv::Mat(5, 5, CV_8U), cv::Point(-1, -1), 1);
	morphologyEx(Image_seg, Image_seg_open, cv::MORPH_OPEN, cv::Mat(3, 3, CV_8U), cv::Point(-1, -1), 1);
	morphologyEx(Image_seg, Image_seg_close, cv::MORPH_CLOSE, cv::Mat(3, 3, CV_8U), cv::Point(-1, -1), 1);

	for (int i = 0; i <Image_seg_erode.rows; i++)
	{
		for (int j = 0; j <Image_seg_erode.cols; j++)
		{
			if (Image_seg_erode.at<uchar>(i, j) != 0)
			{
				ret.Area_erode = ret.Area_erode + 1;
			}
			if (Image_seg_open.at<uchar>(i, j) != 0)
			{
				ret.Area_open = ret.Area_open + 1;
			}
			if (Image_seg_close.at<uchar>(i, j) != 0)
			{
				ret.Area_close = ret.Area_close + 1;
			}
			if (Image_seg.at<uchar>(i, j) != 0)
			{
				ret.Area = ret.Area + 1;
			}
		}
	}
	ret.ratio_erode = ret.Area_erode / ret.Area;
	ret.ratio_open = ret.Area_open / ret.Area;
	ret.ratio_close = ret.Area_close / ret.Area;
	//std::cout << "���:" << ret.Area << endl;
	//cout << "��ʴ�����:" << ret.Area_erode << "\t��ԭ���֮��:" << ret.ratio_erode << endl;
	//cout << "����������:" << ret.Area_open << "\t��ԭ���֮��:" << ret.ratio_open << endl;
	//cout << "����������:" << ret.Area_close << "\t��ԭ���֮��:" << ret.ratio_close << endl;

	//���Բ����//
	float MaxD = -1;//���ֱ��
	vector<cv::Point2f>center(contours.size());
	vector<float>radius(contours.size());
	cv::Point2f p;
	for (int i = 0; i < contours.size(); i++)
	{
		minEnclosingCircle(contours[i], center[i], radius[i]);
		if (2 * radius[i] > MaxD){
			MaxD = 2 * radius[i];
			p.y = center[i].y + 1;
			p.x = center[i].x + 1;
		}

		//center y : row
		//center x: col
		//cout << "center y: " << p.y << endl;
		//cout << "center x: " << p.x << endl;
	}

	//cout << "���ֱ��MaxD:" << MaxD << endl;
	//cout << "���ĵ�:" << p << endl;
	//-------------------------------------------------------------------------------------------------//
	//��Ե�Ҷ�ֵ��ز�������,����Image,cannyImage,binaryImage//
	int counter1 = 0;
	for (int j = 0; j < Image.rows; j++)//˫��ѭ��������Ե���ͼ��
	{
		for (int i = 0; i < Image.cols; i++)
		{
			if (cannyImage.at<uchar>(j, i) != 0)
			{
				ret.edgesum = ret.edgesum + Image.at<uchar>(j, i);
				counter1 = counter1 + 1;
			}
		}
	}
	ret.edgesum = ret.edgesum / counter1;
	for (int j = 0; j < Image.rows; j++)//˫��ѭ��������Ե���ͼ��
	{
		for (int i = 0; i < Image.cols; i++)
		{
			if (cannyImage.at<uchar>(j, i) != 0)
			{
				ret.edgevar = ret.edgevar + pow((Image.at<uchar>(j, i) - ret.edgesum), 2);
			}
		}
	}
	//�ڲ��Ҷ�ֵ��ز�������//
	int counter2 = 0;
	for (int j = 0; j < Image.rows; j++)//˫��ѭ��������ֵͼ��
	{
		for (int i = 0; i < Image.cols; i++)
		{
			if (binaryImage.at<uchar>(j, i) != 0 && cannyImage.at<uchar>(j, i) == 0)
			{
				ret.insidesum = ret.insidesum + Image.at<uchar>(j, i);
				counter2 = counter2 + 1;
			}
		}
	}
	ret.insidesum = ret.insidesum / counter2;
	for (int j = 0; j < Image.rows; j++)//˫��ѭ��������Ե���ͼ��
	{
		for (int i = 0; i < Image.cols; i++)
		{
			if (binaryImage.at<uchar>(j, i) != 0 && cannyImage.at<uchar>(j, i) == 0)
			{
				ret.insidevar = ret.insidevar + pow((Image.at<uchar>(j, i) - ret.insidesum), 2);
			}
		}
	}
	//���Ҷ�ֵ��ز�������
	int counter3 = 0;
	for (int j = 0; j < Image.rows; j++)//˫��ѭ������ͼ��
	{
		for (int i = 0; i < Image.cols; i++)
		{

			if (binaryImage.at<uchar>(j, i) == 0 && cannyImage.at<uchar>(j, i) == 0)
			{
				ret.outsum = ret.outsum + Image.at<uchar>(j, i);
				counter3 = counter3 + 1;
			}
		}
	}
	ret.outsum = ret.outsum / counter3;
	for (int j = 0; j < Image.rows; j++)//˫��ѭ������ͼ��
	{
		for (int i = 0; i < Image.cols; i++)
		{
			if (binaryImage.at<uchar>(j, i) == 0 && cannyImage.at<uchar>(j, i) == 0)
			{
				ret.outvar = ret.outvar + pow((Image.at<uchar>(j, i) - ret.outsum), 2);
			}
		}
	}
	//�ڲ������������ĻҶȾ�ֵ�뷽��//
	ret.edgevar = ret.edgevar / counter1;
	ret.insidevar = ret.insidevar / counter2;
	ret.outvar = ret.outvar / counter3;
	//cout << "�ڲ��ҶȾ�ֵ:" << ret.insidesum << "\t�ڲ��Ҷȷ���:" << ret.insidevar << endl;
	//cout << "��Ե�ҶȾ�ֵ:" << ret.edgesum << "\t��Ե�Ҷȷ���:" << ret.edgevar << endl;
	//cout << "�ⲿ�ҶȾ�ֵ:" << ret.outsum << "\t�ⲿ�Ҷȷ���:" << ret.outvar << endl;

	//-------------------------------------------------------------------------------------------------//
	//��Ե�������Ե�����ڱ�Ե�����ݶȾ�ֵ����//
	vector<double>Gradient_edge, Gradient_out, Gradient_in;
	double dx, dy;
	double Gradient_edge_avg = 0, Gradient_out_avg = 0, Gradient_in_avg = 0;
	//��Ե����//
	for (int i = 0; i < contours[maxContoursIndex].size(); i++)
	{
		cv::Point g = contours[maxContoursIndex].at(i);
		dy = Image.at<uchar>(g.y + 1, g.x) - Image.at<uchar>(g.y, g.x);
		dx = Image.at<uchar>(g.y, g.x + 1) - Image.at<uchar>(g.y, g.x);
		Gradient_edge.push_back(pow(pow(dy + dx, 2), 0.5));
	}
	for (int i = 0; i < Gradient_edge.size(); i++)
	{
		Gradient_edge_avg = Gradient_edge_avg + Gradient_edge[i];
	}
	Gradient_edge_avg = Gradient_edge_avg / Gradient_edge.size();
	//�⡢�ڱ�Ե����//
	//���ñ�Ե�߿�
	int edgewidth = 1;
	int area_current = ret.Area;
	if (area_current < 36 && area_current > 16){
		edgewidth = 2;
	}
	if (area_current < 64 && area_current >= 36){
		edgewidth = 3;
	}
	if (area_current >= 64){
		edgewidth = 4;
	}
	//cout << "��ʼ���õı�Ե��" << edgewidth << endl;
	while (((row_min_perim - edgewidth - 1 < 0) || (col_min_perim - edgewidth - 1 < 0) || (row_max_perim + edgewidth + 1 >= Image.rows) || (col_max_perim + edgewidth + 1 >= Image.cols)) && edgewidth > 1) {
		//cout << "ͼƬ���ù�С�������޸ı�Ե���" << endl;
		edgewidth -= 1;
	}
	//cout << "��ǰ���õı�Ե��" << edgewidth << endl;
	if ((row_min_perim - edgewidth - 1 < 0) || (col_min_perim - edgewidth - 1< 0) || (row_max_perim + edgewidth + 1 >= Image.rows) || (col_max_perim + edgewidth + 1 >= Image.cols)){
		//cout << "ͼƬ���ù�С�������¼���" << endl;
		//fprintf(stderr, "ͼƬ���ù�С���޷������ݶȣ������¼���\n");
		//return ret;
	}
	else{
		for (int i = 0; i < contours[maxContoursIndex].size(); i++)
		{
			cv::Point g = contours[maxContoursIndex].at(i);
			if (g.y <= p.y && g.x>p.x)//��������ĵ�ĵ�һ���� g.y <= p.x&&g.x>p.y
			{
				//Point g = contours[0].at(i);
				dy = Image.at<uchar>(g.y + 1 - edgewidth, g.x + edgewidth) - Image.at<uchar>(g.y - edgewidth, g.x + edgewidth);
				dx = Image.at<uchar>(g.y - edgewidth, g.x + 1 + edgewidth) - Image.at<uchar>(g.y - edgewidth, g.x + edgewidth);
				Gradient_out.push_back(pow(pow(dy + dx, 2), 0.5));
				dy = Image.at<uchar>(g.y + 1 + edgewidth, g.x - edgewidth) - Image.at<uchar>(g.y + edgewidth, g.x - edgewidth);
				dx = Image.at<uchar>(g.y + edgewidth, g.x + 1 - edgewidth) - Image.at<uchar>(g.y + edgewidth, g.x - edgewidth);
				Gradient_in.push_back(pow(pow(dy + dx, 2), 0.5));
			}
			if (g.y < p.y&&g.x <= p.x)//��������ĵ�ĵڶ����� g.y < p.x&&g.x <= p.y
			{
				dy = Image.at<uchar>(g.y + 1 - edgewidth, g.x - edgewidth) - Image.at<uchar>(g.y - edgewidth, g.x - edgewidth);
				dx = Image.at<uchar>(g.y - edgewidth, g.x + 1 - edgewidth) - Image.at<uchar>(g.y - edgewidth, g.x - edgewidth);
				Gradient_out.push_back(pow(pow(dy + dx, 2), 0.5));
				dy = Image.at<uchar>(g.y + 1 + edgewidth, g.x + edgewidth) - Image.at<uchar>(g.y + edgewidth, g.x + edgewidth);
				dx = Image.at<uchar>(g.y + edgewidth, g.x + 1 + edgewidth) - Image.at<uchar>(g.y + edgewidth, g.x + edgewidth);
				Gradient_in.push_back(pow(pow(dy + dx, 2), 0.5));
			}
			if (g.y >= p.y && g.x<p.x)//��������ĵ�ĵ������� g.y >= p.x&&g.x<p.y
			{
				dy = Image.at<uchar>(g.y + 1 + edgewidth, g.x - edgewidth) - Image.at<uchar>(g.y + edgewidth, g.x - edgewidth);
				dx = Image.at<uchar>(g.y + edgewidth, g.x + 1 - edgewidth) - Image.at<uchar>(g.y + edgewidth, g.x - edgewidth);
				Gradient_out.push_back(pow(pow(dy + dx, 2), 0.5));
				dy = Image.at<uchar>(g.y + 1 - edgewidth, g.x + edgewidth) - Image.at<uchar>(g.y - edgewidth, g.x + edgewidth);
				dx = Image.at<uchar>(g.y - edgewidth, g.x + 1 + edgewidth) - Image.at<uchar>(g.y - edgewidth, g.x + edgewidth);
				Gradient_in.push_back(pow(pow(dy + dx, 2), 0.5));
			}
			if (g.y > p.y&&g.x >= p.x)//��������ĵ�ĵ������� g.y > p.x&&g.x >= p.y
			{
				dy = Image.at<uchar>(g.y + 1 + edgewidth, g.x + edgewidth) - Image.at<uchar>(g.y + edgewidth, g.x + edgewidth);
				dx = Image.at<uchar>(g.y + edgewidth, g.x + 1 + edgewidth) - Image.at<uchar>(g.y + edgewidth, g.x + edgewidth);
				Gradient_out.push_back(pow(pow(dy + dx, 2), 0.5));
				dy = Image.at<uchar>(g.y + 1 - edgewidth, g.x - edgewidth) - Image.at<uchar>(g.y - edgewidth, g.x - edgewidth);
				dx = Image.at<uchar>(g.y - edgewidth, g.x + 1 - edgewidth) - Image.at<uchar>(g.y - edgewidth, g.x - edgewidth);
				Gradient_in.push_back(pow(pow(dy + dx, 2), 0.5));
			}
		}
		for (int i = 0; i < Gradient_out.size(); i++)
		{
			Gradient_out_avg = Gradient_out_avg + Gradient_out[i];
		}
		Gradient_out_avg = Gradient_out_avg / Gradient_out.size();
		for (int i = 0; i < Gradient_in.size(); i++)
		{
			Gradient_in_avg = Gradient_in_avg + Gradient_in[i];
		}
		Gradient_in_avg = Gradient_in_avg / Gradient_in.size();
		//cout << "��Ե�ݶ�:" << Gradient_edge_avg << endl;
		//cout << "�ڱ�Ե�ݶ�:" << Gradient_in_avg << endl;
		//cout << "���Ե�ݶ�:" << Gradient_out_avg << endl;
		ret.Gradient_edge_avg = Gradient_edge_avg;
		ret.Gradient_in_avg = Gradient_in_avg;
		ret.Gradient_out_avg = Gradient_out_avg;
	}
	//---------------------------------------------------------------------------------------------//
	//�Ǿ�����//
	vector<double>orientation1, orientation2, orientation3, orientation4;
	vector<double>orientation5, orientation6, orientation7, orientation8;
	double row_temp = 15;//��j��
	double col_temp = 16;//��i��
	double avg_row = p.y;//���ĵ��ڵڼ���
	double avg_col = p.x;//���ĵ��ڵڼ���
	double dy_atan, dx_atan;//������atan
	if (ret.Area > 36)
	{
		for (int j = 0; j < binaryImage.rows; j++)//˫��ѭ������ͼ��
		{
			for (int i = 0; i < binaryImage.cols; i++)//˫��ѭ������ͼ��
			{
				if (binaryImage.at<uchar>(j, i) != 0)
				{
					row_temp = j;//��j��
					col_temp = i;//��i��
					//cout << "������0:"<<row_temp << "  " << col_temp << endl;
					if (j == avg_row && i == avg_col)//j��i�еĵ���������ĵ��ϣ������һ������
					{
						orientation1.push_back(Image.at<uchar>(j, i));
					}
					if (row_temp < avg_row && col_temp > avg_col)//��һ����
					{

						dy_atan = abs(avg_row - row_temp);
						dx_atan = abs(col_temp - avg_col);
						if ((atan(dy_atan / dx_atan)) <= (pi / 4.0))//0-45������
						{

							orientation1.push_back(Image.at<uchar>(row_temp, col_temp));
						}
						if ((atan(dy_atan / dx_atan)) > (pi / 4.0))//45-90������
						{
							orientation2.push_back(Image.at<uchar>(row_temp, col_temp));
						}
					}
					if ((row_temp == avg_row) && (col_temp > avg_col))//����x�������һ����
					{
						orientation1.push_back(Image.at<uchar>(row_temp, col_temp));
					}
					if (row_temp < avg_row && col_temp == avg_col)//����y������ڶ�����
					{
						orientation2.push_back(Image.at<uchar>(row_temp, col_temp));
					}
					if (row_temp < avg_row && col_temp < avg_col)//�ڶ�����
					{
						dy_atan = abs(avg_row - row_temp);
						dx_atan = abs(col_temp - avg_col);
						if ((atan(dy_atan / dx_atan)) < (pi / 4))//135-180������
						{
							orientation4.push_back(Image.at<uchar>(row_temp, col_temp));
						}
						if ((atan(dy_atan / dx_atan)) >= (pi / 4))//90-135������
						{
							orientation3.push_back(Image.at<uchar>(row_temp, col_temp));
						}
					}
					if ((row_temp == avg_row) && (col_temp < avg_col))//����x�������4����
					{
						orientation4.push_back(Image.at<uchar>(row_temp, col_temp));
					}
					if (row_temp > avg_row && col_temp < avg_col)//��������
					{
						dy_atan = abs(avg_row - row_temp);
						dx_atan = abs(col_temp - avg_col);
						if ((atan(dy_atan / dx_atan)) <= (pi / 4))
						{
							orientation5.push_back(Image.at<uchar>(row_temp, col_temp));
						}
						if ((atan(dy_atan / dx_atan)) >(pi / 4))
						{
							orientation6.push_back(Image.at<uchar>(row_temp, col_temp));
						}
					}
					if (row_temp > avg_row && col_temp == avg_col)//����y�������������
					{
						orientation6.push_back(Image.at<uchar>(row_temp, col_temp));
					}
					if (row_temp > avg_row && col_temp > avg_col)
					{
						dy_atan = abs(avg_row - row_temp);
						dx_atan = abs(col_temp - avg_col);
						if ((atan(dy_atan / dx_atan)) <= (pi / 4))
						{
							orientation7.push_back(Image.at<uchar>(row_temp, col_temp));
						}
						if ((atan(dy_atan / dx_atan)) > (pi / 4))
						{
							orientation8.push_back(Image.at<uchar>(row_temp, col_temp));
						}
					}
				}

			}
		}
		double sum = 0, squa = 0;
		for (int i = 0; i < orientation1.size(); i++)
		{
			sum = sum + orientation1[i];
		}
		for (int i = 0; i < orientation1.size(); i++)
		{
			squa = squa + pow((orientation1[i] - sum / orientation1.size()), 2);
		}
		(ret.avg).push_back(sum / orientation1.size());
		(ret.var).push_back(squa / orientation1.size());
		sum = 0, squa = 0;
		for (int i = 0; i < orientation2.size(); i++)
		{
			sum = sum + orientation2[i];
		}
		for (int i = 0; i < orientation2.size(); i++)
		{
			squa = squa + pow((orientation2[i] - sum / orientation2.size()), 2);
		}
		(ret.avg).push_back(sum / orientation2.size());
		(ret.var).push_back(squa / orientation2.size());
		sum = 0, squa = 0;
		for (int i = 0; i < orientation3.size(); i++)
		{
			sum = sum + orientation3[i];
		}
		for (int i = 0; i < orientation3.size(); i++)
		{
			squa = squa + pow((orientation3[i] - sum / orientation3.size()), 2);
		}
		(ret.avg).push_back(sum / orientation3.size());
		(ret.var).push_back(squa / orientation3.size());
		sum = 0, squa = 0;
		for (int i = 0; i < orientation4.size(); i++)
		{
			sum = sum + orientation4[i];
		}
		for (int i = 0; i < orientation4.size(); i++)
		{
			squa = squa + pow((orientation4[i] - sum / orientation4.size()), 2);
		}
		(ret.avg).push_back(sum / orientation4.size());
		(ret.var).push_back(squa / orientation4.size());
		sum = 0, squa = 0;
		for (int i = 0; i < orientation5.size(); i++)
		{
			sum = sum + orientation5[i];
		}
		for (int i = 0; i < orientation5.size(); i++)
		{
			squa = squa + pow((orientation5[i] - sum / orientation5.size()), 2);
		}
		(ret.avg).push_back(sum / orientation5.size());
		(ret.var).push_back(squa / orientation5.size());
		sum = 0, squa = 0;
		for (int i = 0; i < orientation6.size(); i++)
		{
			sum = sum + orientation6[i];
		}
		for (int i = 0; i <orientation6.size(); i++)
		{
			squa = squa + pow((orientation6[i] - sum / orientation6.size()), 2);
		}
		(ret.avg).push_back(sum / orientation6.size());
		(ret.var).push_back(squa / orientation6.size());
		sum = 0, squa = 0;
		for (int i = 0; i < orientation7.size(); i++)
		{
			sum = sum + orientation7[i];
		}
		for (int i = 0; i < orientation7.size(); i++)
		{
			squa = squa + pow((orientation7[i] - sum / orientation7.size()), 2);
		}
		(ret.avg).push_back(sum / orientation7.size());
		(ret.var).push_back(squa / orientation7.size());
		sum = 0, squa = 0;
		for (int i = 0; i < orientation8.size(); i++)
		{
			sum = sum + orientation8[i];
		}
		for (int i = 0; i < orientation8.size(); i++)
		{
			squa = squa + pow((orientation8[i] - sum / orientation8.size()), 2);
		}
		(ret.avg).push_back(sum / orientation8.size());
		(ret.var).push_back(squa / orientation8.size());
		for (int i = 0; i < 8; i++)
		{
			//cout << i + 1 << "������ҶȾ�ֵ" << ret.avg[i] << "\t" << i + 1 << "������Ҷȷ���" << ret.var[i] << endl;
		}

	}
	else
	{
		for (int j = 0; j < binaryImage.rows; j++)//˫��ѭ������ͼ��
		{
			for (int i = 0; i < binaryImage.cols; i++)//˫��ѭ������ͼ��
			{
				if (binaryImage.at<uchar>(j, i) != 0)
				{
					row_temp = j;//��j��
					col_temp = i;//��i��
					if (j == avg_row && i == avg_col)//j��i�еĵ���������ĵ��ϣ������һ������
					{
						orientation1.push_back(Image.at<uchar>(j, i));
					}
					if (row_temp < avg_row && col_temp > avg_col)//��һ����
					{
						orientation1.push_back(Image.at<uchar>(j, i));
					}
					if ((row_temp == avg_row) && (col_temp > avg_col))//����x�������һ����
					{
						orientation1.push_back(Image.at<uchar>(row_temp, col_temp));
					}
					if (row_temp < avg_row && col_temp == avg_col)//����y������ڶ�����
					{
						orientation2.push_back(Image.at<uchar>(row_temp, col_temp));
					}
					if (row_temp < avg_row && col_temp < avg_col)//�ڶ�����
					{
						orientation2.push_back(Image.at<uchar>(row_temp, col_temp));
					}
					if ((row_temp == avg_row) && (col_temp < avg_col))//����x�������������
					{
						orientation3.push_back(Image.at<uchar>(row_temp, col_temp));
					}
					if (row_temp > avg_row && col_temp < avg_col)//��������
					{
						orientation3.push_back(Image.at<uchar>(row_temp, col_temp));
					}
					if (row_temp > avg_row && col_temp == avg_col)//����y�������������
					{
						orientation4.push_back(Image.at<uchar>(row_temp, col_temp));
					}
					if (row_temp > avg_row && col_temp > avg_col)
					{
						orientation4.push_back(Image.at<uchar>(row_temp, col_temp));
					}
				}
			}
		}

		double sum = 0, squa = 0;
		for (int i = 0; i < orientation1.size(); i++)
		{
			sum = sum + orientation1[i];
		}
		for (int i = 0; i < orientation1.size(); i++)
		{
			squa = squa + pow((orientation1[i] - sum / orientation1.size()), 2);
		}
		(ret.avg).push_back(sum / orientation1.size());
		(ret.var).push_back(squa / orientation1.size());
		sum = 0, squa = 0;
		for (int i = 0; i < orientation2.size(); i++)
		{
			sum = sum + orientation2[i];
		}
		for (int i = 0; i < orientation2.size(); i++)
		{
			squa = squa + pow((orientation2[i] - sum / orientation2.size()), 2);
		}
		(ret.avg).push_back(sum / orientation2.size());
		(ret.var).push_back(squa / orientation2.size());
		sum = 0, squa = 0;
		for (int i = 0; i < orientation3.size(); i++)
		{
			sum = sum + orientation3[i];
		}
		for (int i = 0; i < orientation3.size(); i++)
		{
			squa = squa + pow((orientation3[i] - sum / orientation3.size()), 2);
		}
		(ret.avg).push_back(sum / orientation3.size());
		(ret.var).push_back(squa / orientation3.size());
		sum = 0, squa = 0;
		for (int i = 0; i < orientation4.size(); i++)
		{
			sum = sum + orientation4[i];
		}
		for (int i = 0; i < orientation4.size(); i++)
		{
			squa = squa + pow((orientation4[i] - sum / orientation4.size()), 2);
		}
		(ret.avg).push_back(sum / orientation4.size());
		(ret.var).push_back(squa / orientation4.size());
		for (int i = 0; i < 4; i++)
		{
			//cout << i + 1 << "������ҶȾ�ֵ" << ret.avg[i] << "\t" << i + 1 << "������Ҷȷ���" << ret.var[i] << endl;
		}
	}


	//---------------------------------------------------------------------------------------------//
	//M1//
	cv::Point2f p1;
	int x, y;
	vector<double>d;
	for (int i = 0; i <contours[maxContoursIndex].size(); i++)
	{
		p1 = contours[maxContoursIndex].at(i);
		d.push_back(sqrt(pow((p1.x - p.x), 2) + pow((p1.y - p.y), 2)) / MaxD);
	}

	for (int i = 0; i < contours[maxContoursIndex].size(); i++)
	{
		ret.M1 = ret.M1 + d[i];
	}
	ret.M1 = ret.M1 / contours[maxContoursIndex].size();

	//---------------------------------------------------------------------------------------------//
	//M2//
	for (int i = 0; i <contours[maxContoursIndex].size(); i++)
	{
		ret.M2 = ret.M2 + pow((d[i] - ret.M1), 2);
	}
	ret.M2 = sqrt(ret.M2 / (contours[maxContoursIndex].size() - 1));
	//---------------------------------------------------------------------------------------------//
	//M3//
	double maxd, mind, space;
	vector<double>prob(100, 0);
	for (int i = 0; i < contours[maxContoursIndex].size() - 1; i++)//�ж�M1��d�����������Сֵ��
	{
		maxd = d[i];
		mind = d[i];
		if (d[i + 1]>d[i])
		{
			maxd = d[i + 1];
		}
		if (d[i + 1]<d[i])
		{
			mind = d[i + 1];
		}
	}
	space = (maxd - mind) / 100;//100������ļ����С
	for (int i = 0; i < 99; i++)//ѭ��100������
	{
		for (int j = 0; j < contours[maxContoursIndex].size() - 1; j++)//
		{
			if (d[j] >= (mind + space*(i - 1)) && d[j] <= (mind + space*i))
			{
				prob[i] = prob[i] + 1;
			}
		}
	}
	for (int i = 0; i < 99; i++)
	{
		prob[i] = prob[i] / contours[maxContoursIndex].size();
		if (prob[i] == 0)
		{
			prob[i] = 1;
		}
	}
	for (int i = 0; i < 99; i++)
	{
		ret.M3_1D = ret.M3_1D + prob[i] * (log(prob[i]) / log(2));
	}
	ret.M3_1D = -ret.M3_1D;

	double P_i, P_j;
	cv::Mat P_ij = cv::Mat::zeros(cv::Size(256, 256), CV_8UC1);//ָ������Ĵ�С������,����ָ�������ݽ������
	for (int i = 0; i < Image.rows; i++)
	{
		for (int j = 0; j < Image.cols; j++)
		{
			P_i = Image.at<uchar>(i, j);
			if (i == 0 && j == 0)
			{
				P_j = Image.at<uchar>(i, j + 1) + Image.at<uchar>(i + 1, j) + Image.at<uchar>(i + 1, j + 1);
			}
			if ((i == 0) && (j>0) && (j < Image.cols - 1))
			{
				P_j = Image.at<uchar>(i, j - 1) + Image.at<uchar>(i, j + 1) + Image.at<uchar>(i + 1, j - 1) + Image.at<uchar>(i + 1, j) + Image.at<uchar>(i + 1, j + 1);
			}
			if ((i == 0) && (j == (Image.cols - 1)))
			{
				P_j = Image.at<uchar>(i, j - 1) + Image.at<uchar>(i + 1, j) + Image.at<uchar>(i + 1, j - 1);
			}
			if ((i>0) && (j == 0) && (i<Image.rows - 1))
			{
				P_j = Image.at<uchar>(i - 1, j) + Image.at<uchar>(i - 1, j + 1) + Image.at<uchar>(i, j + 1) + Image.at<uchar>(i + 1, j) + Image.at<uchar>(i + 1, j + 1);
			}
			if ((i>0) && (j == (Image.cols - 1)) && (i < Image.rows - 1))
			{
				P_j = Image.at<uchar>(i - 1, j - 1) + Image.at<uchar>(i - 1, j) + Image.at<uchar>(i, j - 1) + Image.at<uchar>(i + 1, j - 1) + Image.at<uchar>(i + 1, j);
			}
			if ((i == (Image.rows - 1)) && (j == 0))
			{
				P_j = Image.at<uchar>(i - 1, j) + Image.at<uchar>(i - 1, j + 1) + Image.at<uchar>(i, j + 1);
			}
			if ((i == (Image.rows - 1)) && (j > 0) && (j < (Image.cols - 1)))
			{
				P_j = Image.at<uchar>(i - 1, j) + Image.at<uchar>(i - 1, j + 1) + Image.at<uchar>(i - 1, j - 1) + Image.at<uchar>(i, j - 1) + Image.at<uchar>(i, j + 1);
			}
			if ((i == (Image.rows - 1)) && (j == (Image.cols - 1)))
			{
				P_j = Image.at<uchar>(i - 1, j) + Image.at<uchar>(i - 1, j - 1) + Image.at<uchar>(i, j - 1);
			}
			if ((i>0) && (j > 0) && (i < (Image.rows - 1)) && (j < (Image.cols - 1)))
			{
				P_j = Image.at<uchar>(i - 1, j - 1) + Image.at<uchar>(i - 1, j) + Image.at<uchar>(i - 1, j + 1) + Image.at<uchar>(i, j + 1) + Image.at<uchar>(i, j - 1) + Image.at<uchar>(i + 1, j - 1) + Image.at<uchar>(i + 1, j) + Image.at<uchar>(i + 1, j + 1);
			}
			P_j = P_j / 8;
			P_ij.at<uchar>(P_i, P_j) = P_ij.at<uchar>(P_i, P_j) + 1;
		}
	}
	for (int i = 0; i < 256; i++)
	{
		for (int j = 0; j < 256; j++)
		{
			double pij = P_ij.at<uchar>(i, j);
			if (pij != 0)
			{
				pij = -(pij / (256 * 256))*log(pij / (256 * 256));
				ret.M3_2D = ret.M3_2D + pij;
			}
		}
	}
	//---------------------------------------------------------------------------------------------//
	//M4//
	cv::Point2f fourPoint[4];
	vector<double>line;
	for (int i = 0; i < contours.size(); i++)
	{
		cv::RotatedRect rectPoint = minAreaRect(contours[i]);
		//��rectPoint�����д洢������ֵ�ŵ� fourPoint��������
		rectPoint.points(fourPoint);
	}
	for (int i = 1; i < 4; i++)
	{
		line.push_back(sqrt(pow((fourPoint[i].y - fourPoint[0].y), 2) + pow((fourPoint[i].x - fourPoint[0].x), 2)));
	}
	vector<double>::iterator biggest = std::max_element(std::begin(line), std::end(line));
	vector<double>::iterator smallest = std::min_element(std::begin(line), std::end(line));
	ret.M4 = *biggest / *smallest;

	//---------------------------------------------------------------------------------------------//
	//M5//
	double coverArea = pi*pow((MaxD / 2), 2);
	ret.M5 = (ret.Area) / coverArea;

	//---------------------------------------------------------------------------------------------//
	//M6//
	ret.M6 = ret.Area / (4 * pi*(ret.Length)*(ret.Length));

	//---------------------------------------------------------------------------------------------//
	//M7//
	for (int i = 1; i < (contours[maxContoursIndex].size() - 1); i++)
	{
		ret.M7 = ret.M7 + abs(d[i] - (d[i - 1] + d[i + 1]) / 2);
	}

	//---------------------------------------------------------------------------------------------//
	//M8//
	ret.M8 = ret.M1 / ret.M2;

	//---------------------------------------------------------------------------------------------//
	//M9//
	double M9_1 = 0, M9_2 = 0;
	int N = contours[maxContoursIndex].size();
	for (int i = 0; i < N; i++)
	{
		M9_1 = M9_1 + pow((d[i] - ret.M1), 4);
		M9_2 = M9_2 + pow((d[i] - ret.M1), 2);
	}
	ret.M9 = (pow(M9_1 / N, 0.25) - pow(M9_2 / N, 0.5)) / (ret.M2);

	//---------------------------------------------------------------------------------------------//


	//return ret;
}

/***************************************************************************************
Function:  ���������㷨
Input:     src ������ԭͼ�� pt ��ʼ������ th ��������ֵ����
Output:    ��ʵ�ʵ����ڵ����� ʵ�����ǰ�ɫ�����������Ǻ�ɫ
Description: �������������Ϊ��ɫ(255),����ɫΪ��ɫ(0)
Return:    Mat
Others:    NULL
***************************************************************************************/
cv::Mat RegionGrow(cv::Mat src, cv::Point2i pt, int th)
{
	cv::Point2i ptGrowing;                      //��������λ��  
	int nGrowLable = 0;                             //����Ƿ�������  
	int nSrcValue = 0;                              //�������Ҷ�ֵ  
	int nCurValue = 0;                              //��ǰ������Ҷ�ֵ  
	cv::Mat matDst = cv::Mat::zeros(src.size(), CV_8UC1);   //����һ���հ��������Ϊ��ɫ  
	//��������˳������  
	int DIR[8][2] = { { -1, -1 }, { 0, -1 }, { 1, -1 }, { 1, 0 }, { 1, 1 }, { 0, 1 }, { -1, 1 }, { -1, 0 } };
	vector<cv::Point2i> vcGrowPt;                     //������ջ  
	vcGrowPt.push_back(pt);                         //��������ѹ��ջ��  
	matDst.at<uchar>(pt.y, pt.x) = 255;               //���������  
	nSrcValue = src.at<uchar>(pt.y, pt.x);            //��¼������ĻҶ�ֵ  

	while (!vcGrowPt.empty())                       //����ջ��Ϊ��������  
	{
		pt = vcGrowPt.back();                       //ȡ��һ��������  
		vcGrowPt.pop_back();

		//�ֱ�԰˸������ϵĵ��������  
		for (int i = 0; i<9; ++i)
		{
			ptGrowing.x = pt.x + DIR[i][0];
			ptGrowing.y = pt.y + DIR[i][1];
			//����Ƿ��Ǳ�Ե��  
			if (ptGrowing.x < 0 || ptGrowing.y < 0 || ptGrowing.x >(src.cols - 1) || (ptGrowing.y > src.rows - 1))
				continue;

			nGrowLable = matDst.at<uchar>(ptGrowing.y, ptGrowing.x);      //��ǰ��������ĻҶ�ֵ  

			if (nGrowLable == 0)                    //�����ǵ㻹û�б�����  
			{
				nCurValue = src.at<uchar>(ptGrowing.y, ptGrowing.x);
				if (abs(nSrcValue - nCurValue) < th)                 //����ֵ��Χ��������  
				{
					matDst.at<uchar>(ptGrowing.y, ptGrowing.x) = 255;     //���Ϊ��ɫ  
					vcGrowPt.push_back(ptGrowing);                  //����һ��������ѹ��ջ��  
				}
			}
		}
	}
	return matDst.clone();
}

//Function:  ��ͨ�����㷨 Two-Pass��
//����ʹ��opencv�Դ���connectedComponents
void icvprCcaByTwoPass(const cv::Mat& _binImg, cv::Mat& _lableImg,int &maxlabel)
{
	// connected component analysis (4-component)  
	// use two-pass algorithm  
	// 1. first pass: label each foreground pixel with a label  
	// 2. second pass: visit each labeled pixel and merge neighbor labels  
	//   
	// foreground pixel: _binImg(x,y) = 1  
	// background pixel: _binImg(x,y) = 0  
	//�����labelimg�б�����label��0

	if (_binImg.empty() ||
		_binImg.type() != CV_8UC1)
	{
		return;
	}

	// 1. first pass  

	_lableImg.release();
	_binImg.convertTo(_lableImg, CV_32SC1);

	int label = 1;  // start by 2  
	std::vector<int> labelSet;
	labelSet.push_back(0);   // background: 0  
	labelSet.push_back(1);   // foreground: 1  

	int rows = _binImg.rows - 1;
	int cols = _binImg.cols - 1;
	for (int i = 1; i < rows; i++)
	{
		int* data_preRow = _lableImg.ptr<int>(i - 1);
		int* data_curRow = _lableImg.ptr<int>(i);
		for (int j = 1; j < cols; j++)
		{
			if (data_curRow[j] == 1)
			{
				std::vector<int> neighborLabels;
				neighborLabels.reserve(2);
				int leftPixel = data_curRow[j - 1];
				int upPixel = data_preRow[j];
				if (leftPixel > 1)
				{
					neighborLabels.push_back(leftPixel);
				}
				if (upPixel > 1)
				{
					neighborLabels.push_back(upPixel);
				}

				if (neighborLabels.empty())
				{
					labelSet.push_back(++label);  // assign to a new label  
					data_curRow[j] = label;
					labelSet[label] = label;
				}
				else
				{
					std::sort(neighborLabels.begin(), neighborLabels.end());
					int smallestLabel = neighborLabels[0];
					data_curRow[j] = smallestLabel;

					// save equivalence  
					for (size_t k = 1; k < neighborLabels.size(); k++)
					{
						int tempLabel = neighborLabels[k];
						int& oldSmallestLabel = labelSet[tempLabel];
						if (oldSmallestLabel > smallestLabel)
						{
							labelSet[oldSmallestLabel] = smallestLabel;
							oldSmallestLabel = smallestLabel;
						}
						else if (oldSmallestLabel < smallestLabel)
						{
							labelSet[smallestLabel] = oldSmallestLabel;
						}
					}
				}
			}
		}
	}

	// update equivalent labels  
	// assigned with the smallest label in each equivalent label set  
	for (size_t i = 2; i < labelSet.size(); i++)
	{
		int curLabel = labelSet[i];
		int preLabel = labelSet[curLabel];
		while (preLabel != curLabel)
		{
			curLabel = preLabel;
			preLabel = labelSet[preLabel];
		}
		labelSet[i] = curLabel;
	}


	// 2. second pass  
	maxlabel=0;
	for (int i = 0; i < rows - rows / 4; i++) //��Сɨ�贲�ĸ���
	{
		int* data = _lableImg.ptr<int>(i);
		for (int j = 0; j < cols; j++)//
		{
			int& pixelLabel = data[j];
			pixelLabel = labelSet[pixelLabel];
			if (pixelLabel >maxlabel){
				maxlabel = pixelLabel;
			}
		}
	}
	for (int i = rows - rows / 4; i < rows; i++){//��Сɨ�贲�ĸ���
		int* data = _lableImg.ptr<int>(i);
		for (int j = 0; j < cols; j++)//
		{
			data[j]=2;
		}
	}
	//cout<<endl<< maxlabel;
}

//Function:  ɸѡ�����������������ͨ�򣬼�Ϊ��ʵ��
void FindLung(const cv::Mat& _lableImg, cv::Mat &res, int maxlabel){
	int rows = _lableImg.rows ;
	int cols = _lableImg.cols ;
	int *labelNum = new int[maxlabel+1]();
	int index;
	for (int i = 0; i < rows; i++){
		for (int j = 0; j < cols; j++){
			index = _lableImg.at<int>(i, j);
			labelNum[index]++;
		}
	}
	//cout << "maxlabel" << maxlabel << endl;
	set<int> label;//��ŷ��ϴ�С������label
	for (int i = 1; i < maxlabel + 1; i++){ 
		if (labelNum[i]>7000 && labelNum[i] < 90000){
			label.insert(i);
			//cout << labelNum[i] << "   " << i << endl;;
		}
	}
	res.release();
	res.create(_lableImg.size(), CV_8UC1);
	res.setTo(0);
	for (int i = 0; i < rows; i++){
		for (int j = 0; j < cols; j++){
			index = _lableImg.at<int>(i, j);
			if (label.count(index) == 1){
				res.at<uchar>(i, j) = 255;
			}
		}
	}
}

// �����䷽� ���۳�������0��Ԫ�أ�
//���� �Ҷ�ͼ
//������ֵ
int otsuThreshold(cv::Mat pic)
{
	const int channels[1] = { 0 };
	const int histSize[1] = { 256 };
	float hranges[2] = { 0, 255 };
	const float* ranges[1] = { hranges };
	cv::MatND hist;
	calcHist(&pic, 1, channels, cv::Mat(), hist, 1, histSize, ranges);

	int T = 0;//��ֵ  
	double gSum0;//��һ��Ҷ���ֵ  
	double gSum1;//�ڶ���Ҷ���ֵ  
	double N0 = 0;//ǰ��������  
	double N1 = 0;//����������  
	double u0 = 0;//ǰ������ƽ���Ҷ�  
	double u1 = 0;//��������ƽ���Ҷ�  
	double w0 = 0;//ǰ�����ص���ռ����ͼ��ı���Ϊ��0  
	double w1 = 0;//�������ص���ռ����ͼ��ı���Ϊ��1  
	double u = 0;//��ƽ���Ҷ�  
	double tempg = -1;//��ʱ��䷽��  
	double g = -1;//��䷽��  
	int height = pic.rows;
	int width = pic.cols;
	double N = width*height - hist.at<float>(0);//���������۳�������0��Ԫ��  

	for (int i = 1; i<256; i++)
	{
		gSum0 = 0;
		gSum1 = 0;
		N0 += hist.at<float>(i);
		N1 = N - N0;
		if (0 == N1)break;//������ǰ�������ص�ʱ������ѭ��  
		w0 = N0 / N;
		w1 = 1 - w0;
		for (int j = 1; j <= i; j++)
		{
			gSum0 += j*hist.at<float>(j);
		}
		u0 = gSum0 / N0;
		for (int k = i + 1; k<256; k++)
		{
			gSum1 += k*hist.at<float>(k);
		}
		u1 = gSum1 / N1;
		//u = w0*u0 + w1*u1;  
		g = w0*w1*(u0 - u1)*(u0 - u1);
		if (tempg<g)
		{
			tempg = g;
			T = i;
		}
	}
	return T;
}

void resize3D(vector<cv::Mat> &src, vector<cv::Mat> &dst, double spacing_x, double spacing_y, double spacing_z){
	double scale_x = 1 / spacing_x;
	double scale_y = 1 / spacing_y;
	double scale_z = 1 / spacing_z;
	int src_z = src.size();
	if (src_z == 0) return;
	int src_x = src[0].cols;
	int src_y = src[0].rows;
	int dst_x = cvRound(src[0].cols*spacing_x);
	int dst_y = cvRound(src[0].rows*spacing_y);
	int dst_z = cvRound(src.size()*spacing_z);
	for (int k = 0; k < dst_z; k++){
		cv::Mat temp;
		temp.create(cv::Size(dst_x, dst_y), CV_8UC1);
		float fk = (float)(k + 0.5)*scale_z - 0.5;
		int sk = floor(fk);
		fk = fk - sk;
		if (sk < 0) {
			fk = 0, sk = 0;
		}
		if (sk >= src_z - 1) {
			fk = 0, sk = src_z - 2;
		}
		for (int j = 0; j < dst_y; j++){
			float fj = (float)(j + 0.5)*scale_y - 0.5;
			int sj = floor(fj);
			fj = fj - sj;
			if (sj < 0) {
				fj = 0, sj = 0;
			}
			if (sj >= src_y - 1) {
				fj = 0, sj = src_y - 2;
			}
			for (int i = 0; i < dst_x; i++){
				float fi = (float)(i + 0.5)*scale_x - 0.5;
				int si = floor(fi);
				fi = fi - si;
				if (si < 0) {
					fi = 0, si = 0;
				}
				if (si >= src_x - 1) {
					fi = 0, si = src_x - 2;
				}

				temp.at<uchar>(j, i) = src[sk].at<uchar>(sj, si)*(1 - fk)*(1 - fi)*(1 - fj) + src[sk + 1].at<uchar>(sj, si)*fk*(1 - fi)*(1 - fj) +
					src[sk].at<uchar>(sj + 1, si)*(1 - fk)*(1 - fi)* fj + src[sk].at<uchar>(sj, si + 1)*(1 - fk)*fi*(1 - fj) +
					src[sk + 1].at<uchar>(sj + 1, si)*fk*(1 - fi)*fj + src[sk + 1].at<uchar>(sj, si + 1)*fk*fi*(1 - fj) +
					src[sk].at<uchar>(sj + 1, si + 1)*(1 - fk)* fi*fj + src[sk + 1].at<uchar>(sj + 1, si + 1)*fk*fi* fj;
				if (temp.at<uchar>(j, i)>127){
					temp.at<uchar>(j, i) = 255;
				}
				else{
					temp.at<uchar>(j, i) = 0;
				}

			}

		}
		dst.push_back(temp);
	}

}

//���㲡������
//����srcΪmaskͼ�񣬼�ֵΪ0�ĵ����������֯��ֵΪ255�ĵ������
//���������������
double surface(vector<cv::Mat> &src){
	if (src.size() == 0) return 0;
	int vertex[8] = { 0 };
	int model[230] = { 0 };
	cv::Mat image, imagenext;
	int vertexSum = 0;
	vector<int> vertexZero(8), vertexOne(8);
	for (int z = 0; z < src.size() - 1; z++){
		int xMax = src[z].cols;
		int yMax = src[z].rows;
		image = src[z];
		imagenext = src[z + 1];
		for (int x = 0; x < xMax - 1; x++){
			for (int y = 0; y < yMax - 1; y++){
				vertexSum = 0;
				vertex[0] = image.at<uchar>(y, x) / 255;
				vertex[1] = image.at<uchar>(y + 1, x) / 255;
				vertex[3] = image.at<uchar>(y + 1, x + 1) / 255;
				vertex[2] = image.at<uchar>(y, x + 1) / 255;
				vertex[4] = imagenext.at<uchar>(y, x) / 255;
				vertex[5] = imagenext.at<uchar>(y + 1, x) / 255;
				vertex[7] = imagenext.at<uchar>(y + 1, x + 1) / 255;
				vertex[6] = imagenext.at<uchar>(y, x + 1) / 255;
				for (int i = 0; i < 8; i++){
					vertexSum += vertex[i];
					if (vertex[i] == 1){
						vertexOne.push_back(i);
					}
					else{
						vertexZero.push_back(i);
					}
				}
				int temp = 0;
				if (vertexSum > 4){
					int size = vertexZero.size();
					for (int i = 0; i < size - 1; i++){
						for (int j = i + 1; j < size; j++){
							temp += vertexTodist(vertexZero[i], vertexZero[j]);
						}
					}
				}
				else{
					int size = vertexOne.size();
					for (int i = 0; i < size - 1; i++){
						for (int j = i + 1; j < size; j++){
							temp += vertexTodist(vertexOne[i], vertexOne[j]);
						}
					}
				}
				if (vertexZero.size()*vertexOne.size() != 0){//ֻ���ڱ߽紦�ż�������
					model[temp]++;
				}
				vertexZero.clear();
				vertexOne.clear();
			}
		}
	}

	/*for (int i = 0; i < 230; i++){
	if (model[i] != 0){
	cout << ":::::"<<i<< endl;
	}
	}*/
	return 0.636*model[0] + 0.669*model[1] + 1.272*model[10] + 1.272*model[100] + 0.554*model[12] + 1.305*model[111]
		+ 1.908*model[30] + 0.927*model[24] + 0.422*model[33] + 1.338*model[222] + 1.573*model[123] + 1.19*model[132] + 2.554*model[60];
}

//���㲡�����
//����srcΪmaskͼ�񣬼�ֵΪ0�ĵ����������֯��ֵΪ255�ĵ������
//������������������ֵΪ0�ĵ����������֯��ֵΪ255�ĵ������
//spacing_zΪ����ͼƬ�е�spaceing_z ,����Ѿ���ֵ���ˣ�����1
double volume(vector<cv::Mat> &src, double spacing_z = 1.0){
	if (src.size() == 0) return 0;
	cv::Mat image;
	vector<int> silceArea(src.size());
	double volume = 0;
	for (int z = 0; z < src.size(); z++){
		int xMax = src[z].cols;
		int yMax = src[z].rows;
		image = src[z];
		int Area = 0;
		for (int x = 0; x < xMax; x++){
			for (int y = 0; y < yMax; y++){
				if (image.at<uchar>(y, x) == 255){
					Area++;
				}
			}
		}
		silceArea.push_back(Area);
	}
	volume = (silceArea[0] + silceArea[silceArea.size() - 1]) / 2;
	for (int i = 1; i < silceArea.size() - 1; i++){
		volume += silceArea[i];
	}
	return volume*spacing_z;
}

int vertexTodist(int x, int y){
	int result = 0, j = 1;
	result = x^y;
	while (result){
		j = j * 10;
		result &= (result - 1);
	}
	return j / 10;
}

//vector<string> getAllFileNames(const string& folder_path)
//{
//	vector<string> name;
//	_finddata_t file;
//	intptr_t   flag;
//	string filename = folder_path + "/*.DCM";//�����ƶ��ļ����ڵ�DCM�ļ�  
//	if ((flag = _findfirst(filename.c_str(), &file)) == -1)//Ŀ¼���Ҳ����ļ�  
//	{
//		cout << "There is no such type file" << endl;
//	}
//	else
//	{
//		//ͨ��ǰ���_findfirst�ҵ���һ���ļ�  
//		//����Ѱ���Ժ���ļ�  
//		name.push_back(string(file.name));
//		while (_findnext(flag, &file) == 0)
//		{
//			name.push_back(string(file.name));
//		}
//	}
//	_findclose(flag);
//	return name;
//}

//int DcmToMat(DicomImage *img, cv::Mat &output)
//{
//	//std::string file_path = "1.dcm";                                      //dcm�ļ�  
//	//DicomImage *img = new DicomImage(file_path.c_str());
//	img->setWindow(WindowsCenter, WindowsWidth);
//	if (img->isMonochrome() && img->getStatus() == EIS_Normal && img != NULL)
//	{
//		if (img->isMonochrome())
//		{
//			int nWidth = img->getWidth();            //���ͼ����  
//			int nHeight = img->getHeight();          //���ͼ��߶�  
//
//			uchar *pixelData = (uchar*)(img->getOutputData(8));   //���8λ��ͼ������ָ��  
//			//std::cout << nWidth << ", " << nHeight << std::endl;
//			if (pixelData != NULL)
//			{
//				//����1  
//				/*IplImage *srcImage = cvCreateImageHeader(cvSize(nWidth, nHeight), IPL_DEPTH_16U, 1);
//				cvSetData(srcImage, pixelData, nWidth*sizeof(unsigned short));
//				cv::Mat dst(srcImage);
//				cv::imshow("image1", dst);*/
//
//				//����2  
//				cv::Mat dst2(nWidth, nHeight, CV_8UC1, cv::Scalar::all(0));
//				uchar* data = nullptr;
//				for (int i = 0; i < nHeight; i++)
//				{
//					data = dst2.ptr<uchar>(i);   //ȡ��ÿһ�е�ͷָ�� Ҳ��ʹ��dst2.at<unsigned short>(i, j) = ?  
//					for (int j = 0; j < nWidth; j++)
//					{
//						*data++ = pixelData[i*nWidth + j];
//					}
//				}
//				output = dst2;
//				//cv::imshow("image2", dst2);
//			}
//		}
//		return 1;
//	}
//
//	//cv::waitKey();
//	delete img;
//	return 0;
//}
void arrrayToMat(jint* arrray, vector<cv::Mat> &output, int roix, int roiy, int roiz){
	cv::Mat temp(roiy, roix,CV_8UC1);
	int size = roix*roiy;
	int low = WindowsCenter - WindowsWidth / 2;
	int high = WindowsCenter + WindowsWidth / 2;
	double pixel = 0;
	//string logpath = "E:\\PostgraduateWork\\LungDetection\\CTdata\\FANG_XIN_FU\\log.txt";
	//ofstream log(logpath, ios::app);
	//log << "low " << low << "high " << high;
	for (int zidx = 0; zidx < roiz; zidx++){
		for (int xidx = 0; xidx < roix; xidx++){
			for (int yidx = 0; yidx < roiy; yidx++){
				pixel = arrray[yidx*roix + xidx + zidx*size];
				
				if (pixel<low){
					pixel = low;
				}
				if (pixel>high){
					pixel = high;
				}
				pixel = 255 * (pixel - low) / WindowsWidth;
				//log << pixel<<",";
				temp.at<uchar>(yidx, xidx) = pixel;
			}
			//log << endl;
		}
		//string path = "E:\\PostgraduateWork\\LungDetection\\CTdata\\FANG_XIN_FU\\c++received\\";
		//cv::imwrite(path + to_string(zidx)+".bmp", temp);
		output.push_back(temp.clone());
		//log << "-------------------"<<endl;
	}
	//log << endl;
}

double  CalEntropy3D(vector<cv::Mat> &src, vector<cv::Mat> &mask){
	if (src.size() < 3){
		return -1;
	}
	double N = 0;
	cv::Mat times;
	cv::Mat ori, mas;
	times.create(cv::Size(256, 256), CV_16UC1);
	times.setTo(0);
	int rows = src[0].rows;
	int cols = src[0].cols;
	for (int z = 0; z < src.size(); z++){
		ori = src[z];
		mas = mask[z];
		for (int i = 1; i < rows - 1; i++){
			for (int j = 1; j < cols - 1; j++){
				if (mas.at<uchar>(i, j) == 255){
					int x = ori.at<uchar>(i, j);
					int y = ori.at<uchar>(i, j + 1) + ori.at<uchar>(i + 1, j) + ori.at<uchar>(i - 1, j)
						+ ori.at<uchar>(i, j - 1);
					if (z != 0 && (z + 1) < src.size()){
						y = y + src[z - 1].at<uchar>(i, j) + src[z + 1].at<uchar>(i, j);
						y = y / 6;
					}
					else{
						y = y / 4;
					}
					times.at<ushort>(x, y) = times.at<ushort>(x, y) + 1;
					N++;
				}
			}
		}
	}
	double p, entropy = 0;
	for (int i = 0; i < 256; i++){
		for (int j = 0; j < 256; j++){
			if (times.at<ushort>(i, j) != 0){
				p = times.at<ushort>(i, j) / N;
				entropy = entropy + p*log(p);
			}
		}
	}
	return -entropy;
}

//��ά��ʴ,ֻʵ����kernal=3�����
void Bin3DMorphologyErode(vector<cv::Mat> &src, vector<cv::Mat> &dst){
	int kernel = 3;//����Ϊ����,��ʱֻ������Ϊ3.
	int zMax = src.size() - kernel / 2;
	//if (zMax == 0){
	//	return;
	//}
	if (src.size() < kernel){//�����ά��ʴ��ȫ�յĻ����˶�����Σ����ά��ʴ
		for (int i = 0; i < src.size(); i++){
			cv::Mat out;
			erode(src[i], out, NULL);//NULLʱ����ʾ����ʹ�òο���λ������3x3�ĺˡ�
			dst.push_back(out);
		}
		return;
	}
	int rMax = src[0].rows - kernel / 2;
	int cMax = src[0].cols - kernel / 2;
	int extension = (kernel - 1) / 2;
	for (int i = 0; i < extension; i++){
		cv::Mat firstimg;
		firstimg.create(src[0].size(), CV_8UC1);
		firstimg.setTo(0);
		dst.push_back(firstimg);//�߽���0���
	}

	for (int z = extension; z < zMax; z++){
		cv::Mat srcImg = src[z];
		cv::Mat srcImgNext = src[z + 1];
		cv::Mat srcImgPre = src[z - 1];
		cv::Mat dstImg;
		dstImg.create(src[0].size(), CV_8UC1);
		dstImg.setTo(0);
		uchar tmp = 0;
		for (int i = extension; i < rMax; i++){
			for (int j = extension; j < cMax; j++){
				tmp = 0;
				tmp = srcImg.at<uchar>(i, j) & srcImg.at<uchar>(i + 1, j);
				tmp = tmp & srcImg.at<uchar>(i - 1, j);
				tmp = tmp & srcImg.at<uchar>(i, j + 1);
				tmp = tmp & srcImg.at<uchar>(i, j - 1);
				tmp = tmp & srcImg.at<uchar>(i + 1, j + 1);
				tmp = tmp & srcImg.at<uchar>(i + 1, j - 1);
				tmp = tmp & srcImg.at<uchar>(i - 1, j - 1);
				tmp = tmp & srcImg.at<uchar>(i - 1, j + 1);
				tmp = tmp & srcImgNext.at<uchar>(i, j);
				tmp = tmp & srcImgPre.at<uchar>(i, j);
				dstImg.at<uchar>(i, j) = tmp;
			}
		}
		dst.push_back(dstImg);
	}

	for (int i = zMax; i < src.size(); i++){
		cv::Mat lastimg;
		lastimg.create(src[0].size(), CV_8UC1);
		lastimg.setTo(0);
		dst.push_back(lastimg);//�߽���0���
	}
}

//��ά����,ֻʵ����kernal=3�����,kernal����״�����������ģ�����4��ͨ����
void Bin3DMorphologyDialate(vector<cv::Mat> &src, vector<cv::Mat> &dst){
	int kernel = 3;//����Ϊ����,��ʱֻ������Ϊ3.
	int zMax = src.size() - kernel / 2;
	//if (zMax == 0){
	//	return;
	//}
	if (src.size() < kernel){//�����ά��ʴ��ȫ�յĻ����˶�����Σ����ά��ʴ
		for (int i = 0; i < src.size(); i++){
			cv::Mat out;
			cv::erode(src[i], out, NULL);//NULLʱ����ʾ����ʹ�òο���λ������3x3�ĺˡ�
			dst.push_back(out);
		}
		return;
	}
	int rMax = src[0].rows - kernel / 2;
	int cMax = src[0].cols - kernel / 2;
	int extension = (kernel - 1) / 2;
	for (int i = 0; i < extension; i++){
		cv::Mat firstimg;
		firstimg.create(src[0].size(), CV_8UC1);
		firstimg.setTo(0);
		dst.push_back(firstimg);//�߽���0���
	}

	for (int z = extension; z < zMax; z++){
		cv::Mat srcImg = src[z];
		cv::Mat srcImgNext = src[z + 1];
		cv::Mat srcImgPre = src[z - 1];
		cv::Mat dstImg;
		dstImg.create(src[0].size(), CV_8UC1);
		dstImg.setTo(0);
		uchar tmp = 0;
		for (int i = extension; i < rMax; i++){
			for (int j = extension; j < cMax; j++){
				tmp = 0;
				tmp = srcImg.at<uchar>(i, j) | srcImg.at<uchar>(i + 1, j);
				tmp = tmp | srcImg.at<uchar>(i - 1, j);
				tmp = tmp | srcImg.at<uchar>(i, j + 1);
				tmp = tmp | srcImg.at<uchar>(i, j - 1);
				tmp = tmp | srcImg.at<uchar>(i + 1, j + 1);
				tmp = tmp | srcImg.at<uchar>(i + 1, j - 1);
				tmp = tmp | srcImg.at<uchar>(i - 1, j - 1);
				tmp = tmp | srcImg.at<uchar>(i - 1, j + 1);
				tmp = tmp | srcImgNext.at<uchar>(i, j);
				tmp = tmp | srcImgPre.at<uchar>(i, j);
				dstImg.at<uchar>(i, j) = tmp;
			}
		}
		dst.push_back(dstImg);
	}

	for (int i = zMax; i < src.size(); i++){
		cv::Mat lastimg;
		lastimg.create(src[0].size(), CV_8UC1);
		lastimg.setTo(0);
		dst.push_back(lastimg);//�߽���0���
	}
}




//outpuOri Ϊ���������ԭͼ
double CalMaocidu(vector<cv::Mat> &resizedImg){

	vector<cv::Mat> ErodeImg;
	int oriArea = 0;
	int ErodeArea = 0;
	Bin3DMorphologyErode(resizedImg, ErodeImg);
	for (int z = 0; z < resizedImg.size(); z++){
		int xMax = resizedImg[z].cols;
		int yMax = resizedImg[z].rows;
		cv::Mat oriimage = resizedImg[z];
		cv::Mat Erodeimage = ErodeImg[z];
		for (int x = 0; x < xMax; x++){
			for (int y = 0; y < yMax; y++){
				if (oriimage.at<uchar>(y, x) == 255){
					oriArea++;
				}
				if (Erodeimage.at<uchar>(y, x) == 255){
					ErodeArea++;
				}
			}
		}
	}
	return double(ErodeArea) / oriArea;
}

//���㳤��
double MajorDiameterXY(cv::Mat img){
	cv::Mat afteropen;
	cv::morphologyEx(img, afteropen, cv::MORPH_OPEN, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3)));
	vector<vector<cv::Point>>precontours;//contours������xΪ�У�yΪ��
	findContours(afteropen.clone(), precontours, cv::RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);
	double maxArea = 0;
	int contouridx = 0;
	if (precontours.size() == 0){
		cv::findContours(img.clone(), precontours, cv::RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);
		if (precontours.size() == 0){
			cout << "error in MajorDiameterXY" << endl;
			return -1;
		}
	}
	for (int i = 0; i< precontours.size(); i++)
	{
		double area = contourArea(precontours[i]);
		if (area>maxArea){
			contouridx = i;
			maxArea = area;
		}
	}
	vector<cv::Point> contours;
	contours = precontours[contouridx];
	cv::Point2f p1, p2;
	double max = 0, temp = 0;
	for (int i = 0; i < contours.size(); i++){
		p1 = contours.at(i);
		for (int j = i + 1; j < contours.size(); j++){
			p2 = contours.at(j);
			temp = sqrt(pow(p1.x - p2.x, 2) + pow(p1.y - p2.y, 2));
			if (temp>max){
				max = temp;
			}
		}
	}
	return max;
}

void solidNodule(vector<cv::Mat> &maskVector, vector<cv::Mat> &imgVector, vector<cv::Mat> &dst){
	int HuThreshold = -650;
	int threshold = 255 * (HuThreshold - (WindowsCenter - WindowsWidth / 2)) / WindowsWidth;
	cv::Mat img, mask;
	int roiYDim = imgVector[0].rows;
	int roiXDim = imgVector[0].cols;
	//vector<Mat> dst;
	for (int i = 0; i < imgVector.size(); i++)
	{
		img = imgVector[i];
		mask = maskVector[i];
		cv::Mat solid(roiYDim, roiXDim, CV_8UC1);
		solid.setTo(0);
		for (int yindex = 0; yindex < roiYDim; yindex++)
		{
			for (int xindex = 0; xindex < roiXDim; xindex++)
			{
				if (mask.at<uchar>(yindex, xindex)>0)
				{

					if (img.at<uchar>(yindex, xindex)>threshold)
					{
						solid.at<uchar>(yindex, xindex) = 255;
					}
				}
			}
		}
		dst.push_back(solid.clone());
	}
}

//����_binImg ��0��ʾ������255Ϊǰ��
//���_lableImg�б���Ϊ0����ͨ����labelΪ1,2,3,4....
void icvprCcaByTwoPass3D(const vector<cv::Mat>& _binImg, vector<cv::Mat>& _lableImg, vector<int> &useLabel)
{

	// connected component analysis (4-component)  
	// use two-pass algorithm  
	// 1. first pass: label each foreground pixel with a label  
	// 2. second pass: visit each labeled pixel and merge neighbor labels  
	//   

	//�����labelimg�б�����label��0

	if (_binImg.empty() ||
		_binImg[0].type() != CV_8UC1 || _binImg[0].empty())
	{
		return;
	}

	// 1. first pass
	_lableImg.resize(_binImg.size() + 1);
	cv::Mat temp;
	for (int i = 1; i < _lableImg.size(); i++){
		_lableImg[i].release();
		temp = _binImg[i - 1] / 255;
		temp.convertTo(_lableImg[i], CV_32SC1);
	}
	_lableImg[0] = cv::Mat(_binImg[0].rows, _binImg[0].cols, CV_32SC1);
	_lableImg[0].setTo(0);

	int label = 1;  // 
	std::vector<int> labelSet;
	labelSet.push_back(0);   // background: 0  
	labelSet.push_back(1);   // foreground: 1  

	int rows = _binImg[0].rows;
	int cols = _binImg[0].cols;
	for (int z = 1; z < _lableImg.size(); z++){
		cv::Mat preImg = _lableImg[z - 1];
		cv::Mat curImg = _lableImg[z];
		for (int i = 1; i < rows; i++)
		{
			int* data_preRow = curImg.ptr<int>(i - 1);
			int* data_curRow = curImg.ptr<int>(i);
			int* pre_data_curRow = preImg.ptr<int>(i);
			int* pre_data_preRow = preImg.ptr<int>(i - 1);
			for (int j = 1; j < cols; j++)
			{
				if (data_curRow[j] == 1)
				{
					std::vector<int> neighborLabels;
					neighborLabels.reserve(7);
					int leftPixel = data_curRow[j - 1];
					int upPixel = data_preRow[j];
					int corPixel = data_preRow[j - 1];
					int pre_mid_Pixel = pre_data_curRow[j];
					int pre_left_Pixel = pre_data_curRow[j - 1];
					int pre_up_Pixel = pre_data_preRow[j];
					int pre_cor_Pixel = pre_data_preRow[j - 1];
					if (leftPixel > 1)
					{
						neighborLabels.push_back(leftPixel);
					}
					if (upPixel > 1)
					{
						neighborLabels.push_back(upPixel);
					}
					if (corPixel > 1)
					{
						neighborLabels.push_back(corPixel);
					}
					if (pre_mid_Pixel > 1)
					{
						neighborLabels.push_back(pre_mid_Pixel);
					}
					if (pre_left_Pixel > 1)
					{
						neighborLabels.push_back(pre_left_Pixel);
					}
					if (pre_up_Pixel > 1)
					{
						neighborLabels.push_back(pre_up_Pixel);
					}
					if (pre_cor_Pixel > 1)
					{
						neighborLabels.push_back(pre_cor_Pixel);
					}

					if (neighborLabels.empty())
					{
						labelSet.push_back(++label);  // assign to a new label  
						data_curRow[j] = label;
						labelSet[label] = label;
					}
					else
					{
						std::sort(neighborLabels.begin(), neighborLabels.end());
						int smallestLabel = neighborLabels[0];
						data_curRow[j] = smallestLabel;

						// save equivalence  
						for (size_t k = 1; k < neighborLabels.size(); k++)
						{
							int tempLabel = neighborLabels[k];
							int& oldSmallestLabel = labelSet[tempLabel];
							if (oldSmallestLabel > smallestLabel)
							{
								labelSet[oldSmallestLabel] = smallestLabel;
								oldSmallestLabel = smallestLabel;
							}
							else if (oldSmallestLabel < smallestLabel)
							{
								labelSet[smallestLabel] = oldSmallestLabel;
							}
						}
					}
				}
			}
		}
	}

	// update equivalent labels  
	// assigned with the smallest label in each equivalent label set  
	for (size_t i = 1; i < labelSet.size(); i++)
	{
		int curLabel = labelSet[i];
		int preLabel = labelSet[curLabel];
		while (preLabel != curLabel)
		{
			curLabel = preLabel;
			preLabel = labelSet[preLabel];
		}
		labelSet[i] = curLabel;
	}


	// 2. second pass  
	//maxlabel = 0;
	vector<int>::iterator it;
	for (int z = 1; z < _lableImg.size(); z++){
		cv::Mat Img = _lableImg[z];
		for (int i = 0; i < rows; i++)
		{
			int* data = Img.ptr<int>(i);
			for (int j = 0; j < cols; j++)//
			{
				int& pixelLabel = data[j];
				pixelLabel = labelSet[pixelLabel];
				if (pixelLabel>0){
					pixelLabel = pixelLabel - 1;
					it = find(useLabel.begin(), useLabel.end(), pixelLabel);
					if (it == useLabel.end()){
						useLabel.push_back(pixelLabel);
					}
				}

				//if (pixelLabel >maxlabel){
				//	maxlabel = pixelLabel;
				//}
			}
		}
	}
	_lableImg.erase(_lableImg.begin());
	//cout<<endl<< maxlabel;
	//if (maxlabel > 0){
	//	FindSolid( _lableImg, maxlabel);
	//}
}

//�ҳ����ظ�������10��ʵ����ͨ�򲿷�
void FindSolid(vector<cv::Mat>& _lableImg, int &maxlabel){

	int pixelThreshold = 10;

	int rows = _lableImg[0].rows;
	int cols = _lableImg[0].cols;
	int *labelNum = new int[maxlabel + 1]();
	int index;
	for (int z = 0; z < _lableImg.size(); z++){
		for (int i = 0; i < rows; i++){
			for (int j = 0; j < cols; j++){
				index = _lableImg[z].at<int>(i, j);
				labelNum[index]++;
			}
		}
	}
	cout << "maxlabel" << maxlabel << endl;
	set<int> label;//��ŷ��ϴ�С������label
	for (int i = 1; i < maxlabel + 1; i++){
		if (labelNum[i]>pixelThreshold){
			label.insert(i);
			cout << labelNum[i] << "   " << i << endl;;
		}
	}
	maxlabel = label.size();
	for (int z = 0; z < _lableImg.size(); z++){
		for (int i = 0; i < rows; i++){
			for (int j = 0; j < cols; j++){
				int& pixelLabel = _lableImg[z].at<int>(i, j);
				if (label.count(pixelLabel) == 0){
					pixelLabel = 0;
				}
			}
		}
	}
}

//���ÿ����ͨ��֧��meanHU,stdHU ��������ɢ�̶�diversity
vector<vector<double>> getSolidConnectHuDiversityProperty(vector<cv::Mat>& connectLabel, vector<cv::Mat>& origin
	, vector<int> &useLabel, vector<double> roiXYZDim ,double &outputDiversity){
	vector<vector<int>> hu(useLabel.size());
	vector<vector<double>> ConnectXYZ(useLabel.size());
	//for (int i = 0; i < solidCon_num; i++){
	//	vector<int> temp;
	//	hu.push_back(temp);
	//}
	int rows = connectLabel[0].rows;
	int cols = connectLabel[0].cols;
	vector<int>::iterator it;
	int label = 0;
	int index = 0;
	for (int z = 0; z < connectLabel.size(); z++){
		for (int i = 0; i < rows; i++){
			for (int j = 0; j < cols; j++){
				label = connectLabel[z].at<int>(i, j);
				if (label> 0){
					it = find(useLabel.begin(), useLabel.end(), label);
					if (it != useLabel.end()){
						index = distance(useLabel.begin(), it);
						hu[index].push_back(origin[z].at<uchar>(i, j));
						ConnectXYZ[index].push_back(j);
						ConnectXYZ[index].push_back(i);
						ConnectXYZ[index].push_back(z);
					}

				}
			}
		}
	}
	double slope = double(WindowsWidth) / 255;
	double bias = double(WindowsCenter) - double(WindowsWidth) / 2;
	vector<vector<double>> huProperty;
	vector<vector<double>> DiversityProperty;
	for (int connect = 0; connect < useLabel.size(); connect++){
		vector<double> oneHuProperty;
		vector<double> oneDiversity;
		double meanHU = 0, stdHU = 0;
		double meanX = 0, meanY = 0,meanZ = 0;
		if (hu[connect].size() != 0){
			for (int i = 0; i < hu[connect].size(); i++){
				meanHU = meanHU + hu[connect][i];
				meanX = meanX + ConnectXYZ[connect][i * 3 + 0];
				meanY = meanY + ConnectXYZ[connect][i * 3 + 1];
				meanZ = meanZ + ConnectXYZ[connect][i * 3 + 2];
			}
			meanHU = meanHU / hu[connect].size();
			meanX = 3 * meanX / hu[connect].size();
			meanY = 3 * meanY / hu[connect].size();
			meanZ = 3 * meanZ / hu[connect].size();
			for (int i = 0; i < hu[connect].size(); i++){
				stdHU = stdHU + (hu[connect][i] - meanHU)*(hu[connect][i] - meanHU);
				//stdX = stdX + (ConnectXYZ[connect][i * 3 + 0] - meanX)*(ConnectXYZ[connect][i * 3 + 0] - meanX);				
				//stdY = stdY + (ConnectXYZ[connect][i * 3 + 1] - meanY)*(ConnectXYZ[connect][i * 3 + 1] - meanY);
				//stdZ = stdZ + (ConnectXYZ[connect][i * 3 + 2] - meanZ)*(ConnectXYZ[connect][i * 3 + 2] - meanZ);
			}
			stdHU = sqrt(stdHU / hu[connect].size());
			//stdX = sqrt(3*stdX / ConnectXYZ[connect].size());
			//stdY = sqrt(3 * stdY / ConnectXYZ[connect].size());
			//stdZ = sqrt(3 * stdZ / ConnectXYZ[connect].size());
			meanHU = slope*meanHU + bias;
			stdHU = slope*stdHU;
		}
		oneHuProperty.push_back(meanHU);
		oneHuProperty.push_back(stdHU);
		huProperty.push_back(oneHuProperty);
		oneDiversity.push_back(meanX);
		oneDiversity.push_back(meanY);
		oneDiversity.push_back(meanZ);
		DiversityProperty.push_back(oneDiversity);
	}
	double DiversityX = 0, DiversityY = 0, DiversityZ = 0;
	double xx = 0, yy = 0, zz = 0;
	if (DiversityProperty.size() != 0){
		for (int i = 0; i < DiversityProperty.size(); i++){
			DiversityX = DiversityX + DiversityProperty[i][0];
			DiversityY = DiversityY + DiversityProperty[i][1];
			DiversityZ = DiversityZ + DiversityProperty[i][2];
		}
		DiversityX = DiversityX / DiversityProperty.size();
		DiversityY = DiversityY / DiversityProperty.size();
		DiversityZ = DiversityZ / DiversityProperty.size();
		for (int i = 0; i < DiversityProperty.size(); i++){
			xx = xx + (DiversityProperty[i][0] - DiversityX)*(DiversityProperty[i][0] - DiversityX);
			yy = yy + (DiversityProperty[i][1] - DiversityY)*(DiversityProperty[i][1] - DiversityY);
			zz = zz + (DiversityProperty[i][2] - DiversityZ)*(DiversityProperty[i][2] - DiversityZ);
		}
		xx = sqrt(xx / DiversityProperty.size());
		yy = sqrt(yy / DiversityProperty.size());
		zz = sqrt(zz / DiversityProperty.size());
	}
	outputDiversity = (xx / roiXYZDim[0] + yy / roiXYZDim[1] + zz / roiXYZDim[2]) / 3.0;
	return huProperty;
}

vector<double> getSolidConnectVolumeProperty(vector<cv::Mat>& connectLabel, vector<int> &useLabel,
	double dxVoxel, double dyVoxel, double dzVoxel){
	vector<double> connectVolume;
	int rows = connectLabel[0].rows;
	int cols = connectLabel[0].cols;
	for (int label = 0; label < useLabel.size(); label++){
		vector<cv::Mat> certainLabel;
		vector<cv::Mat> resizedDst;
		double volSolid = 0;
		certainLabel.clear();
		resizedDst.clear();
		for (int z = 0; z < connectLabel.size(); z++){
			certainLabel.push_back((connectLabel[z] == (useLabel[label])) * 255);
		}
		resize3D(certainLabel, resizedDst, dxVoxel, dyVoxel, dzVoxel);
		volSolid = volume(resizedDst, 1);
		connectVolume.push_back(volSolid);
	}
	return connectVolume;
}

vector<int> getSolidVolumeHistogram(vector<double> solidConnectVolumeProperty, vector<double> xAxis){
	vector<int> volumeHistogram(xAxis.size(),0);
	double volume;
	for (int i = 0; i < solidConnectVolumeProperty.size(); i++){
		volume = solidConnectVolumeProperty[i];
		for (int x = xAxis.size()-1; x >=0; x--){
			if (volume>xAxis[x]){
				volumeHistogram[x] = volumeHistogram[x] + 1;
				break;
			}
		}
	}
	return volumeHistogram;
}