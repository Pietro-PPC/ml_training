#include <iostream>
#include <string>
#include <filesystem>
#include <fstream>
#include <format>

#include <omp.h>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>

#include <boost/property_tree/xml_parser.hpp>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/exceptions.hpp>
#include <boost/algorithm/string.hpp>

#include "knn/knn.hpp"

namespace pt = boost::property_tree;
namespace fs = std::filesystem;

const int HIST_SIZE{256};

void drawRect(cv::Mat &img, cv::Point2f points[]){
    for (int i = 0; i < 4; ++i)
            cv::line(img, points[i], points[(i+1)%4], cv::Scalar(0, 0, 255));
}

void cropRotatedRect(const cv::Mat &img, cv::Mat &croppedImg, const cv::RotatedRect &rotRect){
    cv::Point2f srcPoints[4], dstPoints[4];

    rotRect.points(srcPoints); // get source points

    cv::Size2f newSize{rotRect.size};
    if(rotRect.angle <= 45){ // get destiny points and change size if necessary
        std::swap(newSize.width, newSize.height);
        dstPoints[0] = cv::Point2f{newSize.width,newSize.height}; // starts with bottom right
        dstPoints[1] = cv::Point2f{0,newSize.height};
        dstPoints[2] = cv::Point2f{0,0};
        dstPoints[3] = cv::Point2f{newSize.width,0};
    } else {
        dstPoints[0] = cv::Point2f{0,newSize.height}; // starts with bottom left
        dstPoints[1] = cv::Point2f{0,0};
        dstPoints[2] = cv::Point2f{newSize.width,0};
        dstPoints[3] = cv::Point2f{newSize.width,newSize.height};
    }
    cv::Mat pt = cv::getPerspectiveTransform(srcPoints, dstPoints);
    cv::warpPerspective(img, croppedImg, pt, newSize);
}

void segmentImgs(const std::string &src_path, const std::string &dest_path){

    // ofstream logFile("logs.txt");

    for (const auto &dirEnt : fs::recursive_directory_iterator(src_path)){
        if (fs::is_directory(dirEnt) || dirEnt.path().extension() != ".jpg") continue;

        // create directories
        std::string file_pref{dirEnt.path().stem()};
        std::string cur_dir{dirEnt.path().parent_path()};
        cur_dir.erase(0, src_path.size()); cur_dir += "/";

        std::string empty_dir = dest_path + cur_dir + "Empty/";
        std::string occup_dir = dest_path + cur_dir + "Occupied/";

        fs::create_directories(empty_dir);
        fs::create_directories(occup_dir);

        const std::string img_input = src_path + cur_dir + file_pref + ".jpg";
        const std::string xml_input = src_path + cur_dir + file_pref + ".xml";

        // read image
        cv::Mat cv_img = cv::imread(img_input);

        pt::ptree tree;

        if (!fs::exists(xml_input)){
            std::cout << "File " << xml_input << " not found." << std::endl;
            continue;
        }
        read_xml(xml_input, tree);
        pt::ptree parking = tree.get_child("parking");
        pt::ptree::const_iterator parking_it = parking.begin();
        while (parking_it->first != "space") parking_it++;

        for (; parking_it != parking.end(); ++parking_it){
            int occ, id = parking_it->second.get<int>("<xmlattr>.id");

            try{
                occ = parking_it->second.get<int>("<xmlattr>.occupied");
            } catch(pt::ptree_error &exc){
                std::cerr << "id " << id << " of " << xml_input << " not processed\n"; // for now, segments without occupied attribute are ignored
                continue;
            }

            cv::RotatedRect rotRect;

            // get xml data
            rotRect.center.x = parking_it->second.get<float>("rotatedRect.center.<xmlattr>.x");
            rotRect.center.y = parking_it->second.get<float>("rotatedRect.center.<xmlattr>.y");
            
            rotRect.size.width = parking_it->second.get<float>("rotatedRect.size.<xmlattr>.w");
            rotRect.size.height = parking_it->second.get<float>("rotatedRect.size.<xmlattr>.h");

            rotRect.angle = parking_it->second.get<float>("rotatedRect.angle.<xmlattr>.d");

            // Crop rectangle
            cv::Mat cropped_img;
            cropRotatedRect(cv_img, cropped_img, rotRect);
            
            std::string fPath = dest_path + cur_dir;
            fPath += occ ? "Occupied/" : "Empty/";
            fPath += file_pref + "#" + std::format("{:03}",id) + ".jpg";

            cv::imwrite(fPath, cropped_img);
        }
    }

}

cv::Mat *getLBP(const cv::Mat &img){
    cv::Mat *lbp_img{ new cv::Mat{ cv::Mat::zeros(cv::Size(img.cols-2, img.rows-2), CV_8UC1)} };

    std::vector<int> seqi{-1,0,1,1,1,0,-1,-1};
    std::vector<int> seqj{1,1,1,0,-1,-1,-1,0};

    for (int i{1}; i < img.size[0]-1; ++i){
        for (int j{1}; j < img.size[1]-1; ++j){
            unsigned char lbp{0};
            for (unsigned int k = 0; k < seqi.size(); ++k)
                lbp |= ( img.at<uchar>(i,j) >= img.at<uchar>(i+seqi[k],j+seqj[k]) ? 1 : 0 ) << k;
            lbp_img->at<uchar>(i-1,j-1) = lbp;
        }
    }
    
    return lbp_img;
}

void appendHistogram(std::ofstream &fout, const cv::Mat &lbpHist, int cla, 
    const std::string &place, const std::string &w){
    
    for (int i = 0; i < lbpHist.size[0]; ++i) 
        fout << lbpHist.at<float>(i) << ";";

    fout << cla << ";" << place << ";" << w << "\n";
}

void getHistogram(cv::Mat &hist, const cv::Mat &img){
    const int histSize[] = {256};
    const float histRange1[] = {0,256};
    const float *histRange[] = {histRange1};
    const int channels[] = {0};

    hist = cv::Mat::zeros(256, 1, CV_32FC1);

    cv::calcHist(&img, 1, channels, cv::Mat(), hist, 1, histSize, histRange, true, false);
}

void getHistograms(const fs::path &dsDir, const fs::path &csvOut){
    std::ofstream fout;
    fout.open(csvOut);

    cv::Mat *lbp;
    int i{1};
    for (const auto &dirEnt : fs::recursive_directory_iterator(dsDir)){
        if (fs::is_directory(dirEnt.path())) continue;
        if (i % 1000 == 0) 
            std::cout << "\rImages processed: " << i << std::flush;
        
        // get parking lot, weather condition and status of parking place
        std::vector<std::string> spl;
        boost::split(spl, dirEnt.path().string(), boost::is_any_of("/"));
        std::string lugar{spl[3]};
        std::string condicao{spl[4]};
        std::string status{spl[6]};
        int isOcc = (status == "Empty") ? 0 : 1;

        // Read image, convert to lbp and append histogram to csv
        cv::Mat img{cv::imread(dirEnt.path())};
        cv::cvtColor(img, img, cv::COLOR_BGR2GRAY);
        
        cv::Mat lbpHist;
        lbp = getLBP(img);

        getHistogram(lbpHist, *lbp);
        appendHistogram(fout, lbpHist, isOcc, lugar, condicao);

        delete lbp;
        i++;
    }
    std::cout << std::endl;

    fout.close();

}

void splitTrainTest(const fs::path &csvPath, const fs::path &trainPath, const fs::path &testPath){
    const std::string tstPklot{"UFPR05"};

    std::ifstream csvFile{csvPath};
    std::ofstream trainFile{trainPath};
    std::ofstream testFile{testPath};
    std::string ln;

    std::vector<int> cntEmptyOcc{ {0,0} };
    int i{0};
    while (std::getline(csvFile, ln) ){
        if (i % 1000 == 0)  std::cout << "\rLines processed: " << i << std::flush;
        std::vector<std::string> spl;
        boost::split(spl, ln, boost::is_any_of(";"));
        
        std::vector<double> histogram(256);
        for (int it{0}; it < HIST_SIZE; ++it)
            histogram[it] = stod(spl[it]);
        cv::normalize (histogram, histogram, 1.0, 0.0, cv::NORM_MINMAX);

        std::ofstream &output = ( spl[spl.size()-2] != tstPklot) ? trainFile : testFile;
        
        for (int it{0}; it < HIST_SIZE; ++it)
            output << histogram[it] << ";" ;
        output << spl[spl.size()-3] << "\n";

        i++;
    }
    std::cout << "\rLines processed: " << i << std::flush;

}

void extractCsvData(std::vector< std::pair<std::vector<double>,int> > &data, const fs::path csvPath, bool verbose=true){
    std::ifstream csvStream{csvPath};
    std::string ln;
    // build data vector
    int i{0};
    while (std::getline(csvStream, ln) ){
        if (verbose && i % 10000 == 0)  std::cout << "\rLines processed: " << i << std::flush;
        std::vector<std::string> spl;
        boost::split(spl, ln, boost::is_any_of(";"));
        
        std::vector<double> curData;
        for (unsigned int it{0}; it < spl.size()-1; ++it)
            curData.push_back( stod(spl[it]) );
        
        data.emplace_back(curData, stoi(spl[ spl.size()-1 ]));
        i++;
        // if (i >= 10000) break;
    }
    if (verbose) std::cout << "\rLines processed: " << i << std::endl;

    csvStream.close();
}

const int THREAD_NUM = 8;
int main(){

    fs::path trnPath{"data/trainSet.csv"};
    fs::path tstPath{"data/testSet.csv"};
    
    std::vector< std::pair<std::vector<double>, int> > trnData, tstData;

    std::cout << "Extracting CSV data from " << trnPath << "..." << std::endl;
    extractCsvData(trnData, trnPath);
    std::cout << "Extraction finished." << std::endl;

    knn classifier;
    std::cout << "Training classifier..." << std::flush;
    classifier.train(trnData);
    std::cout << " done\n" << std::endl;

    std::cout << "Extracting CSV data from " << tstPath << "..." << std::endl;
    extractCsvData(tstData, tstPath);
    std::cout << "Extraction finished.\n" << std::endl;

    std::cout << "Fitting testing set to model..." << std::endl;
    std::vector<int> confMatrix(4, 0);
    int *confMatrixPtr = &confMatrix[0];
    // std::vector< std::pair<std::vector<double>, int> > tstSubset(&tstData[0], &tstData[0] + 20000);
    std::vector< std::pair<std::vector<double>, int> > tstSubset(tstData.begin(), tstData.end());

    int progress = 0;
    int step = 20;
    int sz = tstSubset.size();

    omp_set_num_threads(THREAD_NUM);
    #pragma omp parallel for reduction(+:confMatrixPtr[:4])
    for (int i =0; i<sz; ++i){
        std::pair<std::vector<double>, int> &dpair = tstSubset[i];
        int pred = classifier.fit(dpair.first, 3);
        confMatrixPtr[dpair.second*2 + pred]++;

        if (i%step == 0){
            #pragma omp critical
            {
                std::cout << "\r" << (++progress)*step << "/" << sz << std::flush;
            }
        }
    }
    
    std::cout << "\nFitting finished!" << std::endl;

    std::cout << "\nHere are your results\n";
    for (int i{0}; i < 2; ++i){
        for (int j{0}; j < 2; ++j)
            std::cout << std::setw(6) << confMatrix[i*2 + j] << " ";
        std::cout << "\n";
    }
    
    return 0;
}
