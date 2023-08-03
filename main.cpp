#include <iostream>
#include <string>
#include <filesystem>
#include <fstream>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>

#include "include/rapidxml-1.13/rapidxml.hpp"

namespace fs = std::filesystem;
namespace rx = rapidxml;

void lsDir(const std::string &path){
    for (const auto &entry : fs::directory_iterator(path))
        std::cout << entry.path() << std::endl;
}

std::string getFileString(const std::string &fname){
    std::ifstream fstr; fstr.open(fname);
    
    std::string ret = "";
    std::string buf;

    while (fstr >> buf)
        ret += buf;

    return ret;
}

int main(){
    const std::string DS_PATH = "../PKLot/PKLot/";
    
    // PARTE 1: Mostrar Imagem
    const std::string img_sample = DS_PATH + "UFPR04/Cloudy/2012-12-12/2012-12-12_10_00_05.jpg";

    cv::Mat img;
    img = cv::imread(img_sample, cv::IMREAD_COLOR);

    if (!img.data){
        std::cerr << "No image data" << std::endl;
        return 1;
    }

    cv::imshow("Parking Lot", img);
    cv::waitKey(0);

    // PARTE 2: Ler XML
    const std::string xml_sample = DS_PATH + "UFPR04/Cloudy/2012-12-12/2012-12-12_10_00_05.xml";
    
    std::cout << getFileString(xml_sample) << std::endl;

    return 0;
}
