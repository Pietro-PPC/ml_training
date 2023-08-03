#include <iostream>
#include <string>
#include <filesystem>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>

namespace fs = std::filesystem;

void lsDir(const std::string &path){
    for (const auto &entry : fs::directory_iterator(path))
        std::cout << entry.path() << std::endl;
}

int main(){
    const std::string DS_PATH = "../PKLot/PKLot/";
    
    const std::string img_sample = DS_PATH + "UFPR04/Cloudy/2012-12-12/2012-12-12_10_00_05.jpg";

    cv::Mat img;
    img = cv::imread(img_sample, cv::IMREAD_COLOR);

    if (!img.data){
        std::cerr << "No image data" << std::endl;
        return 1;
    }

    cv::imshow("Parking Lot", img);

    cv::waitKey(0);

    return 0;
}
