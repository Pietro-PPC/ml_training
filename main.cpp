#include <iostream>
#include <string>
#include <filesystem>
#include <fstream>
#include <format>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>

#include "include/rapidxml-1.13/rapidxml.hpp"

namespace fs = std::filesystem;
namespace xml = rapidxml;

void lsDir(const std::string &path){
    for (const auto &entry : fs::directory_iterator(path))
        std::cout << entry.path() << std::endl;
}

void drawRect(cv::Mat &img, cv::Point2f points[]){
    for (int i = 0; i < 4; ++i)
            cv::line(img, points[i], points[(i+1)%4], cv::Scalar(0, 0, 255));
}

xml::xml_document<> *getXmlDoc(const std::string &fname){
    std::ifstream fnameStream(fname);
    std::vector<char> fContent( (std::istreambuf_iterator<char>(fnameStream)), std::istreambuf_iterator<char>());
    xml::xml_document<> *doc = new xml::xml_document{};
    doc->parse<0>(&fContent[0]);    // 0 means default parse flag
    return doc;
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

int main(){
    const std::string DS_PATH = "../PKLot/PKLot/";
    const std::string DS_SEG_PATH = "../PKLot/MyPKLotSeg/";
    const std::string cur_dir = "PUCPR/Cloudy/2012-09-12/";
    const std::string file_pref = "2012-09-12_06_10_30";

    const std::string img_fname = DS_PATH + cur_dir + file_pref + ".jpg";
    const std::string xml_fname = DS_PATH + cur_dir + file_pref + ".xml";

    fs::create_directories(DS_SEG_PATH + cur_dir + "Empty/");
    fs::create_directories(DS_SEG_PATH + cur_dir + "Occupied/");

    // read image
    cv::Mat cv_img = cv::imread(img_fname);

    // get parking node

    xml::xml_document<> *doc = getXmlDoc(xml_fname);
    xml::xml_node<> *parking = doc->first_node("parking");
    xml::xml_node<> *space = parking->first_node();
    for(; space; space = space->next_sibling()){
        xml::xml_node<> *rect_node = space->first_node("rotatedRect");
        
        int id = atoi(space->first_attribute("id")->value());
        int occ = atoi(space->first_attribute("occupied")->value());

        cv::RotatedRect rotRect;

        // get xml data
        xml::xml_node<> *center_node = rect_node->first_node("center");
        rotRect.center.x = atof(center_node->first_attribute("x")->value());
        rotRect.center.y = atof(center_node->first_attribute("y")->value());

        xml::xml_node<> *size_node = rect_node->first_node("size");
        rotRect.size.width  = atof(size_node->first_attribute("w")->value());
        rotRect.size.height = atof(size_node->first_attribute("h")->value());

        xml::xml_node<> *angle_node = rect_node->first_node("angle");
        rotRect.angle = atof(angle_node->first_attribute("d")->value());

        // Crop rectangle
        cv::Mat cropped_img;
        cropRotatedRect(cv_img, cropped_img, rotRect);
        

        std::string fPath = DS_SEG_PATH + cur_dir;
        fPath += occ ? "Occupied/" : "Empty/";
        fPath += file_pref + "#" + std::format("{:03}",id) + ".jpg";

        cv::imwrite(fPath, cropped_img);
    }

    delete doc;

    return 0;
}
