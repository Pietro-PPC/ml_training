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

void joinStrs(std::string &res, const std::vector<std::string> &strs, const int uBound, const std::string &sep){
    res = "";
    for(int i = 0; i < uBound; ++i)
        res += strs[i] + sep;
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

    const std::string DS_PATH = "../PKLot/PKLot/UFPR05";
    const std::string DS_SEG_PATH = "../PKLot/MyPKLotSeg/UFPR05";

    for (const auto &dirEnt : fs::recursive_directory_iterator(DS_PATH)){
        if (fs::is_directory(dirEnt) || dirEnt.path().extension() != ".jpg") continue;

        std::cout << "Processing " << dirEnt << " ..." << std::endl;

        std::string file_pref{dirEnt.path().stem()};
        std::string cur_dir{dirEnt.path().parent_path()};
        cur_dir.erase(0, DS_PATH.size()); cur_dir += "/";

        std::string empty_dir = DS_SEG_PATH + cur_dir + "Empty/";
        std::string occup_dir = DS_SEG_PATH + cur_dir + "Occupied/";

        fs::create_directories(empty_dir);
        fs::create_directories(occup_dir);

        const std::string img_input = DS_PATH + cur_dir + file_pref + ".jpg";
        const std::string xml_input = DS_PATH + cur_dir + file_pref + ".xml";

        cv::Mat cv_img = cv::imread(img_input);

        xml::xml_document<> *doc = getXmlDoc(xml_input);
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
    }

    return 0;
}
