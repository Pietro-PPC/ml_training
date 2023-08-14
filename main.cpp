#include <iostream>
#include <string>
#include <filesystem>
#include <fstream>
#include <format>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>

#include <boost/property_tree/xml_parser.hpp>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/exceptions.hpp>

namespace pt = boost::property_tree;
namespace fs = std::filesystem;

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
    // cv::Mat *lbp_img{new cv::Mat::zeros(cv::Size{img.size[0]-2, img.size[1]-2}, cv::CV_8UC1)};
    cv::Mat *lbp_img{ new cv::Mat{ cv::Mat::zeros(cv::Size(img.cols-2, img.rows-2), CV_8UC1)} };

    std::vector<int> seqi{-1,0,1,1,1,0,-1,-1};
    std::vector<int> seqj{1,1,1,0,-1,-1,-1,0};

    for (unsigned int i{1}; i < img.size[0]-1; ++i){
        for (unsigned int j{1}; j < img.size[1]-1; ++j){
            unsigned char lbp{0};
            for (unsigned int k = 0; k < seqi.size(); ++k)
                lbp |= ( img.at<uchar>(i,j) >= img.at<uchar>(i+seqi[k],j+seqj[k]) ? 1 : 0 ) << k;
            lbp_img->at<uchar>(i-1,j-1) = lbp;
        }
    }
    
    return lbp_img;
}

int main(){
    // const std::string sample_img_path{"data/PKLot/MyPKLotSeg/PUCPR/Cloudy/2012-09-12/Empty/2012-09-12_06_05_16#001.jpg"};
    // const std::string sample_pklot_path{"data/PKLot/PKLot/PUCPR/Cloudy/2012-09-12/2012-09-12_06_05_16.jpg"}; 
    const std::string man{"data/samples/man.png"};

    cv::Mat sample_img{cv::imread(man)};
    cv::cvtColor(sample_img, sample_img, cv::COLOR_BGR2GRAY);
    
    cv::Mat *lbp{ getLBP(sample_img)};

    cv::imshow("image", *lbp);
    cv::waitKey(0);

    delete lbp;

    return 0;
}
