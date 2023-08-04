#include <iostream>
#include <string>
#include <filesystem>
#include <fstream>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>

#include "include/rapidxml-1.13/rapidxml.hpp"

namespace fs = std::filesystem;
namespace xml = rapidxml;

void lsDir(const std::string &path){
    for (const auto &entry : fs::directory_iterator(path))
        std::cout << entry.path() << std::endl;
}

void print_attributes(const xml::xml_node<>* const node){
    for (xml::xml_attribute<> *attr = node->first_attribute();
        attr; attr = attr->next_attribute()){

        std::cout << "\t" << attr->name() << " = " << attr->value() << "\n";
    }
}

int main(){
    const std::string DS_PATH = "../PKLot/PKLot/";
    const std::string xml_sample = DS_PATH + "PUCPR/Cloudy/2012-09-12/2012-09-12_06_10_30.xml";

    std::ifstream fname(xml_sample);
    std::vector<char> content( (std::istreambuf_iterator<char>(fname)), std::istreambuf_iterator<char>());
    xml::xml_document<> doc; doc.parse<0>(&content[0]);    // 0 means default parse flags

    xml::xml_node<> *root = doc.first_node();
    xml::xml_node<> *space = root->first_node();
    for(; space; space = space->next_sibling()){
        std::cout << "Node: " << space->name() << "\n";
        print_attributes(space);

        xml::xml_node<> *contour = space->first_node("contour");
        xml::xml_node<> *point = contour->first_node();
        std::pair<unsigned int, unsigned int> rect[4];
        for (int it = 0; point && it < 4; point = point->next_sibling(), ++it){
            rect[it].first  = atoi(point->first_attribute("x")->value());
            rect[it].second = atoi(point->first_attribute("y")->value());
            std::cout << "\t" << rect[it].first << " " << rect[it].second << std::endl;
        }

        std::cout << std::endl;
    }

    return 0;
}
