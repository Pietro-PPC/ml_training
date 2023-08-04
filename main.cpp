#include <iostream>
#include <string>
#include <filesystem>
#include <fstream>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>

#include "include/rapidxml-1.13/rapidxml.hpp"

namespace fs = std::filesystem;
using namespace rapidxml;
using namespace std;

void lsDir(const std::string &path){
    for (const auto &entry : fs::directory_iterator(path))
        std::cout << entry.path() << std::endl;
}

void print_attributes(const xml_node<>* const node){
    for (xml_attribute<> *attr = node->first_attribute();
        attr; attr = attr->next_attribute()){

        cout << "\t" << attr->name() << " = " << attr->value() << "\n";
    }
}

int main(){
    const std::string DS_PATH = "../PKLot/PKLot/";
    const std::string xml_sample = DS_PATH + "PUCPR/Cloudy/2012-09-12/2012-09-12_06_10_30.xml";

    std::ifstream fname(xml_sample);
    std::vector<char> content( (istreambuf_iterator<char>(fname)), istreambuf_iterator<char>());
    xml_document<> doc; doc.parse<0>(&content[0]);    // 0 means default parse flags

    xml_node<> *root = doc.first_node();

    xml_node<> *space = root->first_node();
    for(; space; space = space->next_sibling()){
        cout << "Node: " << space->name() << "\n";
        print_attributes(space);

        xml_node<> *contour = space->first_node("contour");
        xml_node<> *point = contour->first_node();
        for (; point; point = point->next_sibling())
            print_attributes(point);
        

        std::cout << std::endl;
    }

    // print_attributes(node);

    return 0;
}
