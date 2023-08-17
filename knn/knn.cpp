#include <vector>
#include <utility>
#include <algorithm>
#include <cmath>
#include <queue>

#include "knn.hpp"

void knn::train(const std::vector< std::pair<std::vector<double>, int> > &data){
    for (auto d : data)
        this->trainedData.push_back(d);
}


int knn::fit( const std::vector<double> &pattern, const int k=1) const{
    // std::vector<std::pair<double, int>> dists;
    std::priority_queue< 
        std::pair<double, int>, 
        std::vector<std::pair<double, int>>, 
        std::greater<std::pair<double, int>> > dists;

    for (auto &hist : this->trainedData){
        double curDist = this->calcDist(pattern, hist.first);
        if (dists.size() < k) // if heap's not full, fill it
            dists.emplace(curDist, hist.second);
        else if (curDist < dists.top().first){ // current distance is one of the k nearest so far
            dists.pop();
            dists.emplace(curDist, hist.second);
        }
    }

    int votes[] = {0,0};
    std::pair<double, int> dist;
    while (!dists.empty()){
        dist = dists.top(); dists.pop();
        votes[ dist.second ]++;
    }
    
    if (votes[0] > votes[1])
        return 0;
    return 1;
};

double knn::calcDist(const std::vector<double> &a, const std::vector<double> &b) const{
    double dst = 0.0;
    int sz = a.size();
    for (int i{0}; i < sz; ++i){
        double sub = a[i]-b[i];
        dst += sub*sub;
    }
    return dst;
}