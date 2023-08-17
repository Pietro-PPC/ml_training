#include <vector>

class knn {
    public: 
        void train(const std::vector< std::pair<std::vector<double>,int> > &data);
        int fit(const std::vector<double> &pattern, const int k) const;

    private:
        std::vector< std::pair<std::vector<double>, int> > trainedData;
        double calcDist(const std::vector<double> &a, const std::vector<double> &b) const;
};