//
// Created by clarence on 18-5-24.
//

#ifndef LOCAL_MAP_GVG_H
#define LOCAL_MAP_GVG_H

#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/core/core.hpp"

class GVG
{
public:
    GVG();
    ~GVG();

    void voronoi(cv::Mat &img);

    void restructure(cv::Mat &map, cv::Mat &tangent_map, cv::Mat &restructured_map, std::vector<cv::Point3i> &clusters, float radius, float threshold);

    void thinning(cv::Mat &img);

    cv::Mat tangent_vector(cv::Mat &input_img, int window_size);

private:
    float point_sqr_dist(cv::Point &p1, cv::Point &p2);

    void draw_voronoi( cv::Mat& img, cv::Subdiv2D& subdiv );


};

#endif //LOCAL_MAP_GVG_H
