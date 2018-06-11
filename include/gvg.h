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

    void cluster_filter(cv::Mat &map, cv::Mat &tangent_map, cv::Mat &restructured_map, std::vector<std::vector<cv::Point>> &result_cluster, int min_points, float radius, float threshold);

    void thinning(cv::Mat &img);

    cv::Mat tangent_vector(cv::Mat &input_img, int window_size, float fit_threshold = 100.f);

    cv::Mat restructure(cv::Mat &area_map, std::vector<std::vector<cv::Point>> &cluster, int min_line_length, float max_dist, float max_slope_error); /// max_slope_error < 1.0

private:
    float point_sqr_dist(cv::Point &p1, cv::Point &p2);

    float point_dist(cv::Point &p1, cv::Point &p2);

    void draw_voronoi( cv::Mat& img, cv::Subdiv2D& subdiv );

    float point_to_line_dist(cv::Point line_a, cv::Point line_b, cv::Point p, float line_length);

    float point_to_line_sqr_dist(cv::Point line_a, cv::Point line_b, cv::Point p, float line_length);

    bool line_fit(cv::Mat &area_map, std::vector<cv::Point> &cluster, cv::Vec4f &line_para, std::vector<cv::Point> &temp_cluster, std::vector<cv::Point> &end_points_this, int max_step);

    bool in_mat_range(cv::Point &p, cv::Mat &area_map, int shrink_size = 0);
};

#endif //LOCAL_MAP_GVG_H
