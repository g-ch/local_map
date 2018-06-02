//
// Created by clarence on 18-6-1.
//

#include <string>
#include <iostream>
#include <vector>
#include <cmath>
#include "gvg.h"
#include <iostream>
#include <flann/flann.hpp>

using namespace std;
using namespace cv;

GVG::GVG() {}

GVG::~GVG() {}

void GVG::draw_voronoi( Mat& img, Subdiv2D& subdiv )
{
    vector<vector<Point2f> > facets;
    vector<Point2f> centers;
    subdiv.getVoronoiFacetList(vector<int>(), facets, centers);

    vector<Point> ifacet;

    for( size_t i = 0; i < facets.size(); i++ )
    {
        ifacet.resize(facets[i].size());
        for( size_t j = 0; j < facets[i].size(); j++ )
            ifacet[j] = facets[i][j];

        Scalar color;
        color = 150;
        //fillConvexPoly(img, ifacet, color, 8, 0);

        int isize = ifacet.size();

        for(size_t k = 0; k < isize - 1; k++)
        {
            if(ifacet[k].y > 0 && ifacet[k].y < img.rows && ifacet[k].x > 0 && ifacet[k].x < img.cols)
            {
                if(img.ptr<unsigned char>(ifacet[k].y)[ifacet[k].x] > 200)
                {
                    if(img.ptr<unsigned char>(ifacet[k+1].y)[ifacet[k+1].x] > 200)
                        line(img, ifacet[k], ifacet[k+1], color, 1);
                }
            }
        }
        if(img.ptr<unsigned char>(ifacet[0].y)[ifacet[0].x] > 200)
        {
            if(img.ptr<unsigned char>(ifacet[isize-1].y)[ifacet[isize-1].x] > 200)
                line(img, ifacet[isize-1], ifacet[0], color, 1);
        }
    }
}


void GVG::voronoi(cv::Mat &img)
{
    /// Find contours
    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;

    cv::findContours(img, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);


    /// Remove points on the edge of the image, Keep obstacle points
    std::vector<cv::Point2f> obstacle_points;
    for(int i = 0; i < contours[0].size(); i++)
    {
        if(contours[0][i].y != 0 && contours[0][i].y != img.rows - 1 && contours[0][i].x != 0 && contours[0][i].x != img.cols - 1)
            obstacle_points.push_back(contours[0][i]);
    }

    /// Create a safe road area by large contour size
    cv::drawContours(img, contours, 0, cv::Scalar(50), 5);

    cv::Size size = img.size();
    cv::Rect rect(0, 0, size.width, size.height);

    cv::Subdiv2D subdiv(rect);
    subdiv.insert(obstacle_points);

    draw_voronoi(img, subdiv);
}


void GVG::restructure(cv::Mat &map, cv::Mat &tangent_map, cv::Mat &restructured_map, std::vector<cv::Point3i> &clusters, float radius, float threshold)
{
    if(map.rows != tangent_map.rows || map.cols != tangent_map.cols) return;
    vector<Point> points;

    float square_radius = radius * radius;

    for(int i = 0; i < map.rows; i++)
    {
        for (int j = 0; j < map.cols; j++)
        {
            if(map.ptr<unsigned char>(i)[j] > 0)
            {
                Point temp_point;
                temp_point.x = j;
                temp_point.y = i;
                points.push_back(temp_point);
            }

        }
    }

    int valid_num = points.size();
    std::vector<int> points_cluster_seq(valid_num, 0);  /// Important answer

    int cluster_num = 1;
    while(true) /// Loop for clusters
    {
        vector<Point> points_search;  /// To generate a seed
        vector<int> points_search_seq;  /// From zero

        for(int i = 0; i < valid_num; i++)
        {
            if(points_cluster_seq[i] == 0) /// Virgin points
            {
                points_search.push_back(points[i]);
                points_search_seq.push_back(i);
            }

        }
        int search_num = points_search.size();
        if(search_num < 1) break;

        int seed_seq_search = rand() % search_num; /// 0 ~ virgin_num-1
        int seed_seq_valid = points_search_seq[seed_seq_search];

        Point seed_point = points_search[seed_seq_search];

        points_cluster_seq[seed_seq_valid] = cluster_num;

        points_search.erase(points_search.begin() + seed_seq_search);
        points_search_seq.erase(points_search_seq.begin() + seed_seq_search);

        search_num -= 1;

        /// To store intersection sequence
        std::vector<int> intersection_queue;
        int quene_size = 0;
        bool if_quene = false;

        while(true) /// Find points in one cluster
        {
            int counter_this_seed = 0;


            std::vector<Point> near_points;
            std::vector<int> near_points_seq;
            /// Search points satisfy the condition for this seed
            for(int j = 0; j < search_num; j++)
            {
                /// Verify if it has been allocated to a cluster
                int if_virgin = points_cluster_seq[points_search_seq[j]];

                if(if_virgin == 0)
                {
                    float dist = point_sqr_dist(points_search[j], seed_point);
                    if(dist < square_radius)
                    {
                        if(fabsf(tangent_map.ptr<float>(seed_point.y)[seed_point.x] - tangent_map.ptr<float>(points_search[j].y)[points_search[j].x]) < threshold)
                        {
                            counter_this_seed ++;
                            near_points.push_back(points_search[j]);
                            near_points_seq.push_back(j);
                        }
                    }
                }
            }

            if(counter_this_seed >= 2)
            {
                intersection_queue.push_back(seed_seq_search);
                quene_size ++;
            }
            else if(counter_this_seed == 0 && if_quene)
            {
                quene_size --;
                points_search.erase(points_search.end() - 1); /// Delete last one
            }


            if(counter_this_seed == 0 && quene_size < 1) break;

            /// Choose one as a new seed
            if(counter_this_seed > 0)
            {
                seed_point = near_points[0];
                seed_seq_search = near_points_seq[0];
                seed_seq_valid = points_search_seq[seed_seq_search];
                points_cluster_seq[seed_seq_valid] = cluster_num;
                if_quene = false;
            }
            else
            {
                seed_seq_search = intersection_queue[quene_size - 1];
                seed_point = points_search[seed_seq_search];
                seed_seq_valid = points_search_seq[seed_seq_search];
                if_quene = true;
            }

        }

        cluster_num ++;
    }

    /// Test showing
    cout<<cluster_num<<endl;
    for(int i = 0; i < points_cluster_seq.size(); i++)
    {
        cout<<"cluster: "<<points_cluster_seq[i] << ", ";
    }
    cout<<endl;


    /// Give result
    clusters.clear();
    for(int i=0; i < valid_num; i++)
    {
        Point3i p;
        p.x = points[i].x;
        p.y = points[i].y;
        p.z = points_cluster_seq[i];
        clusters.push_back(p);
    }

}


cv::Mat GVG::tangent_vector(cv::Mat &input_img, int window_size)
{
    /// 0~1.0, 0:invalid; 0.1-1.0: 0~180 degree
    cv::Mat output_img(input_img.rows, input_img.cols, CV_32F, cv::Scalar(0));

    for(int i = 0; i < input_img.rows; i++)
    {
        for(int j = 0; j < input_img.cols; j++)
        {
            if(input_img.ptr<unsigned char>(i)[j] == 0) continue;

            /// If point is not black, find nearby non black points in a defined size window
            vector<Point> nearby_points;  /// Self included
            for(int m = i - window_size; m <= i + window_size; m++) //row, y
            {
                for(int n = j - window_size; n <= j + window_size; n++) //col, x
                {
                    if(m >= 0 && n >= 0 && m < input_img.rows && n < input_img.cols)
                    {
                        if(input_img.ptr<unsigned char>(m)[n] > 0)
                        {
                            Point temp_point;
                            temp_point.x = n;
                            temp_point.y = m;
                            nearby_points.push_back(temp_point);
                        }
                    }

                }
            }

            /// Calculate tangent vector, average for directions between every two points
            int nearby_num = nearby_points.size();
            int counter = 0;
            double tangent = 0.0;
            if(nearby_num > 1)
            {
                for(int k = 0; k < nearby_num - 1; k++)
                {
                    for(int h = k + 1; h < nearby_num; h++)
                    {
                        int delt_x = nearby_points[k].x - nearby_points[h].x;
                        int delt_y = abs(nearby_points[k].y - nearby_points[h].y);

                        tangent += atan2((double)delt_y, (double)delt_x);
                        counter ++;
                    }
                }
                tangent = tangent / counter; //0 ~ PI
                output_img.ptr<float>(i)[j] = tangent / 3.5 + 0.1; /// 0.1-1.0: 0~180 degree
            }
        }
    }

    return output_img;
}



float GVG::point_sqr_dist(cv::Point &p1, cv::Point &p2)
{
    return (p1.x - p2.x)*(p1.x - p2.x) + (p1.y - p2.y)*(p1.y - p2.y);
}

void GVG::thinning(cv::Mat &img)
{
    while(true)
    {
        int counter = 0;
        cv::Mat map_zero1(img.rows, img.cols, CV_8UC1, cv::Scalar(0));
        cv::Mat map_zero2(img.rows, img.cols, CV_8UC1, cv::Scalar(0));

        /**
         * Test Method 1
         */
        std::vector<std::vector<cv::Point>> contours;
        std::vector<cv::Vec4i> hierarchy;
        cv::findContours(img, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);

        for(int k = 0; k < contours[0].size(); k++)
        {
            int i = contours[0][k].y;
            int j = contours[0][k].x;

            if(i == 0 || i == img.rows - 1 ||  j == 0 || j == img.cols - 1)
                continue;

            int neighbour[8];
            neighbour[0] = (img.ptr<unsigned char>(i-1)[j] > 0)? 1:0;
            neighbour[1] = (img.ptr<unsigned char>(i-1)[j+1] > 0)? 1:0;
            neighbour[2] = (img.ptr<unsigned char>(i)[j+1] > 0)? 1:0;
            neighbour[3] = (img.ptr<unsigned char>(i+1)[j+1] > 0)? 1:0;
            neighbour[4] = (img.ptr<unsigned char>(i+1)[j] > 0)? 1:0;
            neighbour[5] = (img.ptr<unsigned char>(i+1)[j-1] > 0)? 1:0;
            neighbour[6] = (img.ptr<unsigned char>(i)[j-1] > 0)? 1:0;
            neighbour[7] = (img.ptr<unsigned char>(i-1)[j-1] > 0)? 1:0;
            int bp1 = neighbour[0] + neighbour[1] + neighbour[2] + neighbour[3] + neighbour[4] + neighbour[5] + neighbour[6] + neighbour[7];
            int ap1 = 0;
            for(int m=0; m<7; m++)
            {
                if(neighbour[m] == 0 && neighbour[m+1] == 1)
                    ap1 ++;
            }

            if(bp1 >= 2 && bp1 <= 6 && ap1 == 1 && neighbour[0]*neighbour[2]*neighbour[4]==0 && neighbour[2]*neighbour[4]*neighbour[6] == 0){
                map_zero1.ptr<unsigned char>(i)[j] = 255;
                //img.ptr<unsigned char>(i)[j] = 0;
                counter ++;
            }
        }
        img = img - map_zero1;

        for(int k = 0; k < contours[0].size(); k++)
        {
            int i = contours[0][k].y;
            int j = contours[0][k].x;

            if(i == 0 || i == img.rows - 1 ||  j == 0 || j == img.cols - 1)
                continue;

            int neighbour[8];
            neighbour[0] = (img.ptr<unsigned char>(i-1)[j] > 0)? 1:0;
            neighbour[1] = (img.ptr<unsigned char>(i-1)[j+1] > 0)? 1:0;
            neighbour[2] = (img.ptr<unsigned char>(i)[j+1] > 0)? 1:0;
            neighbour[3] = (img.ptr<unsigned char>(i+1)[j+1] > 0)? 1:0;
            neighbour[4] = (img.ptr<unsigned char>(i+1)[j] > 0)? 1:0;
            neighbour[5] = (img.ptr<unsigned char>(i+1)[j-1] > 0)? 1:0;
            neighbour[6] = (img.ptr<unsigned char>(i)[j-1] > 0)? 1:0;
            neighbour[7] = (img.ptr<unsigned char>(i-1)[j-1] > 0)? 1:0;
            int bp1 = neighbour[0] + neighbour[1] + neighbour[2] + neighbour[3] + neighbour[4] + neighbour[5] + neighbour[6] + neighbour[7];
            int ap1 = 0;
            for(int m=0; m<7; m++)
            {
                if(neighbour[m] == 0 && neighbour[m+1] == 1)
                    ap1 ++;
            }

            if(bp1 >= 2 && bp1 <= 6 && ap1 == 1 && neighbour[0]*neighbour[2]*neighbour[6]==0 && neighbour[0]*neighbour[4]*neighbour[6] == 0) {
                map_zero2.ptr<unsigned char>(i)[j] = 255;
                //img.ptr<unsigned char>(i)[j] = 0;
                counter++;
            }
        }
        img = img - map_zero2;

        /**
        * Test Method 2
        */
//
//        for(int i=1; i<img.rows-1; i++)
//        {
//            for(int j=1; j<img.cols-1; j++)
//            {
//                if( img.ptr<unsigned char>(i)[j] > 0)
//                {
//                    int neighbour[8];
//                    neighbour[0] = (img.ptr<unsigned char>(i-1)[j] > 0)? 1:0;
//                    neighbour[1] = (img.ptr<unsigned char>(i-1)[j+1] > 0)? 1:0;
//                    neighbour[2] = (img.ptr<unsigned char>(i)[j+1] > 0)? 1:0;
//                    neighbour[3] = (img.ptr<unsigned char>(i+1)[j+1] > 0)? 1:0;
//                    neighbour[4] = (img.ptr<unsigned char>(i+1)[j] > 0)? 1:0;
//                    neighbour[5] = (img.ptr<unsigned char>(i+1)[j-1] > 0)? 1:0;
//                    neighbour[6] = (img.ptr<unsigned char>(i)[j-1] > 0)? 1:0;
//                    neighbour[7] = (img.ptr<unsigned char>(i-1)[j-1] > 0)? 1:0;
//                    int bp1 = neighbour[0] + neighbour[1] + neighbour[2] + neighbour[3] + neighbour[4] + neighbour[5] + neighbour[6] + neighbour[7];
//                    int ap1 = 0;
//                    for(int m=0; m<7; m++)
//                    {
//                        if(neighbour[m] == 0 && neighbour[m+1] == 1)
//                            ap1 ++;
//                    }
//
//                    if(bp1 >= 2 && bp1 <= 6 && ap1 == 1 && neighbour[0]*neighbour[2]*neighbour[4]==0 && neighbour[2]*neighbour[4]*neighbour[6] == 0){
//                        map_zero1.ptr<unsigned char>(i)[j] = 255;
//                        counter ++;
//                    }
//                }
//            }
//        }
//        img = img - map_zero1;
//
//        for(int i=1; i<img.rows-1; i++)
//        {
//            for(int j=1; j<img.cols-1; j++)
//            {
//                if( img.ptr<unsigned char>(i)[j] > 0)
//                {
//                    int neighbour[8];
//                    neighbour[0] = (img.ptr<unsigned char>(i-1)[j] > 0)? 1:0;
//                    neighbour[1] = (img.ptr<unsigned char>(i-1)[j+1] > 0)? 1:0;
//                    neighbour[2] = (img.ptr<unsigned char>(i)[j+1] > 0)? 1:0;
//                    neighbour[3] = (img.ptr<unsigned char>(i+1)[j+1] > 0)? 1:0;
//                    neighbour[4] = (img.ptr<unsigned char>(i+1)[j] > 0)? 1:0;
//                    neighbour[5] = (img.ptr<unsigned char>(i+1)[j-1] > 0)? 1:0;
//                    neighbour[6] = (img.ptr<unsigned char>(i)[j-1] > 0)? 1:0;
//                    neighbour[7] = (img.ptr<unsigned char>(i-1)[j-1] > 0)? 1:0;
//                    int bp1 = neighbour[0] + neighbour[1] + neighbour[2] + neighbour[3] + neighbour[4] + neighbour[5] + neighbour[6] + neighbour[7];
//                    int ap1 = 0;
//                    for(int m=0; m<7; m++)
//                    {
//                        if(neighbour[m] == 0 && neighbour[m+1] == 1)
//                            ap1 ++;
//                    }
//
//                    if(bp1 >= 2 && bp1 <= 6 && ap1 == 1 && neighbour[0]*neighbour[2]*neighbour[6]==0 && neighbour[0]*neighbour[4]*neighbour[6] == 0){
//                        map_zero2.ptr<unsigned char>(i)[j] = 255;
//                        counter ++;
//                    }
//                }
//            }
//        }
//        img = img - map_zero2;

        if(counter < 1) break;
    }
}