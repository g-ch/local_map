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
        color = 150; /// The color for voronoi skeleton

        int isize = ifacet.size();

        for(size_t k = 0; k < isize - 1; k++)
        {
            if(in_mat_range(ifacet[k],img) && img.ptr<unsigned char>(ifacet[k].y)[ifacet[k].x] > 200)
                line(img, ifacet[k], ifacet[k], color, 1);
        }

//        for(size_t k = 0; k < isize - 1; k++)  /// Condition: in the image range, in white area
//        {
//            if(ifacet[k].y > 0 && ifacet[k].y < img.rows && ifacet[k].x > 0 && ifacet[k].x < img.cols)
//            {
//                if(ifacet[k+1].y > 0 && ifacet[k+1].y < img.rows && ifacet[k+1].x > 0 && ifacet[k+1].x < img.cols)
//                {
//                    if(img.ptr<unsigned char>(ifacet[k].y)[ifacet[k].x] > 200)
//                    {
//                        if(img.ptr<unsigned char>(ifacet[k+1].y)[ifacet[k+1].x] > 200)
//                            line(img, ifacet[k], ifacet[k+1], color, 1);
//                    }
//                }
//                else
//                {
//                    if(img.ptr<unsigned char>(ifacet[k].y)[ifacet[k].x] > 200)
//                    {
//                        line(img, ifacet[k], ifacet[k], color, 1);
//                    }
//                }
//
//            }
//        }
//        if(img.ptr<unsigned char>(ifacet[0].y)[ifacet[0].x] > 200 && ifacet[0].y > 0 && ifacet[0].y < img.rows && ifacet[0].x > 0 && ifacet[0].x < img.cols)
//        {
//            if(img.ptr<unsigned char>(ifacet[isize-1].y)[ifacet[isize-1].x] > 200 && ifacet[isize-1].y > 0 && ifacet[isize-1].y < img.rows && ifacet[isize-1].x > 0 && ifacet[isize-1].x < img.cols)
//                line(img, ifacet[isize-1], ifacet[0], color, 1);
//        }
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


void GVG::cluster_filter(cv::Mat &map, cv::Mat &tangent_map, cv::Mat &restructured_map, std::vector<std::vector<cv::Point>> &result_cluster, int min_points, float radius, float threshold)
{
    if(map.rows != tangent_map.rows || map.cols != tangent_map.cols) return;
    vector<Point> points;  /// to store all voronoi points

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
    cout<<"Start clustering!"<<endl;
    while(true) /// Loop for clusters
    {
        vector<Point> points_search;  /// To generate a seed
        vector<int> points_search_seq;  /// From zero, for the search group in each loop

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

        /// To store intersection sequence
        std::vector<int> intersection_queue;
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
                        float t1 = tangent_map.ptr<float>(seed_point.y)[seed_point.x];
                        float t2 = tangent_map.ptr<float>(points_search[j].y)[points_search[j].x];
                        if(t1 > 0.01f && t2 > 0.01f && fabsf(t1 - t2) < threshold)
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
            }

            if(if_quene)
            {
                intersection_queue.erase(intersection_queue.end()-1);
                if_quene = false;
            }

            if(counter_this_seed == 0 && intersection_queue.size() < 1) break;

            /// Choose one as a new seed
            if(counter_this_seed > 0)
            {
                seed_point = near_points[0];
                seed_seq_search = near_points_seq[0];
                seed_seq_valid = points_search_seq[seed_seq_search];
                points_cluster_seq[seed_seq_valid] = cluster_num;

            }
            else
            {
                seed_seq_search = intersection_queue[intersection_queue.size() - 1];
                seed_point = points_search[seed_seq_search];
                seed_seq_valid = points_search_seq[seed_seq_search];
                if_quene = true;
            }

        }

        cluster_num ++;
    } ///end of loop

    cluster_num -= 1;
    /// Test showing
    cout<<"Original cluster number = "<< cluster_num<<endl;

    /// Store cluster result
    for(int i=0; i<cluster_num; i++)
    {
        std::vector<cv::Point> temp_cluster;
        for(int j = 0; j< valid_num; j++)
        {
            if(points_cluster_seq[j] == i)
            {
                cv::Point p;
                p.x = points[j].x;
                p.y = points[j].y;
                temp_cluster.push_back(p);
            }
        }
        if(temp_cluster.size() > min_points)
            result_cluster.push_back(temp_cluster);
    }
    cout<<"Keep cluster number = "<< result_cluster.size()<<endl;
}


cv::Mat GVG::restructure(cv::Mat &area_map, std::vector<std::vector<cv::Point>> &cluster, float max_dist_error, float max_slope_error)
{
    /// This will restructure the area with the clusters and valid area
    cv::Mat output_img(area_map.rows, area_map.cols, CV_8UC1, cv::Scalar(0));

    float max_dist_error_sqr = max_dist_error * max_dist_error;
    float min_slope_transvection = 1.f - max_slope_error;

    cout << "min_slope_transvection=" << min_slope_transvection << endl;

    int max_step = sqrt(area_map.rows * area_map.rows + area_map.cols * area_map.cols);
    cout<<"max_step="<<max_step<<endl;
    std::vector<std::vector<cv::Point>> artificial_clusters;
    std::vector<cv::Vec4f> line_para_clusters;
    std::vector<std::vector<cv::Point>> end_points_clusters;

    cv::namedWindow("TEST");


    /// Now we fit all the points in each cluster to a line
    for(int i=0; i<cluster.size(); i++)
    {
        std::vector<cv::Point> temp_cluster;
        std::vector<cv::Point> end_points_this;
        cv::Vec4f line_para;

        if(line_fit(area_map, cluster[i], line_para, temp_cluster, end_points_this, max_step))
        {
            line_para_clusters.push_back(line_para);
            artificial_clusters.push_back(temp_cluster);
            end_points_clusters.push_back(end_points_this);
        }
        else /// Remove lines that can not be fitted (The reason is the line is too close to the edge of the map)
        {
            cluster.erase(cluster.begin() + i);
            i -= 1;
            cout<<"Removed one cluster!!"<<endl;
        }

    }

    /// For debug
    cv::Mat test_img = output_img.clone();
    for(int i = 0; i < end_points_clusters.size(); i++)
    {
        cout<<"final cluster = "<< end_points_clusters[i][0] << ", "<<end_points_clusters[i][1] << endl;
        cv::line(test_img, end_points_clusters[i][0], end_points_clusters[i][1], cv::Scalar(255), 2);
    }
    cv::imshow("TEST", test_img);
    cv::waitKey();
    /// End of debug code

    cout<<"end_points_clusters=" << end_points_clusters.size()<<endl;
//    for(int i = 0; i < end_points_clusters.size(); i++)
//    {
//        cv::line(output_img, end_points_clusters[i][0], end_points_clusters[i][1], cv::Scalar(150), 2);
//    }

    /// Merge lines
    for(int h=0; h<artificial_clusters.size()-1; h++)
    {
        for(int k=h+1; k<artificial_clusters.size(); k++)
        {

            bool dist_check_passed = false;
            bool slope_check_passed = false;

//            /// Check distance threshold by calculating distance from end points of one line to every points of another line
//            if(!dist_check_passed)
//            {
//                for(int i=0; i<artificial_clusters[k].size(); i++)  /// end points of h to points in k
//                {
//                    float dist1 = point_sqr_dist(end_points_clusters[h][0], artificial_clusters[k][i]);
//                    float dist2 = point_sqr_dist(end_points_clusters[h][1], artificial_clusters[k][i]);
//
//                    if(dist1 < max_dist_error_sqr || dist2 < max_dist_error_sqr)
//                    {
//                        dist_check_passed = true;
//                        break;
//                    }
//                }
//            }
//
//            if(!dist_check_passed)
//            {
//                for(int i=0; i<artificial_clusters[h].size(); i++)  /// end points of k to points in h
//                {
//                    float dist1 = point_sqr_dist(end_points_clusters[k][0], artificial_clusters[h][i]);
//                    float dist2 = point_sqr_dist(end_points_clusters[k][1], artificial_clusters[h][i]);
//
//                    if(dist1 < max_dist_error_sqr || dist2 < max_dist_error_sqr)
//                    {
//                        dist_check_passed = true;
//                        break;
//                    }
//                }
//            }

            /// Check distance between every two points to give the distance between two lines
            /// NOTE: need to be improved!!!!
            if(!dist_check_passed)
            {
                for(int i=0; i<artificial_clusters[k].size(); i++)
                {
                    for(int j=0; j<artificial_clusters[h].size(); j++)
                    {
                        float dist1 = point_sqr_dist(end_points_clusters[h][j], artificial_clusters[k][i]);
                        if(dist1 < max_dist_error_sqr)
                        {
                            dist_check_passed = true;
                            break;
                        }
                    }

                    if(dist_check_passed) break;
                }
            }

            /// Now check slope
            if(dist_check_passed)
            {
                /// Use transvection of the two normalized direction vectors to judge if the two lines are nealy parallel
                float slope_transvection = fabsf(line_para_clusters[h][0] * line_para_clusters[k][0] + line_para_clusters[h][1] * line_para_clusters[k][1]);
                cout<<"slope_transvection="<<slope_transvection<<endl;

                if(slope_transvection > min_slope_transvection)
                {
                    slope_check_passed = true;
                }
            }

            /// Now merge if necessary, push the values of h to k and continue with h+1
            if(dist_check_passed && slope_check_passed)
            {

                std::vector<cv::Point> cluster_merged;
                std::vector<cv::Point> temp_cluster;
                std::vector<cv::Point> end_points_this;
                cv::Vec4f line_para;

                /// Merge all points of line k and h

                /// NOTE: WHY??????? use original data would give wrong fitted line
                for(int m = 0; m < cluster[h].size(); m++)
                {
                    cluster_merged.push_back(cluster[h][m]);
                }
                for(int n = 0; n < cluster[k].size(); n++)
                {
                    cluster_merged.push_back(cluster[k][n]);
                }

//                for(int m = 0; m < artificial_clusters[h].size(); m++)
//                {
//                    cluster_merged.push_back(artificial_clusters[h][m]);
//                }
//                for(int n = 0; n < artificial_clusters[k].size(); n++)
//                {
//                    cluster_merged.push_back(artificial_clusters[k][n]);
//                }


                line_fit(area_map, cluster_merged, line_para, temp_cluster, end_points_this, max_step);

                /// Erase and insert
                artificial_clusters.erase(artificial_clusters.begin() + h);
                artificial_clusters.erase(artificial_clusters.begin() + k-1);

                cluster.erase(cluster.begin() + h);
                cluster.erase(cluster.begin() + k-1);

                line_para_clusters.erase(line_para_clusters.begin() + h);
                line_para_clusters.erase(line_para_clusters.begin() + k-1);

                end_points_clusters.erase(end_points_clusters.begin() + h);
                end_points_clusters.erase(end_points_clusters.begin() + k-1);

                artificial_clusters.insert(artificial_clusters.begin()+k-1, temp_cluster);
                line_para_clusters.insert(line_para_clusters.begin()+k-1, line_para);
                end_points_clusters.insert(end_points_clusters.begin()+k-1, end_points_this);
                cluster.insert(cluster.begin()+k-1, cluster_merged);


                /// For debug
                cv::Mat test_img = output_img.clone();
                for(int i = 0; i < end_points_clusters.size(); i++)
                {
                    cout<<"final cluster = "<< end_points_clusters[i][0] << ", "<<end_points_clusters[i][1] << endl;
                    cv::line(test_img, end_points_clusters[i][0], end_points_clusters[i][1], cv::Scalar(255), 2);
                }
                cv::imshow("TEST", test_img);
                cv::waitKey();
                /// End of debug code


                /// For next loop
                cout<<"Merge condition: "<<dist_check_passed<<", "<<slope_check_passed<<endl;
                h = h - 1;  /// Important
                break;

            }

            cout<<"Merge condition: "<<dist_check_passed<<", "<<slope_check_passed<<endl;
        }
    }


    cout<<"end_points_clusters=" << end_points_clusters.size()<<endl;
    for(int i = 0; i < end_points_clusters.size(); i++)
    {
        cout<<"final cluster = "<< end_points_clusters[i][0] << ", "<<end_points_clusters[i][1] << endl;
        cv::line(output_img, end_points_clusters[i][0], end_points_clusters[i][1], cv::Scalar(255), 2);
    }


    return output_img;
}


bool GVG::in_mat_range(cv::Point &p, cv::Mat &area_map, int shrink_size)
{
    if(p.x > shrink_size && p.x < area_map.cols - shrink_size && p.y > shrink_size && p.y < area_map.rows - shrink_size)
        return true;
    else
        return  false;
}

bool GVG::line_fit(cv::Mat &area_map, std::vector<cv::Point> &cluster, cv::Vec4f &line_para, std::vector<cv::Point> &artificial_cluster, std::vector<cv::Point> &end_points_this, int max_step)
{
    cv::fitLine(cluster, line_para, cv::DIST_L2, 0, 1e-2, 1e-2);


    float length = sqrt(line_para[0]*line_para[0] + line_para[1]*line_para[1]);
    float direction_x = line_para[0] / length;
    float direction_y = line_para[1] / length;
    line_para[0] = direction_x;  /// Store normalized direction vector
    line_para[1] = direction_y;

    cv::Point point0, point1, point2;
    /// NOTE: This point is not among points cluster, but a created one. So we need to find one
    point1.x = line_para[2] + 10.f*direction_x;
    point1.y = line_para[3] + 10.f*direction_y;
    point2.x = line_para[2] - 10.f*direction_x;
    point2.y = line_para[3] - 10.f*direction_y;

    float line_dist = 20.f;
    float min_dist_temp = 10000.f;

    point0.x = 0; /// Initialize
    point0.y = 0;

    for(int i = 0; i < cluster.size(); i++)
    {
        float dist_temp = point_to_line_dist(point1, point2, cluster[i], line_dist);
        if(dist_temp < min_dist_temp && in_mat_range(cluster[i], area_map, 1))
        {
            min_dist_temp = dist_temp;
            point0 = cluster[i];
        }
    }

    if(point0.x + point0.y == 0) /// No point was given
        return false;

    cout<<line_para[0]<<","<<line_para[1]<<","<<point0.x<<","<<point0.y<<endl;
    artificial_cluster.push_back(point0);

    cv::Point point_end1, point_end2;

    /// Grow lines
    /// One direction
    cv::Point last_check_point;
    last_check_point = point0;

//    if( max_step > cluster.size() * 2)
//        max_step = cluster.size() * 2;

    for(int j=0; j<max_step; j++)
    {
        int delt_x = j * direction_x;
        int delt_y = j * direction_y;

        cv::Point check_point;
        check_point.x = point0.x + delt_x;
        check_point.y = point0.y + delt_y;

        if(!in_mat_range(check_point, area_map, 1)) /// edge point in the map
        {
            point_end1.x = last_check_point.x;  /// point_end1
            point_end1.y = last_check_point.y;  /// point_end1
            //cout << "break" <<endl;
            break;
        }

        /// Check neighbourhood
        int condition_num = 0;
        int check_condition_size = 3;
        int left_side = (int)(check_condition_size / 2);

        for(int a=0; a<check_condition_size; a++)
        {
            for(int b=0; b<check_condition_size; b++)
            {
                cv::Point temp_point;
                temp_point.x = check_point.x - left_side + a;
                temp_point.y = check_point.y - left_side + b;
                if(area_map.ptr<unsigned char>(temp_point.y)[temp_point.x] == 255)
                {
                    condition_num ++;
                }
            }
        }

        if(condition_num >= check_condition_size * check_condition_size)
        {
            artificial_cluster.push_back(check_point);
            last_check_point.x = check_point.x;
            last_check_point.y = check_point.y;
        }
        else
        {
            point_end1.x = last_check_point.x;  /// point_end1
            point_end1.y = last_check_point.y;  /// point_end1
            //cout << "break" <<endl;
            break;
        }

        /// To fix j == max_step break without given endpoint problem
        if(j == max_step-1)
            point_end2 = check_point;
    }
    /// Opposite direction
//    for(int j=0; j<max_step; j++)
//    {
//        int delt_x = -j * direction_x;
//        int delt_y = -j * direction_y;
//
//        cv::Point check_point;
//        check_point.x = point0.x + delt_x;
//        check_point.y = point0.y + delt_y;
//
//        if(area_map.ptr<unsigned char>(check_point.y)[check_point.x] == 255 && in_mat_range(check_point, area_map))
//        {
//            artificial_cluster.push_back(check_point);
//            last_check_point.x = check_point.x;
//            last_check_point.y = check_point.y;
//        }
//        else
//        {
//            point_end2.x = last_check_point.x;  /// point_end2
//            point_end2.y = last_check_point.y;   /// point_end2
//            break;
//        }
//    }

    /// Opposite direction
    last_check_point = point0;
    for(int j=0; j<max_step; j++)
    {
        int delt_x = -j * direction_x;
        int delt_y = -j * direction_y;

        cv::Point check_point;
        check_point.x = point0.x + delt_x;
        check_point.y = point0.y + delt_y;

        if(!in_mat_range(check_point, area_map, 1)) /// edge point in the map
        {
            point_end2.x = last_check_point.x;  /// point_end2
            point_end2.y = last_check_point.y;  /// point_end2
            //cout << "break" <<endl;
            break;
        }

        /// Check neighbourhood
        int condition_num = 0;
        int check_condition_size = 3;
        int left_side = (int)(check_condition_size / 2);

        for(int a=0; a<check_condition_size; a++)
        {
            for(int b=0; b<check_condition_size; b++)
            {
                cv::Point temp_point;
                temp_point.x = check_point.x - left_side + a;
                temp_point.y = check_point.y - left_side + b;
                if(area_map.ptr<unsigned char>(temp_point.y)[temp_point.x] == 255)
                {
                    condition_num ++;
                }
            }
        }

        if(condition_num >= check_condition_size * check_condition_size)
        {
            artificial_cluster.push_back(check_point);
            last_check_point.x = check_point.x;
            last_check_point.y = check_point.y;
        }
        else
        {
            point_end2.x = last_check_point.x;  /// point_end2
            point_end2.y = last_check_point.y;  /// point_end2
            //cout << "break" <<endl;
            break;
        }

        /// To fix j == max_step break without given endpoint problem
        if(j == max_step-1)
            point_end2 = check_point;

    }

    cout<<endl<<"point_end1="<<point_end1.x<<"," << point_end1.y<<"; ";
    cout<<endl<<"point_end2="<<point_end2.x<<"," << point_end2.y<<endl;
    end_points_this.push_back(point_end1);
    end_points_this.push_back(point_end2);

    return true;
}



cv::Mat GVG::tangent_vector(cv::Mat &input_img, int window_size, float fit_threshold)
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

            /// This is an old method
//            /// Calculate tangent vector, average for directions between every two points
//            int nearby_num = nearby_points.size();
//            int counter = 0;
//            double tangent = 0.0;
//            if(nearby_num > 1)
//            {
//                for(int k = 0; k < nearby_num - 1; k++)
//                {
//                    for(int h = k + 1; h < nearby_num; h++)
//                    {
//                        int delt_x = nearby_points[k].x - nearby_points[h].x;
//                        int delt_y = abs(nearby_points[k].y - nearby_points[h].y);
//
//                        tangent += atan2((double)delt_y, (double)delt_x);
//                        counter ++;
//                    }
//                }
//                tangent = tangent / counter; //0 ~ PI
//                output_img.ptr<float>(i)[j] = tangent / 3.5 + 0.1; /// 0.1-1.0: 0~180 degree
//            }

            /// This is the new method
            /// Calculate tangent vector, use two points as seeds to generate a line and find the best fit
            int nearby_num = nearby_points.size();

            if(nearby_num > 1)
            {
                double tangent = 0.0;
                float min_dist = window_size * 10.f;
                int def_p1_seq;
                int def_p2_seq;

                for(int k = 0; k < nearby_num - 1; k++)
                {
                    for(int h = k + 1; h < nearby_num; h++)
                    {
                        float line_length = point_dist(nearby_points[k], nearby_points[h]);
                        float dist_temp = 0.f;
                        for(int w=0; w<nearby_num; w++)
                        {
                            //dist_temp += point_to_line_sqr_dist(nearby_points[k], nearby_points[h], nearby_points[w], line_length);
                            dist_temp += point_to_line_dist(nearby_points[k], nearby_points[h], nearby_points[w], line_length);
                        }

                        /// A weight between distance of center point to line and nearby points to line
                        cv::Point center_point;
                        center_point.x = j;
                        center_point.y = i;
                        dist_temp = 0.1*point_to_line_dist(nearby_points[k], nearby_points[h],center_point,line_length) + 0.9*dist_temp;

                        if(dist_temp < min_dist)
                        {
                            min_dist = dist_temp;
                            def_p1_seq = k;
                            def_p2_seq = h;
                        }

                    }
                }

                if(min_dist < fit_threshold)
                {
                    int delt_x = nearby_points[def_p1_seq].x - nearby_points[def_p2_seq].x;
                    int delt_y = abs(nearby_points[def_p1_seq].y - nearby_points[def_p2_seq].y);

                    tangent = atan2((double)delt_y, (double)delt_x);
                    output_img.ptr<float>(i)[j] = tangent / 3.5 + 0.1; /// 0.1-1.0: 0~180 degree
                }

            }
        }
    }

    return output_img;
}


float GVG::point_to_line_dist(cv::Point line_a, cv::Point line_b, cv::Point p, float line_length)
{
    return sqrt(point_to_line_sqr_dist(line_a, line_b, p, line_length)); /// return length of Line PC
}


float GVG::point_to_line_sqr_dist(cv::Point line_a, cv::Point line_b, cv::Point p, float line_length)
{
    float ap_x = p.x - line_a.x;
    float ap_y = p.y - line_a.y;
    float ab_x = line_b.x - line_a.x;
    float ab_y = line_b.y - line_a.y;

    float ab_i_x = ab_x / line_length;
    float ab_i_y = ab_y / line_length;

    /// C is P's projective point on line AB
    float ac_length = (ap_x * ab_x + ap_y * ab_y) / line_length;

    float ac_x = ac_length * ab_i_x;
    float ac_y = ac_length * ab_i_y;

    float pc_x = ac_x - ap_x;
    float pc_y = ac_y - ap_y;

    return pc_x*pc_x + pc_y*pc_y; /// return square length of Line PC

}

float GVG::point_dist(cv::Point &p1, cv::Point &p2)
{
    return sqrt(p1.x - p2.x)*(p1.x - p2.x) + (p1.y - p2.y)*(p1.y - p2.y);
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