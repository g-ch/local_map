//
// Created by clarence on 18-5-17.
//

#ifndef LOCAL_MAP_CLOUD_FILTER_H
#define LOCAL_MAP_CLOUD_FILTER_H

#include <string>
#include <iostream>
#include <vector>
#include <cmath>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/features/normal_3d.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/segmentation/conditional_euclidean_clustering.h>
#include <pcl/search/search.h>
#include <pcl/search/kdtree.h>
#include <pcl/segmentation/region_growing.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/surface/mls.h>
#include "gvg.h"


class CloudProcess
{
public:
    CloudProcess();
    ~CloudProcess();

    /**
     * Main process function
     */
    void filter_process();

    /**
     * deploy process function and show
     * To use this function, you must have a thread running  cloud_process.cloud_viewer->spinOnce(100)
     * Can not be used with any other show function in the same time
     */
    void filter_process_and_show();

    /**
     * Process PointXYZI cloud  with obstacle and free space
     *
     */
    void process_cloud_all();

    /**
    * To split free space and obstacle PointXYZ cloud from input PointXYZI cloud, judging by intensity
    * @param cloud
    * @param free_space_cloud
    * @param obstacle_cloud
    */
    void freespace_obstacle_split(pcl::PointCloud<pcl::PointXYZI>::Ptr cloud, pcl::PointCloud<pcl::PointXYZ>::Ptr free_space_cloud, pcl::PointCloud<pcl::PointXYZ>::Ptr obstacle_cloud, float threshold);

    /**
     * NOTE!!!  To do
     * Most variables should be private and only have public functions to read them
     */

    /**
     * Global mat for cloud
     */
    /// Only obstacle cloud, input
    pcl::PointCloud<pcl::PointXYZ> input_cloud;

    /// Only obstacle cloud, output
    pcl::PointCloud<pcl::PointXYZ> output_cloud;

    /// Obstacle: intensity = 0.f and free space :intensity = 1.f, input
    pcl::PointCloud<pcl::PointXYZI> input_cloud_all;

    /// Obstacle: intensity = 0.f and free space :intensity = 1.f, output
    pcl::PointCloud<pcl::PointXYZI> output_cloud_all;

    /// Clusters to store segmented indices
    std::vector<pcl::PointIndices> clusters;

    /// Cloud pieces stored to show
    std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr> colored_cloud_e;

    /// Clusters to store vertical segmented indices, like walls
    std::vector<pcl::PointIndices> vertical_clusters;

    /// Clusters to store horizontal segmented indices, like roof, ground and surface of table
    std::vector<pcl::PointIndices> horizontal_clusters;


    /**
     * Global mat for 2D map image
     */
    cv::Mat map;
    cv::Mat map_intensity;
    cv::Mat map_filted_ob;

    /**
     * Parameters for structure features
     */

    /** Parameter to represent the roof and ground existence situation
     * Judging by Walls(Vertical Point Clusters)
     * 0: found walls with upper and lower bounds (eg: ordinary hall)
     * 1: found walls with only upper bound (eg: high hall, fly near roof)
     * 2: found walls with only lower bound (eg: high hall, fly near ground)
     * 3: found walls without bounds (eg: high hall, fly in the middle)
     * 4: found no walls (eg: in the air of a large space)
     */
    int vertical_structure_type;

    /** Parameter to represent the roof and ground existence situation
     * 0: with roof and ground
     * 1: with only roof
     * 2: with only ground
     * 3: without roof or ground
     * 4: with artificial roof, but without ground
     * 5: with artificial ground, but without roof
     * 6: with roof and artificial ground
     * 7: with ground and artificial roof
     * 8: with artificial roof and artificial ground
     */
    int horizontal_structure_type;

    /**
     * Parameters to represent the z position of roof and ground
     */
    float upper_bound;
    float lower_bound;
    double roof_height;
    double ground_height;
    double space_height;

    /**
     * Cloud viewer
     */
    boost::shared_ptr<pcl::visualization::PCLVisualizer> cloud_viewer;

    /**
     * Parameters about the robot
     */
    float robot_upper_height; /// Distance from the camera to the top of the robot
    float robot_lower_height; /// Distance from the camera to the bottom of the robot

    /**
     * Parameters for filters or smoothers
     */
    /// Neighbour radius for normal calculation
    double Normal_Radius;

    /// Limit to judge if a standardized normal a vertical or horizontal normal
    float Vertical_Normal_Limit;
    float Horizontal_Normal_Limit_Sqr;

    /// Area size. Should be the same as local "ewok_ring_buffer" map
    /// NOTE: Area_Length / Voxel_Length should be less than 64
    float Area_Length;
    float Voxel_Length;

    /// Threshold to treat as free space
    float fs_min_val;

    /// Voxel_Grid_Filter
    struct Voxel_Grid_Filter
    {
        bool Use;
        float Leaf_X;
        float Leaf_Y;
        float Leaf_Z;
    }vg_f;

    /// Statistical_Outlier_Removal_Filter
    struct Statistical_Outlier_Removal_Filter
    {
        bool Use;
        int Mean_K;
        double Stddev_Mul_Thresh;
    }sor_f;

    /// Moving_Least_Squares_Reconstruction
    struct Moving_Least_Squares_Reconstruction
    {
        bool Use;
        bool Polynomial_Fit;
        double Search_Radius;
        float Dilation_Voxel_Size;
        double Sqr_Gauss_Param;
    }mls_r;

    /// Conditional_Euclidean_Clustering
    struct Conditional_Euclidean_Clustering
    {
        bool Use;
        double Normal_Radius;
        float Cluster_Tolerance;
        int Point_Size_Min_Dividend;
        int Point_Size_Max_Dividend;
    }ce_c;

    /// Region_Growing_Segmentation
    struct Region_Growing_Segmentation
    {
        bool Use;
        int Point_Size_Min_Dividend;
        int Point_Size_Max_Dividend;
        int Number_Of_Neighbours;
        float Smoothness_Threshold;
        float Curvature_Threshold;
        int Indice_Size_Threshold;
    }rg_s;


private:

    /**
     * PointXYZ Viewer
     * @param cloud
     * @return
     */
    boost::shared_ptr<pcl::visualization::PCLVisualizer> PointXYZViewer(pcl::PointCloud<pcl::PointXYZ>::ConstPtr cloud);

    /**
     * PointXYZRGB Color Viewer
     * @param cloud
     * @return
     */
    boost::shared_ptr<pcl::visualization::PCLVisualizer> PointXYZRGBViewer (std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr> clouds_Ptr);


    /**
     * deploy process function and show output_cloud
     * To use this function, you must have a thread running  cloud_process.cloud_viewer->spinOnce(100)
     * Can not be used with any other show function in the same time
     */
    void filter_process_and_show_cloud();

    /**
     * deploy process function and show segmented cloud
     * To use this function, you must have a thread running  cloud_process.cloud_viewer->spinOnce(100)
     * Can not be used with any other show function in the same time
     */
    void filter_process_and_show_result();

    /**
     * Get a small local 2D map (opencv Mat) to generate GVG
     * Both original cloud data and filtered cloud with be used to enhance reliability
     */
    void two_dimension_map_generate();

    /**
     * Process vertical clusters. Normally, walls
     */
    void vertical_clusters_process();

    /**
     * Process horizontal clusters. Normally, roof and ground
     */
    void horizontal_clusters_process();

    /**
   * Display PointXYZ with input cloud Ptr
   * @param cloud_ptr
   */
    void viewPointXYZPtr(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_ptr);

    /**
     * Display PointXYZ with input cloud
     * @param cloud
     */
    void viewPointXYZ(pcl::PointCloud<pcl::PointXYZ> cloud);

    /**
     * Display PointXYZRGB clouds
     * @param cloud
     */
    void viewPointXYZRGBPtr(std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr> clouds_Ptr);

};

#endif //LOCAL_MAP_CLOUD_FILTER_H
