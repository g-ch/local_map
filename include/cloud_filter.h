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

class CloudProcess
{
public:
    CloudProcess();
    ~CloudProcess();

    /**
     * Main process function
     */
    void process();

    /**
     * deploy process function and show
     * To use this function, you must have a thread running  cloud_process.cloud_viewer->spinOnce(100)
     * Can not be used with any other show function in the same time
     */
    void process_and_show();


    /**
     * Global mat for cloud
     */
    pcl::PointCloud<pcl::PointXYZ> input_cloud;

    pcl::PointCloud<pcl::PointXYZ> output_cloud;

    std::vector<pcl::PointIndices> clusters; //clusters to store segmented indices

    std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr> colored_cloud_e; //store cloud pieces

    std::vector<pcl::PointIndices> vertical_clusters; //clusters to store vertical segmented indices, like walls

    std::vector<pcl::PointIndices> horizontal_clusters; //clusters to store horizontal segmented indices, like roof, ground and surface of table

    /**
     * Cloud viewer
     */
    boost::shared_ptr<pcl::visualization::PCLVisualizer> cloud_viewer;

    /**
     * Paremeters for filters or smoothers
     */
    double Normal_Radius;

    float Vertical_Normal_Limit;
    float Horizontal_Normal_Limit_Sqr;

    struct Voxel_Grid_Filter
    {
        bool Use;
        float Leaf_X;
        float Leaf_Y;
        float Leaf_Z;
    }vg_f;

    struct Statistical_Outlier_Removal_Filter
    {
        bool Use;
        int Mean_K;
        double Stddev_Mul_Thresh;
    }sor_f;

    struct Moving_Least_Squares_Reconstruction
    {
        bool Use;
        bool Polynomial_Fit;
        double Search_Radius;
        float Dilation_Voxel_Size;
        double Sqr_Gauss_Param;
    }mls_r;

    struct Conditional_Euclidean_Clustering
    {
        bool Use;
        double Normal_Radius;
        float Cluster_Tolerance;
        int Point_Size_Min_Dividend;
        int Point_Size_Max_Dividend;
    }ce_c;

    struct Region_Growing_Segmentation
    {
        bool Use;
        int Point_Size_Min_Dividend;
        int Point_Size_Max_Dividend;
        int Number_Of_Neighbours;
        float Smoothness_Threshold;
        float Curvature_Threshold;
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

    /**
     * deploy process function and show output_cloud
     * To use this function, you must have a thread running  cloud_process.cloud_viewer->spinOnce(100)
     * Can not be used with any other show function in the same time
     */
    void process_and_show_cloud();

    /**
     * deploy process function and show segmented cloud
     * To use this function, you must have a thread running  cloud_process.cloud_viewer->spinOnce(100)
     * Can not be used with any other show function in the same time
     */
    void process_and_show_result();
};

#endif //LOCAL_MAP_CLOUD_FILTER_H
