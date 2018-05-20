//
// Created by clarence on 18-5-17.
//

#ifndef LOCAL_MAP_CLOUD_FILTER_H
#define LOCAL_MAP_CLOUD_FILTER_H

#include <string>
#include <iostream>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/visualization/pcl_visualizer.h>

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
     * Display function
     */
    void viewPointXYZPtr(pcl::PointCloud<pcl::PointXYZ>::ConstPtr cloud_ptr);
    void viewPointXYZ(pcl::PointCloud<pcl::PointXYZ> cloud);

    /**
     * Global mat for cloud
     */
    pcl::PointCloud<pcl::PointXYZ> input_cloud;
    pcl::PointCloud<pcl::PointXYZ> output_cloud;

    /**
     * Cloud viewer
     */
    boost::shared_ptr<pcl::visualization::PCLVisualizer> cloud_viewer;

    /**
     * Paremeters for filters
     */
    struct Statistical_Filter
    {
        bool Use;
        int Mean_K;
        double Stddev_Mul_Thresh;
    }sf;

private:
    boost::shared_ptr<pcl::visualization::PCLVisualizer> PointXYZViewer(pcl::PointCloud<pcl::PointXYZ>::ConstPtr cloud);
};

#endif //LOCAL_MAP_CLOUD_FILTER_H
