//
// Created by clarence on 18-5-17.
//

#include "../include/cloud_filter.h"


pcl::PointCloud<pcl::PointXYZ>::Ptr input_cloud_ptr( new pcl::PointCloud<pcl::PointXYZ>);

CloudProcess::CloudProcess()
{
    sf.Use = true;
    sf.Mean_K = 20;
    sf.Stddev_Mul_Thresh = 0.1;
}

CloudProcess::~CloudProcess()
{

}

void CloudProcess::process()
{
    std::cout<<"start process"<<std::endl;

    input_cloud_ptr = input_cloud.makeShared(); //ISSUE

    /* Create the filtering object */
    // Statistical Filter
    if(sf.Use)
    {
        pcl::StatisticalOutlierRemoval<pcl::PointXYZ> sor;
        sor.setInputCloud (input_cloud_ptr);
        sor.setMeanK (sf.Mean_K);
        sor.setStddevMulThresh (sf.Stddev_Mul_Thresh);
        sor.filter (output_cloud);
    }


    std::cout<<"Process finished"<<endl;
    std::cout<<output_cloud<<endl;
}

boost::shared_ptr<pcl::visualization::PCLVisualizer> CloudProcess::PointXYZViewer(pcl::PointCloud<pcl::PointXYZ>::ConstPtr cloud)
{
    boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer (new pcl::visualization::PCLVisualizer ("3D Viewer"));
    viewer->setBackgroundColor (0, 0, 0);

    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> single_color (cloud, 0, 255, 0); //add green color
    viewer->addPointCloud<pcl::PointXYZ> (cloud, single_color, "cloud");

    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "cloud");
    viewer->addCoordinateSystem(1.0);
    viewer->initCameraParameters();
    return (viewer);
}

void CloudProcess::viewPointXYZPtr(pcl::PointCloud<pcl::PointXYZ>::ConstPtr cloud_ptr)
{
    cloud_viewer = PointXYZViewer(cloud_ptr);
}

void CloudProcess::viewPointXYZ(pcl::PointCloud<pcl::PointXYZ> cloud)
{
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_ptr( new pcl::PointCloud<pcl::PointXYZ>);
    cloud_ptr = cloud.makeShared();
    cloud_viewer = PointXYZViewer(cloud_ptr);
}
