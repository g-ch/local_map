//
// Created by clarence on 18-5-17.
//

#include "../include/cloud_filter.h"
#include "boost/bind.hpp"

pcl::PointCloud<pcl::PointXYZ>::Ptr input_cloud_ptr( new pcl::PointCloud<pcl::PointXYZ>);

bool customRegionGrowing (const pcl::PointNormal& point_a, const pcl::PointNormal& point_b, float squared_distance)
{
    Eigen::Map<const Eigen::Vector3f> point_a_normal = point_a.getNormalVector3fMap (), point_b_normal = point_b.getNormalVector3fMap ();
    if (squared_distance < 10000)
    {
        if (fabs (point_a_normal.dot (point_b_normal)) < 0.06) //normal continuity
            return (true);
    }

    return (false);
}

CloudProcess::CloudProcess()
{
    sor_f.Use = true;
    sor_f.Mean_K = 20;
    sor_f.Stddev_Mul_Thresh = 0.1;

    vg_f.Use = false;
    vg_f.Leaf_X = 0.1;
    vg_f.Leaf_Y = 0.1;
    vg_f.Leaf_Z = 0.1;

    Normal_Radius = 0.2;

    ce_c.Use = false;
    ce_c.Cluster_Tolerance = 1.0;
    ce_c.Point_Size_Min_Dividend = 500;
    ce_c.Point_Size_Max_Dividend = 3;

    rg_s.Use = true;
    rg_s.Point_Size_Min_Dividend = 500;
    rg_s.Point_Size_Max_Dividend = 3;
    rg_s.Number_Of_Neighbours = 30;
    rg_s.Smoothness_Threshold = 3.0 / 180.0 * M_PI;
    rg_s.Curvature_Threshold = 0.2 * M_PI;

}

CloudProcess::~CloudProcess()
{

}

void CloudProcess::process()
{
    std::cout<<"start process"<<std::endl;

    input_cloud_ptr = input_cloud.makeShared(); //ISSUE

    /* Create the filtering object */
    // VoxelGrid filter
    if(vg_f.Use)
    {
        pcl::VoxelGrid<pcl::PointXYZ> vg;
        vg.setInputCloud (input_cloud_ptr);
        vg.setLeafSize (0.1f, 0.1f, 0.1f);
        vg.filter (output_cloud);
    }

    // Statistical Outlier Removal Filter
    if(sor_f.Use)
    {
        pcl::StatisticalOutlierRemoval<pcl::PointXYZ> sor;
        sor.setInputCloud (input_cloud_ptr);
        sor.setMeanK (sor_f.Mean_K);
        sor.setStddevMulThresh (sor_f.Stddev_Mul_Thresh);
        sor.filter (output_cloud);
    }

    // Normal Estimation
    pcl::PointCloud<pcl::PointXYZ>::Ptr output_cloud_ptr( new pcl::PointCloud<pcl::PointXYZ>);
    output_cloud_ptr = output_cloud.makeShared(); //ISSUE
    pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> ne;
    pcl::PointCloud<pcl::Normal>::Ptr normals (new pcl::PointCloud<pcl::Normal>);
    pcl::search::Search<pcl::PointXYZ>::Ptr tree = boost::shared_ptr<pcl::search::Search<pcl::PointXYZ> >(new pcl::search::KdTree<pcl::PointXYZ>);
    ne.setInputCloud (output_cloud_ptr);
    ne.setSearchMethod (tree);
    ne.setRadiusSearch (Normal_Radius) ;
    ne.compute (*normals);


    // Conditional Euclidean Clustering
    if(ce_c.Use)
    {
        pcl::PointCloud<pcl::PointNormal>::Ptr cloud_with_normals (new pcl::PointCloud<pcl::PointNormal>);
        pcl::concatenateFields (*output_cloud_ptr, *normals, *cloud_with_normals); //save normals and cloud in cloud_with_normals

        pcl::ConditionalEuclideanClustering<pcl::PointNormal> cec (true);
        cec.setInputCloud (cloud_with_normals);
        cec.setConditionFunction (&customRegionGrowing);
        cec.setClusterTolerance (ce_c.Cluster_Tolerance);
        cec.setMinClusterSize ((int)(cloud_with_normals->points.size () / ce_c.Point_Size_Min_Dividend));
        cec.setMaxClusterSize ((int)(cloud_with_normals->points.size () / ce_c.Point_Size_Max_Dividend));
        cec.segment (clusters);

        std::cout<<clusters.size()<<endl;
    }

    if(rg_s.Use)
    {
        pcl::RegionGrowing<pcl::PointXYZ, pcl::Normal> reg;
        reg.setMinClusterSize ((int)(output_cloud_ptr->points.size () / rg_s.Point_Size_Min_Dividend));
        reg.setMaxClusterSize ((int)(output_cloud_ptr->points.size () / rg_s.Point_Size_Max_Dividend));
        reg.setSearchMethod (tree);
        reg.setNumberOfNeighbours (rg_s.Number_Of_Neighbours);
        reg.setInputCloud (output_cloud_ptr);
        //reg.setIndices (indices);
        reg.setInputNormals (normals);
        reg.setSmoothnessThreshold (rg_s.Smoothness_Threshold);
        reg.setCurvatureThreshold (rg_s.Curvature_Threshold);

        reg.extract (clusters);

        std::cout << "Number of clusters is equal to " << clusters.size () << std::endl;
        std::cout << "First cluster has " << clusters[0].indices.size () << " points." << std::endl;
    }


    /*Visualization*/
    //pcl::PointCloud<pcl::PointXYZRGB>::Ptr colored_cloud_e[clusters.size()] ;
    //colored_cloud_e.insert(colored_cloud_e.end(),);//reserve(clusters.size());
    //std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr> colored_cloud_e;

    std::vector<unsigned int> colors(3,1);
    int counter = 0;

    //将分割好的点云设置颜色并保存出来（这一步单纯是为了显示）
    for(size_t i_cluster = 0; i_cluster < clusters.size(); i_cluster++)
    {
        colored_cloud_e.push_back((new pcl::PointCloud<pcl::PointXYZRGB>)->makeShared());
        counter = 0;
        colors[0] = rand() % 256;
        colors[1] = rand() % 256;
        colors[2] = rand() % 256;
        colored_cloud_e[i_cluster]->is_dense = output_cloud_ptr->is_dense;
        for(size_t i_point = 0; i_point <clusters[i_cluster].indices.size();i_point++)
        {
            pcl::PointXYZRGB point;
            point.x = (output_cloud_ptr->points[clusters[i_cluster].indices[i_point]].x);
            point.y = (output_cloud_ptr->points[clusters[i_cluster].indices[i_point]].y);
            point.z = (output_cloud_ptr->points[clusters[i_cluster].indices[i_point]].z);
            point.r = colors[0];
            point.g = colors[1];
            point.b = colors[2];
            colored_cloud_e[i_cluster]->points.push_back(point);

            counter++;
        }
        colored_cloud_e[i_cluster]->width = counter;
        colored_cloud_e[i_cluster]->height = 1;
        //std::stringstream ss;
        //ss << "cloud_cluster_e_" << i_cluster << ".pcd";
        //pcl::io::savePCDFileASCII(ss.str(), *colored_cloud_e[i_cluster]);
    }

    std::cout<<"Process finished"<<endl;
    std::cout<<output_cloud<<endl;
}

void CloudProcess::process_and_show_cloud()
{
    process();
    viewPointXYZ(output_cloud);
}

void CloudProcess::process_and_show_result()
{
    process();
    viewPointXYZRGBPtr(colored_cloud_e);
}

boost::shared_ptr<pcl::visualization::PCLVisualizer> CloudProcess::PointXYZViewer(pcl::PointCloud<pcl::PointXYZ>::ConstPtr cloud)
{
    boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer (new pcl::visualization::PCLVisualizer ("PointXYZViewer"));
    viewer->setBackgroundColor (0, 0, 0);

    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> single_color (cloud, 0, 255, 0); //add green color
    viewer->addPointCloud<pcl::PointXYZ> (cloud, single_color, "cloud");

    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "cloud");
    viewer->addCoordinateSystem(1.0);
    viewer->initCameraParameters();
    return (viewer);
}


boost::shared_ptr<pcl::visualization::PCLVisualizer> CloudProcess::PointXYZRGBViewer (std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr> cloud)
{
    boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer (new pcl::visualization::PCLVisualizer ("PointXYZRGBViewer"));
    viewer->setBackgroundColor (0, 0, 0);
    for(int i = 0; i < cloud.size(); i++)
    {
        std::string cloud_name = "cloud" + std::to_string(i);
        pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb(cloud[i]);
        viewer->addPointCloud<pcl::PointXYZRGB> (cloud[i], rgb, cloud_name);
        viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, cloud_name);
    }

    viewer->addCoordinateSystem (1.0);
    viewer->initCameraParameters ();
    return (viewer);
}


void CloudProcess::viewPointXYZPtr(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_ptr)
{
    cloud_viewer = PointXYZViewer(cloud_ptr);
}

void CloudProcess::viewPointXYZRGBPtr(std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr> clouds_Ptr)
{
    cloud_viewer = PointXYZRGBViewer(clouds_Ptr);
}

void CloudProcess::viewPointXYZ(pcl::PointCloud<pcl::PointXYZ> cloud)
{
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_ptr( new pcl::PointCloud<pcl::PointXYZ>);
    cloud_ptr = cloud.makeShared();
    cloud_viewer = PointXYZViewer(cloud_ptr);
}


