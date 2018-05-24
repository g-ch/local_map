//
// Created by clarence on 18-5-17.
//

#include "../include/cloud_filter.h"
#include "boost/bind.hpp"

pcl::PointCloud<pcl::PointXYZ>::Ptr input_cloud_ptr( new pcl::PointCloud<pcl::PointXYZ>);
pcl::PointCloud<pcl::PointXYZ>::Ptr output_cloud_ptr( new pcl::PointCloud<pcl::PointXYZ>);

pcl::PointCloud<pcl::PointXYZI>::Ptr input_cloud_all_ptr( new pcl::PointCloud<pcl::PointXYZI>);

pcl::PointCloud<pcl::PointXYZ>::Ptr free_space_ptr( new pcl::PointCloud<pcl::PointXYZ>);
pcl::PointCloud<pcl::PointXYZ>::Ptr obstacle_ptr( new pcl::PointCloud<pcl::PointXYZ>);


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
    /// Area size. Should be the same as local "ewok_ring_buffer" map
    /// NOTE: Area_Length / Voxel_Length should be less than 64
    Area_Length = 6.4;
    Voxel_Length = 0.1;

    /// Threshold to treat as free space
    fs_min_val = 0.5;

    /// Voxel_Grid_Filter
    vg_f.Use = false;
    vg_f.Leaf_X = 0.15;
    vg_f.Leaf_Y = 0.15;
    vg_f.Leaf_Z = 0.15;

    /// Statistical_Outlier_Removal_Filter
    sor_f.Use = true;
    sor_f.Mean_K = 20;
    sor_f.Stddev_Mul_Thresh = 0.1;

    /// Neighbour radius for normal calculation
    Normal_Radius = 0.6;

    /// Limit to judge if a standardized normal a vertical or horizontal normal
    Vertical_Normal_Limit = 0.2f; //dot multiply
    Horizontal_Normal_Limit_Sqr = 0.0225f; //cross multiply

    /// Moving_Least_Squares_Reconstruction
    mls_r.Use = true;
    mls_r.Search_Radius = 1.0;  //1.5;//0.8;
    mls_r.Dilation_Voxel_Size = 0.2; //Important
    mls_r.Polynomial_Fit = true;

    /// Conditional_Euclidean_Clustering
    ce_c.Use = false;
    ce_c.Cluster_Tolerance = 1.0;
    ce_c.Point_Size_Min_Dividend = 500;
    ce_c.Point_Size_Max_Dividend = 3;

    /// Region_Growing_Segmentation
    rg_s.Use = true;
    rg_s.Point_Size_Min_Dividend = 500;
    rg_s.Point_Size_Max_Dividend = 3;
    rg_s.Number_Of_Neighbours = 72; //Important
    rg_s.Smoothness_Threshold = 5.0 / 180.0 * M_PI;
    rg_s.Curvature_Threshold = 2.0; //1.4; //Important
    rg_s.Indice_Size_Threshold = 30; //To remove clusters with too few points

}

CloudProcess::~CloudProcess()
{

}

void CloudProcess::filter_process()
{
    std::cout<<"start process"<<std::endl;

    input_cloud_ptr = input_cloud.makeShared();
    std::cout<<"input_cloud"<<std::endl<<input_cloud<<std::endl;

    /// Definition of cloud with normals, will be calculated either in
    pcl::PointCloud<pcl::PointNormal>::Ptr cloud_with_normals (new pcl::PointCloud<pcl::PointNormal>);
    /// Definition of normals, will be calculated Normal Estimation
    pcl::PointCloud<pcl::Normal>::Ptr normals (new pcl::PointCloud<pcl::Normal>);
    /// Link of output_cloud ptr
    output_cloud_ptr = output_cloud.makeShared();

    ///*1. First we filter*

    ///* Create the filtering object */
    /// VoxelGrid filter, to cut number
    if(vg_f.Use)
    {

        pcl::VoxelGrid<pcl::PointXYZ> vg;
        vg.setInputCloud (input_cloud_ptr);
        vg.setLeafSize (vg_f.Leaf_X, vg_f.Leaf_Y, vg_f.Leaf_Z);
        vg.filter (output_cloud);
        std::cout<<"VoxelGrid filter"<<std::endl<<output_cloud<<std::endl;
    } else
    {
        pcl::copyPointCloud(input_cloud, output_cloud);
    }

    /// Statistical Outlier Removal Filter, will cut number and remove outpoints, may not be useful
    if(sor_f.Use)
    {
        pcl::StatisticalOutlierRemoval<pcl::PointXYZ> sor;
        output_cloud_ptr = output_cloud.makeShared();
        sor.setInputCloud (output_cloud_ptr);
        sor.setMeanK (sor_f.Mean_K);
        sor.setStddevMulThresh (sor_f.Stddev_Mul_Thresh);
        sor.filter (output_cloud);
        std::cout<<"Statistical Outlier Removal"<<std::endl<<output_cloud<<std::endl;
    }

    /// Moving_Least_Squares_Reconstruction, to homogenize, will cut number
    if(mls_r.Use)
    {
        /// Create a KD-Tree
        pcl::search::KdTree<pcl::PointXYZ>::Ptr tree_sm (new pcl::search::KdTree<pcl::PointXYZ>);
        /// Output has the PointNormal type in order to store the normals calculated by MLS
        pcl::PointCloud<pcl::PointNormal> mls_points;
        /// Init object (second point type is for the normals, even if unused)
        pcl::MovingLeastSquares<pcl::PointXYZ, pcl::PointNormal> mls;

        mls.setComputeNormals (true);

        /// Set parameters
        output_cloud_ptr = output_cloud.makeShared();
        mls.setInputCloud (output_cloud_ptr);
        mls.setPolynomialFit (true);
        mls.setSearchMethod (tree_sm);
        mls.setSearchRadius (mls_r.Search_Radius);
        mls.setUpsamplingMethod(pcl::MovingLeastSquares<pcl::PointXYZ, pcl::PointNormal>::VOXEL_GRID_DILATION); //IMPORTANT
        mls.setDilationVoxelSize(mls_r.Dilation_Voxel_Size);
        /// Reconstruct
        mls.process (mls_points);

        std::cout<<"Voxel Size "<< mls.getDilationVoxelSize()<<std::endl;
        std::cout<<"Iterations "<< mls.getDilationIterations()<<std::endl;
        pcl::copyPointCloud(mls_points, output_cloud);
        std::cout<<"Moving_Least_Squares_Reconstruction"<<std::endl<<output_cloud<<std::endl;

        /// NORMALS SET TO "normals" HERE
        normals->resize(mls_points.size());
        for (size_t i = 0; i < mls_points.points.size(); ++i)
        {
            normals->points[i].normal_x = mls_points.points[i].normal_x;
            normals->points[i].normal_y = mls_points.points[i].normal_y;
            normals->points[i].normal_z = mls_points.points[i].normal_z;
            normals->points[i].curvature = mls_points.points[i].curvature;
        }

    } else
    {
        /// Normal Estimation
        pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> ne;
        pcl::search::Search<pcl::PointXYZ>::Ptr tree = boost::shared_ptr<pcl::search::Search<pcl::PointXYZ> >(new pcl::search::KdTree<pcl::PointXYZ>);
        output_cloud_ptr = output_cloud.makeShared();
        ne.setInputCloud (output_cloud_ptr);
        ne.setSearchMethod (tree);
        ne.setRadiusSearch (Normal_Radius) ;
        ne.compute (*normals);

        output_cloud_ptr = output_cloud.makeShared();
        pcl::concatenateFields (*output_cloud_ptr, *normals, *cloud_with_normals); //save normals and cloud in cloud_with_normals
    }

    ///*2. Now start to cluster*

    /// Conditional Euclidean Clustering
    if(ce_c.Use)
    {

        pcl::ConditionalEuclideanClustering<pcl::PointNormal> cec (true);
        cec.setInputCloud (cloud_with_normals);
        cec.setConditionFunction (&customRegionGrowing);
        cec.setClusterTolerance (ce_c.Cluster_Tolerance);
        cec.setMinClusterSize ((int)(cloud_with_normals->points.size () / ce_c.Point_Size_Min_Dividend));
        cec.setMaxClusterSize ((int)(cloud_with_normals->points.size () / ce_c.Point_Size_Max_Dividend));
        cec.segment (clusters);

        std::cout<<clusters.size()<<std::endl;
    }

    ///Region Growing Segmentation
    if(rg_s.Use)
    {
        pcl::RegionGrowing<pcl::PointXYZ, pcl::Normal> reg;
        pcl::search::Search<pcl::PointXYZ>::Ptr tree_rg = boost::shared_ptr<pcl::search::Search<pcl::PointXYZ> > (new pcl::search::KdTree<pcl::PointXYZ>);
        output_cloud_ptr = output_cloud.makeShared();
        reg.setMinClusterSize ((int)(output_cloud_ptr->points.size () / rg_s.Point_Size_Min_Dividend));
        reg.setMaxClusterSize ((int)(output_cloud_ptr->points.size () / rg_s.Point_Size_Max_Dividend));
        reg.setSearchMethod (tree_rg);
        reg.setNumberOfNeighbours (rg_s.Number_Of_Neighbours);
        reg.setInputCloud (output_cloud_ptr);
        //reg.setIndices (indices);
        reg.setInputNormals (normals);
        reg.setSmoothnessThreshold (rg_s.Smoothness_Threshold);
        reg.setCurvatureThreshold (rg_s.Curvature_Threshold);

        reg.extract (clusters);

        ///Remove too small indices
        int number_of_valid_points = 0;
        for(int i = 0; i < clusters.size (); i++)
        {
            std::vector<pcl::PointIndices>::iterator it;
            //clusters.begin(); i < clusters.end();
            if(clusters[i].indices.size() < rg_s.Indice_Size_Threshold)
            {
                std::vector<pcl::PointIndices>::iterator it = clusters.begin() + i;
                clusters.erase(it);
            }
            else
                number_of_valid_points += clusters[i].indices.size();
        }

        std::cout << "Number of clusters is equal to " << clusters.size () << std::endl;
        std::cout << "Number of valid points is equal to " << number_of_valid_points << std::endl;
        std::cout << "First cluster has " << clusters[0].indices.size () << " points." << std::endl;
        std::cout << "First point position x = " << output_cloud_ptr->points[clusters[0].indices[1]].x <<std::endl;
        std::cout << "First point position y = " << output_cloud_ptr->points[clusters[0].indices[1]].y <<std::endl;
        std::cout << "First point position z = " << output_cloud_ptr->points[clusters[0].indices[1]].z <<std::endl;
    }


    ///*3. Only keep vertical and horizontal plane cluster && Visualization clusters*/
    if(rg_s.Use)
    {
        std::vector<unsigned int> colors(3,1);
        int counter = 0;

        colored_cloud_e.clear(); //clear to start from beginning
        vertical_clusters.clear(); //clear to start from beginning
        horizontal_clusters.clear(); //clear to start from beginning

        for(size_t i_cluster = 0; i_cluster < clusters.size(); i_cluster++)
        {

            ///*3.1 Plane clustering by normals*
            /***NOTE: Consider to use RANSAC plan fitting to get normal in one cluster
             * rather than calculate average normal of normals calculated before
            */
            int plane_type; //0: vertical; 1: horizontal, 2:others

            ///Average normals for points in one cluster
            float norm_x_avg = 0.f;
            float norm_y_avg = 0.f;
            float norm_z_avg = 0.f;

            for(size_t j_point = 0; j_point <clusters[i_cluster].indices.size();j_point++)
            {
                float norm_x = normals->points[clusters[i_cluster].indices[j_point]].normal_x;
                float norm_y = normals->points[clusters[i_cluster].indices[j_point]].normal_y;
                float norm_z = normals->points[clusters[i_cluster].indices[j_point]].normal_z;

                float length = sqrt(norm_x*norm_x + norm_y*norm_y + norm_z*norm_z);
                norm_x_avg += norm_x / length;
                norm_y_avg += norm_y / length;
                norm_z_avg += norm_z / length;
            }

            norm_x_avg = norm_x_avg / clusters[i_cluster].indices.size();
            norm_y_avg = norm_y_avg / clusters[i_cluster].indices.size();
            norm_z_avg = norm_z_avg / clusters[i_cluster].indices.size();

            ///Judge if vertical or horizontal or others
            if(fabsf(norm_z_avg) < Vertical_Normal_Limit) //vertical, norm*(0,0,1)=0
            {
                plane_type = 0;
                vertical_clusters.push_back(clusters[i_cluster]);
            }
            else if(norm_x_avg * norm_x_avg + norm_y_avg * norm_y_avg < Horizontal_Normal_Limit_Sqr) //vertical, ||norm x (0,0,1)|| = 0
            {
                plane_type = 1;
                horizontal_clusters.push_back(clusters[i_cluster]);
            }
            else
                plane_type = 2;

            ///*3.2 Give different clusters different colors to show later*
            ///Add one cluster ptr
            colored_cloud_e.push_back((new pcl::PointCloud<pcl::PointXYZRGB>)->makeShared());
            counter = 0;
            ///Generate one color. Set random colors to display
            if(plane_type == 0) //Vertical; Red
            {
                colors[0] = rand() % 256 + 30;
                if(colors[0] > 255) colors[0] = 255;
                colors[1] = 0;
                colors[2] = 0;
            }
            else if(plane_type == 1) //Horizontal; Green
            {
                colors[0] = 0;
                colors[1] = rand() % 256 + 30;
                if(colors[1] > 255) colors[1] = 255;
                colors[2] = 0;
            }
            else  //Others; Blue / black
            {
                colors[0] = 0;
                colors[1] = 0;
                colors[2] = 0; //rand() % 256;
            }

            output_cloud_ptr = output_cloud.makeShared(); //link

            colored_cloud_e[i_cluster]->is_dense = output_cloud_ptr->is_dense;

            ///Set color and insert points sequence
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
    }


    ///*4. Process all vertical and  horizontal clusters to find or make a roof and a ground *
    /// Now, it is about logic :)  See specific functions



    std::cout<<"Process finished"<<std::endl;
    std::cout<<output_cloud<<std::endl;
}


void CloudProcess::filter_process_and_show()
{
    if(rg_s.Use)
        filter_process_and_show_result();
    else
        filter_process_and_show_cloud();

}

void CloudProcess::filter_process_and_show_cloud()
{
    filter_process();
    viewPointXYZ(output_cloud);
}

void CloudProcess::filter_process_and_show_result()
{
    filter_process();
    viewPointXYZRGBPtr(colored_cloud_e);
}

void CloudProcess::process_cloud_all()
{
    input_cloud_all_ptr = input_cloud_all.makeShared();

    /// Split freespace and obstacle
    freespace_obstacle_split(input_cloud_all_ptr, free_space_ptr, obstacle_ptr, fs_min_val);
    pcl::copyPointCloud(*obstacle_ptr, input_cloud);
    filter_process_and_show();
    two_dimension_map_generate();
}

void CloudProcess::two_dimension_map_generate()
{
    /// NOTE: x, y, z values in "input_cloud_all" should all be with in [0, Area_Length]
    ///       if not, please transform the coordinate by cutting the center position
    int length = (int) (Area_Length / Voxel_Length);
    std::cout << "***Length = " << length <<std::endl;
    cv::Mat map(length, length, CV_8UC1, cv::Scalar(0));
    cv::Mat map_ob(length, length, CV_8UC1, cv::Scalar(0)); /// NOTE: Area_Length / Voxel_Length should be less than 64
    cv::Mat map_fs(length, length, CV_8UC1, cv::Scalar(0)); /// NOTE: Area_Length / Voxel_Length should be less than 64

    input_cloud_all_ptr = input_cloud_all.makeShared();

    /// NOTE: No need to split again. Consider to improve efficiency later according to real input
    for(int i = 0; i < input_cloud_all_ptr->width; i++)
    {
        if(input_cloud_all_ptr->points[i].intensity > fs_min_val)  /// free space
        {
            map_fs.ptr<unsigned char>((int)(input_cloud_all_ptr->points[i].x / Voxel_Length))[(int)(input_cloud_all_ptr->points[i].y / Voxel_Length)] += 1;
        }
        else /// obstacle
        {
            map_ob.ptr<unsigned char>((int)(input_cloud_all_ptr->points[i].x / Voxel_Length))[(int)(input_cloud_all_ptr->points[i].y / Voxel_Length)] += 1;
        }
    }

//    for(int i = 0; i < length; i++) /// row
//    {
//        for(int j = 0; j < length; j++) /// col
//        {
//            if(map_fs.ptr<unsigned char>(i)[j] > 0 || map_ob.ptr<unsigned char>(i)[j] > 0)
//            {
//                /// If any point detected, give 128 as basic value. Detect in given height
//                map.ptr<unsigned char>(i)[j] = 128 +
//            }
//        }
//    }

    cv::imshow("ob", map_ob);
    cv::waitKey(100);
    cv::imshow("fs", map_fs);
    cv::waitKey(100);

}


void CloudProcess::vertical_clusters_process()
{
    //vertical_clusters
    output_cloud_ptr = output_cloud.makeShared();

}

void CloudProcess::horizontal_clusters_process()
{
    //horizontal_clusters
    output_cloud_ptr = output_cloud.makeShared();
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
        pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb(cloud[i]); //add cloud with name ID
        viewer->addPointCloud<pcl::PointXYZRGB> (cloud[i], rgb, cloud_name);
        viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, cloud_name);
    }

    viewer->addCoordinateSystem (1.0);
    viewer->initCameraParameters ();
    return (viewer);
}

void CloudProcess::freespace_obstacle_split(pcl::PointCloud<pcl::PointXYZI>::Ptr cloud, pcl::PointCloud<pcl::PointXYZ>::Ptr free_space_cloud, pcl::PointCloud<pcl::PointXYZ>::Ptr obstacle_cloud, float threshold)
{
    free_space_cloud->clear();
    obstacle_cloud->clear();

    for(int i = 0; i < cloud->width; i++)
    {
        if(cloud->points[i].intensity > threshold)
        {
            pcl::PointXYZ pclp;
            pclp.x = cloud->points[i].x;
            pclp.y = cloud->points[i].y;
            pclp.z = cloud->points[i].z;
            free_space_cloud->points.push_back(pclp);
        } else
        {
            pcl::PointXYZ pclp;
            pclp.x = cloud->points[i].x;
            pclp.y = cloud->points[i].y;
            pclp.z = cloud->points[i].z;
            obstacle_cloud->points.push_back(pclp);
        }
    }
}



