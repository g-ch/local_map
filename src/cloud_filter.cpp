//
// Created by clarence on 18-5-17.
//

#include "../include/cloud_filter.h"
#include "boost/bind.hpp"

using namespace std;
using namespace cv;

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
    /// NOTE: Area_Length / Voxel_Length should be smaller than 255. (uchar index)
    Area_Length = 6.4;
    Voxel_Length = 0.1;

    /// Threshold to treat as free space
    fs_min_val = 0.5;

    /// Voxel_Grid_Filter
    /// NOTE: Maximum cloud points number after this down sampling filter is 65536. You know why.
    /// Change on int data type is not suggested. Because it would be too slow even if you change the data type to avoid mistake
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
    /// Note: these parameters need to be tuned carefully later !!!!
    rg_s.Use = true;
    rg_s.Point_Size_Min_Dividend = 500;
    rg_s.Point_Size_Max_Dividend = 3;
    rg_s.Number_Of_Neighbours = 72; //Important
    rg_s.Smoothness_Threshold = 5.0 / 180.0 * M_PI;
    rg_s.Curvature_Threshold = 2.0; //1.4; //Important
    rg_s.Indice_Size_Threshold = 30; //To remove clusters with too few points

    /// Distance from the camera to the top \ bottom of the robot
    robot_upper_height = 0.3f;
    robot_lower_height = 0.3f;

    /// Paremeters about 2D map
    valid_fspoints_search_radius =1.f;
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
            /***NOTE:  To do
             * Consider to use RANSAC plan fitting to get normal in one cluster
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
            /// Can be removed to run faster
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

    if(rg_s.Use)
    {
        vertical_clusters_process();
        horizontal_clusters_process();

        space_height = roof_height - ground_height;

        std::cout<<"space_height = " << space_height<<std::endl;
        std::cout<<"roof_height = " << roof_height<<std::endl;
        std::cout<<"ground_height = " << ground_height<<std::endl;
        std::cout<<"Space type = (vertical, horizontal) = (" << vertical_structure_type <<", "<< horizontal_structure_type <<")"<< std::endl;

    }

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


    /// Too slow with little use. Just abort it....
    //filter_process_and_show();

    /// Use this one directly is better.
    two_dimension_map_generate();
}

void CloudProcess::two_dimension_map_generate()
{
    /// NOTE: x, y, z values in "input_cloud_all" should all be with in [0, Area_Length]
    ///       if not, please transform the coordinate by cutting the center position, where the robot is.
    int length = (int) (Area_Length / Voxel_Length);
    std::cout << "***Length = " << length <<std::endl;

    /// Initialize maps
    map = cv::Mat::zeros(length, length, CV_8UC1);
    cv::Mat map_ob(length, length, CV_8UC1, cv::Scalar(0));
    cv::Mat map_fs(length, length, CV_8UC1, cv::Scalar(0));

    /// Ratio of obstacle and free space to whole available space along z axis
    std::vector<std::vector<float>> map_ob_ratio;
    std::vector<std::vector<float>> map_fs_ratio;

    input_cloud_all_ptr = input_cloud_all.makeShared();

    /// Find roof and ground height
    int number_bound = (int) (pow(valid_fspoints_search_radius, 3) * 4.2 / pow(Voxel_Length, 3) * 0.4); ///limitation set by volume

    pcl::PointCloud<pcl::PointXYZ>::Ptr input_cloud_xyz_ptr (new pcl::PointCloud<pcl::PointXYZ>);

    pointXYZItoXYZ(input_cloud_all_ptr, input_cloud_xyz_ptr);

    pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
    kdtree.setInputCloud(input_cloud_xyz_ptr);
    pcl::PointXYZ searchPoint;

    float highest_z = 0.0;
    float lowest_z = 6.4;

    for(int i = 0; i < input_cloud_all_ptr->width; i++)
    {
        if(input_cloud_xyz_ptr->points[i].z > highest_z)
        {
            std::vector<int> pointIdxRadiusSearch;
            std::vector<float> pointRadiusSquaredDistance;
            searchPoint = input_cloud_xyz_ptr->points[i];

            if(kdtree.radiusSearch (searchPoint, valid_fspoints_search_radius, pointIdxRadiusSearch, pointRadiusSquaredDistance) > number_bound )
            {
                highest_z = searchPoint.z;
            }
        }

        if(input_cloud_xyz_ptr->points[i].z < lowest_z)
        {
            std::vector<int> pointIdxRadiusSearch;
            std::vector<float> pointRadiusSquaredDistance;
            searchPoint = input_cloud_xyz_ptr->points[i];

            if(kdtree.radiusSearch (searchPoint, valid_fspoints_search_radius, pointIdxRadiusSearch, pointRadiusSquaredDistance) > number_bound )
            {
                lowest_z = searchPoint.z;
            }
        }

    }

    std::cout << "Found highest_z = " << highest_z << std::endl;
    std::cout << "Found lowest_z = " << lowest_z << std::endl;

    roof_height = highest_z;
    ground_height = lowest_z;

    /// NOTE: No need to split again. Consider to improve efficiency later according to real input
    float z_min_demand = ground_height; // + 0.1f;
    float z_max_demand = roof_height; // - 0.1f;

    for(int i = 0; i < input_cloud_all_ptr->width; i++)
    {


        /// Only count points in reasonable height range
        if(input_cloud_all_ptr->points[i].z < z_min_demand)
            continue;
        if(input_cloud_all_ptr->points[i].z >  z_max_demand)
            continue;

        /// Add free space and obstacle points
        if(input_cloud_all_ptr->points[i].intensity > fs_min_val)  /// free space
        {
            map_fs.ptr<unsigned char>((int)(input_cloud_all_ptr->points[i].x / Voxel_Length))[(int)(input_cloud_all_ptr->points[i].y / Voxel_Length)] += 1;
        }
        else /// obstacle
        {
            map_ob.ptr<unsigned char>((int)(input_cloud_all_ptr->points[i].x / Voxel_Length))[(int)(input_cloud_all_ptr->points[i].y / Voxel_Length)] += 1;
        }
    }

    /**
     * 2D map with processed points
     */

    /// Add useful original points data to map
    float height_max_voxels = (z_max_demand - z_min_demand) / Voxel_Length;
    int temp_intensity = 0;


    for(int i = 0; i < length; i++) /// row
    {
        std::vector<float> ob_ratio_temp_vec;
        std::vector<float> fs_ratio_temp_vec;

        for(int j = 0; j < length; j++) /// col
        {
            if(map_fs.ptr<unsigned char>(i)[j] > 0 || map_ob.ptr<unsigned char>(i)[j] > 0)
            {
                /// If any point detected, give 128 as basic value.
                /// Free space has a higher intensity while obstacle has a lower.
                /// Obstacles has 1 times reliability

                //temp_intensity = 128 - map_filted_ob.ptr<unsigned char>(i)[j] * 100 + (float)map_fs.ptr<unsigned char>(i)[j] / height_max_voxels * 127.f - (float)map_ob.ptr<unsigned char>(i)[j] / height_max_voxels * 127.f;

                /// Ratio calculate
                float ob_ratio = (float)map_ob.ptr<unsigned char>(i)[j] / height_max_voxels;
                float fs_ratio = (float)map_fs.ptr<unsigned char>(i)[j] / height_max_voxels;

                ob_ratio_temp_vec.push_back(ob_ratio);
                fs_ratio_temp_vec.push_back(fs_ratio);

                /// Intensity in map
                temp_intensity = 128 +  fs_ratio* 127.f -  ob_ratio* 127.f;

                if(temp_intensity < 1) temp_intensity = 1;
                else if(temp_intensity > 255) temp_intensity = 255;

                map.ptr<unsigned char>(i)[j] = (unsigned char) temp_intensity;
            }
        }

        map_ob_ratio.push_back(ob_ratio_temp_vec);
        map_fs_ratio.push_back(fs_ratio_temp_vec);

    }
    map_intensity = map.clone();

    /// Intensity filter
    map = map > 200;

    /// Erode and dilate
    cv::Mat map_eroded = map.clone();
    cv::Mat element = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5,5));
    cv::erode(map, map_eroded, element); //Opening operation
    cv::dilate(map_eroded, map_eroded, element);


    /// Flood fill to keep only one connected region
    cv::floodFill(map_eroded, cv::Point(length/2-1, length/2-1), cv::Scalar(100), 0, cv::Scalar(10), cv::Scalar(10), 8); /// Square area
    map_eroded = map_eroded == 100;

    /// Remove small black pieces inside. Might be obstacles like pedestrians
    cv::Mat element2 = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5,5));
    cv::dilate(map_eroded, map_eroded, element); /// Closing operation
    cv::erode(map_eroded, map_eroded, element);

    cv::Mat voronoi_map = map_eroded.clone();
    GVG gvg;
    gvg.voronoi(voronoi_map);

    cv::Mat generalized_voronoi_map(length, length, CV_8UC1, cv::Scalar(0));

    for(int i = 0; i < length; i++) /// row
    {
        for(int j = 0; j < length; j++) /// col
        {
            /// Only keep gray contour part
            if(voronoi_map.ptr<unsigned char>(i)[j] == 150) {
                generalized_voronoi_map.ptr<unsigned char>(i)[j] = 255;
            }
            else
                generalized_voronoi_map.ptr<unsigned char>(i)[j] = 0;  /// Black: unknown
        }
    }

    cv::Mat tangent_map = gvg.tangent_vector(generalized_voronoi_map, 2, 10);
    cv::Mat restructured_map = cv::Mat::zeros(length, length, CV_8UC1);
    std::vector<std::vector<cv::Point>> clusters;
    gvg.cluster_filter(generalized_voronoi_map, tangent_map, restructured_map, clusters, 3, 4, 0.3);

    /// Just to show the clusters
    cv::Mat cluster_map(length, length, CV_8UC1, cv::Scalar(0));
    for(int i = 0; i<clusters.size(); i++)
    {
        //int color = rand() % 200 + 55;
        int color = 50*(i+1);
        for(int j=0; j<clusters[i].size();j++)
        {
            //if(i == 0)
            cluster_map.ptr<unsigned char>(clusters[i][j].y)[clusters[i][j].x] = color;
        }
    }
    cv::Mat final_map = gvg.restructure(map_eroded, clusters, 4, 0.1);

//    /// For testing the effect of finding polygon contour to pre-process
//    vector<vector<Point>> contours;
//    vector<Vec4i> hierarcy;
//    findContours(map_eroded, contours, hierarcy, 0, CV_CHAIN_APPROX_NONE);
//
//    vector<vector<Point>> contours_poly(contours.size());
//
//    for (int i = 0; i<contours.size(); i++)
//    {
//        approxPolyDP(Mat(contours[i]), contours_poly[i], 15, true);
//        drawContours(map_eroded, contours_poly, i, Scalar(150), 2, 8);
//    }



    //gvg.thinning(generalized_voronoi_map);

    cv::Mat tangent_show_img(length, length, CV_8UC1, cv::Scalar(0));
    for(int i=0; i < tangent_show_img.rows; i++)
    {
        for(int j=0; j<tangent_show_img.cols; j++)
        {
            tangent_show_img.ptr<unsigned char>(i)[j] = (unsigned char)(tangent_map.ptr<float>(i)[j] * 255);
        }
    }

    /// Save and show
    cv::imwrite("/home/clarence/catkin_ws/src/local_map/data/Original/map.jpg", map);
    cv::imwrite("/home/clarence/catkin_ws/src/local_map/data/Original/map_intensity.jpg", map_intensity);
    cv::imwrite("/home/clarence/catkin_ws/src/local_map/data/Original/map_eroded.jpg", map_eroded);
    cv::imwrite("/home/clarence/catkin_ws/src/local_map/data/Original/generalized_voronoi_map.jpg", generalized_voronoi_map);
    cv::imwrite("/home/clarence/catkin_ws/src/local_map/data/Original/voronoi_map.jpg", voronoi_map);
    cv::imwrite("/home/clarence/catkin_ws/src/local_map/data/Original/restructured_map.jpg", restructured_map);
    cv::imwrite("/home/clarence/catkin_ws/src/local_map/data/Original/cluster_map.jpg", cluster_map);
    cv::imwrite("/home/clarence/catkin_ws/src/local_map/data/Original/final_map.jpg", final_map);
    cv::imwrite("/home/clarence/catkin_ws/src/local_map/data/Original/tangent_show_img.jpg", tangent_show_img);

    cv::waitKey(100);
}


void CloudProcess::vertical_clusters_process()
{
    //vertical_clusters
    output_cloud_ptr = output_cloud.makeShared();

    int length = (int) (Area_Length / Voxel_Length);
    map_filted_ob = cv::Mat::zeros(length, length, CV_8UC1);

    if(vertical_clusters.size() == 0)
    {
        vertical_structure_type = 4; /// found no walls
        return;
    } else
    {
        /// Find the lowest and the highest point
        upper_bound = 0.f;
        lower_bound = Area_Length;
        int point_seq;

        for(int i = 0; i < vertical_clusters.size(); i++)
        {
            for(int j = 0; j < vertical_clusters[i].indices.size(); j++)
            {
                point_seq = vertical_clusters[i].indices[j];

                float z = output_cloud_ptr->points[point_seq].z;
                if(z > upper_bound) upper_bound = z;
                if(z < lower_bound) lower_bound = z;

                map_filted_ob.ptr<unsigned char>((int)(output_cloud_ptr->points[point_seq].x / Voxel_Length))[(int)(output_cloud_ptr->points[point_seq].y / Voxel_Length)] += 1;
            }
        }


        /// Set vertical_structure_type
        ///NOTE: Sometimes the camera view angle is too small to reach the Area_Length height in a cubic local map
        ///      In that case, change a better camera is suggested, but you can also narrow the bound range below. Eg: 0.2-0.8
        if(upper_bound < 0.9 * Area_Length)
        {
            if(lower_bound > 0.1 * Area_Length)
                vertical_structure_type = 0;  /// found walls with upper and lower bounds (eg: ordinary hall)
            else
                vertical_structure_type = 1;  /// found walls with only upper bound

        } else if(lower_bound > 0.1 * Area_Length)
        {
            vertical_structure_type = 2; /// found walls with only lower bound
        } else
        {
            vertical_structure_type = 3; /// found walls without bounds
        }
        return;
    }

}

void CloudProcess::horizontal_clusters_process()  /// Note: Can only be excuted after vertical_clusters_process()
{
    //horizontal_clusters
    output_cloud_ptr = output_cloud.makeShared();

    if(horizontal_clusters.size() == 0) /// No real horizontal plain
    {
        switch(vertical_structure_type)
        {
            case 0: /// Found walls with upper and lower bounds
            {
                roof_height = upper_bound;
                ground_height = lower_bound;
                horizontal_structure_type = 8; /// with artificial roof and artificial ground
                break;
            }

            case 1: /// found walls with only upper bound
            {
                roof_height = upper_bound;
                ground_height = 0.f;
                horizontal_structure_type = 4; /// with artificial roof, but without ground
                break;
            }

            case 2: /// found walls with only lower bound
            {
                roof_height = Area_Length;
                ground_height = lower_bound;
                horizontal_structure_type = 5; /// with artificial ground, but without roof
                break;
            }

            case 3: ///found walls without bounds
            {
                roof_height = Area_Length;
                ground_height = 0.f;
                horizontal_structure_type = 3; /// without roof or ground
                break;
            }
            default: /// 4: found no walls
            {
                roof_height = Area_Length;
                ground_height = 0.f;
                horizontal_structure_type = 3; /// without roof or ground
                break;
            }
        }

    }
    else /// Found real horizontal plain
    {

        /// Calculate average height for plane in each horizontal clusters
        double horizontal_plain_heights[horizontal_clusters.size()]; /// Store average height
        int points_number_plain[horizontal_clusters.size()]; /// Store points number in each cluster

        for(int i = 0; i < horizontal_clusters.size(); i++)
        {
            points_number_plain[i] = horizontal_clusters[i].indices.size();
            double z_acc = 0.0;
            for(int j = 0; j < horizontal_clusters[i].indices.size(); j++)
            {
                z_acc += output_cloud_ptr->points[horizontal_clusters[i].indices[j]].z;
            }
            horizontal_plain_heights[i] = z_acc / horizontal_clusters[i].indices.size();
        }

        /// Find or create a roof and a ground plane
        float search_roof_height, search_ground_height;

        switch(vertical_structure_type)
        {
            /// Found walls with upper and lower bounds
            case 0:
            {
                ///* In this case, treat mean plains' height as roof or ground height
                /// if those plains can be found in a reasonable range defined by vertical bounds *
                /// Otherwise we create artificial roof or ground *
                search_roof_height = upper_bound - 0.1 * Area_Length;
                search_ground_height = lower_bound + 0.1 * Area_Length;

                bool found_roof = false;
                bool found_ground = false;

                /// Variables for weighted mean
                double roof_height_acc = 0.0;
                int roof_number_acc = 0;
                double ground_height_acc = 0.0;
                int ground_number_acc = 0;

                for(int i = 0; i < horizontal_clusters.size(); i++)
                {
                    if(horizontal_plain_heights[i] > search_roof_height)
                    {
                        found_roof = true;
                        roof_height_acc += horizontal_plain_heights[i] * points_number_plain[i];
                        roof_number_acc += points_number_plain[i];
                    }
                    else if(horizontal_plain_heights[i] < search_ground_height)
                    {
                        found_ground = true;
                        ground_height_acc += horizontal_plain_heights[i] * points_number_plain[i];
                        ground_number_acc += points_number_plain[i];
                    }
                }

                if(found_roof && found_ground)
                {
                    roof_height = roof_height_acc / roof_number_acc;
                    ground_height = ground_height_acc / ground_number_acc;
                    horizontal_structure_type = 0; /// with roof and ground
                }
                else if(found_roof)
                {
                    roof_height = roof_height_acc / roof_number_acc;
                    ground_height = lower_bound;
                    horizontal_structure_type = 6; /// with roof and artificial ground
                }
                else if(found_ground)
                {
                    roof_height = upper_bound;
                    ground_height = ground_height_acc / ground_number_acc;
                    horizontal_structure_type = 7; /// with ground and artificial roof
                }
                else
                {
                    roof_height = upper_bound;
                    ground_height = lower_bound;
                    horizontal_structure_type = 8; /// with artificial roof and artificial ground
                }

                break;
            }

            /// found walls with only upper bound
            case 1:
            {
                ///* In this case, treat mean plains' height as roof height
                /// if those plains can be found in a reasonable range defined by vertical bounds *
                /// Otherwise we create artificial roof*

                search_roof_height = upper_bound - 0.1 * Area_Length;

                bool found_roof = false;

                /// Variables for weighted mean
                double roof_height_acc = 0.0;
                int roof_number_acc = 0;


                for(int i = 0; i < horizontal_clusters.size(); i++)
                {
                    if(horizontal_plain_heights[i] > search_roof_height)
                    {
                        found_roof = true;
                        roof_height_acc += horizontal_plain_heights[i] * points_number_plain[i];
                        roof_number_acc += points_number_plain[i];
                    }
                }

                if(found_roof)
                {
                    roof_height = roof_height_acc / roof_number_acc;
                    ground_height = 0.0;
                    horizontal_structure_type = 1; /// with only roof
                }

                else
                {
                    roof_height = upper_bound;
                    ground_height = 0.0;
                    horizontal_structure_type = 4; /// with artificial roof, but without ground
                }
                break;
            }

            /// found walls with only lower bound
            case 2:
            {
                ///* In this case, treat mean plains' height as ground height
                /// if those plains can be found in a reasonable range defined by vertical bounds *
                /// Otherwise we create artificial ground *


                search_ground_height = lower_bound + 0.1 * Area_Length;

                bool found_ground = false;

                /// Variables for weighted mean
                double ground_height_acc = 0.0;
                int ground_number_acc = 0;

                for(int i = 0; i < horizontal_clusters.size(); i++)
                {
                    if(horizontal_plain_heights[i] < search_ground_height)
                    {
                        found_ground = true;
                        ground_height_acc += horizontal_plain_heights[i] * points_number_plain[i];
                        ground_number_acc += points_number_plain[i];
                    }
                }

                if(found_ground)
                {
                    roof_height = Area_Length;
                    ground_height = ground_height_acc / ground_number_acc;
                    horizontal_structure_type = 2; /// with only ground
                }
                else
                {
                    roof_height = Area_Length;
                    ground_height = lower_bound;
                    horizontal_structure_type = 5; /// with artificial ground, but without roof
                }
                break;
            }

            ///found walls without bounds
            case 3:
            {
                ///* In this case , we just consider it as a place with high walls and the depth camera
                /// can not see the ground or the roof *
                ///* What if the roof is not like a "---___" with height change??
                /// Oh. We just consider the higher part then. The drone won't use this map to avoid collision

                roof_height = Area_Length;
                ground_height = 0.0;
                horizontal_structure_type = 3; /// without roof or ground

                break;
            }

            /// 4: found no walls
            default:
            {
                ///* This means that the drone is in a large room where the depth camera can not reach the walls
                ///* but it can reach the roof or ground or both
                ///* In this case , we just treat the weighted average plane height over the collision height of
                ///  the drone as roof. The ground is similarly defined. *
                ///* What if there are walls but they none of them are detected?
                ///  Then you really need to consider change you camera or your test environment. :) *

                search_roof_height = Area_Length / 2 + robot_upper_height;
                search_ground_height =  Area_Length / 2 + robot_lower_height;

                bool found_roof = false;
                bool found_ground = false;

                /// Variables for weighted mean
                double roof_height_acc = 0.0;
                int roof_number_acc = 0;
                double ground_height_acc = 0.0;
                int ground_number_acc = 0;

                for(int i = 0; i < horizontal_clusters.size(); i++)
                {
                    if(horizontal_plain_heights[i] > search_roof_height)
                    {
                        found_roof = true;
                        roof_height_acc += horizontal_plain_heights[i] * points_number_plain[i];
                        roof_number_acc += points_number_plain[i];
                    }
                    else if(horizontal_plain_heights[i] < search_ground_height)
                    {
                        found_ground = true;
                        ground_height_acc += horizontal_plain_heights[i] * points_number_plain[i];
                        ground_number_acc += points_number_plain[i];
                    }
                }

                if(found_roof && found_ground)
                {
                    roof_height = roof_height_acc / roof_number_acc;
                    ground_height = ground_height_acc / ground_number_acc;
                    horizontal_structure_type = 0; /// with roof and ground
                }
                else if(found_roof)
                {
                    roof_height = roof_height_acc / roof_number_acc;
                    ground_height = 0.0;
                    horizontal_structure_type = 1; /// with only roof
                }
                else if(found_ground)
                {
                    roof_height = Area_Length;
                    ground_height = ground_height_acc / ground_number_acc;
                    horizontal_structure_type = 2; /// with only ground
                }
                else
                {
                    roof_height = Area_Length;
                    ground_height = 0.0;
                    horizontal_structure_type = 3; /// without roof or ground
                }


                break;
            }
        }
    }

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


void CloudProcess::pointXYZItoXYZ(pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_xyzi, pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_xyz)
{
    cloud_xyz->clear();
    for(int i = 0; i < cloud_xyzi->width; i++)
    {
        pcl::PointXYZ pclp;
        pclp.x = cloud_xyzi->points[i].x;
        pclp.y = cloud_xyzi->points[i].y;
        pclp.z = cloud_xyzi->points[i].z;
        cloud_xyz->points.push_back(pclp);
    }
}



