//
// Created by clarence on 18-5-18.
//

#include "../include/cloud_filter.h"
#include "ros/ros.h"
#include "sensor_msgs/image_encodings.h"
#include <stdio.h>
#include <sys/time.h>

using namespace std;

// global mat for cloud
pcl::PointCloud<pcl::PointXYZ> input_cloud;
pcl::PointCloud<pcl::PointXYZ> output_cloud;
pcl::PointCloud<pcl::PointXYZI> input_cloud_all;
pcl::PointCloud<pcl::PointXYZI> output_cloud_all;

CloudProcess cloud_process;

sensor_msgs::PointCloud2 cloud2;
ros::Publisher cloud_pub;

ros::Time last_time;

/**
 * Define work mode
 */
enum WorkMode{
    TOPIC_PROCESS = 0,  //Process clouds given by ros topic
    TOPIC_SAVE,        //Save a cloud given by ros topic
    CLOUD_PROCESS   //Process a cloud loaded from a .pcd file
};

//keyboard
char key_pressed = 'a';
WorkMode mode = CLOUD_PROCESS;

/**
 * Handle ros topic cloud only contain obstacle
 * @param msg
 */
void cloudCallback(const sensor_msgs::PointCloud2ConstPtr &msg)
{
    if(mode != CLOUD_PROCESS)
    {
        /*Update data*/
        cout << "new cloud " << endl;
        pcl::fromROSMsg(*msg, input_cloud);

        /*Filter process*/
        cloud_process.input_cloud = input_cloud;
        cloud_process.filter_process();

        /*Publish*/
        output_cloud = cloud_process.output_cloud;
        pcl::toROSMsg(output_cloud, cloud2);
        cloud2.header.frame_id = "world";
        cloud_pub.publish(cloud2);
        cout << "cloud publish to rviz! \n" << endl;
    }

    /*Save cloud if press's'*/
    if(mode == TOPIC_SAVE && key_pressed == 's')
    {
        struct timeval tv;
        if(gettimeofday(&tv,NULL) == 0)
        {
            string name = "/home/clarence/catkin_ws/src/local_map/data/cloud" + to_string(tv.tv_sec) + ".pcd";
            pcl::io::savePCDFileASCII (name, input_cloud);
            key_pressed = 'a';
        }
        else cout<<"can not get system time"<<endl;
    }
}

/**
 * Handle ros topic cloud only contain obstacle and free space
 * @param msg
 */
void cloudAllCallback(const sensor_msgs::PointCloud2ConstPtr &msg)
{

    if(mode != CLOUD_PROCESS)
    {
        /*Update data*/
        cout << "new cloud " << endl;
        pcl::fromROSMsg(*msg, input_cloud_all);
    }

    /*Save cloud if press's'*/
    if(mode == TOPIC_SAVE && key_pressed == 's')
    {
        struct timeval tv;
        if(gettimeofday(&tv,NULL) == 0)
        {
            string name = "/home/clarence/catkin_ws/src/local_map/data/cloud" + to_string(tv.tv_sec) + ".pcd";
            pcl::io::savePCDFileASCII (name, input_cloud_all);
            key_pressed = 'a';
        }
        else cout<<"can not get system time"<<endl;
    }
}

/**
 * Handle a .pcd cloud only contain obstacle
 * @param path
 */
void read_cloud_process_XYZ(string path)
{
    pcl::PCDReader reader;
    reader.read (path, input_cloud);

    last_time = ros::Time::now();
    cloud_process.input_cloud = input_cloud;
    cloud_process.filter_process_and_show(); //must be used with cloud_process.cloud_viewer->spinOnce(100)
    cout<<"Time = "<<ros::Time::now().toSec()-last_time.toSec()<<endl;
}

/**
 * Handle a .pcd cloud only contain obstacle and free space
 * @param path
 */
void read_cloud_process_XYZI(string path)
{
    pcl::PCDReader reader;
    reader.read (path, input_cloud_all);


    last_time = ros::Time::now();

    cloud_process.input_cloud_all = input_cloud_all;
    cloud_process.process_cloud_all();

    cout<<"Time = "<<ros::Time::now().toSec()-last_time.toSec()<<endl;
}

/** Ros node main
 * @function main
 * */
int main( int argc, char **argv )
{
    ros::init(argc, argv, "cloud_filter");
    ros::NodeHandle n;

    ros::Subscriber cloud_sub_obs = n.subscribe("/ring_buffer/cloud2", 2, cloudCallback);
    ros::Subscriber cloud_sub_all = n.subscribe("/ring_buffer/cloud_all", 2, cloudAllCallback);

    cloud_pub = n.advertise<sensor_msgs::PointCloud2>("/local_map/filtered_cloud", 2);

    if(mode == CLOUD_PROCESS)
    {
        //read_cloud_process_XYZ("/home/clarence/catkin_ws/src/local_map/data/2.pcd");
        read_cloud_process_XYZI("/home/clarence/catkin_ws/src/local_map/data/3m_all/all-3m-T-Intersection-2.pcd"); //all-3m-hallway-2.pcd"); //
        //read_cloud_process_XYZI("/home/clarence/catkin_ws/src/local_map/data/3m_all/all-3m-hallway-1.pcd"); //all-3m-T-Intersection-2.pcd"); //all-3m-hallway-2.pcd");
    }

    ros::Rate loop_rate(20);

    /*Main loop*/
    while(ros::ok())
    {
        if(mode == TOPIC_SAVE)
        {
            key_pressed = getchar();
            cout<<"Pressed "<< key_pressed <<endl;
            if(key_pressed == 'q')
                break;
        }
//        else if(mode == CLOUD_PROCESS && !cloud_process.cloud_viewer->wasStopped ())
//        {
//            cloud_process.cloud_viewer->spinOnce(100); //Display update, Important
//        }

        ros::spinOnce();
        loop_rate.sleep();
    }

    return 0;
}