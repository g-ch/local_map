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

CloudProcess cloud_process;

sensor_msgs::PointCloud2 cloud2;
ros::Publisher cloud_pub;

enum WorkMode{
    TOPIC_PROCESS = 0,
    TOPIC_SAVE,
    CLOUD_PROCESS
};

//keyboard
char key_pressed = 'a';
WorkMode mode = CLOUD_PROCESS;

void cloudCallback(const sensor_msgs::PointCloud2ConstPtr &msg)
{
    if(mode != CLOUD_PROCESS)
    {
        /*Update data*/
        cout << "new cloud " << endl;
        pcl::fromROSMsg(*msg, input_cloud);

        /*Filter process*/
        cloud_process.input_cloud = input_cloud;
        cloud_process.process();

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

void read_cloud_process(string path)
{
    pcl::PCDReader reader;
    reader.read (path, input_cloud);

    cloud_process.input_cloud = input_cloud;
    cloud_process.process();

    cloud_process.viewPointXYZ(cloud_process.output_cloud);
}

/* @function main */
int main( int argc, char **argv )
{
    ros::init(argc, argv, "cloud_filter");
    ros::NodeHandle n;

    ros::Subscriber cloud_sub = n.subscribe("/ring_buffer/cloud2", 2, cloudCallback);
    cloud_pub = n.advertise<sensor_msgs::PointCloud2>("/local_map/filtered_cloud", 2);

    if(mode == CLOUD_PROCESS)
    {
        read_cloud_process("/home/clarence/catkin_ws/src/local_map/data/cloud1526614377.pcd");
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
        else if(mode == CLOUD_PROCESS && !cloud_process.cloud_viewer->wasStopped ())
        {
            cloud_process.cloud_viewer->spinOnce(100); //Display update, Important
        }

        ros::spinOnce();
        loop_rate.sleep();
    }

    return 0;
}