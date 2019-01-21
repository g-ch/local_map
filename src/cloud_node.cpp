//
// Created by clarence on 18-5-18.
//

#include "../include/cloud_filter.h"
#include "ros/ros.h"
#include "sensor_msgs/image_encodings.h"
#include <stdio.h>
#include <sys/time.h>
#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/time_synchronizer.h>

using namespace std;
using namespace message_filters;

CloudProcess cloud_process;

ros::Publisher cloud_pub;

ros::Time last_time;

//pcl::PointCloud<pcl::PointXYZI> cloud_all;

bool cloud_updated = false;
bool initialized = false;

/**
 * Handle a .pcd cloud only contain obstacle and free space
 * @param path
 */
void read_cloud_process_XYZI(string path)
{
    pcl::PointCloud<pcl::PointXYZI> input_cloud_all;
    pcl::PCDReader reader;
    reader.read (path, input_cloud_all);


    last_time = ros::Time::now();

    cloud_process.input_cloud_all = input_cloud_all;
    cloud_process.process_cloud_all();

    cout<<"Time = "<<ros::Time::now().toSec()-last_time.toSec()<<endl;
}

void cloudCallback(const sensor_msgs::PointCloud2ConstPtr& cloud_ob, const sensor_msgs::PointCloud2ConstPtr& cloud_fs)
{
    if(!cloud_updated)
    {
        pcl::PointCloud<pcl::PointXYZ> cloud_obstacle;
        pcl::PointCloud<pcl::PointXYZ> cloud_free_space;

        pcl::fromROSMsg(*cloud_ob, cloud_obstacle);
        pcl::fromROSMsg(*cloud_fs, cloud_free_space);

        cloud_process.input_cloud_all.clear();

        for(int i = 0; i < cloud_obstacle.width; i++)
        {
            pcl::PointXYZI p1;
            p1.x = cloud_obstacle.points[i].x;
            p1.y = cloud_obstacle.points[i].y;
            p1.z = cloud_obstacle.points[i].z;
            p1.intensity = 0.f;
            cloud_process.input_cloud_all.push_back(p1);
        }

        for(int j = 0; j < cloud_free_space.width; j++)
        {
            pcl::PointXYZI p2;
            p2.x = cloud_free_space.points[j].x;
            p2.y = cloud_free_space.points[j].y;
            p2.z = cloud_free_space.points[j].z;
            p2.intensity = 1.f;
            cloud_process.input_cloud_all.push_back(p2);
        }

        cout<<"cloud_process.input_cloud_all: "<<cloud_process.input_cloud_all.width<<endl;

        cloud_updated = true;
        initialized = true;
        cout << "Updated" <<endl;
    }
}

/** Ros node main
 * @function main
 * */
int main( int argc, char **argv )
{
    ros::init(argc, argv, "cloud_filter");
    ros::NodeHandle n;

    read_cloud_process_XYZI("/home/clarence/catkin_ws/src/local_map/data/3m_all/all-3m-T-Intersection-3.pcd");

    message_filters::Subscriber<sensor_msgs::PointCloud2> cloud_sub_ob(n, "/ring_buffer/cloud_ob", 1);
    message_filters::Subscriber<sensor_msgs::PointCloud2> cloud_sub_fs(n, "/ring_buffer/cloud_fs", 1);
    typedef sync_policies::ApproximateTime<sensor_msgs::PointCloud2, sensor_msgs::PointCloud2> MySyncPolicy;
    Synchronizer<MySyncPolicy> sync(MySyncPolicy(10), cloud_sub_ob, cloud_sub_fs);
    sync.registerCallback(boost::bind(&cloudCallback, _1, _2));

    ros::Rate loop_rate(10);

    while(ros::ok())
    {
        if(cloud_updated && initialized)
        {
            last_time = ros::Time::now();

            /*** Calculate direction and publish*/
            //cloud_process.input_cloud_all = cloud_all;
            cloud_process.process_cloud_all();

            cout<<"Time = "<<ros::Time::now().toSec()-last_time.toSec()<<endl;
            cloud_updated = false;
        }

        ros::spinOnce();
        loop_rate.sleep();
    }


    return 0;
}