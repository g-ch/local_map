//
// Created by clarence on 18-5-17.
//

#include "../include/cloud_filter.h"

using namespace std;

// global mat for cloud
pcl::PointCloud<pcl::PointXYZ> input_cloud;
pcl::PointCloud<pcl::PointXYZ> output_cloud;
sensor_msgs::PointCloud2 cloud2;
ros::Publisher cloud_pub;


void cloudCallback(const sensor_msgs::PointCloud2ConstPtr &msg)
{
    /*Update data*/
    cout << "new cloud " << endl;
    input_cloud.points.clear();
    pcl::fromROSMsg(*msg, input_cloud);

    /*Filter process*/


    /*Publish*/
    pcl::toROSMsg(output_cloud, cloud2);
    cloud2.header.frame_id = "world";
    cloud_pub.publish(cloud2);
    cout << "cloud publish to rviz! \n" << endl;
}



/* @function main */
int main( int argc, char **argv )
{
    ros::init(argc, argv, "cloud_filter");
    ros::NodeHandle n;

    ros::Subscriber cloud_sub = n.subscribe("/ring_buffer/cloud2", 1, cloudCallback);
    cloud_pub = n.advertise<sensor_msgs::PointCloud2>("/local_map/filtered_cloud", 2);


    ros::Rate loop_rate(10);

    /*Main loop*/
    while(ros::ok)
    {
        ros::spinOnce();
        loop_rate.sleep();
    }

    return 0;
}
