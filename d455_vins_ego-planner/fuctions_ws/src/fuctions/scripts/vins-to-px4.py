#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import tf
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped, Point, Quaternion
import numpy as np

class VinsToMavros:
    def __init__(self):
        rospy.init_node('vins_to_px4_bridge', anonymous=False)

        # 发布频率（PX4 EKF2 推荐 30-50Hz）
        self.publish_rate = rospy.get_param('~publish_rate', 50)
        self.last_pub_time = rospy.Time(0)
        self.min_interval = rospy.Duration(1.0 / self.publish_rate)

        # 坐标系转换参数
        self.swap_xy = rospy.get_param('~swap_xy', True)
        self.swap_xz = rospy.get_param('~swap_xz', False)
        self.negate_x = rospy.get_param('~negate_x', False)
        self.negate_y = rospy.get_param('~negate_y', True)
        self.negate_z = rospy.get_param('~negate_z', False)
        
        # 四元数旋转（不使用 scipy）
        self.apply_yaw_rotation = rospy.get_param('~apply_yaw_rotation', True)
        self.yaw_rotation_deg = rospy.get_param('~yaw_rotation_deg', 90.0)
        
        # 发布到 MAVROS
        self.pub = rospy.Publisher(
            '/mavros/vision_pose/pose',
            PoseStamped,
            queue_size=10
        )

        # TF 广播器
        self.tf_broadcaster = tf.TransformBroadcaster()

        # 订阅 VINS 里程计
        rospy.Subscriber(
            '/vins_estimator/odometry',
            Odometry,
            self.odom_cb,
            queue_size=10
        )

        rospy.loginfo("VINS -> PX4 桥接节点已启动 (频率: %d Hz)", self.publish_rate)
        rospy.loginfo("坐标系转换: swap_xy=%s, negate_x=%s, negate_y=%s, negate_z=%s", 
                      self.swap_xy, self.negate_x, self.negate_y, self.negate_z)

    def transform_position(self, pos):
        x, y, z = pos.x, pos.y, pos.z
        
        if self.swap_xy:
            x, y = y, x
        if self.swap_xz:
            x, z = z, x
        if self.negate_x:
            x = -x
        if self.negate_y:
            y = -y
        if self.negate_z:
            z = -z
        
        return Point(x, y, z)
    
    def quat_multiply(self, q1, q2):
        """四元数乘法: q1 * q2"""
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        return [
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2
        ]

    def transform_orientation(self, quat):
        """使用纯 Python 实现姿态旋转（不依赖 scipy）"""
        if not self.apply_yaw_rotation:
            return Quaternion(quat.x, quat.y, quat.z, quat.w)
        
        # 绕 Z 轴旋转指定角度
        theta = np.radians(self.yaw_rotation_deg)
        half_theta = theta / 2.0
        rot_q = [np.cos(half_theta), 0, 0, np.sin(half_theta)]  # w, x, y, z
        
        orig_q = [quat.w, quat.x, quat.y, quat.z]  # w, x, y, z
        
        # 应用旋转: q_new = rot_q * q_orig
        q_new = self.quat_multiply(rot_q, orig_q)
        
        return Quaternion(q_new[1], q_new[2], q_new[3], q_new[0])  # x, y, z, w

    def odom_cb(self, msg):
        now = rospy.Time.now()
        if (now - self.last_pub_time) < self.min_interval:
            return
        self.last_pub_time = now

        stamp = msg.header.stamp
        if stamp == rospy.Time(0):
            stamp = now

        pose_msg = PoseStamped()
        pose_msg.header.stamp = stamp
        pose_msg.header.frame_id = "map"

        pose_msg.pose.position = self.transform_position(msg.pose.pose.position)
        pose_msg.pose.orientation = self.transform_orientation(msg.pose.pose.orientation)

        self.pub.publish(pose_msg)

        # 广播 TF
        pos = pose_msg.pose.position
        ori = pose_msg.pose.orientation
        self.tf_broadcaster.sendTransform(
            (pos.x, pos.y, pos.z),
            (ori.x, ori.y, ori.z, ori.w),
            stamp,
            "base_link",
            "map"
        )
        
        if rospy.get_param('~debug', False):
            rospy.loginfo("Pub: x=%.3f, y=%.3f, z=%.3f, yaw=%.1f°", 
                         pos.x, pos.y, pos.z, 
                         np.degrees(tf.transformations.euler_from_quaternion([ori.x, ori.y, ori.z, ori.w])[2]))

if __name__ == '__main__':
    try:
        VinsToMavros()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
