"""ROS 2 adapter for the QP governor code.
"""
import json
from dataclasses import fields
import time
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from std_msgs.msg import String
from .qp import Parameters, govern


class TorqueSupervisorQP(Node):
    def __init__(self):
        super().__init__('torque_supervisor_qp')
        defaults = Parameters()
        for f in fields(defaults):
            self.declare_parameter(f.name,getattr(defaults,f.name))
        self.p = Parameters(**{f.name:self.get_parameter(f.name).value for f in fields(defaults)})
        for name,value in [('input_topic','/cmd_vel_mpc'),('output_topic','/governor/cmd_vel'),
                           ('odom_topic','/tyr/odom'),('debug_topic','/governor/debug'),
                           ('timeout_s',0.5)]:
            self.declare_parameter(name,value)
        topic = lambda n: str(self.get_parameter(n).value)
        if topic('input_topic') == topic('output_topic'):
            raise ValueError('input and output topics must differ')
        self.timeout = float(self.get_parameter('timeout_s').value)
        if self.timeout <= 0:
            raise ValueError('timeout_s must be positive')
        self.nominal = self.state = None
        self.t_cmd = self.t_odom = -float('inf')
        self.previous = (0.,0.)
        self.create_subscription(Twist,topic('input_topic'),self.on_cmd,10)
        self.create_subscription(Odometry,topic('odom_topic'),self.on_odom,10)
        self.pub = self.create_publisher(Twist,topic('output_topic'),10)
        self.debug = self.create_publisher(String,topic('debug_topic'),10)
        self.create_timer(self.p.sample_time,self.step)

    def on_cmd(self,msg):
        self.nominal = (float(msg.linear.x),float(msg.angular.z))
        self.t_cmd = time.monotonic()

    def on_odom(self,msg):
        self.state = (float(msg.twist.twist.linear.x),float(msg.twist.twist.angular.z))
        self.t_odom = time.monotonic()

    def step(self):
        now = time.monotonic()
        info = {'implementation':'qp-command-governor'}
        if self.nominal is None or self.state is None or max(now-self.t_cmd,now-self.t_odom)>self.timeout:
            command,info['status'] = (0.,0.),'stale_input_stop_request'
        else:
            try:
                result = govern(self.nominal,self.state,self.previous,self.p)
                command = result.command or (0.,0.)
                info.update(status=result.status,torques=result.torques,max_residual=result.max_residual)
            except ValueError as exc:
                command = (0.,0.)
                info.update(status='invalid_input_stop_request',error=str(exc))
        self.previous = command
        msg = Twist();msg.linear.x=float(command[0]);msg.angular.z=float(command[1])
        self.pub.publish(msg)
        dbg = String();dbg.data=json.dumps(info,allow_nan=False);self.debug.publish(dbg)


def main():
    rclpy.init()
    node = TorqueSupervisorQP()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node();rclpy.shutdown()


if __name__ == '__main__':
    main()
