#!/usr/bin/env python3
import rclpy
import pandas 
from rclpy.node import Node
from example_interfaces.msg import Int64
 
class CSVNode(Node): # MODIFY NAME
    def __init__(self):
        super().__init__("csv_node") # MODIFY NAME

        self.publisher = self.create_publisher(Int64, "number", 10)
        self.timer = self.create_timer(3, self.publish_row)
        self.get_logger().info("CSV publisher has been started.")
    
    def publish_row(self):
        row = Int64()
        names = ["time", "lat", "lon","stiff", "r2", "leg"]
        data = pandas.read_csv("/Users/natalie/Desktop/heatmap_csvs/2024-6-18_Mh24_Loc1_Path1_10_12_am_Trial3.csv",
                        sep = ",", header=0,names=names, usecols=["lat","lon","stiff"])
        
        

        self.publisher.publish(number)

def main(args=None):
    rclpy.init(args=args)
    node = MyCustomNode() 
    rclpy.spin(node)
    rclpy.shutdown()
 
 
if __name__ == "__main__":
    main()