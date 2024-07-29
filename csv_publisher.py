#!/usr/bin/env python3
import rclpy
import pandas 
from rclpy.node import Node
from example_interfaces.msg import Float64MultiArray
 
class CSVNode(Node): # MODIFY NAME
    def __init__(self):
        super().__init__("csv_node") # MODIFY NAME

        self.publisher = self.create_publisher(Float64MultiArray, "number", 10)
        names = ["time", "lat", "lon","stiff", "r2", "leg"]
        self.data = pandas.read_csv("/Users/natalie/Desktop/heatmap_csvs/2024-6-18_Mh24_Loc1_Path1_10_12_am_Trial3.csv",
                        sep = ",", header=0,names=names, usecols=["lat","lon","stiff"])
        
        self.row_index = 0
        self.timer = self.create_timer(3, self.publish_row)
        self.get_logger().info("CSV publisher has been started.")
    
    def publish_row(self):
        if self.row_index < len(self.data):
            row_data = self.data.iloc[self.row_index]
            msg = Float64MultiArray()
            msg.data = [float(row_data['lon'], float(row_data['lat']), float(row_data['stiff']))]

            self.publisher.publish(msg)
            self.row_index += 1
            self.get_logger().info(f"Published row{self.row_index}: {msg.data}")
        
        else:
            self.get_logger.info("All data has been published.")
            self.timer.cancel()

def main(args=None):
    rclpy.init(args=args)
    node = CSVNode() 
    rclpy.spin(node)
    rclpy.shutdown()
 
 
if __name__ == "__main__":
    main()