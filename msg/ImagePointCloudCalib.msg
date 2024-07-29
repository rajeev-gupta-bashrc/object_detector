# ImagePointCloudCalib.msg

float32[] points
uint32 num_point_features
sensor_msgs/Image[] image_all

float32[] projection_all
uint32 projection_rows 
uint32 projection_cols 

float32[] transformation_all
uint32 transformation_rows 
uint32 transformation_cols 

float32[] rectification_all
uint32 rectification_rows 
uint32 rectification_cols 

uint32 num_cameras