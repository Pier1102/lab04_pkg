import math
import numpy as np
import matplotlib.pyplot as plt
from std_msgs.msg import String
from nav_msgs.msg import Odometry
from rosbag2_reader_py import Rosbag2Reader 
from tf_transformations import euler_from_quaternion
from rclpy.time import Time
from scipy.interpolate import interp1d
from landmark_msgs.msg import LandmarkArray 

bag_path = "/home/miky/ros2_ws/src/lab04_pkg/lab04_pkg/rosbag2_lab04pier"
reader = Rosbag2Reader(bag_path)

odom_topic= "/odom"
velocity_topic = "/cmd_vel"
scan_topic = "/scan"
camera_topic = "/camera/landmarks"

topics_task_3 = [camera_topic, scan_topic, velocity_topic,odom_topic]

reader.set_filter(topics_task_3)

current_rob_x = 0.0
current_rob_y = 0.0
current_rob_theta = 0.0
has_robot_pose = False


data = {
    scan_topic:  {"t": [], "x": [], "y": [], "theta": []},
    camera_topic: {"t": [], "x": [], "y": [], "theta": []},
    velocity_topic:{"t": [], "v": [], "w": []},
    odom_topic:{"t": [], "x": [], "y": [], "theta": []}
}
# Import necessari (assicurati di averli in cima al file)
# from landmark_msgs.msg import LandmarkArray
# import numpy as np

# --- STRUTTURA DATI ---
data = {
    scan_topic:     {"t": [], "x": [], "y": [], "theta": []}, # Chiave presente ma lasciata vuota
    camera_topic:   {"t": [], "x": [], "y": [], "theta": []},
    velocity_topic: {"t": [], "v": [], "w": []},
    odom_topic:     {"t": [], "x": [], "y": [], "theta": []}
}

# === CICLO DI LETTURA ===
for topic_name, msg, t in reader:
    
    # ---------------------------------------------------------
    # 1. ODOMETRIA (Aggiorna posizione robot e salva dati)
    # ---------------------------------------------------------
    if topic_name == odom_topic: # Verifica topic (es. '/odom')
        # Timestamp
        stamp = Time.from_msg(msg.header.stamp).nanoseconds * 1e-9
        
        # Estrai Posa
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        _, _, theta = euler_from_quaternion([q.x, q.y, q.z, q.w])

        # >>> AGGIORNAMENTO STATO GLOBALE (Fondamentale per la Camera) <<<
        current_rob_x = x
        current_rob_y = y
        current_rob_theta = theta
        has_robot_pose = True
        # ----------------------------------------------------------------

        # Salva i dati
        data[topic_name]["t"].append(stamp)
        data[topic_name]["x"].append(x)
        data[topic_name]["y"].append(y)
        data[topic_name]["theta"].append(theta)

    # ---------------------------------------------------------
    # 2. CAMERA (Calcola posizione globale Target)
    # ---------------------------------------------------------
    elif topic_name == camera_topic: 
        
        if has_robot_pose and isinstance(msg, LandmarkArray):
            
            if len(msg.landmarks) > 0:
                stamp = Time.from_msg(msg.header.stamp).nanoseconds * 1e-9
                
                lm = msg.landmarks[0] # Prendiamo il primo target
                r = lm.range          # Distanza
                b = lm.bearing        # Angolo relativo
                
                glob_target_x = current_rob_x + r * np.cos(current_rob_theta + b)
                glob_target_y = current_rob_y + r * np.sin(current_rob_theta + b)
                
                
                data[topic_name]["t"].append(stamp)
                data[topic_name]["x"].append(glob_target_x)
                data[topic_name]["y"].append(glob_target_y)
                data[topic_name]["theta"].append(0.0) 

    # ---------------------------------------------------------
    # 3. VELOCITÀ (Cmd_vel)
    # ---------------------------------------------------------
    elif topic_name == velocity_topic: 
        
        
        stamp = Time.from_msg(msg.header.stamp).nanoseconds * 1e-9 if hasattr(msg, 'header') else t * 1e-9
        
        # Salva le velocità
        data[topic_name]["t"].append(stamp)
        data[topic_name]["v"].append(msg.linear.x)
        data[topic_name]["w"].append(msg.angular.z)

# Reset filtro (se necessario dal tuo reader)
reader.reset_filter()

odom_t  = np.array(data["/odom"]["t"])
odom_x  = np.array(data["/odom"]["x"])
odom_y  = np.array(data["/odom"]["y"])
odom_th = np.array(data["/odom"]["theta"])

t_cmd = np.array(data['/cmd_vel']['t'])
v_cmd = np.array(data['/cmd_vel']['v'])
w_cmd = np.array(data['/cmd_vel']['w'])

landmark_t=np.array(data['/camera/landmarks']['t'])
landmark_x=np.array(data["/camera/landmarks"]['x'])
landmark_y=np.array(data["/camera/landmarks"]['y'])
landmark_th=np.array(data["/camera/landmarks"]['theta'])

odom_data = np.vstack((odom_x,odom_y,odom_th)).T
camera_data = np.vstack((landmark_x,landmark_y,landmark_th)).T

plt.figure(figsize=(10,5)) 
plt.plot(landmark_x, landmark_y, label='/camera/landmarks', color='red', linewidth=2, linestyle='-')
plt.plot(odom_x,odom_y,color='green',label='/odom', linewidth=2, linestyle='--')
plt.title("2D Trajectory Analysis")
plt.xlabel("X Position [m]")
plt.ylabel("Y Position [m]")
plt.legend(loc="best") 
plt.grid(True) 
plt.axis('equal') 
plt.savefig("trajectory_aligned.png", dpi=300)
plt.show()