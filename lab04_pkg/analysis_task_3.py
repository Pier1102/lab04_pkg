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
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from lab04_pkg.utils import rmse , normalize_angle

#bag_path = "/home/pier/Desktop/rosbag2_PIER"
bag_path = "/home/pier/Desktop/rosbag2_Michele"
#bag_path = "/home/pier/Desktop/rosbag2Cesare"
reader = Rosbag2Reader(bag_path)

odom_topic= "/odom"
velocity_topic = "/cmd_vel"
scan_topic = "/scan"
camera_topic = "/camera/landmarks"
status_topic = "/dwa/status"

topics_task_3 = [camera_topic, scan_topic, velocity_topic,odom_topic]

#=====COMPUTE SUCCESS RATE ======

reader.set_filter([status_topic])

# contatori
n_goal = 0
n_collision = 0
n_timeout = 0

for topic_name, msg, t in reader:
    if topic_name == status_topic and isinstance(msg, String):
        status = msg.data
        
        if status in ["Goal Reached"]:
            n_goal += 1
        elif status == "Collision":
            n_collision += 1
        elif status == "Timeout":
            n_timeout += 1
     
reader.reset_filter()

n_total = n_goal + n_collision + n_timeout
print(f"Total navigation events: {n_total}")
print(f"   Times the robot reach the goal : {n_goal}")
print(f"  Collision           : {n_collision}")
print(f"  Timeout             : {n_timeout}")

if n_total > 0:
    success_rate = 100.0 * n_goal / n_total
    print(f"\nSuccess Rate = {success_rate:.1f} %")
else:
    print("\nNo status messages found in the bag.")



#=================== Overall average and minimum distance [m] from the obstacles ===================

reader.set_filter([scan_topic])
ranges = []

for topic_name, msg, t in reader:
    if topic_name == scan_topic and isinstance(msg, LaserScan):
        r = np.array(msg.ranges, dtype=float)
        # sostituisco NaN e +inf con un valore molto grande
        r = np.nan_to_num(r, nan=np.inf, posinf=np.inf)

        ranges.append(r)

reader.reset_filter()
if len(ranges) == 0:
    print("\nNo LaserScan messages found in  /scan.")
else:
    # Calcolo delle statistiche
    all_ranges = np.concatenate(ranges)

    # tolgo solo valori inf e <= 0 (niente maschere sugli angoli ecc)
    finite_ranges = all_ranges[np.isfinite(all_ranges) & (all_ranges > 0.0)]

    if finite_ranges.size == 0:
        print("Nessuna distanza valida trovata (tutto inf o 0)")
    else:
        avg_distance = np.mean(finite_ranges)
        min_distance = np.min(finite_ranges)

        print(f"Distanza media: {avg_distance:.2f} m")
        print(f"Distanza minima: {min_distance:.2f} m")

#=====compute time tracking ======
follow_dist     = 0.2                 # distanza ideale 
dist_tol        = 0.3                 # tolleranza sulla distanza
bearing_tol_rad = math.radians(60.0)   # 60 gradi


#=====PLOT =====
reader.set_filter(topics_task_3)

current_rob_x = 0.0
current_rob_y = 0.0
current_rob_theta = 0.0
has_robot_pose = False


data = {
    scan_topic:     {"t": [], "x": [], "y": [], "theta": []}, 
    camera_topic:   {"t": [], "x": [], "y": [], "theta": []},
    velocity_topic: {"t": [], "v": [], "w": []},
    odom_topic:     {"t": [], "x": [], "y": [], "theta": []}
}

# === CICLO DI LETTURA ===
for topic_name, msg, t in reader:
    if topic_name == odom_topic and isinstance(msg, Odometry): 
        stamp = Time.from_msg(msg.header.stamp).nanoseconds * 1e-9
        
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        _, _, theta = euler_from_quaternion([q.x, q.y, q.z, q.w])

        # Aggiorna la posa corrente del robot
        current_rob_x = x
        current_rob_y = y
        current_rob_theta = theta
        has_robot_pose = True
        
        # Salva i dati
        data[topic_name]["t"].append(stamp)
        data[topic_name]["x"].append(x)
        data[topic_name]["y"].append(y)
        data[topic_name]["theta"].append(theta)

    # 2. CAMERA (Calcola posizione globale Target)

    elif topic_name == camera_topic and isinstance(msg, LandmarkArray):
        if has_robot_pose :
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

  
    elif topic_name == velocity_topic and isinstance(msg, Twist): 
        stamp =  t 
        # Salva le velocità
        data[topic_name]["t"].append(stamp)
        data[topic_name]["v"].append(msg.linear.x)
        data[topic_name]["w"].append(msg.angular.z)


reader.reset_filter()

odom_t  = np.array(data["/odom"]["t"],dtype=float)
odom_x  = np.array(data["/odom"]["x"])
odom_y  = np.array(data["/odom"]["y"])
odom_th = np.array(data["/odom"]["theta"])

t_cmd = np.array(data['/cmd_vel']['t'])
v_cmd = np.array(data['/cmd_vel']['v'])
w_cmd = np.array(data['/cmd_vel']['w'])

# tempo relativo
t_cmd = 1e-9 * (t_cmd - t_cmd[0])

# ANGULAR MAX DEL TURTLEBOT3 introdotto perchè avevamo un problema con una bag corrotta
w_max = 2.84  # rad/s

w_valid = np.abs(w_cmd) <= w_max

t_cmd = t_cmd[w_valid]
v_cmd = v_cmd[w_valid]
w_cmd = w_cmd[w_valid]

landmark_t=np.array(data['/camera/landmarks']['t'])
landmark_x=np.array(data["/camera/landmarks"]['x'])
landmark_y=np.array(data["/camera/landmarks"]['y'])
landmark_th=np.array(data["/camera/landmarks"]['theta'])

odom_data = np.vstack((odom_x,odom_y,odom_th)).T
camera_data = np.vstack((landmark_x,landmark_y,landmark_th)).T

interp_target = interp1d(
    landmark_t,
    camera_data,
    axis=0,
    fill_value="extrapolate",
    kind="linear"
)

target_on_robot = interp_target(odom_t)
tgt_x = target_on_robot[:, 0]
tgt_y = target_on_robot[:, 1]
tgt_th = target_on_robot[:, 2]  

dx = tgt_x - odom_x
dy = tgt_y - odom_y
dist = np.hypot(dx, dy)

angle_to_target = np.arctan2(dy, dx)
bearing = normalize_angle(angle_to_target - odom_th)  # wrap in [-pi, pi]

isdist_ok = np.abs(dist - follow_dist) <= dist_tol
isbearing_ok = np.abs(bearing) <= bearing_tol_rad

tracking_mask = isdist_ok & isbearing_ok

# ================== INTEGRAZIONE NEL TEMPO ==================

dt = np.diff(odom_t, prepend=odom_t[0])  # dt[0] = 0
total_time = odom_t[-1] - odom_t[0]
tracking_time = np.sum(dt[tracking_mask])
tracking_percent = 100.0 * tracking_time / total_time

print("\n===== TIME OF TRACKING =====")
print(f"Total time:       {total_time:.2f} s")
print(f"Tracking time:    {tracking_time:.2f} s")
print(f"Time of tracking: {tracking_percent:.1f} %")


#========= COMPUTE RMSE ==========

rmse_x = rmse(odom_data[:, 0], target_on_robot[:, 0])
rmse_y = rmse(odom_data[:, 1], target_on_robot[:, 1])

# direzione dal robot verso il target
dx = target_on_robot[:, 0] - odom_data[:, 0]
dy = target_on_robot[:, 1] - odom_data[:, 1]
angle_to_target = np.arctan2(dy, dx)

# errore di heading del robot rispetto al target
heading_err = normalize_angle(odom_data[:, 2] - angle_to_target)

# RMSE angolare
rmse_heading = math.sqrt(np.mean(heading_err**2))


print("\n===== RMSE  =====")
print(f"RMSE X:        {rmse_x:.3f} m")
print(f"RMSE Y:        {rmse_y:.3f} m")
print(f"RMSE Heading:  {math.degrees(rmse_heading):.3f} deg")




plt.figure(figsize=(10,5)) 
plt.plot(landmark_x, landmark_y, label='/camera/landmarks', color='red', linewidth=2, linestyle='-')
plt.plot(odom_x,odom_y,color='green',label='/odom', linewidth=2, linestyle='--')
plt.title("2D Trajectory Analysis")
plt.xlabel("X Position [m]")
plt.ylabel("Y Position [m]")
plt.legend(loc="best") 
plt.grid(True) 
plt.axis('equal') 
plt.show()


plt.figure(figsize=(10,8)) # Figura un po' più alta per contenere due grafici

# --- Primo Grafico: Velocità Lineare (v) ---
plt.subplot(2, 1, 1) #
plt.plot(t_cmd, v_cmd, label='Linear Velocity (v)', color='blue', linewidth=2, linestyle='-')
plt.title("Command Signal Profiles (v, w)") 
plt.ylabel("Linear Vel [m/s]")
plt.grid(True)
plt.legend(loc="upper right")

# --- Secondo Grafico: Velocità Angolare (w) ---
plt.subplot(2, 1, 2) #
plt.plot(t_cmd, w_cmd, label='Angular Velocity (w)', color='orange', linewidth=2, linestyle='-')
plt.xlabel("Time [s]") 
plt.ylabel("Angular Vel [rad/s]")
plt.grid(True)
plt.legend(loc="upper right")
plt.show()