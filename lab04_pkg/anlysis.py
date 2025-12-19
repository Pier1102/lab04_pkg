import math
import numpy as np
import matplotlib.pyplot as plt
from std_msgs.msg import String
from nav_msgs.msg import Odometry
from rosbag2_reader_py import Rosbag2Reader 
from tf_transformations import euler_from_quaternion
from rclpy.time import Time
from scipy.interpolate import interp1d
from lab04_pkg.utils import rmse , normalize_angle
from sensor_msgs.msg import LaserScan

bag_path = "/home/pier/ros2_ws/src/lab04_pkg/lab04_pkg/rosbag_lab04_task1_task2"
#bag_path = "/home/pier/Desktop/rosbag2_PIER"
reader = Rosbag2Reader(bag_path)

status_topic = "/dwa/status"

topics_status = [status_topic]
reader.set_filter(topics_status)

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

follow_dist     = 0.2                 # distanza ideale che avevi nel DWA
dist_tol        = 0.3                 # tolleranza sulla distanza
bearing_tol_rad = math.radians(60.0)   # 60 gradi

robot_topic  = "/ground_truth"
target_topic = "/dynamic_goal_pose"
odom_topic= "/odom"
velocity_topic = "/cmd_vel"
scan_topic = "/scan"

topics_tracking = [robot_topic, target_topic, velocity_topic,odom_topic]
reader.set_filter(topics_tracking)

data = {
    robot_topic:  {"t": [], "x": [], "y": [], "theta": []},
    target_topic: {"t": [], "x": [], "y": [], "theta": []},
    velocity_topic:{"t": [], "v": [], "w": []},
    odom_topic:{"t": [], "x": [], "y": [], "theta": []}
}

for topic_name, msg, t in reader:
    if topic_name in topics_tracking and isinstance(msg, Odometry):
        stamp = Time.from_msg(msg.header.stamp).nanoseconds * 1e-9

        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        _, _, theta = euler_from_quaternion([q.x, q.y, q.z, q.w])

        data[topic_name]["t"].append(stamp)
        data[topic_name]["x"].append(x)
        data[topic_name]["y"].append(y)
        data[topic_name]["theta"].append(theta)
    elif topic_name=="/cmd_vel":

        stamp = Time.from_msg(msg.header.stamp).nanoseconds * 1e-9 if hasattr(msg, 'header') else t # Se t è il tempo del bag

        # Estrai direttamente le velocità
        v = msg.linear.x      # Velocità Lineare
        w = msg.angular.z     # Velocità Angolare (Ricorda: Z, non X!)

        # Salva i dati
        data[topic_name]["t"].append(stamp)
        data[topic_name]["v"].append(v)
        data[topic_name]["w"].append(w)

reader.reset_filter()

# conversione in array numpy
robot_t  = np.array(data["/ground_truth"]["t"])
robot_x  = np.array(data["/ground_truth"]["x"])
robot_y  = np.array(data["/ground_truth"]["y"])
robot_th = np.array(data["/ground_truth"]["theta"])

target_t  = np.array(data["/dynamic_goal_pose"]["t"])
target_x  = np.array(data["/dynamic_goal_pose"]["x"])
target_y  = np.array(data["/dynamic_goal_pose"]["y"])
target_th = np.array(data["/dynamic_goal_pose"]["theta"])

odom_t  = np.array(data["/odom"]["t"])
odom_x  = np.array(data["/odom"]["x"])
odom_y  = np.array(data["/odom"]["y"])
odom_th = np.array(data["/odom"]["theta"])

t_cmd = np.array(data['/cmd_vel']['t'])
v_cmd = np.array(data['/cmd_vel']['v'])
w_cmd = np.array(data['/cmd_vel']['w'])

robot_data = np.vstack((robot_x, robot_y,robot_th)).T   
odom_data = np.vstack((odom_x,odom_y,odom_th)).T
target_data = np.vstack((target_x, target_y, target_th)).T  


offset_x = robot_data[0, 0] - odom_data[0, 0]
offset_y = robot_data[0, 1] - odom_data[0, 1]

#2. Crea una copia dei dati odom per non sovrascrivere gli originali (utile per debug)
odom_aligned = odom_data.copy()

#3. Applica l'offset a tutte le righe
odom_aligned[:, 0] = odom_data[:, 0] + offset_x  # Allinea X
odom_aligned[:, 1] = odom_data[:, 1] + offset_y  # Allinea Y

print("Campioni robot:", len(robot_t))
print("Campioni target:", len(target_t))

if len(robot_t) < 2 or len(target_t) < 2:
    raise RuntimeError("Dati insufficienti per calcolare il time of tracking")
#interp1d richiede almeno 2 punti per costruire la funzione,con un solo punto non può fare interpolazione 

#================== INTERPOLAZIONE TARGET SUI TEMPI DEL ROBOT ==================

target_data = np.vstack((target_x, target_y, target_th)).T

interp_target = interp1d(
    target_t,
    target_data,
    axis=0,
    fill_value="extrapolate",
    kind="linear"
)

target_on_robot = interp_target(robot_t)
tgt_x = target_on_robot[:, 0]
tgt_y = target_on_robot[:, 1]
tgt_th = target_on_robot[:, 2]  # se ti serve

# ================== DISTANZA E BEARING ==================

dx = tgt_x - robot_x
dy = tgt_y - robot_y
dist = np.hypot(dx, dy)

angle_to_target = np.arctan2(dy, dx)
bearing = normalize_angle(angle_to_target - robot_th)  # wrap in [-pi, pi]

dist_ok = np.abs(dist - follow_dist) <= dist_tol
bearing_ok = np.abs(bearing) <= bearing_tol_rad
tracking_mask = dist_ok & bearing_ok

# ================== INTEGRAZIONE NEL TEMPO ==================

dt = np.diff(robot_t, prepend=robot_t[0])  # dt[0] = 0
total_time = robot_t[-1] - robot_t[0]
tracking_time = np.sum(dt[tracking_mask])

tracking_percent = 100.0 * tracking_time / total_time


print("\n===== TIME OF TRACKING =====")
print(f"Total time:       {total_time:.2f} s")
print(f"Tracking time:    {tracking_time:.2f} s")
print(f"Time of tracking: {tracking_percent:.1f} %")

# --- RMSE POSIZIONE ---
rmse_x = rmse(robot_data[:, 0], target_on_robot[:, 0])
rmse_y = rmse(robot_data[:, 1], target_on_robot[:, 1])


# --- RMSE ORIENTAZIONE ---
# direzione dal robot verso il target
dx = target_on_robot[:, 0] - robot_data[:, 0]
dy = target_on_robot[:, 1] - robot_data[:, 1]
angle_to_target = np.arctan2(dy, dx)

# errore di heading del robot rispetto al target
heading_err = normalize_angle(robot_data[:, 2] - angle_to_target)

# RMSE angolare
rmse_heading = math.sqrt(np.mean(heading_err**2))


print("\n===== RMSE  =====")
print(f"RMSE X:        {rmse_x:.3f} m")
print(f"RMSE Y:        {rmse_y:.3f} m")
print(f"RMSE Heading:  {math.degrees(rmse_heading):.3f} deg")

#=================== Overall average and minimum distance [m] from the obstacles ===================
topics_to_read = ["/scan"]
reader.set_filter(topics_to_read)
ranges = []

for topic_name, msg, t in reader:
    if topic_name == "/scan" and isinstance(msg, LaserScan):
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



# ================== PLOT TRAJECTORY 2D ==================
plt.figure(figsize=(10,5)) 
plt.plot(robot_x, robot_y, label='/ground_truth', color='red', linewidth=2, linestyle='-')
plt.plot(target_x,target_y,label='/dynamic_goal_pose', color='green', linewidth=2, linestyle='--')
plt.plot(odom_aligned[:,0],odom_aligned[:,1],label="/odom",color="tab:blue",linewidth=2,linestyle='--')
plt.title("2D Trajectory Analysis")
plt.xlabel("X Position [m]")
plt.ylabel("Y Position [m]")
plt.legend(loc="best") 
plt.grid(True) 
plt.axis('equal') 
#plt.savefig("trajectory_aligned.png", dpi=300)
plt.show()

if len(t_cmd) > 0: #allineamento del tempo 
    t_cmd = t_cmd - t_cmd[0]

# 2. Creazione Plot usando la tua "forma"
plt.figure(figsize=(10,8)) # Figura un po' più alta per contenere due grafici

# --- Primo Grafico: Velocità Lineare (v) ---
plt.subplot(2, 1, 1) # 2 righe, 1 colonna, grafico numero 1
plt.plot(t_cmd, v_cmd, label='Linear Velocity (v)', color='blue', linewidth=2, linestyle='-')
plt.title("Command Signal Profiles (v, w)") # Titolo in alto
plt.ylabel("Linear Vel [m/s]")
plt.grid(True)
plt.legend(loc="upper right")

# --- Secondo Grafico: Velocità Angolare (w) ---
plt.subplot(2, 1, 2) # 2 righe, 1 colonna, grafico numero 2
plt.plot(t_cmd, w_cmd, label='Angular Velocity (w)', color='orange', linewidth=2, linestyle='-')
plt.xlabel("Time [s]") # L'asse X (Tempo) si mette solo in basso
plt.ylabel("Angular Vel [rad/s]")
plt.grid(True)
plt.legend(loc="upper right")
plt.show()

