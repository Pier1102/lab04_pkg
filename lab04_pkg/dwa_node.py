import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
import tf_transformations
import math
import numpy as np
from lab04_pkg.utils import * 
from std_msgs.msg import String, Float32
from landmark_msgs.msg import LandmarkArray  

class Dwa_node(Node):
    def __init__(self):
        super().__init__('dwa_node')
        self.get_logger().info('dwa_node started.')
        # === PARAMETRI ===
        self.declare_parameter("dt", 0.1) # time interval for simulation
        self.declare_parameter("sim_time", 1.0) #simulation time for each trajectory
        self.declare_parameter("time_granularity", 0.1) #time granularity for simulation

        self.declare_parameter("v_samples", 10) #how many values of linear velocity to sample
        self.declare_parameter("w_samples", 20) #how many values of angular velocity to sample

        self.declare_parameter("goal_dist_tol", 0.3) # distance from goal to consider it reached
        self.declare_parameter("collision_tol", 0.3) # distance from obstacles to consider a collision

        self.declare_parameter("weight_angle", 0.2) # weight for angle in objective function
        self.declare_parameter("weight_vel", 0.35) # weight for velocity in objective function
        self.declare_parameter("weight_obs", 0.1) # weight for obstacles in objective function
        self.declare_parameter("weight_target", 0.15) 

        self.declare_parameter("obstacle_max_dist", 3.0) # maximum distance to consider obstacles
        self.declare_parameter("max_num_steps", 1000) # maximum number of steps to reach the goal
        self.declare_parameter("obst_tolerance", 0.5) # obstacle tolerance distance
        self.declare_parameter("frequency", 15.0) # go_to_pose_callback frequency
        self.declare_parameter("num_ranges", 27) # ranges to consider from laser scan
        self.declare_parameter("robot_radius", 0.15) #  la dimensione del  robot


        self.dt = self.get_parameter("dt").value
        self.sim_time = self.get_parameter("sim_time").value
        self.time_granularity = self.get_parameter("time_granularity").value

        self.v_samples = self.get_parameter("v_samples").value
        self.w_samples = self.get_parameter("w_samples").value

        self.goal_dist_tol = self.get_parameter("goal_dist_tol").value
        self.collision_tol = self.get_parameter("collision_tol").value
        
        #pesi
        self.weight_angle = self.get_parameter("weight_angle").value
        self.weight_vel = self.get_parameter("weight_vel").value
        self.weight_obs = self.get_parameter("weight_obs").value
        self.weight_target=self.get_parameter("weight_target").value

        self.obstacle_max_dist = self.get_parameter("obstacle_max_dist").value
        self.max_num_steps = self.get_parameter("max_num_steps").value
        self.obst_tolerance = self.get_parameter("obst_tolerance").value 
        self.frequency = self.get_parameter("frequency").value
        self.num_ranges = self.get_parameter("num_ranges").value
        self.robot_radius = self.get_parameter("robot_radius").value

        self.goal_received = False
        self.goal_x = None
        self.goal_y = None
        self.max_linear_acc = 0.5
        self.max_ang_acc = 1.
        self.min_linear_vel=0.0
        self.max_linear_vel=0.15
        self.min_angular_vel= -1.
        self.max_angular_vel=1.
        self.sim_step = round(self.sim_time / self.time_granularity)
        
        self.feedback_steps_max=50
        self.initial_feedback_step=0

        self.x = 0.0
        self.y = 0.0
        self.yaw = 0.0
        self.robot_velocity = np.array([0.0, 0.0])

        self.dist_soglia = 0.3  # distanza soglia per rallenatare il robot quando si avvicina al goal
        self.follow_dist= 0.2 #dist ideale dal target t2
       
        
        # Inizializziamo obstacles_xy vuoto per evitare crash se non c'è scan
        self.obstacles_xy = np.empty((0, 2)) #2D array with 0 rows and 2 columns as  initial list of obstacles 
        self.filtered_obstacles = np.full(self.num_ranges, np.inf)

        #=== TIMER ===
        self.timer = self.create_timer(1.0 / self.frequency, self.go_to_pose_callback)

       # === PUB / SUB ===
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.odom_sub = self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        self.scan_sub = self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)
        #self.goal_sub = self.create_subscription(Odometry,'/dynamic_goal_pose',self.goal_callback,10)

        self.step_count = 0          # contatore globale dei passi di controllo

        # publisher per feedback distanza goal
        self.feedback_pub = self.create_publisher(Float32, '/dwa/goal_distance', 10)

        # publisher per evento finale Goal / Collision / Timeout
        self.status_pub = self.create_publisher(String, '/dwa/status', 10)

        self.landmark_sub = self.create_subscription(LandmarkArray,'/camera/landmarks',self.landmark_callback,10)


    def odom_callback(self, msg: Odometry):
        self.x = msg.pose.pose.position.x # get x position
        self.y = msg.pose.pose.position.y # get y position
        q = msg.pose.pose.orientation # get orientation quaternion
        quat = [q.x, q.y, q.z, q.w] # convert quaternion to list
        _, _, self.yaw = tf_transformations.euler_from_quaternion(quat) # get yaw from quaternion
        self.robot_velocity=np.array([msg.twist.twist.linear.x,msg.twist.twist.angular.z])

    def scan_callback(self, msg: LaserScan):
     self.raw_ranges = np.array(msg.ranges) #array con tutte le misure originali, comprese NaN e inf, se presenti

     self.obstacles1 = np.nan_to_num(self.raw_ranges,nan=msg.range_min,posinf=msg.range_max) #replace nan with range_min and inf with range_max

     self.obstacles = np.zeros(len(self.obstacles1)) #rray di zeri della stessa lunghezza di self.obstacles1

     for i in range(len(self.obstacles1)): 
         if self.obstacles1[i] <=3.5 :
             self.obstacles[i] = self.obstacles1[i]
         else:
            self.obstacles[i] = np.inf

         if self.obstacles[i]  <= self.obst_tolerance:
          self.get_logger().info('detected obstacle at: {:.2f} m'.format(self.obstacles[i])) 
    
     #Filter scan ranges 
     self.filtered_obstacles = np.zeros(self.num_ranges) #array to store filtered obstacles of dimension num_ranges(ossia quanti settori voglio dividere il campo visivo del laser)
     step = int(len(self.obstacles)/self.num_ranges) #dimensione vari settori
     for i in range(self.num_ranges):
        self.filtered_obstacles[i] = min(self.obstacles[(i*step):(i+1)*step]) # prendo il minimo di ogni settore e lo metto in filtered_obstacles
    
     self.obstacles_xy = []    # <--- lista delle coordinate (x,y) degli ostacoli 
    
     angle_min = msg.angle_min #prende l'angolo minimo del laser
     angle_inc = msg.angle_increment #passo angolare del laser

     # angolo centrale di ogni settore
     for i in range(int(self.num_ranges)):
        if self.filtered_obstacles[i] == np.inf:
            continue  # salta se non c'è ostacolo in questo settore
        dist = self.filtered_obstacles[i]

        # calcolo dell’indice del centro del settore
        center_idx = int(i*step + step/2)

        # angolo relativo del laser
        angle = angle_min + center_idx * angle_inc

        # coordinate ostacolo nel frame del ROBOT
        ox_r = dist * np.cos(angle)
        oy_r = dist * np.sin(angle)

        # trasformazione nel WORLD frame
        ox_w = self.x + ox_r * np.cos(self.yaw) - oy_r * np.sin(self.yaw)
        oy_w = self.y + ox_r * np.sin(self.yaw) + oy_r * np.cos(self.yaw)

        self.obstacles_xy.append([ox_w, oy_w])

        # convertiamo in numpy array
     self.obstacles_xy = np.array(self.obstacles_xy)
        # ora self.obstacles_xy è ciò che devi dare all’algoritmo DWA

    def landmark_callback(self, msg: LandmarkArray):
        """
        Callback to process landmark data from the camera.
        Converts landmark positions from robot frame to world frame.
        """
        if len(msg.landmarks) == 0:
            return  # No landmarks detected

        # scegli il tag che ti interessa; qui prendo il primo
        lm = msg.landmarks[0]
        
        # ATTENZIONE: controlla i nomi dei campi nel tuo Landmark.msg
        r = lm.range      # distanza dal tag [m]
        beta = lm.bearing # angolo rispetto all'asse del robot [rad]
        self.get_logger().info(f"[Landmark] range = {r:.3f} m, bearing = {beta:.3f} rad")


        # 1) coordinate tag nel frame ROBOT (base_link)
        x_tag_base = r * math.cos(beta)
        y_tag_base = r * math.sin(beta)

        # 2) trasformazione nel frame odom usando la posa del robot
        x_r = self.x
        y_r = self.y
        yaw_r = self.yaw

        x_tag_odom = x_r + x_tag_base * math.cos(yaw_r) - y_tag_base * math.sin(yaw_r)
        y_tag_odom = y_r + x_tag_base * math.sin(yaw_r) + y_tag_base * math.cos(yaw_r)

        # 3) aggiorna il goal del DWA
        self.goal_x = x_tag_odom
        self.goal_y = y_tag_odom
        self.goal_received = True

        self.get_logger().info(
            f"New goal from AprilTag: ({self.goal_x:.2f}, {self.goal_y:.2f})")

       
    # def goal_callback(self, msg: Odometry): #goal callback to get dynamic goal (GoalManager node)
    #     self.goal_x = msg.pose.pose.position.x
    #     self.goal_y = msg.pose.pose.position.y
    #     self.goal_received = True
    #     self.get_logger().info('New goal received: ({:.2f}, {:.2f})'.format(self.goal_x, self.goal_y))


    def simulate_paths(self, n_paths, pose, u): #pose given by odometry
        """
        Simulate trajectory at constant velocity u=(v,w)
        """
        sim_paths = np.zeros((n_paths, self.sim_step, pose.shape[0]))
        sim_paths[:, 0] = pose.copy()

        for i in range(1, self.sim_step): #for each values i'm simulating the trajectory
            sim_paths[:, i, 0] = sim_paths[:, i - 1, 0] + u[:, 0] * np.cos(sim_paths[:, i - 1, 2]) * self.dt
            sim_paths[:, i, 1] = sim_paths[:, i - 1, 1] + u[:, 0] * np.sin(sim_paths[:, i - 1, 2]) * self.dt
            sim_paths[:, i, 2] = sim_paths[:, i - 1, 2] + u[:, 1] * self.dt

        return sim_paths
    

    def get_trajectories(self, robot_pose): 
    
        # calculate reachable range of velocity and angular velocity in the dynamic window
        min_lin_vel, max_lin_vel, min_ang_vel, max_ang_vel = self.compute_dynamic_window(self.robot_velocity)
        
        v_values = np.linspace(min_lin_vel, max_lin_vel, self.v_samples)
        w_values = np.linspace(min_ang_vel, max_ang_vel, self.w_samples) # i'm creating a grid of linear and angular velocities, and there are self.vsamples elements for the linear velocitiy and self.wsamples elements for the angular velocity

        # list of all paths and velocities
        n_paths = w_values.shape[0]*v_values.shape[0]
        sim_paths = np.zeros((n_paths, self.sim_step, robot_pose.shape[0]))
        velocities = np.zeros((n_paths, 2))

        # evaluate all velocities and angular velocities combinations    
        vv, ww = np.meshgrid(v_values, w_values)
        velocities = np.dstack([vv,ww]).reshape(n_paths, 2)
        sim_paths = self.simulate_paths(n_paths, robot_pose, velocities)

        return sim_paths, velocities

    def compute_cmd(self, goal_pose, robot_state, obstacles):

        """
        Compute the next velocity command u=(v,w) according to the DWA algorithm.
        The velocity leading to the highest scored trajectory is selected.
        """
        # create path
        paths, velocities = self.get_trajectories(robot_state) # simulate all the trajectories

        # evaluate path
        opt_idx = self.evaluate_paths(paths, velocities, goal_pose, robot_state, obstacles) # evaluate all the paths and select the best one
        u = velocities[opt_idx] # select the optimal velocity
        return u
    
    def compute_dynamic_window(self, robot_vel): 
        """
        Calculate the dynamic window composed of reachable linear velocity and angular velocity according to robot's kinematic limits.
        """
        #given my velocity value, what is the min and max velocity I can reach in the next dt time
        # linear velocity

        min_vel = robot_vel[0] - self.dt * self.max_linear_acc

        max_vel = robot_vel[0] + self.dt * self.max_linear_acc

        # minimum
        if min_vel < self.min_linear_vel:
            min_vel = self.min_linear_vel
        # maximum
        if max_vel > self.max_linear_vel:
            max_vel = self.max_linear_vel

        # angular velocity
        min_ang_vel = robot_vel[1] - self.dt * self.max_ang_acc
        max_ang_vel = robot_vel[1] + self.dt * self.max_ang_acc
        # minimum
        if min_ang_vel < self.min_angular_vel:
            min_ang_vel = self.min_angular_vel
        # maximum
        if max_ang_vel > self.max_angular_vel:
            max_ang_vel = self.max_angular_vel

        return min_vel, max_vel, min_ang_vel, max_ang_vel


    def evaluate_paths(self, paths, velocities, goal_pose, robot_pose, obstacles):
        """
        Evaluate the simulated paths using the objective function.
        J = w_h * heading + w_v * vel + w_o * obst_dist
        """
        # detect nearest obstacle
        nearest_obs = calc_nearest_obs(robot_pose, obstacles)

        # Compute the scores for the generated path
        # (1) heading_angle and goal distance
        score_heading_angles = self.score_heading_angle(paths, goal_pose)
        # (2) velocity
        score_vel = self.score_vel(velocities, paths, goal_pose)
        # (3) obstacles
        score_obstacles = self.score_obstacles(paths, nearest_obs) #most complex thing, this will be the distance from the obstacles
        # (4) distance from ideal target following distance
        score_dist_target = self.score_dist_target(paths, goal_pose)

        # Scores Normalization
        score_heading_angles = normalize(score_heading_angles)
        score_vel = normalize(score_vel)
        score_obstacles = normalize(score_obstacles)
        score_dist_target = normalize(score_dist_target)

        # Compute the idx of the optimal path according to the overall score
        opt_idx = np.argmax(np.sum(
            np.array([score_heading_angles, score_vel, score_obstacles,score_dist_target]) #(3,N)
            * np.array([[self.weight_angle, self.weight_vel, self.weight_obs, self.weight_target]]).T, #.T la transposta per avere (3,1)
            axis=0,
        ))

        try:
            return opt_idx #this is the indx of the velocity that gives the best trajectory
        except:
            raise Exception("Not possible to find an optimal path")

    def score_heading_angle(self, path, goal_pose):
        """
        Go towards the target objective: score trajectory according to the heading angle to the goal
        """
        #path[:, -1, :] è lo stato finale (ultimo istante) di ogni traiettoria
        last_x = path[:, -1, 0] #contiene il valore x finale di ciascuna traiettoria
        last_y = path[:, -1, 1] #contiene il valore y finale di ciascuna traiettoria
        last_th = path[:, -1, 2] #contiene l'angolo theta finale di ciascuna traiettoria

        # calculate angle
        angle_to_goal = np.arctan2(goal_pose[1] - last_y, goal_pose[0] - last_x)
        #Quindi è l angolo (in radianti) della direzione che va dal punto finale della traiettoria al goal

        # calculate score
        score_angle = angle_to_goal - last_th  #Questo è l'errore angolare fra la direzione a cui il robot è orientato alla fine della traiettoria e la direzione del goal
        score_angle = np.fabs(normalize_angle(score_angle)) #normalize_angle porta l angolo nel range [-pi, pi). np.fabs fa il valore assoluto
        score_angle = np.pi - score_angle  #pi - errore angolare, in questo modo un errore di 0 dà il punteggio massimo (pi), un errore di pi dà punteggio 0

        return score_angle

    def score_vel(self, u, path, goal_pose):
        """
        Maximum velocity objective: score trajectory according to the forward velocity. When the robot is near the goal, slow down.
        """

        vel = u[:,0] # linear velocity value is the score itself (to maximize)
        dist_to_goal = np.linalg.norm(path[:, -1, 0:2] - goal_pose, axis=-1)
        score = np.zeros(len(vel))
         # slow down when near the goal
        for i in range(len(vel)):
            if dist_to_goal[i] <= self.dist_soglia:
                score[i] = vel[i] * np.exp(-dist_to_goal[i] / self.goal_dist_tol)
            else: #lontano dal goal massimizza la velocità
                score[i] = vel[i]
    
        return score
    

    def score_obstacles(self, path, obstacles):
        """
        Obstacle avoidance objective: score trajectory according to the distance to the nearest obstacle.
        """
        score_obstacle = 3.0*np.ones((path.shape[0]))

        for obs in obstacles:
            dx = path[:, :, 0] - obs[0]
            dy = path[:, :, 1] - obs[1]
            dist = np.hypot(dx, dy) #np.hypot calcola la distanza euclidea nel piano

            min_dist = np.min(dist, axis=-1) # la distanza minima, nel tempo, fra la traiettoria k e quell ostacolo specifico.
            score_obstacle[min_dist < score_obstacle] = min_dist[min_dist < score_obstacle]
        
            # collision with obstacle
            score_obstacle[score_obstacle < self.robot_radius + self.collision_tol] = -100 # heavy penalty for collision
               
        return score_obstacle
    
    def score_dist_target(self, path, goal_pose): #seconda parte task 2
        """
        Task 2 seconda parte: score trajectory according to the distance to an ideal following distance from the target.
        """
        last_x = path[:, -1, 0] 
        last_y = path[:, -1, 1]
        last_th = path[:, -1, 2]
        # calculate score

        dist_to_goal = np.linalg.norm(path[:, -1, 0:2] - goal_pose, axis=-1)
        dist_score = -np.abs(dist_to_goal - self.follow_dist)  # penalizzo se mi allontano dalla distanza ideale
            # angolo di visibilità
        angle_to_target = np.arctan2(
            goal_pose[1] - last_y,
            goal_pose[0] - last_x
        )
        angle_diff = normalize_angle(angle_to_target - last_th)
        angle_score = np.cos(angle_diff)  # premia se l'angolo è vicino a 0 (cioè se il robot è orientato verso il target)
        for i in range(len(angle_score)):
            if angle_score[i] < 0:
             angle_score[i] = 0  # non premiare angoli maggiori di 90 gradi
        score = dist_score * angle_score
        return score


    
 
    def go_to_pose_callback(self): #core of the program, this generates command to reach a goal
     
     if not self.goal_received:
        self.get_logger().info("Waiting for goal...")
        return
     if self.obstacles_xy.shape[0] == 0: #in questo modo controllo se ho ricevuto almeno un laser scan
        self.get_logger().warn("No obstacles yet, skipping DWA step")
        return
     
     msg=Twist()

     self.step_count += 1
     self.initial_feedback_step+=1

     # stato attuale e distanza dal goal
     self.robot_state=np.array([self.x,self.y,self.yaw])
     self.goal_position=np.array([self.goal_x,self.goal_y])
     self.robot_x_y=np.array([self.x,self.y])

     self.dist_to_goal=np.linalg.norm(self.robot_x_y - self.goal_position)

     #self.get_logger().info(f"Distance to goal: {self.dist_to_goal:.2f}")

     
     collision_event_thresh = self.robot_radius + 0.02  # molto vicino al contatto

     min_dist = np.min(self.filtered_obstacles)
     #self.get_logger().info(f"min(filtered_obstacles) = {min_dist:.3f}")
    #safety check
     if min_dist < collision_event_thresh:
              
              self.get_logger().info('Obstacle too close, stop the robot.')
              cmd = Twist()
              cmd.linear.x = 0.0
              cmd.angular.z = 0.0
              self.cmd_pub.publish(cmd)
              
              msg = String()
              msg.data = "Collision"
              self.status_pub.publish(msg)
            

              # reset stato di navigazione
              self.goal_received = False
              self.step_count = 0
              self.initial_feedback_step = 0
              return

     else:
        self.command=self.compute_cmd(self.goal_position,self.robot_state,self.obstacles_xy)
        msg.linear.x=self.command[0]
        msg.angular.z=self.command[1]
        #self.get_logger().info(f"Publishing vel: {msg}")
        self.cmd_pub.publish(msg)

     self.robot_state=np.array([self.x,self.y,self.yaw])
     self.goal_position=np.array([self.goal_x,self.goal_y])
     self.robot_x_y=np.array([self.x,self.y])
     self.dist_to_goal=np.linalg.norm(self.robot_x_y - self.goal_position)
     
     if self.initial_feedback_step % self.feedback_steps_max == 0: # print feedback every feedback_steps_max steps
        feedback_msg = Float32()
        feedback_msg.data = float(self.dist_to_goal)
        self.feedback_pub.publish(feedback_msg)
        #self.get_logger().info(f"Distance to goal: {self.dist_to_goal:.2f}")

        self.goal_reached=False

     if self.dist_to_goal < self.goal_dist_tol:
        self.goal_reached = True
        cmd = Twist()
        cmd.linear.x = 0.0
        cmd.angular.z = 0.0
        self.cmd_pub.publish(cmd)

        msg = String()
        msg.data = "Goal Reached"
        self.status_pub.publish(msg)
        
        #self.get_logger().info("Goal successfully reached!")
        self.goal_received = False
        self.step_count = 0
        self.initial_feedback_step = 0
        return

    # controllo Timeout sul numero massimo di passi
     if self.step_count > self.max_num_steps:
        #self.get_logger().warn("Timeout reached, stopping robot.")
        cmd = Twist()
        cmd.linear.x = 0.0
        cmd.angular.z = 0.0
        self.cmd_pub.publish(cmd)
        msg = String()
        msg.data = "Timeout"
        self.status_pub.publish(msg)
    
        
        self.goal_received = False
        self.step_count = 0
        self.initial_feedback_step = 0
        return

def main():
    rclpy.init()
    node = Dwa_node()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
    
     

     



