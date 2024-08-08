import argparse
import rospy
from sensor_msgs.msg import Image, JointState
from cv_bridge import CvBridge
from geometry_msgs.msg import (
    PoseStamped,
    Pose,
    Point,
    Quaternion,)

import cv2 as cv
import torch 
import numpy as np
import os,sys
from policy import ACTPolicy
import pickle
from einops import rearrange

# sawyer
import intera_interface
import intera_external_devices

from intera_interface import CHECK_VERSION
from intera_core_msgs.msg import (
    IODeviceStatus,
    EndpointState,)


target_color = 'red'
box_color = 'red'

task_name = 'sorting_program_sawyer21'
ckpt_dir = '/home/boxjod/RLBench_ACT_Sawyer/Trainings/sorting_program_sawyer212/50demo_36step_20chunk_8batch_efficientnet_b0'
ckpt_name = 'policy_best_epoch1000.pth'
learning_rate = 1e-5

task_name_21 = ''
ckpt_dir_21 = ''
ckpt_name_21 = ''

task_name_22 = ''
ckpt_dir_22 = ''
ckpt_name_22 = ''

GRIPPER_PARAM = 0.02
IF_AUTO = False

def observation_to_action(policy, max_timesteps, ckpt_dir):
  print("start get the robot observation and do the model... ")
  bridge = CvBridge()
    
  gpos = []
  qpos = []
  gripper_state = 1
  stats_path = os.path.join(ckpt_dir, f'dataset_stats.pkl')
  with open(stats_path, 'rb') as f:
    stats = pickle.load(f)
    
  pre_process_qpos = lambda s_qpos: (s_qpos - stats['qpos_mean']) / stats['qpos_std']
  pre_process_gpos = lambda s_gpos: (s_gpos - stats['gpos_mean']) / stats['gpos_std']
  pre_process_action_history = lambda s_action: (s_action - stats['action_mean']) / stats['action_std']
  post_process = lambda a: a * stats['action_std'] + stats['action_mean']
  
  query_frequency = 1
  chunk_size = 20 # chunking_size 
  action_dim = 8
  hidden_dim = 512
  
  if task_name == 'sorting_program_sawyer21':
    max_timesteps = int(max_timesteps * 2.4)  
  elif task_name == 'sorting_program_sawyer22':
    max_timesteps = int(max_timesteps * 2.0)  
  all_time_actions = torch.zeros([max_timesteps, max_timesteps+chunk_size, 8]).cuda() 
  image_list = [] # for visualization
  
  tray_defaul_waypoint = {'right_j0': 0.0568505859375, 'right_j1': -0.1841328125, 'right_j2': -0.0026962890625, 'right_j3': 0.8043798828125, 'right_j4': -0.019640625, 'right_j5': 0.950361328125, 'right_j6': -1.4002099609375}
  
  # intera structure
  side = 'right'
  limb = intera_interface.Limb(side)
  head = intera_interface.Head()
  gripper = intera_interface.Gripper(side + '_gripper')
  joints_name = limb.joint_names()
  
  qpos_initial = [] 
  gpos_initial = []
  
  # update qpos_initial and gpos_initial
  def refresh_initial_pos():
    nonlocal qpos_initial, gpos_initial
    qpos_dict = limb.joint_angles()
    
    qpos_initial = [qpos_dict['right_j0'], qpos_dict['right_j1'], qpos_dict['right_j2'], qpos_dict['right_j3'], qpos_dict['right_j4'], qpos_dict['right_j5'], qpos_dict['right_j6']]
    
    gpos_dict = limb.endpoint_pose()
    position_initial = gpos_dict['position']
    orientation_initial = gpos_dict['orientation']
    
    gpos_initial = [position_initial.x, position_initial.y, position_initial.z, orientation_initial.x, orientation_initial.y, orientation_initial.z, orientation_initial.w]  
  
  if task_name == 'sorting_program_sawyer22':
    refresh_initial_pos()
  
  timestep = 0
  image_get_count = 0
  clac_rate = 10 
  
  def go_to_next_gpos(target_gpos):
    nonlocal timestep, limb, all_time_actions, gripper_state, max_timesteps
    global task_name, ckpt_name, ckpt_dir
    
    # hover_distance = -0.135 
    tip_name = 'right_gripper_tip'
    
    approach = Pose()
    approach.position.x = target_gpos[0]
    approach.position.y = target_gpos[1]
    approach.position.z = target_gpos[2]
    approach.orientation.x = target_gpos[3]
    approach.orientation.y = target_gpos[4]
    approach.orientation.z = target_gpos[5]
    approach.orientation.w = target_gpos[6]
    # approach.position.z = approach.position.z # + hover_distance 
    timestep = timestep + 1
    # print(timestep,end=' ')

    joint_angles = limb.ik_request(approach, tip_name) 
    
    limb.set_joint_position_speed(0.06) # 0.06 is good
    done = limb.move_to_joint_positions(joint_angles, timeout = 1/clac_rate) 
    
    if gripper_state == 1 and target_gpos[7] <= 0.3:
      print("close the gripper:",target_gpos[7])
      gripper.close()
      if task_name == 'sorting_program_sawyer21':
        timestep = max_timesteps
      
    elif timestep > 60 and gripper_state == 0 and target_gpos[7] >= 0.1 :
      print("open the gripper")
      gripper.open()
      
      
    if timestep >= max_timesteps:
      subscriber_control(0)
      
      if IF_AUTO:
        if task_name == 'sorting_program_sawyer21':
          print("continue to the  the step 2")
          task_name = task_name_22
          ckpt_dir = ckpt_dir_22
          ckpt_name = ckpt_name_22
          subscriber_control(0)

          policy, max_timesteps = buil_model(ckpt_dir, ckpt_name)
          observation_to_action(policy, max_timesteps, ckpt_dir)
          
        elif task_name == 'sorting_program_sawyer22' :
          print("operate done")
          os._exit(0)
          
      else:
        if task_name == 'sorting_program_sawyer21':
          print("you can try agqin input '1' and '2' to next setp")
        elif   task_name == 'sorting_program_sawyer22':
          print("you can try agqin input '1'")
          
        done = False
        while not done and not rospy.is_shutdown():
          c = intera_external_devices.getch()
          if c in ['\x1b', '\x03']:
            done = True
            rospy.signal_shutdown("Example finished.")
          elif c == '1':
            if task_name == 'sorting_program_sawyer21':
              go_to_initial_position()
              all_time_actions = torch.zeros([max_timesteps, max_timesteps+chunk_size, 8]).cuda() 
              print("retest the step 1")
              timestep = 0
              subscriber_control(1)
              done = True
            elif task_name == 'sorting_program_sawyer22':
              print("restart the task")
              task_name = task_name_21
              ckpt_dir = ckpt_dir_21
              ckpt_name = ckpt_name_21
              subscriber_control(0)
            
              policy, max_timesteps = buil_model(ckpt_dir, ckpt_name)
              observation_to_action(policy, max_timesteps, ckpt_dir)
            
          elif task_name == 'sorting_program_sawyer21' and c =='2':
            print("continue to the  the step 2")
            task_name = task_name_22
            ckpt_dir = ckpt_dir_22
            ckpt_name = ckpt_name_22
            subscriber_control(0)

            policy, max_timesteps = buil_model(ckpt_dir, ckpt_name)
            observation_to_action(policy, max_timesteps, ckpt_dir)
            if IF_AUTO:
              sys.exit()
      
  history_action = np.zeros((chunk_size,) + (action_dim,), dtype=np.float32)
  history_image_feature = np.zeros((2,chunk_size,) + (hidden_dim,), dtype=np.float32)
  
  def policy_model_calc(policy, curr_image, qpos_current, qpos_diff, gpos_current, gpos_diff):
    nonlocal history_action, history_image_feature, max_timesteps, timestep
    
    with torch.inference_mode(): 
      image_list.append({'wrist':curr_image})

      qpos_numpy = np.array(qpos_current) # 7 + 1 + 7 = 15
      qpos_numpy = np.array(np.append(qpos_numpy, qpos_diff)) # 7 + 1 + 7 = 15
      qpos_numpy = pre_process_qpos(qpos_numpy)
      qpos_numpy = torch.from_numpy(qpos_numpy).float().cuda().unsqueeze(0)
      
      gpos_numpy = np.array(gpos_current) # 7 + 1 + 7 = 15
      gpos_numpy = np.array(np.append(gpos_numpy, gpos_diff)) # 7 + 1 + 7 = 15
      gpos_numpy = pre_process_gpos(gpos_numpy)
      gpos_numpy = torch.from_numpy(gpos_numpy).float().cuda().unsqueeze(0)
      
      history_action_numpy = np.array(history_action)
      history_action_numpy = pre_process_action_history(history_action_numpy)
      history_action_numpy = torch.from_numpy(history_action_numpy).float().cuda()
    
      is_pad_history = np.zeros(max_timesteps)
      is_pad_history[timestep:] = 1
      is_pad_history = torch.from_numpy(is_pad_history).bool().cuda()
      
      ### query policy
      if timestep % query_frequency == 0:
        command_embedding = None
        all_actions, image_feature = policy(qpos_numpy, gpos_numpy, curr_image, 
                             history_image_feature, history_action_numpy, 
                             is_pad_history=is_pad_history,
                             actions=None, is_pad_action=None, 
                             command_embedding=command_embedding) 
        

      all_time_actions[[timestep], timestep: timestep+chunk_size] = all_actions
      actions_for_curr_step = all_time_actions[:, timestep]
      
      actions_populated = torch.all(actions_for_curr_step != 0, axis=1)
      actions_for_curr_step = actions_for_curr_step[actions_populated]
      k = 0.01
      exp_weights = np.exp(-k * np.arange(len(actions_for_curr_step)))
      exp_weights = exp_weights / exp_weights.sum() 
      exp_weights = torch.from_numpy(exp_weights).cuda().unsqueeze(dim=1) 
      raw_action = (actions_for_curr_step * exp_weights).sum(dim=0, keepdim=True)
          
      ### post-process actions
      raw_action = raw_action.squeeze(0).cpu().numpy()
      action = post_process(raw_action) 
      go_to_next_gpos(action)   
      
      history_action = np.insert(history_action, 0, action, axis=0)[:chunk_size]
      history_image_feature[0] = np.insert(history_image_feature[0], 0, image_feature[0], axis=0)[:chunk_size]
      history_image_feature[1] = np.insert(history_image_feature[1], 0, image_feature[1], axis=0)[:chunk_size]
        

  def get_observations(data): # 30Hz
    nonlocal image_get_count, image_list, qpos, gripper_state, gpos, timestep, timestep, qpos_initial, gpos_initial
    
    image_get_count = image_get_count + 1
    if (image_get_count >= 30 / clac_rate) and (timestep < max_timesteps): 
      image_get_count = 0
      
      image = bridge.imgmsg_to_cv2(data,  desired_encoding='rgb8')
      wrist_image_current = image
      # if task_name == 'sorting_program_sawyer21':
      wrist_image_current = cv.resize(image, (0, 0), fx=0.25, fy=0.25, interpolation=cv.INTER_AREA) # wrist_rgb
      
      curr_images = []
      curr_image = rearrange(wrist_image_current, 'h w c -> c h w')
      curr_images.append(curr_image)    
      curr_image = np.stack(curr_images, axis=0)
      curr_image = torch.from_numpy(curr_image / 255.0).float().cuda().unsqueeze(0)
      
      qpos_diff = [a-b for a,b in zip(qpos, qpos_initial)]
      qpos_current = qpos + [gripper_state]
      
      gpos_diff = [a-b for a,b in zip(gpos, gpos_initial)]
      gpos_current = gpos + [gripper_state] 
      
      # print(f"{len(gpos_initial)=},{len(gpos_diff)=}")
      policy_model_calc(policy, curr_image, qpos_current, qpos_diff, gpos_current, gpos_diff)


  def get_gpos(data): # gpos 
    nonlocal gpos
    position = data.pose.position
    orientation = data.pose.orientation
    gpos = [position.x, position.y, position.z - 0.135, orientation.x, orientation.y, orientation.z, orientation.w] 
    
  def get_qpos(data): # gpos  
    nonlocal qpos, timestep
    if len(data.position) == 9:
      qpos = [data.position[1], data.position[2], data.position[3], data.position[4], data.position[5], data.position[6], data.position[7]]
      effort = sum(data.effort)
      if effort > -20:
        
        if gripper_state == 0:
          gripper.open()
        else:
          print("detect collision")
          subscriber_control(0)
        timestep = max_timesteps
        
        
  def get_gripper_state(data): # 100Hz
    nonlocal gripper_state
    gripper_state = float(data.signals[10].data[1:-1]) 
    if gripper_state >= GRIPPER_PARAM:
      gripper_state = 1
    else:
      gripper_state = 0
  
  def subscriber_control(ctrl):
    nonlocal wrist_cam_Subscriber, endpoint_state_Subscriber, joint_states_Subscriber, right_gripper_Subscriber
    if ctrl == 1:
      wrist_cam_Subscriber = rospy.Subscriber("/camera/color/image_raw", Image, get_observations) # wrist rgb
      endpoint_state_Subscriber = rospy.Subscriber("/robot/limb/right/endpoint_state", EndpointState, get_gpos) # wrist rgb
      joint_states_Subscriber = rospy.Subscriber("/robot/joint_states", JointState, get_qpos) # wrist rgb
      right_gripper_Subscriber = rospy.Subscriber("/io/end_effector/right_gripper/state", IODeviceStatus, get_gripper_state) # wrist rgb
    else:
      wrist_cam_Subscriber.unregister()
      endpoint_state_Subscriber.unregister()
      joint_states_Subscriber.unregister()
      right_gripper_Subscriber.unregister()
      
  def go_to_initial_position():
    
    print("going to the initial position...")
    
    gripper.open()
    init_positions=[0, -1.046, 0, 1.046, 0, 1.57, -1.48]
    start_angles = dict(zip(joints_name, init_positions))
    
    limb.set_joint_position_speed(0.2)
    limb.move_to_joint_positions(start_angles,timeout=6.0)
    limb.set_joint_position_speed(0.1)
    head.set_pan(0.0)
    refresh_initial_pos()
    
  
  def go_to_waypoint0_position():
      nonlocal all_time_actions, timestep, tray_defaul_waypoint
      go_to_initial_position() 
      
      limb.move_to_joint_positions(tray_defaul_waypoint,timeout=8.0)
      
      print("you can use ' joint_position_keyboard.py ' random the waypoint0, \nand put the block in the gripper \nif ok then input '1'")
      done = False
      while not done and not rospy.is_shutdown(): 
        c = intera_external_devices.getch()
        if c in ['\x1b', '\x03']:
          done = True
          rospy.signal_shutdown("Example finished.")
        elif c == '1':
          tray_defaul_waypoint = limb.joint_angles()
          return
      
  if task_name == 'sorting_program_sawyer21':
    go_to_initial_position()
  elif task_name == 'sorting_program_sawyer22':
    # go_to_waypoint0_position()
    x = 1
  
  wrist_cam_Subscriber = rospy.Subscriber("/camera/color/image_raw", Image, get_observations) # wrist rgb
  endpoint_state_Subscriber = rospy.Subscriber("/robot/limb/right/endpoint_state", EndpointState, get_gpos) # wrist rgb
  joint_states_Subscriber = rospy.Subscriber("/robot/joint_states", JointState, get_qpos) # wrist rgb
  right_gripper_Subscriber = rospy.Subscriber("/io/end_effector/right_gripper/state", IODeviceStatus, get_gripper_state) # wrist rgb
  
  rospy.spin()  


def buil_model(ckpt_dir, ckpt_name):
  print("buil the model structure")
  from constants import SIM_TASK_CONFIGS
  task_config = SIM_TASK_CONFIGS[task_name]

  max_timesteps = task_config['episode_len']
  camera_names = task_config['camera_names']
  
  enc_layers = 4
  dec_layers = 7
  nheads = 8 
  policy_config = {'lr': learning_rate,
                    'chunk_size': 20,
                    'kl_weight': 10,
                    'hidden_dim': 512,
                    'dim_feedforward': 3200,
                    'lr_backbone': 1e-5,
                    'backbone': 'efficientnet_b0' if task_name == 'sorting_program_sawyer21' else 'efficientnet_b0',
                    'enc_layers': enc_layers,
                    'dec_layers': dec_layers,
                    'nheads': nheads,
                    'camera_names': camera_names,
                    }
  
  ckpt_path = os.path.join(ckpt_dir, ckpt_name)
  policy = ACTPolicy(policy_config)
  loading_status = policy.load_state_dict(torch.load(ckpt_path))
  print(loading_status)
  policy.cuda()
  policy.eval() 
  print(f'Loaded: {ckpt_path}')
  return policy, max_timesteps


def main():
  global target_color, box_color, IF_AUTO, task_name, ckpt_dir, ckpt_name
  global task_name_21, ckpt_dir_21, ckpt_name_21, task_name_22, ckpt_dir_22, ckpt_name_22
  parser = argparse.ArgumentParser()
  parser.add_argument("--target_color", type=str, default="red",
                      help="the target color to pick up")
  parser.add_argument("--box_color", type=str, default="red", help="the box color which to put down the picked target block")
  args = parser.parse_args()
  
  target_color = args.target_color
  box_color = args.box_color
  
  if target_color != None:
    IF_AUTO = True

  task_name_21 = 'sorting_program_sawyer21'
  task_name_22 = 'sorting_program_sawyer22'
  if target_color == 'red':
    ckpt_dir_21 = '/home/boxjod/RLBench_ACT_Sawyer/Trainings/sorting_program_sawyer211/50demo_36step_20chunk_8batch_efficientnet_b0'
    ckpt_name_21 = 'policy_best_epoch4000.pth'
    if box_color == 'red': # red to red
      ckpt_dir_22 = '/home/boxjod/RLBench_ACT_Sawyer/Trainings/sorting_program_sawyer221/50demo_77step_20chunk_8batch_efficientnet_b0'
      ckpt_name_22 = 'policy_best_epoch1000.pth'
      
  elif target_color == 'blue':
    ckpt_dir_21 = '/home/boxjod/RLBench_ACT_Sawyer/Trainings/sorting_program_sawyer212/50demo_36step_20chunk_8batch_efficientnet_b0'
    ckpt_name_21 = 'policy_best_epoch4000.pth'
    if box_color == 'green': # blue to green
      ckpt_dir_22 = '/home/boxjod/RLBench_ACT_Sawyer/Trainings/sorting_program_sawyer222/50demo_77step_20chunk_8batch_efficientnet_b0'
      ckpt_name_22 = 'policy_best_epoch1000.pth'
      
  elif target_color == 'green':
    ckpt_dir_21 = '/home/boxjod/RLBench_ACT_Sawyer/Trainings/sorting_program_sawyer213/50demo_36step_20chunk_8batch_efficientnet_b0'
    ckpt_name_21 = 'policy_best_epoch4000.pth'
    if box_color == 'red': # green to red
      ckpt_dir_22 = '/home/boxjod/RLBench_ACT_Sawyer/Trainings/sorting_program_sawyer223/50demo_77step_20chunk_8batch_efficientnet_b0'
      ckpt_name_22 = 'policy_best_epoch2000.pth'
      
  task_name = task_name_21
  ckpt_dir = ckpt_dir_21
  ckpt_name = ckpt_name_21

  
  print("Initializing node... ")
  rospy.init_node('verification_sawyer_node')
       
  print("Getting robot state... ", end=' ')
  rs = intera_interface.RobotEnable(CHECK_VERSION)
  init_state = rs.state().enabled
  print(f"{init_state}")
  
  def clean_shutdown():
    print("\nExiting example.")
  rospy.on_shutdown(clean_shutdown)
  
  if init_state == False:
    rospy.loginfo("Enabling robot...")
    rs.enable()
  
  policy, max_timesteps = buil_model(ckpt_dir, ckpt_name)
  observation_to_action(policy, max_timesteps, ckpt_dir)
  
  print("Done.")
  
if __name__ == '__main__':
    main()
