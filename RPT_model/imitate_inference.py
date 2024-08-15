import torch
import numpy as np
import os
import pickle
import argparse
import matplotlib.pyplot as plt
from copy import deepcopy
from tqdm import tqdm
from einops import rearrange

from constants import DT
from constants import PUPPET_GRIPPER_JOINT_OPEN
from utils import load_data # data functions
from utils import compute_dict_mean, set_seed, detach_dict # helper functions
from policy import ACTPolicy, CNNMLPPolicy, DiffusionPolicy
from visualize_episodes import save_videos

from sim_env import BOX_POSE
from pyrep.errors import ConfigurationPathError
from command_script.command_utils import initialize_model_and_tokenizer, encode_text
import IPython
import cv2 as cv

e = IPython.embed

def main(args):
    
    np.set_printoptions(linewidth=300)
    set_seed(1)
    # command line parameters
    is_eval = args['eval']
    policy_class = args['policy_class']
    onscreen_render = args['onscreen_render']
    task_name = args['task_name']
    batch_size_train = args['batch_size'] 
    batch_size_val = args['batch_size']
    num_epochs = args['num_epochs']
    num_verification = args['num_verification']
    variation = args['variation']
    multi_gpu = args["multi_gpu"]
    
    if args["gpu"] is not None and not multi_gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = f"{args['gpu']}"
        assert torch.cuda.is_available()
    
    is_sim = True 
    if is_sim:
        from constants import SIM_TASK_CONFIGS
        task_config = SIM_TASK_CONFIGS[task_name]

    dataset_dir = task_config['dataset_dir']
    num_episodes = task_config['num_episodes'] * task_config['num_variation'] 
    print(f"{task_config['num_episodes']=}, {task_config['num_variation']=}, {num_episodes=}, ")
    
    episode_len = task_config['episode_len']
    camera_names = task_config['camera_names']
    
    commands = args["command"].split(",") if args["command"] else [] # task_config['commands'] # 
    use_language = args["use_language"]
    language_encoder = args["language_encoder"]
    
    # max_skill_len = (args["max_skill_len"] if args["max_skill_len"] is not None else episode_len)
    max_skill_len = episode_len
    action_dim = 8 # 7 + 1
        
    if 'ACT1' in policy_class: # qpos => gpos 1
        action_is_qpos = False
        use_gpos = False 
        state_dim =  8
        
    elif 'ACT2' in policy_class: # (qpos)+(gpos)=> gpos 2
        action_is_qpos = False
        use_gpos = True # True
        state_dim =  8 
        
    elif 'ACT3' in policy_class: #  # (qpos+qdiff)+(gpos+gdiff)=> gpos 3
        action_is_qpos = False 
        use_gpos = True 
        state_dim =  15 
    else: # basic model
        action_is_qpos = True
        use_gpos = False
        state_dim =  8 
    
    lr_backbone = 1e-5
    backbone = args['backbone']
    print("backbone:",backbone)
    if 'ACT' in policy_class: # policy_class == 'ACT':
        enc_layers = 4
        dec_layers = 7
        nheads = 8 
        policy_config = {'policy_class': policy_class,
                         'camera_names': camera_names,
                         'backbone': backbone,
                         'state_dim': state_dim,
                         'use_gpos': use_gpos,

                         'hidden_dim': args['hidden_dim'],
                         'dim_feedforward': args['dim_feedforward'],
                         'enc_layers': enc_layers,
                         'dec_layers': dec_layers,
                         'nheads': nheads,
                         
                         'lr': args['lr'],
                         'kl_weight': args['kl_weight'],
                         'lr_backbone': lr_backbone,
                         
                         'action_dim': action_dim,
                         'chunk_size': args['chunk_size'],
                         }
    elif policy_class == "Diffusion":
        policy_config = {
            "lr": args["lr"],
            'state_dim': state_dim,
            "camera_names": camera_names,
            'weight_decay': args['weight_decay'],
            "observation_horizon": 1,
            "action_horizon": 8,  # TODO not used
            "prediction_horizon": args["chunk_size"],
            "chunk_size": args["chunk_size"],
            "num_inference_timesteps": 10,
            "ema_power": 0.75,
            "vq": False,
            "backbone": backbone,
            "multi_gpu": multi_gpu,
            "is_eval": is_eval,
            'state_dim': state_dim,
            
            'action_dim': action_dim,
            'hidden_dim': args['hidden_dim'],
        }
    elif policy_class == "CNNMLP":
        policy_config = {
            "lr": args["lr"],
            "lr_backbone": lr_backbone,
            "backbone": backbone,
            "chunk_size": 1,
            "camera_names": camera_names,
            
            'state_dim': state_dim,
            'action_dim': action_dim,
            'hidden_dim': args['hidden_dim'],
        }
    else:
        raise NotImplementedError
    
    chunk_size = args['chunk_size']
    batch_size = args['batch_size']
    ckpt_dir = args['ckpt_dir'] + f'/{task_name}/{policy_class}_{num_episodes}demo_{episode_len}step_{chunk_size}chunk_{batch_size}batch_{backbone}'
    print(f"train on {camera_names}")
    config = {
        'num_epochs': num_epochs,
        'ckpt_dir': ckpt_dir,
        'episode_len': episode_len,
        'lr': args['lr'],
        'policy_class': policy_class,
        'onscreen_render': onscreen_render,
        'policy_config': policy_config,
        'task_name': task_name,
        'seed': args['seed'],
        'temporal_agg': args['temporal_agg'],
        'camera_names': camera_names,
        'real_robot': not is_sim,
        "use_language": use_language,
        "language_encoder": language_encoder,
        "max_skill_len": max_skill_len,
    }
    
    use_diff = False if state_dim == 8 else True
    print(f"{action_is_qpos=}, {use_gpos=}, {state_dim=}, {use_diff=}")
    if is_eval: 
        ckpt_names = [f'policy_best_epoch{num_epochs}.pth']
        results = []
        for ckpt_name in ckpt_names:
            success_rate, avg_return = eval_bc(config, ckpt_name, save_episode=True, num_verification=num_verification,variation=variation, use_diff=use_diff, action_is_qpos=action_is_qpos)
            results.append([ckpt_name, success_rate, avg_return])

        for ckpt_name, success_rate, avg_return in results:
            print(f'{ckpt_name}: {success_rate=} {avg_return=}')

    else:
        train_dataloader, val_dataloader, stats, _ = load_data(dataset_dir, num_episodes, camera_names, batch_size_train, batch_size_val, policy_class, 
                                                            max_len=max_skill_len, chunk_size=chunk_size, command_list=commands, use_language=use_language, language_encoder=language_encoder,
                                                            use_gpos=use_gpos, use_diff=use_diff, action_is_qpos=action_is_qpos)

        # save dataset stats
        if not os.path.isdir(ckpt_dir):
            os.makedirs(ckpt_dir)
            
        stats_path = os.path.join(ckpt_dir, f'dataset_stats.pkl') 
        with open(stats_path, 'wb') as f: 
            pickle.dump(stats, f)
            
        train_bc(train_dataloader, val_dataloader, config) 
    

def make_policy(policy_class, policy_config):
    if 'ACT' in policy_class: 
        policy = ACTPolicy(policy_config)
    elif policy_class == "Diffusion":
        policy = DiffusionPolicy(policy_config)
    elif policy_class == "CNNMLP":
        policy = CNNMLPPolicy(policy_config)
    else:
        raise NotImplementedError
    return policy

def make_optimizer(policy_class, policy):
    if 'ACT' in policy_class:
        optimizer = policy.configure_optimizers()
    elif policy_class == "Diffusion":
        optimizer = policy.configure_optimizers()
    elif policy_class == "CNNMLP":
        optimizer = policy.configure_optimizers()
    else:
        raise NotImplementedError
    return optimizer

from rlbench.backend.utils import float_array_to_rgb_image
from rlbench.backend.const import DEPTH_SCALE
def get_image(ts, camera_names, policy_class): 
    curr_images = []
    
    # print(f'{camera_names=}')
    wrist_rgb = ts.wrist_rgb 
    if "ACT" in policy_class: # and policy_class!= "ACT0E0":
        wrist_rgb = cv.resize(wrist_rgb, (0, 0), fx=0.25, fy=0.25, interpolation=cv.INTER_AREA)
    
    
    curr_image = rearrange(wrist_rgb, 'h w c -> c h w')
    curr_images.append(curr_image)    
    
    # if "wrist_depth" in camera_names: # not recommend
    #     # wrist_depth = float_array_to_rgb_image(ts.wrist_depth, scale_factor=DEPTH_SCALE)
    #     # wrist_depth = np.clip(np.array(wrist_depth), 0, 255).astype(np.uint8)
    #     import cv2
    #     wrist_depth0 = ts.wrist_depth*255.0*5
    #     wrist_depth = cv2.applyColorMap(cv2.convertScaleAbs(wrist_depth0, alpha=1), cv2.COLORMAP_JET)
        
    #     curr_image = rearrange(wrist_depth, 'h w c -> c h w')
    #     curr_images.append(curr_image)
        
    if "head" in camera_names:    
        head_rgb = ts.head_rgb
        if "ACT" in policy_class: # and policy_class!= "ACT0E0":
            head_rgb = cv.resize(head_rgb, (0, 0), fx=0.25, fy=0.25, interpolation=cv.INTER_AREA)
        curr_image = rearrange(ts.head_rgb, 'h w c -> c h w')
        
        curr_images.append(curr_image)
        
    curr_image = np.stack(curr_images, axis=0)
    curr_image = torch.from_numpy(curr_image / 255.0).float().cuda().unsqueeze(0)
    return curr_image

def generate_command_embedding(command, t, language_encoder, tokenizer, model, instructor=None):

    command_embedding = encode_text(command, language_encoder, tokenizer, model)
    command_embedding = torch.tensor(command_embedding).cuda()
    if instructor is not None:
        command_embedding = instructor.get_nearest_embedding(command_embedding)[0]
    return command_embedding
    
    
def eval_bc(config, ckpt_name, save_episode=True, num_verification=50, variation=0, use_gpos=True, use_diff=True, action_is_qpos=False):
    seed = 10
    set_seed(seed)
    ckpt_dir = config['ckpt_dir']
    state_dim = config['policy_config']['state_dim']
    action_dim = config['policy_config']['action_dim']
    
    real_robot = config['real_robot']
    policy_class = config['policy_class']
    policy_config = config['policy_config']
    camera_names = config['camera_names']
    max_timesteps = config['episode_len']
    task_name = config['task_name']
    temporal_agg = config['temporal_agg']
    
    hidden_dim = policy_config['hidden_dim']
    use_language = config["use_language"]
    language_encoder = config["language_encoder"]
    max_skill_len = config["max_skill_len"]
    if use_language:
        tokenizer, model = initialize_model_and_tokenizer(language_encoder)
        assert tokenizer is not None and model is not None
    
    # load policy and stats
    ckpt_path = os.path.join(ckpt_dir, ckpt_name)
    policy = make_policy(policy_class, policy_config)
    loading_status = policy.load_state_dict(torch.load(ckpt_path))
    ckpt_name0 =ckpt_name.split('.')[0]
    
    print(loading_status)
    policy.cuda()
    policy.eval() 
    print(f'Loaded: {ckpt_path}')
    stats_path = os.path.join(ckpt_dir, f'dataset_stats.pkl')
    with open(stats_path, 'rb') as f:
        stats = pickle.load(f)
    
    pre_process_qpos = lambda s_qpos: (s_qpos - stats['qpos_mean']) / stats['qpos_std']
    pre_process_gpos = lambda s_gpos: (s_gpos - stats['gpos_mean']) / stats['gpos_std']
    pre_process_action_history = lambda s_action: (s_action - stats['action_mean']) / stats['action_std']
    
    if policy_class == "Diffusion":
        post_process = (lambda a: ((a + 1) / 2) * (stats["action_max"] - stats["action_min"]) + stats["action_min"])
    else: 
        post_process = lambda a: a * stats['action_std'] + stats['action_mean']
    
    # load environment
    if not real_robot:
        from sim_env import make_sim_env
        env = make_sim_env(task_name, policy_class)
        print("random seed = ", seed)
        env_max_reward = 1 # env.task.max_rewardz
    query_frequency = policy_config['chunk_size']
    if temporal_agg:
        query_frequency = 1
        chunk_size = policy_config['chunk_size']

    max_timesteps = int(max_timesteps * 1.3) # may increase for real-world tasks

    num_rollouts = num_verification 
    command_list = []
    command_embedding = None
    episode_returns = []
    highest_rewards = []
    for rollout_id in range(num_rollouts):
        gripper_flag = 0
        
        if variation >= 0:
            env.set_variation(variation) 
        else:
            random_variation = np.random.randint(3)
            env.set_variation(random_variation) 
        descriptions, ts_obs = env.reset() 

        ### evaluation loop
        if temporal_agg:
            all_time_actions = torch.zeros([max_timesteps, max_timesteps+chunk_size, action_dim]).cuda()

        qpos_history = torch.zeros((1, max_timesteps, state_dim)).cuda()
        gpos_history = torch.zeros((1, max_timesteps, state_dim)).cuda()
        
        image_list = [] # for visualization
        qpos_list = []
        gpos_list = []
        target_gpos_list = []
        rewards = []
        
        history_action = np.zeros((chunk_size,) + (action_dim,), dtype=np.float32)
        history_image_feature = np.zeros((2,chunk_size,) + (hidden_dim,), dtype=np.float32)
        
        with torch.inference_mode():
            path = []
            t = 0
            for timestep in range(max_timesteps):
                obs = ts_obs

                if use_diff and timestep == 0:
                    qpos_initial = obs.joint_positions
                    gpos_initial = obs.gripper_pose
                
                is_pad_history = np.zeros(max_timesteps)
                is_pad_history[timestep:] = 1
                is_pad_history = torch.from_numpy(is_pad_history).bool().cuda()
                
                if(rollout_id%5 == 0):
                    # image_list.append({'wrist':obs.wrist_rgb, 'head':obs.head_rgb, })
                    image_list.append({'front':obs.front_rgb, 'head':obs.head_rgb, 'wrist':obs.wrist_rgb})
                
                qpos_numpy = np.array(np.append(obs.joint_positions, obs.gripper_open)) # 7 + 1 = 8
                gpos_numpy = np.array(np.append(obs.gripper_pose, obs.gripper_open)) # 7 + 1 = 8
                
                if use_diff:
                    qpos_diff = [a - b for a,b in zip(obs.joint_positions, qpos_initial)]
                    qpos_numpy = np.array(np.append(qpos_numpy, qpos_diff)) # 7 + 1 + 7 = 15

                    gpos_diff = [a - b for a,b in zip(obs.gripper_pose, gpos_initial)] 
                    gpos_numpy = np.array(np.append(gpos_numpy, gpos_diff)) # 7 + 1 + 7 = 15
                
                qpos = pre_process_qpos(qpos_numpy)
                qpos = torch.from_numpy(qpos).float().cuda().unsqueeze(0)
                qpos_history[:, t] = qpos

                gpos = pre_process_gpos(gpos_numpy)
                gpos = torch.from_numpy(gpos).float().cuda().unsqueeze(0)
                gpos_history[:, t] = gpos
                
                history_action_numpy = np.array(history_action)
                history_action_numpy = pre_process_action_history(history_action_numpy)
                history_action_numpy = torch.from_numpy(history_action_numpy).float().cuda()
                
                curr_image = get_image(obs, camera_names, policy_class) 
                
                ### query policy
                if 'ACT' in config['policy_class'] or 'Diffusion' in config["policy_class"]:  # config['policy_class'] == "ACT":
                    if t % query_frequency == 0:
                        
                        if use_language and (t % max_skill_len == 0) :
                            # Check if an intervention is needed; if so, language correction
                            command = descriptions[0] 
                            command_embedding = generate_command_embedding(command, t, language_encoder, tokenizer, model)
                            # print(command_embedding)
                        
                        if not use_gpos:
                            gpos = None    
                        
                        if 'ACT' in config['policy_class']:
                            all_actions, image_feature = policy(qpos, gpos, curr_image, 
                                                history_image_feature, history_action_numpy, is_pad_history=is_pad_history, 
                                                actions=None, is_pad_action=None, command_embedding=command_embedding) 
                        if 'Diffusion' in config['policy_class']:
                             all_actions = policy(curr_image, None, qpos)
                             

                        language_correction = False
                        if use_language:
                            prefix = "user" if language_correction else "prediction"
                            command_list.append(f"{prefix}: {command}")
                        
                    if temporal_agg: # Action Chunking
                        all_time_actions[[t], t:t+chunk_size] = all_actions
                        actions_for_curr_step = all_time_actions[:, t]
                        
                        actions_populated = torch.all(actions_for_curr_step != 0, axis=1)
                        actions_for_curr_step = actions_for_curr_step[actions_populated]
                        k = 0.01
                        exp_weights = np.exp(-k * np.arange(len(actions_for_curr_step)))
                        exp_weights = exp_weights / exp_weights.sum() 
                        exp_weights = torch.from_numpy(exp_weights).cuda().unsqueeze(dim=1)
                        
                        raw_action = (actions_for_curr_step * exp_weights).sum(dim=0, keepdim=True)
                        
                    else: 
                        raw_action = all_actions[:, t % query_frequency]
                        
                elif config['policy_class'] == "CNNMLP":
                    raw_action = policy(qpos, curr_image) 
                else:
                    raise NotImplementedError
                
                ### post-process actions
                raw_action = raw_action.squeeze(0).cpu().numpy()
                action = post_process(raw_action)  
                # target_qpos = action
                
                if 'ACT' in config['policy_class']:
                    history_action = np.insert(history_action, 0, action, axis=0)[:chunk_size]
                    history_image_feature[0] = np.insert(history_image_feature[0], 0, image_feature[0], axis=0)[:chunk_size]
                    history_image_feature[1] = np.insert(history_image_feature[1], 0, image_feature[1], axis=0)[:chunk_size]

                if action_is_qpos:
                    ts_obs, reward, terminate = env.step(action) # qpos could deal with gripper command
                    qpos_list.append(qpos_numpy)
                    target_gpos_list.append(action)
                    rewards.append(reward) 
                else:
                    try:
                        next_gripper_position = action[0:3] # next 
                        next_gripper_quaternion = action[3:7]
                        gripper_state = action[7]
                        path.append(env._robot.arm.get_linear_path(position=next_gripper_position, quaternion=next_gripper_quaternion, steps=10, relative_to=env._robot.arm, ignore_collisions=True))
                        
                        # deal with gripper
                        if gripper_state < 0.6 :
                            gripper_state = 0
                            for g_obj in env._task.get_graspable_objects():
                                env._robot.gripper.grasp(g_obj)
                                    
                            if gripper_flag < 2 : # 
                                gripper_flag = gripper_flag + 1 
                                # print("attach the target")
                                
                                # clear history information
                                if gripper_flag==2:
                                    history_action = np.zeros((chunk_size,) + (action_dim,), dtype=np.float32)
                                    history_image_feature = np.zeros((2,chunk_size,) + (hidden_dim,), dtype=np.float32)
                                    qpos_initial = obs.joint_positions
                                    gpos_initial = obs.gripper_pose

                        elif gripper_state >= 0.6 : 
                            gripper_state = 1
                            if gripper_flag >= 2:
                                gripper_flag = gripper_flag - 1
                                # print("release the target")
                                env._robot.gripper.release() 
                                # clear history information
                                history_action = np.zeros((chunk_size,) + (action_dim,), dtype=np.float32)
                                history_image_feature = np.zeros((2,chunk_size,) + (hidden_dim,), dtype=np.float32)
                                qpos_initial = obs.joint_positions
                                gpos_initial = obs.gripper_pose
                        
                        # print(f'action = {gripper_state=}')
                        env._action_mode.gripper_action_mode.action(env._scene, np.array((gripper_state,)))

                        path[t].visualize() 
                        
                        done = False 
                        while done != 1: 
                            done = path[t].step()
                            env._scene.step() 
                        
                        ts_obs = env._scene.get_observation()
                        reward, _ = env._task.success() 
                        qpos_list.append(qpos_numpy)
                        gpos_list.append(gpos_numpy)
                        target_gpos_list.append(action)
                        rewards.append(reward) 

                    except ConfigurationPathError: 
                        print("ConfigurationPathError ", t) # , "path lens: ",len(path))
                        break 

                if reward == env_max_reward:
                    break # if already success, directly break this episode to speed up the eval process

                t = t + 1
                
            plt.close()
            
        # for i in range(t+1): # clear the path history
        #     path[i].clear_visualization()
        rewards = np.array(rewards) # 
        episode_return = np.sum(rewards[rewards!=None])
        episode_returns.append(episode_return)
        episode_highest_reward = np.max(rewards)  
        highest_rewards.append(episode_highest_reward)
        # print(f'Rollout {rollout_id}\n{episode_return=}, {episode_highest_reward=}, {env_max_reward=}, Success: {episode_highest_reward==env_max_reward}')
        print(f'{rollout_id} Rollout with {t} steps for [{descriptions[0]}]: {episode_highest_reward==env_max_reward}')
        # if(rollout_id % 5 == 0): 
        #     if save_episode:
        #         save_videos(image_list, DT, video_path=os.path.join(ckpt_dir, f'video_{ckpt_name0}_{rollout_id}_{episode_highest_reward==env_max_reward}.mp4'))

    success_rate = np.mean(np.array(highest_rewards) == env_max_reward) 
    avg_return = np.mean(episode_returns) 
    summary_str = f'\nSuccess rate: {success_rate}\nAverage return: {avg_return}\n\n'
    for r in range(env_max_reward+1):
        more_or_equal_r = (np.array(highest_rewards) >= r).sum()
        more_or_equal_r_rate = more_or_equal_r / num_rollouts
        summary_str += f'Reward >= {r}: {more_or_equal_r}/{num_rollouts} = {more_or_equal_r_rate*100}%\n'

    print(summary_str)

    # save success rate to txt
    result_file_name = 'result_' + ckpt_name0 + f'({more_or_equal_r_rate*100}%).txt'
    with open(os.path.join(ckpt_dir, result_file_name), 'w') as f:
        f.write(summary_str)
        # f.write(repr(episode_returns))
        f.write('\n\n')
        f.write(repr(highest_rewards)) 

    return success_rate, avg_return


def forward_pass(data, policy, policy_class=None, use_gpos=True, use_language=False):
    
    if use_language:  # use_language
        image_data, qpos_data, gpos_data, history_images_data, history_action_data, is_pad_history, action_data, is_pad_action, command_embedding = data
        command_embedding = command_embedding.cuda()
    else: # len = 8
        image_data, qpos_data, gpos_data, history_images_data, history_action_data, is_pad_history, action_data, is_pad_action = data
        command_embedding = None
        
    image_data, qpos_data, gpos_data, = image_data.cuda(), qpos_data.cuda(), gpos_data.cuda()
    history_images_data, history_action_data, is_pad_history = history_images_data.cuda(),history_action_data.cuda(), is_pad_history.cuda()
    action_data, is_pad_action = action_data.cuda(), is_pad_action.cuda()
    
    if use_gpos:
        return policy(qpos_data, gpos_data, image_data, history_images_data, history_action_data, is_pad_history, action_data, is_pad_action, command_embedding) # TODO remove None
    else:
        gpos_data = None
        if 'ACT' in policy_class:
            return policy(qpos_data, gpos_data, image_data, history_images_data, history_action_data, is_pad_history, action_data, is_pad_action, command_embedding) # TODO remove None 
        elif 'Diffusion' in policy_class:
            return policy(image_data, None, qpos_data, action_data, is_pad_action)
        
        elif 'CNNMLP' in policy_class:
            return policy(qpos_data, image_data, action_data, is_pad_action, command_embedding)


def train_bc(train_dataloader, val_dataloader, config):
    num_epochs = config['num_epochs']
    ckpt_dir = config['ckpt_dir']
    seed = config['seed']
    policy_class = config['policy_class']
    policy_config = config['policy_config']
    if "ACT" in policy_class:
        use_gpos = config['policy_config']['use_gpos']
    else:
        use_gpos = False
    use_language = config["use_language"]
    set_seed(seed)
    
# 1. make policy
    policy = make_policy(policy_class, policy_config)
    policy.cuda()
    optimizer = make_optimizer(policy_class, policy)

    # load model ckpt
    start_epoch = 0
    min_val_loss = np.inf
    train_history = []
    validation_history = []
    for last_history_epoch in range(num_epochs-2,-1,-1):
        ckpt_path = os.path.join(ckpt_dir, f'policy_epoch{last_history_epoch + 1}_seed{seed}.ckpt')
         
        if os.path.exists(ckpt_path): # Load the history trained weights of epoch
            print(f'Load the history trained weights of epoch={last_history_epoch+1}')
            checkpoint = torch.load(ckpt_path, map_location=torch.device('cpu'))
            policy.load_state_dict(checkpoint['net']) 
            optimizer.load_state_dict(checkpoint['optimizer'])  
            start_epoch = checkpoint['epoch']  
            min_val_loss = checkpoint['min_val_loss']
            train_history = checkpoint['train_history']
            validation_history = checkpoint['validation_history']
            start_epoch = start_epoch + 1
            save_1000 = 0
            break 
    
    best_ckpt_info = None 
    # 2. do train epoch
    epoch = start_epoch
    for epoch in tqdm(range(start_epoch, num_epochs)): 
        
        # 2.1 validation and summary the last epoch：
        with torch.inference_mode():
            policy.eval() 
            epoch_dicts = []

            for batch_idx, data in enumerate(val_dataloader):
                forward_dict = forward_pass(data, policy, policy_class=policy_class, use_gpos=use_gpos, use_language=use_language) 
                epoch_dicts.append(forward_dict)
            
            # print(f'{forward_dict=}')
            epoch_summary = compute_dict_mean(epoch_dicts) 
            validation_history.append(epoch_summary)

            epoch_val_loss = epoch_summary['loss']
            if epoch_val_loss < min_val_loss:
                min_val_loss = epoch_val_loss 
                best_ckpt_info = (epoch, min_val_loss, deepcopy(policy.state_dict()))
        
        # 2.2. training epoch 
        policy.train() 
        optimizer.zero_grad()
        
        for batch_idx, data in enumerate(train_dataloader):
            forward_dict = forward_pass(data, policy, policy_class=policy_class, use_gpos=use_gpos, use_language=use_language) 
            loss = forward_dict['loss'] 
            loss.backward() 
            
            optimizer.step() 
            optimizer.zero_grad()
            
            train_history.append(detach_dict(forward_dict)) 
    
    # save the policy weight for continue train
    ckpt_path = os.path.join(ckpt_dir, f'policy_epoch{epoch + 1}_seed{seed}.ckpt')
    checkpoint = {
        "net": policy.state_dict(),
        'optimizer':optimizer.state_dict(),
        "epoch": epoch,
        "min_val_loss": min_val_loss,
        "train_history": train_history,
        "validation_history": validation_history
    }
    torch.save(checkpoint, ckpt_path)
    
    # save best checkpoint
    best_epoch, min_val_loss, best_state_dict = best_ckpt_info  
    pth_path = os.path.join(ckpt_dir, f'policy_best_epoch{epoch + 1}.pth')
    torch.save(best_state_dict, pth_path) 
    
    print(f'Training finished:\nSeed {seed}, val loss {min_val_loss:.6f} at epoch {best_epoch}')
    # save training curves
    plot_history(train_history, validation_history, num_epochs, ckpt_dir, seed)
    return best_ckpt_info


def plot_history(train_history, validation_history, num_epochs, ckpt_dir, seed):
    # save training curves
    for key in train_history[0]:
        plot_path = os.path.join(ckpt_dir, f'trainval_{key}_epoch_seed{seed}.png') # {num_epochs}
        
        plt.figure()
        train_values = [summary[key].item() for summary in train_history]
        val_values = [summary[key].item() for summary in validation_history]
        
        plt.plot(np.linspace(0, num_epochs-1, len(train_history)), train_values, label='train')
        plt.plot(np.linspace(0, num_epochs-1, len(validation_history)), val_values, label='validation')
        
        # plt.ylim([-0.1, 1])
        plt.tight_layout()
        plt.legend()
        plt.title(key)
        plt.savefig(plot_path)
        
    print(f'Saved plots to {ckpt_dir}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--num_verification',  default=50, type=int, help='number of verification') 
    parser.add_argument('--onscreen_render', action='store_true') 
    parser.add_argument('--ckpt_dir', action='store', type=str, help='ckpt_dir', required=True)
    
    parser.add_argument('--policy_class', action='store', type=str, help='policy_class, capitalize', required=True) 
    parser.add_argument('--task_name', action='store', type=str, help='task_name', required=True)
    
    parser.add_argument('--batch_size', action='store', type=int, help='batch_size', required=True) 
    parser.add_argument('--seed', action='store', type=int, help='seed', required=True)
    parser.add_argument('--num_epochs', action='store', type=int, help='num_epochs', required=True) 
    parser.add_argument('--lr', action='store', type=float, help='lr', required=True) # learning rate 

    # for ACT
    parser.add_argument('--kl_weight', action='store', type=int, help='KL Weight', required=False)
    parser.add_argument('--chunk_size', action='store', type=int, help='chunk_size', required=False) 
    parser.add_argument('--hidden_dim', action='store', type=int, help='hidden_dim', required=False) 
    parser.add_argument('--dim_feedforward', action='store', type=int, help='dim_feedforward', required=False) 
    parser.add_argument('--temporal_agg', action='store_true')

    parser.add_argument("--backbone", type=str, default='resnet18', choices=['resnet18', 'resnet34', 'resnet50', 'efficientnet_b0', 'efficientnet_b3', 'resnet18film', 'resnet34film', 'resnet50film','efficientnet_b0film', 'efficientnet_b3film', 'efficientnet_b5film'], help="Which image encoder to use for the BC policy.")
    
    # for LLMs
    parser.add_argument('--command', action='store', type=str, help='comma-separated list of commands', default='', required=False)
    parser.add_argument('--use_language', action='store_true')
    parser.add_argument('--language_encoder', action='store', type=str, choices=['distilbert', 'clip'], default='distilbert', help='Type of language encoder to use: distilbert or clip', required=False)
    
    # variation
    parser.add_argument('--variation', action='store', type=int, default=0, help='the variations of the task', required=False)
    
    # for gpu
    parser.add_argument('--gpu', action='store', type=int, help='gpu', default=0, required=False)
    parser.add_argument('--multi_gpu', action='store_true')
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    main(vars(parser.parse_args()))
