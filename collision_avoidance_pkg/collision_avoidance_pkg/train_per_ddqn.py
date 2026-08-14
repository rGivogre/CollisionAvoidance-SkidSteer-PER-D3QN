import json
import rclpy
import numpy as np
import torch
import os
import argparse
from datetime import datetime
from .gazebo_env import GazeboEnv
from .agents.PER_DDQN_agent import PERDDQNAgent

# Training parameters
MAX_EPISODES = 3000
MAX_STEPS_PER_EPISODE = 1100    
SAVE_EVERY = 250  

def main():
    parser = argparse.ArgumentParser(description="Skidbot PER-DDQN Training Node")
    parser.add_argument('--speed', type=float, default=0.3, help='Linear velocity')
    parser.add_argument('--learning_rate', type=float, default=2.5e-4, help='Learning rate')

    args, _ = parser.parse_known_args() 
    speed = args.speed
    lr = args.learning_rate

    rclpy.init()
    env = GazeboEnv(linear_speed=speed, lock_step=True, map_name='multi_maps')
    agent = PERDDQNAgent(learning_rate=lr)
    
    models_base_dir = 'models'
    plot_data_base_dir = 'plot_data'
    run_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_name = run_timestamp + "_PER" # Added suffix to easily identify PER runs
    
    models_dir = os.path.join(models_base_dir, run_name)
    plot_data_dir = os.path.join(plot_data_base_dir, run_name)
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(plot_data_dir, exist_ok=True)

    run_config = {
        'run_name': run_name,
        'start_time': run_timestamp,
        'agent_key': 'PER-DDQN',
        'speed': speed,
        'learning_rate': lr,
        'max_steps_per_episode': MAX_STEPS_PER_EPISODE,
        'save_every': SAVE_EVERY,
        'models_dir': models_dir,
        'plot_data_dir': plot_data_dir,
        'checkpoint_template': f"per_ddqn_ep{{episode:04d}}.pth",
    }

    for config_dir in [models_dir, plot_data_dir]:
        with open(os.path.join(config_dir, 'runconfig.json'), 'w') as config_file:
            json.dump(run_config, config_file, indent=2)

    reward_history = []
    action_history = []
    crash_history = []
    start_history = []
    epsilon_history = []
    print("--- Starting Skidbot Training (PER-DDQN) ---")

    try:
        for episode in range(1, MAX_EPISODES + 1):
            state, start_coords = env.reset()
            start_history.append(start_coords)
            episode_reward = 0
            episode_actions = [] 

            for step in range(MAX_STEPS_PER_EPISODE):
                action_idx = agent.get_action(state)
                episode_actions.append(action_idx) 
                
                next_state, reward, done, crash_coords = env.step(action_idx)
                
                agent.store_transition(state, action_idx, reward, next_state, done)
                agent.train_step()
                
                state = next_state
                episode_reward += reward
                
                if done:
                    crash_history.append(crash_coords) 
                    break
            
            reward_history.append(episode_reward)
            epsilon_history.append(agent.epsilon)
            
            if len(episode_actions) < MAX_STEPS_PER_EPISODE:
                episode_actions.extend([-1] * (MAX_STEPS_PER_EPISODE - len(episode_actions)))
            action_history.append(episode_actions)

            agent.update_epsilon()

            formatted_crash = f"({crash_coords[0]:.2f}, {crash_coords[1]:.2f})" if done else 'No'
            print(f"Episode: {episode}/{MAX_EPISODES}, Reward: {episode_reward:.2f}, Epsilon: {agent.epsilon:.3f}, Crash: {formatted_crash}")

            if (episode % SAVE_EVERY  == 0):
                model_path = os.path.join(models_dir, f"per_ddqn_ep{episode:04d}.pth")
                checkpoint = {
                    'model_state': agent.policy_net.state_dict(),
                    'optimizer_state': agent.optimizer.state_dict()
                }
                torch.save(checkpoint, model_path)
                
                # Use cpprb native save method instead of pickle
                memory_path = os.path.join(models_dir, f"per_ddqn_ep{episode:04d}_memory.npz")
                agent.memory.save_transitions(memory_path)
                
                np.save(os.path.join(plot_data_dir, 'rewards.npy'), np.array(reward_history))
                np.save(os.path.join(plot_data_dir, 'actions.npy'), np.array(action_history))
                np.save(os.path.join(plot_data_dir, 'crashes.npy'), np.array(crash_history))
                np.save(os.path.join(plot_data_dir, 'epsilons.npy'), np.array(epsilon_history))
                
                print(f"Model and logs saved at episode {episode}")

    except (KeyboardInterrupt, RuntimeError) as e:
        # FIXED TYPO: "convert" instead of "covert"
        if isinstance(e, RuntimeError) and "Unable to convert call argument" not in str(e):
            raise e
        print("\n--- Training interrupted by user (CTRL-C) ---")
        print("Saving full checkpoint and memory buffer before exiting...")
        
        interrupted_ep = episode if 'episode' in locals() else 0
        
        model_path = os.path.join(models_dir, f"per_ddqn_ep{interrupted_ep:04d}_interrupted.pth")
        checkpoint = {
            'model_state': agent.policy_net.state_dict(),
            'optimizer_state': agent.optimizer.state_dict()
        }
        torch.save(checkpoint, model_path)
        
        memory_path = os.path.join(models_dir, f"per_ddqn_ep{interrupted_ep:04d}_interrupted_memory.npz")
        agent.memory.save_transitions(memory_path)
            
        np.save(os.path.join(plot_data_dir, 'rewards.npy'), np.array(reward_history))
        np.save(os.path.join(plot_data_dir, 'actions.npy'), np.array(action_history))
        np.save(os.path.join(plot_data_dir, 'crashes.npy'), np.array(crash_history))
        np.save(os.path.join(plot_data_dir, 'epsilons.npy'), np.array(epsilon_history))
        
        print(f"Interrupted checkpoint safely stored at: {model_path}")

    finally:
        print("Cleaning up and terminating nodes safely...")
        env.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        print("Training session closed.")

if __name__ == '__main__':
    main()