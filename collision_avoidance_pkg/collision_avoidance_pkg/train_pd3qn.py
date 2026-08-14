import json
import rclpy
import numpy as np
import torch
import os
import argparse
import pickle
from datetime import datetime
from .gazebo_env import GazeboEnv
from .agents.PER_D3QN_agent import PERD3QNAgent

# Training parameters (from the paper)
MAX_EPISODES = 3000
MAX_STEPS_PER_EPISODE = 1100    # With each step lasting 0.3s (because we wait for 3 new scans after each action), this allows for a max episode duration of around 5.5 minutes.
SAVE_EVERY = 250  # Save the model every x episodes

def main():
    parser = argparse.ArgumentParser(description="Skidbot PER-D3QN Training Node") # Aggiornato in PER-D3QN
    parser.add_argument('--speed', type=float, default=0.3, help='Linear velocity (LINEAR_SPEED) for the robot navigation')
    parser.add_argument('--learning_rate', type=float, default=2.5e-4, help='Learning rate for the neural network')

    args, _ = parser.parse_known_args() # Extract known args so it doesn't crash on ROS 2 internal arguments
    speed = args.speed
    lr = args.learning_rate

    # Initialize ROS 2 system, environment node, and the PER-D3QN agent
    rclpy.init()
    env = GazeboEnv(linear_speed=speed, lock_step=True, map_name='multi_maps')
    agent = PERD3QNAgent(learning_rate=lr) # Istanza modificata in PERD3QNAgent
    
    # Setup project directories for saving models and plot data (assuming execution from repo root)
    models_base_dir = 'models'
    plot_data_base_dir = 'plot_data'
    
    # Generate unique subfolder names based on current timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    models_dir = os.path.join(models_base_dir, f"per_d3qn_{timestamp}") # Aggiornato nome cartella
    plot_data_dir = os.path.join(plot_data_base_dir, f"per_d3qn_{timestamp}") # Aggiornato nome cartella
    
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(plot_data_dir, exist_ok=True)
    
    # Save training hyperparameters for full experiment traceability
    hyperparams = {
        "max_episodes": MAX_EPISODES,
        "max_steps_per_episode": MAX_STEPS_PER_EPISODE,
        "learning_rate": lr,
        "linear_speed": speed
    }
    with open(os.path.join(plot_data_dir, 'hyperparameters.json'), 'w') as f:
        json.dump(hyperparams, f, indent=4)

    # Initialize tracking lists for evaluating agent performance
    reward_history = []
    action_history = []
    crash_history = []
    epsilon_history = []
    
    print("\n==============================================")
    print("      STARTING PER-D3QN TRAINING EXPERIMENT    ") # Aggiornato log di logica
    print("==============================================\n")
    
    try:
        for episode in range(1, MAX_EPISODES + 1):
            state, start_coords = env.reset()
            episode_reward = 0
            episode_actions = []
            
            for step in range(1, MAX_STEPS_PER_EPISODE + 1):
                # Request action from agent based on current environmental scan state
                action = agent.get_action(state)
                episode_actions.append(action)
                
                # Execute action in gazebo environment
                next_state, reward, done, crash = env.step(action)
                
                # Buffer the transition tuple in prioritized replay memory
                agent.store_transition(state, action, reward, next_state, done)
                
                # Execute a single gradient optimization step
                agent.train_step()
                
                episode_reward += reward
                state = next_state
                
                if done:
                    # Log the definitive outcome status of the episode
                    crash_history.append(1 if crash else 0)
                    break
            else:
                # If loop finishes without breaking, max steps were hit without crashing
                crash_history.append(0)
                
            # Decay exploration rate at the end of every episode
            agent.update_epsilon()
            
            # Save episode metrics
            reward_history.append(episode_reward)
            action_history.append(episode_actions)
            epsilon_history.append(agent.epsilon)
            
            # Print periodic training updates to console
            print(f"Episode {episode:04d}/{MAX_EPISODES} | Reward: {episode_reward:7.2f} | "
                  f"Steps: {step:04d} | Epsilon: {agent.epsilon:.4f} | Terminal: {'CRASH' if crash_history[-1] else 'TIMEOUT'}")
            
            # Periodically save model checkpoints and snapshot metrics
            if episode % SAVE_EVERY == 0:
                model_path = os.path.join(models_dir, f"per_d3qn_ep{episode:04d}.pth") # Aggiornato nome file pth
                checkpoint = {
                    'model_state': agent.policy_net.state_dict(),
                    'optimizer_state': agent.optimizer.state_dict()
                }
                torch.save(checkpoint, model_path)
                print(f"--> Saved checkpoint at episode {episode} to {model_path}")
                
                # Interim saving of plot metrics to avoid data loss
                np.save(os.path.join(plot_data_dir, 'rewards.npy'), np.array(reward_history))
                np.save(os.path.join(plot_data_dir, 'actions.npy'), np.array(action_history, dtype=object))
                np.save(os.path.join(plot_data_dir, 'crashes.npy'), np.array(crash_history))
                np.save(os.path.join(plot_data_dir, 'epsilons.npy'), np.array(epsilon_history))

        print("\n==============================================")
        print("          TRAINING COMPLETED SUCCESSFULLY      ")
        print("==============================================\n")

    except KeyboardInterrupt:
        print("\n--- Training interrupted by user (CTRL-C) ---")
        print("Saving full checkpoint and memory buffer before exiting...")
        
        # Determine current episode
        interrupted_ep = episode if 'episode' in locals() else 0
        
        # Save model and optimizer
        model_path = os.path.join(models_dir, f"per_d3qn_ep{interrupted_ep:04d}_interrupted.pth") # Aggiornato nome file pth
        checkpoint = {
            'model_state': agent.policy_net.state_dict(),
            'optimizer_state': agent.optimizer.state_dict()
        }
        torch.save(checkpoint, model_path)
        
        # Save Replay Buffer
        memory_path = os.path.join(models_dir, f"per_d3qn_ep{interrupted_ep:04d}_interrupted_memory.pkl") # Aggiornato nome file pkl
        with open(memory_path, 'wb') as f:
            pickle.dump(agent.memory, f)
            
        # Save final history logs
        np.save(os.path.join(plot_data_dir, 'rewards.npy'), np.array(reward_history))
        np.save(os.path.join(plot_data_dir, 'actions.npy'), np.array(action_history, dtype=object))
        np.save(os.path.join(plot_data_dir, 'crashes.npy'), np.array(crash_history))
        np.save(os.path.join(plot_data_dir, 'epsilons.npy'), np.array(epsilon_history))
        
        print(f"Interrupted checkpoint safely stored at: {models_dir}")
        
    finally:
        # Shutdown environment node and standard ROS 2 stack
        env.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()