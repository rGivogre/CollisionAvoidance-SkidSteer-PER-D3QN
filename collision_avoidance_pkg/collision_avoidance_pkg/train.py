import rclpy
import numpy as np
import torch
import os  # Added to manage directory creation
from .gazebo_env import GazeboEnv
from .agent import DDQNAgent

# Parametri di addestramento (dal paper)
MAX_EPISODES = 3000
MAX_STEPS_PER_EPISODE = 500
SAVE_EVERY = 100  # Salva il modello ogni 100 episodi

def main():
    # 1. Inizializza il sistema ROS 2 e l'ambiente
    rclpy.init()
    env = GazeboEnv()
    
    # 2. Inizializza l'agente (DDQN)
    agent = DDQNAgent()
    
    # --- LOGGING INITIALIZATION ---
    reward_history = []
    action_history = []
    os.makedirs('logs', exist_ok=True)
    # ------------------------------
    
    print("--- Inizio Addestramento Skidbot (DDQN) ---")

    #3000
    for episode in range(1, MAX_EPISODES + 1):
        # 3. Reset dell'ambiente (Pseudocode: Initialize state s1)
        state = env.reset()
        episode_reward = 0
        episode_actions = []  # Track actions taken this episode

        #500
        for step in range(MAX_STEPS_PER_EPISODE):
            # 4. Scegli un'azione con epsilon-greedy (Pseudocode: Select action a_t)
            action_idx = agent.get_action(state)
            episode_actions.append(action_idx)  # Log action
            
            # 5. Esegui l'azione in Gazebo (Pseudocode: Execute a_t, observe r_t, s_t+1)
            next_state, reward, done = env.step(action_idx)
            
            # 6. Salva l'esperienza nella memoria (Pseudocode: Store transition)
            agent.store_transition(state, action_idx, reward, next_state, done)
            
            # 7. Addestra la rete (Pseudocode: Sample random minibatch & update)
            agent.train_step()
            
            state = next_state
            episode_reward += reward
            
            if done:
                break
        
        # 8. Aggiorna epsilon (esplorazione) dopo ogni episodio
        agent.update_epsilon()
        
        # --- APPEND EPISODE DATA ---
        reward_history.append(episode_reward)
        # Pad episode actions if the robot crashed early so it fits cleanly into a matrix later
        if len(episode_actions) < MAX_STEPS_PER_EPISODE:
            episode_actions.extend([-1] * (MAX_STEPS_PER_EPISODE - len(episode_actions)))
        action_history.append(episode_actions)
        # ---------------------------

        # Log dei progressi
        print(f"Episodio: {episode}/{MAX_EPISODES}, Reward: {episode_reward}, Epsilon: {agent.epsilon:.3f}")

        # 9. Salvataggio del modello e dei log data
        if episode % SAVE_EVERY == 0:
            torch.save(agent.policy_net.state_dict(), f"ddqn_skidbot_ep{episode}.pth")
            
            # Save data arrays for plot.py
            np.save('logs/rewards.npy', np.array(reward_history))
            np.save('logs/actions.npy', np.array(action_history))
            print(f"Modello e log salvati all'episodio {episode}")

    # Pulizia finale
    print("Addestramento completato.")
    env.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()