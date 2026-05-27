import rclpy
from rclpy.node import Node
import numpy as np
import torch
import os
from .gazebo_env import GazeboEnv
from .agent import QNetwork, STATE_SIZE, ACTION_SIZE

# Nota: Creiamo una classe di test che eredita da Node per gestire nativamente i parametri ROS 2
class SkidbotTestManager(Node):
    def __init__(self):
        super().__init__('skidbot_test_manager')
        
        # 1. Dichiara i parametri ROS 2 con i loro valori di default
        self.declare_parameter('model_path', 'models/ddqn_skidbot_ep3000.pth')
        self.declare_parameter('num_test_episodes', 5)
        self.declare_parameter('max_steps_per_episode', 500)
        
        # Nota: Il parametro 'test_world' viene solitamente passato al launch file di Gazebo, 
        # ma lo dichiariamo qui nel caso in cui la tua logica o i tuoi log debbano tracciarlo.
        self.declare_parameter('test_world', 'map2.world')

        # 2. Recupera i valori effettivi (quelli di default o quelli passati da riga di comando)
        self.model_path = self.get_parameter('model_path').get_parameter_value().string_value
        self.num_episodes = self.get_parameter('num_test_episodes').get_parameter_value().integer_value
        self.max_steps = self.get_parameter('max_steps_per_episode').get_parameter_value().integer_value
        self.test_world = self.get_parameter('test_world').get_parameter_value().string_value

def main():
    rclpy.init()
    
    # Inizializza il gestore dei parametri di test
    test_manager = SkidbotTestManager()
    
    # Inizializza l'ambiente di simulazione Gazebo
    env = GazeboEnv()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[TEST] Uso del dispositivo: {device}")
    print(f"[TEST] World configurato: {test_manager.test_world}")
    print(f"[TEST] Episodi totali: {test_manager.num_episodes}")
    print(f"[TEST] Max passi per episodio: {test_manager.max_steps}")
    
    # Configura l'architettura della rete neurale
    policy_net = QNetwork(STATE_SIZE, ACTION_SIZE).to(device)
    
    # Carica i pesi dal percorso dinamico recuperato dai parametri
    if os.path.exists(test_manager.model_path):
        print(f"[TEST] Caricamento del modello da: {test_manager.model_path}")
        policy_net.load_state_dict(torch.load(test_manager.model_path, map_location=device))
        policy_net.eval()
    else:
        print(f"[TEST] Errore: Il file di modello '{test_manager.model_path}' non esiste.")
        env.destroy_node()
        test_manager.destroy_node()
        rclpy.shutdown()
        return

    print("\n--- Inizio Fase di Test dello Skidbot ---")

    for episode in range(1, test_manager.num_episodes + 1):
        state, start_coords = env.reset()
        episode_reward = 0
        steps = 0
        
        print(f"\nAvvio Episodio di Test {episode}...")

        for step in range(test_manager.max_steps):
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            
            with torch.no_grad():
                q_values = policy_net(state_tensor)
            
            action_idx = torch.argmax(q_values).item()
            next_state, reward, done, crash_coords = env.step(action_idx)
            
            state = next_state
            episode_reward += reward
            steps += 1
            
            if done:
                print(f"Episodio terminato per collisione dopo {steps} passi.")
                break
        
        print(f"Risultato Episodio {episode} -> Passi completati: {steps}, Reward Totale: {episode_reward}")

    print("\nFase di test completata.")
    env.destroy_node()
    test_manager.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()