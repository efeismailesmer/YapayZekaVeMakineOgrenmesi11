import pygame
import numpy as np
import random
import math
from collections import defaultdict

# Initialize pygame
pygame.init()

# Constants
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 400
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)

# CartPole physics parameters
GRAVITY = 9.8
CART_MASS = 1.0
POLE_MASS = 0.1
POLE_LENGTH = 0.5  # Half the pole's length
FORCE_MAG = 10.0
TIME_STEP = 0.02
THETA_THRESHOLD = 12 * math.pi / 180  # 12 degrees
X_THRESHOLD = 2.4

# Q-learning parameters
LEARNING_RATE = 0.1
DISCOUNT_FACTOR = 0.99
EXPLORATION_RATE = 1.0
MIN_EXPLORATION_RATE = 0.01
EXPLORATION_DECAY = 0.995

class CartPole:
    def __init__(self):
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("CartPole with AI")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont(None, 24)
        self.reset()
        
    def reset(self):
        # State: [cart_position, cart_velocity, pole_angle, pole_angular_velocity]
        self.state = [0.0, 0.0, 0.0, 0.0]
        self.steps = 0
        self.done = False
        return self._discretize_state(self.state)
    
    def step(self, action):
        # Extract state variables
        x, x_dot, theta, theta_dot = self.state
        
        # Apply force based on action (0 = left, 1 = right)
        force = -FORCE_MAG if action == 0 else FORCE_MAG
        
        # Physics calculations
        # Formula from: http://coneural.org/florian/papers/05_cart_pole.pdf
        costheta = math.cos(theta)
        sintheta = math.sin(theta)
        
        temp = (force + POLE_MASS * POLE_LENGTH * theta_dot**2 * sintheta) / (CART_MASS + POLE_MASS)
        theta_acc = (GRAVITY * sintheta - costheta * temp) / (POLE_LENGTH * (4.0/3.0 - POLE_MASS * costheta**2 / (CART_MASS + POLE_MASS)))
        x_acc = temp - POLE_MASS * POLE_LENGTH * theta_acc * costheta / (CART_MASS + POLE_MASS)
        
        # Update state with Euler integration
        x = x + TIME_STEP * x_dot
        x_dot = x_dot + TIME_STEP * x_acc
        theta = theta + TIME_STEP * theta_dot
        theta_dot = theta_dot + TIME_STEP * theta_acc
        
        self.state = [x, x_dot, theta, theta_dot]
        self.steps += 1
        
        # Check if episode is done
        self.done = bool(
            x < -X_THRESHOLD or
            x > X_THRESHOLD or
            theta < -THETA_THRESHOLD or
            theta > THETA_THRESHOLD
        )
        
        # Reward: 1 for each step survived
        reward = 0.0 if self.done else 1.0
        
        return self._discretize_state(self.state), reward, self.done
    
    def _discretize_state(self, state):
        """Convert continuous state to discrete for Q-learning"""
        x, x_dot, theta, theta_dot = state
        
        # Discretize each dimension into bins
        x_bins = np.linspace(-X_THRESHOLD, X_THRESHOLD, 10)
        x_dot_bins = np.linspace(-5, 5, 10)
        theta_bins = np.linspace(-THETA_THRESHOLD, THETA_THRESHOLD, 10)
        theta_dot_bins = np.linspace(-10, 10, 10)
        
        # Find bin indices
        x_idx = np.digitize(x, x_bins)
        x_dot_idx = np.digitize(x_dot, x_dot_bins)
        theta_idx = np.digitize(theta, theta_bins)
        theta_dot_idx = np.digitize(theta_dot, theta_dot_bins)
        
        return (x_idx, x_dot_idx, theta_idx, theta_dot_idx)
    
    def render(self, episode, total_reward):
        self.screen.fill(WHITE)
        
        # Draw cart
        cart_x = SCREEN_WIDTH // 2 + int(self.state[0] * 50)  # Scale position for display
        cart_y = SCREEN_HEIGHT // 2
        cart_width, cart_height = 50, 30
        pygame.draw.rect(self.screen, BLACK, [cart_x - cart_width//2, cart_y - cart_height//2, cart_width, cart_height])
        
        # Draw pole
        pole_length = int(POLE_LENGTH * 200)  # Scale length for display
        pole_end_x = cart_x + math.sin(self.state[2]) * pole_length
        pole_end_y = cart_y - math.cos(self.state[2]) * pole_length
        pygame.draw.line(self.screen, RED, (cart_x, cart_y), (pole_end_x, pole_end_y), 6)
        
        # Draw base line
        pygame.draw.line(self.screen, BLUE, (0, cart_y + cart_height//2), (SCREEN_WIDTH, cart_y + cart_height//2), 2)
        
        # Display info
        episode_text = self.font.render(f"Episode: {episode}", True, BLACK)
        steps_text = self.font.render(f"Steps: {self.steps}", True, BLACK)
        reward_text = self.font.render(f"Total Reward: {total_reward}", True, BLACK)
        
        self.screen.blit(episode_text, (10, 10))
        self.screen.blit(steps_text, (10, 40))
        self.screen.blit(reward_text, (10, 70))
        
        pygame.display.flip()
        self.clock.tick(60)

class QLearningAgent:
    def __init__(self, actions=[0, 1]):
        self.actions = actions
        self.q_table = defaultdict(lambda: np.zeros(len(actions)))
        self.exploration_rate = EXPLORATION_RATE
        
    def get_action(self, state):
        # Exploration: choose random action
        if random.random() < self.exploration_rate:
            return random.choice(self.actions)
        
        # Exploitation: choose best action from Q-table
        return np.argmax(self.q_table[state])
    
    def update_q_table(self, state, action, reward, next_state, done):
        # Q-learning update formula
        best_next_action = np.argmax(self.q_table[next_state])
        td_target = reward + (0 if done else DISCOUNT_FACTOR * self.q_table[next_state][best_next_action])
        td_error = td_target - self.q_table[state][action]
        self.q_table[state][action] += LEARNING_RATE * td_error
    
    def decay_exploration(self):
        self.exploration_rate = max(MIN_EXPLORATION_RATE, self.exploration_rate * EXPLORATION_DECAY)

def main():
    env = CartPole()
    agent = QLearningAgent()
    
    num_episodes = 1000
    max_steps = 500
    
    for episode in range(num_episodes):
        state = env.reset()
        total_reward = 0
        
        for step in range(max_steps):
            # Process pygame events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return
            
            # Get action from agent
            action = agent.get_action(state)
            
            # Take action in environment
            next_state, reward, done = env.step(action)
            total_reward += reward
            
            # Update Q-table
            agent.update_q_table(state, action, reward, next_state, done)
            
            # Render environment
            env.render(episode, total_reward)
            
            state = next_state
            
            if done:
                break
        
        # Decay exploration rate after each episode
        agent.decay_exploration()
        
        print(f"Episode: {episode}, Steps: {env.steps}, Total Reward: {total_reward}, Exploration Rate: {agent.exploration_rate:.4f}")
        
        # If we've solved the problem (balanced for 500 steps), slow down to watch
        if env.steps >= 500:
            pygame.time.delay(1000)  # Pause for a second to celebrate

if __name__ == "__main__":
    main()
    pygame.quit()