import pygame
import numpy as np
import random
import pickle
import os
import time
from collections import defaultdict

# Initialize pygame
pygame.init()

# Constants
SCREEN_WIDTH = 600
SCREEN_HEIGHT = 750  # Increased height for the new button
BOARD_SIZE = 3
CELL_SIZE = 150
MARGIN = 75

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (200, 200, 200)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
GREEN = (0, 255, 0)
PURPLE = (128, 0, 128)
ORANGE = (255, 165, 0)

# Q-learning parameters
LEARNING_RATE = 0.1
DISCOUNT_FACTOR = 0.9
EXPLORATION_RATE = 0.3
MIN_EXPLORATION_RATE = 0.01
EXPLORATION_DECAY = 0.999

class TicTacToe:
    def __init__(self):
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("Tic-Tac-Toe with Learning AI")
        self.font = pygame.font.SysFont(None, 36)
        self.small_font = pygame.font.SysFont(None, 24)
        self.reset()
        self.auto_training = False
        self.auto_train_games = 0
        self.auto_train_start_time = 0
        
    def reset(self):
        # 0 = empty, 1 = X (player), 2 = O (AI)
        self.board = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=int)
        self.current_player = 1  # Player starts
        self.winner = None
        self.game_over = False
        self.moves_history = []  # To track states and actions for learning
        
    def get_state(self):
        # Convert board to a tuple for dictionary key
        return tuple(self.board.flatten())
    
    def get_valid_actions(self):
        # Return indices of empty cells
        actions = []
        for i in range(BOARD_SIZE):
            for j in range(BOARD_SIZE):
                if self.board[i][j] == 0:
                    actions.append((i, j))
        return actions
    
    def make_move(self, row, col):
        if self.board[row][col] == 0 and not self.game_over:
            # Store state before move for learning
            prev_state = self.get_state()
            
            # Make move
            self.board[row][col] = self.current_player
            
            # Store move in history
            self.moves_history.append((prev_state, (row, col)))
            
            # Check for win or draw
            if self.check_win():
                self.winner = self.current_player
                self.game_over = True
            elif len(self.get_valid_actions()) == 0:
                self.game_over = True  # Draw
            else:
                # Switch player
                self.current_player = 3 - self.current_player  # 1 -> 2, 2 -> 1
            
            return True
        return False
    
    def check_win(self):
        player = self.current_player
        
        # Check rows
        for i in range(BOARD_SIZE):
            if all(self.board[i, j] == player for j in range(BOARD_SIZE)):
                return True
        
        # Check columns
        for j in range(BOARD_SIZE):
            if all(self.board[i, j] == player for i in range(BOARD_SIZE)):
                return True
        
        # Check diagonals
        if all(self.board[i, i] == player for i in range(BOARD_SIZE)):
            return True
        if all(self.board[i, BOARD_SIZE - 1 - i] == player for i in range(BOARD_SIZE)):
            return True
        
        return False
    
    def render(self, ai_stats):
        self.screen.fill(WHITE)
        
        # Draw board grid
        for i in range(1, BOARD_SIZE):
            # Horizontal lines
            pygame.draw.line(self.screen, BLACK, 
                            (MARGIN, MARGIN + i * CELL_SIZE), 
                            (MARGIN + BOARD_SIZE * CELL_SIZE, MARGIN + i * CELL_SIZE), 
                            4)
            # Vertical lines
            pygame.draw.line(self.screen, BLACK, 
                            (MARGIN + i * CELL_SIZE, MARGIN), 
                            (MARGIN + i * CELL_SIZE, MARGIN + BOARD_SIZE * CELL_SIZE), 
                            4)
        
        # Draw X's and O's
        for i in range(BOARD_SIZE):
            for j in range(BOARD_SIZE):
                center_x = MARGIN + j * CELL_SIZE + CELL_SIZE // 2
                center_y = MARGIN + i * CELL_SIZE + CELL_SIZE // 2
                
                if self.board[i, j] == 1:  # X
                    pygame.draw.line(self.screen, BLUE, 
                                    (center_x - 40, center_y - 40), 
                                    (center_x + 40, center_y + 40), 
                                    10)
                    pygame.draw.line(self.screen, BLUE, 
                                    (center_x + 40, center_y - 40), 
                                    (center_x - 40, center_y + 40), 
                                    10)
                elif self.board[i, j] == 2:  # O
                    pygame.draw.circle(self.screen, RED, 
                                      (center_x, center_y), 
                                      40, 8)
        
        # Display game status
        status_text = ""
        if self.auto_training:
            status_text = "AI Auto-Training Mode"
            text_color = PURPLE
        elif self.game_over:
            if self.winner == 1:
                status_text = "You Win!"
                text_color = BLUE
            elif self.winner == 2:
                status_text = "AI Wins!"
                text_color = RED
            else:
                status_text = "Draw!"
                text_color = BLACK
        else:
            if self.current_player == 1:
                status_text = "Your Turn (X)"
                text_color = BLUE
            else:
                status_text = "AI's Turn (O)"
                text_color = RED
        
        status_surface = self.font.render(status_text, True, text_color)
        self.screen.blit(status_surface, (SCREEN_WIDTH // 2 - status_surface.get_width() // 2, 20))
        
        # Display AI stats
        stats_y = MARGIN + BOARD_SIZE * CELL_SIZE + 30
        
        games_text = self.small_font.render(f"Games Played: {ai_stats['games']}", True, BLACK)
        self.screen.blit(games_text, (MARGIN, stats_y))
        
        wins_text = self.small_font.render(f"AI Wins: {ai_stats['ai_wins']}", True, RED)
        self.screen.blit(wins_text, (MARGIN, stats_y + 30))
        
        losses_text = self.small_font.render(f"AI Losses: {ai_stats['ai_losses']}", True, BLUE)
        self.screen.blit(losses_text, (MARGIN, stats_y + 60))
        
        draws_text = self.small_font.render(f"Draws: {ai_stats['draws']}", True, BLACK)
        self.screen.blit(draws_text, (MARGIN, stats_y + 90))
        
        exploration_text = self.small_font.render(f"AI Exploration Rate: {ai_stats['exploration_rate']:.3f}", True, GREEN)
        self.screen.blit(exploration_text, (MARGIN + 200, stats_y))
        
        learning_text = self.small_font.render(f"AI Knowledge: {len(ai_stats['q_table'])} states", True, GREEN)
        self.screen.blit(learning_text, (MARGIN + 200, stats_y + 30))
        
        # Auto-training stats
        if self.auto_training:
            auto_train_time = time.time() - self.auto_train_start_time
            auto_train_text = self.small_font.render(
                f"Auto-Training: {self.auto_train_games} games ({auto_train_time:.1f}s)", 
                True, PURPLE
            )
            self.screen.blit(auto_train_text, (MARGIN + 200, stats_y + 60))
        
        # New game button
        pygame.draw.rect(self.screen, GREEN, (SCREEN_WIDTH // 2 - 150, stats_y + 60, 140, 40))
        new_game_text = self.small_font.render("New Game", True, WHITE)
        self.screen.blit(new_game_text, (SCREEN_WIDTH // 2 - 150 + 70 - new_game_text.get_width() // 2, stats_y + 70))
        
        # Auto-train button
        button_color = PURPLE if self.auto_training else ORANGE
        pygame.draw.rect(self.screen, button_color, (SCREEN_WIDTH // 2 + 10, stats_y + 60, 140, 40))
        auto_train_text = self.small_font.render("Auto-Train: " + ("ON" if self.auto_training else "OFF"), True, WHITE)
        self.screen.blit(auto_train_text, (SCREEN_WIDTH // 2 + 10 + 70 - auto_train_text.get_width() // 2, stats_y + 70))
        
        pygame.display.flip()
        
    def is_new_game_button_clicked(self, pos):
        stats_y = MARGIN + BOARD_SIZE * CELL_SIZE + 30
        button_rect = pygame.Rect(SCREEN_WIDTH // 2 - 150, stats_y + 60, 140, 40)
        return button_rect.collidepoint(pos)
    
    def is_auto_train_button_clicked(self, pos):
        stats_y = MARGIN + BOARD_SIZE * CELL_SIZE + 30
        button_rect = pygame.Rect(SCREEN_WIDTH // 2 + 10, stats_y + 60, 140, 40)
        return button_rect.collidepoint(pos)
    
    def toggle_auto_training(self):
        self.auto_training = not self.auto_training
        if self.auto_training:
            self.auto_train_games = 0
            self.auto_train_start_time = time.time()
            self.reset()  # Start with a fresh game
        
class QLearningAgent:
    def __init__(self):
        self.q_table = defaultdict(lambda: defaultdict(float))
        self.exploration_rate = EXPLORATION_RATE
        self.games_played = 0
        self.ai_wins = 0
        self.ai_losses = 0
        self.draws = 0
        self.load_q_table()
        
    def get_action(self, state, valid_actions, player=2):
        if not valid_actions:
            return None
        
        # Exploration: choose random action
        if random.random() < self.exploration_rate:
            return random.choice(valid_actions)
        
        # Exploitation: choose best action from Q-table
        state_q_values = self.q_table[state]
        
        # If no values exist for this state or for valid actions, choose randomly
        if not state_q_values or all(state_q_values.get(action, 0) == 0 for action in valid_actions):
            return random.choice(valid_actions)
        
        # Choose action with highest Q-value (for AI) or lowest Q-value (for simulated player)
        if player == 2:  # AI
            return max(valid_actions, key=lambda action: state_q_values.get(action, 0))
        else:  # Simulated player (in auto-training)
            # For auto-training, we want the simulated player to be somewhat smart but not perfect
            # Mix of random and strategic play
            if random.random() < 0.3:  # 30% random play
                return random.choice(valid_actions)
            else:  # 70% strategic play (choose worst move for AI)
                return min(valid_actions, key=lambda action: state_q_values.get(action, 0))
    
    def learn_from_game(self, moves_history, winner):
        if not moves_history:
            return
        
        # Determine rewards based on game outcome
        if winner == 2:  # AI won
            final_reward = 1.0
            self.ai_wins += 1
        elif winner == 1:  # AI lost
            final_reward = -1.0
            self.ai_losses += 1
        else:  # Draw
            final_reward = 0.2
            self.draws += 1
        
        self.games_played += 1
        
        # Filter moves made by AI (player 2)
        ai_moves = [(state, action) for i, (state, action) in enumerate(moves_history) 
                   if i % 2 == 1 or (i == 0 and moves_history[0][0].count(2) > moves_history[0][0].count(1))]
        
        # Update Q-values backwards (from last move to first)
        cumulative_reward = final_reward
        for state, action in reversed(ai_moves):
            # Convert action tuple to a hashable form
            action_key = (action[0], action[1])
            
            # Update Q-value using Q-learning formula
            old_q_value = self.q_table[state][action_key]
            self.q_table[state][action_key] = old_q_value + LEARNING_RATE * (cumulative_reward - old_q_value)
            
            # Discount reward for earlier actions
            cumulative_reward *= DISCOUNT_FACTOR
        
        # Decay exploration rate
        self.decay_exploration()
        
        # Save Q-table periodically
        if self.games_played % 10 == 0:
            self.save_q_table()
    
    def decay_exploration(self):
        self.exploration_rate = max(MIN_EXPLORATION_RATE, self.exploration_rate * EXPLORATION_DECAY)
    
    def save_q_table(self):
        # Convert defaultdict to regular dict for saving
        q_dict = {str(state): dict(actions) for state, actions in self.q_table.items()}
        
        with open('q_table.pkl', 'wb') as f:
            pickle.dump(q_dict, f)
    
    def load_q_table(self):
        if os.path.exists('q_table.pkl'):
            try:
                with open('q_table.pkl', 'rb') as f:
                    q_dict = pickle.load(f)
                
                # Convert back to defaultdict and tuple keys
                for state_str, actions in q_dict.items():
                    state = eval(state_str)
                    for action_str, value in actions.items():
                        action = eval(action_str)
                        self.q_table[state][action] = value
                
                print(f"Loaded Q-table with {len(self.q_table)} states")
            except Exception as e:
                print(f"Error loading Q-table: {e}")
                self.q_table = defaultdict(lambda: defaultdict(float))
    
    def get_stats(self):
        return {
            'games': self.games_played,
            'ai_wins': self.ai_wins,
            'ai_losses': self.ai_losses,
            'draws': self.draws,
            'exploration_rate': self.exploration_rate,
            'q_table': self.q_table
        }

def main():
    game = TicTacToe()
    agent = QLearningAgent()
    running = True
    clock = pygame.time.Clock()
    
    while running:
        # Process events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN and not game.auto_training:
                pos = pygame.mouse.get_pos()
                
                # Check if auto-train button was clicked
                if game.is_auto_train_button_clicked(pos):
                    game.toggle_auto_training()
                    continue
                
                # Check if new game button was clicked
                if game.is_new_game_button_clicked(pos):
                    if game.game_over:
                        agent.learn_from_game(game.moves_history, game.winner)
                    game.reset()
                    continue
                
                # Handle player move
                if game.current_player == 1 and not game.game_over:
                    # Convert mouse position to board coordinates
                    if MARGIN <= pos[0] <= MARGIN + BOARD_SIZE * CELL_SIZE and \
                       MARGIN <= pos[1] <= MARGIN + BOARD_SIZE * CELL_SIZE:
                        col = (pos[0] - MARGIN) // CELL_SIZE
                        row = (pos[1] - MARGIN) // CELL_SIZE
                        
                        game.make_move(row, col)
            elif event.type == pygame.MOUSEBUTTONDOWN and game.auto_training:
                pos = pygame.mouse.get_pos()
                
                # Only allow clicking the auto-train button during auto-training
                if game.is_auto_train_button_clicked(pos):
                    game.toggle_auto_training()
                    continue
        
        # Auto-training mode
        if game.auto_training:
            # If game is over, learn from it and start a new game
            if game.game_over:
                agent.learn_from_game(game.moves_history, game.winner)
                game.auto_train_games += 1
                game.reset()
                
                # Save more frequently during auto-training
                if game.auto_train_games % 100 == 0:
                    agent.save_q_table()
            
            # AI plays both sides in auto-training mode
            state = game.get_state()
            valid_actions = game.get_valid_actions()
            
            # Choose action based on current player
            action = agent.get_action(state, valid_actions, game.current_player)
            
            if action:
                # Faster gameplay during auto-training
                game.make_move(action[0], action[1])
        else:
            # Normal gameplay mode
            # AI's turn
            if game.current_player == 2 and not game.game_over:
                state = game.get_state()
                valid_actions = game.get_valid_actions()
                action = agent.get_action(state, valid_actions)
                
                if action:
                    pygame.time.delay(500)  # Add delay for better visualization
                    game.make_move(action[0], action[1])
            
            # If game is over, learn from it
            if game.game_over and game.moves_history:
                agent.learn_from_game(game.moves_history, game.winner)
                game.moves_history = []  # Clear history after learning
        
        # Render game
        game.render(agent.get_stats())
        
        # Control frame rate
        if game.auto_training:
            clock.tick(60)  # Faster in auto-training mode
        else:
            clock.tick(30)  # Normal speed in regular mode
    
    # Save Q-table before exiting
    agent.save_q_table()
    pygame.quit()

if __name__ == "__main__":
    main()
