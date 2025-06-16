import cv2
import numpy as np
import mediapipe as mp
import torch
import torch.nn as nn
import time
import joblib
import random
import math

# Load scaler and classes
scaler = joblib.load('scaler.pkl')
classes = np.load("gesture_classes.npy", allow_pickle=True)
num_classes = len(classes)

# Model definition
class GestureClassifier(nn.Module):
    def __init__(self, input_size=10, num_classes=4):
        super(GestureClassifier, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, num_classes)
        )
    def forward(self, x):
        return self.model(x)

# Load model
optimized_model = torch.jit.load("gesture_classifier_traced.pt")
optimized_model.eval()

# MediaPipe setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_drawing = mp.solutions.drawing_utils

# Game Classes
class Player:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.size = 30
        self.color = (0, 255, 0)
        self.health = 100
        self.max_health = 100
        self.speed = 15
        self.invulnerable_time = 0
        self.jump_velocity = 0
        self.on_ground = True
        self.gravity = 1.5
        
    def update(self, win_height):
        # Apply gravity
        if not self.on_ground:
            self.jump_velocity += self.gravity
            self.y += self.jump_velocity
            
        # Check ground collision
        if self.y >= win_height - 100 - self.size:  # Ground level
            self.y = win_height - 100 - self.size
            self.on_ground = True
            self.jump_velocity = 0
        else:
            self.on_ground = False
            
        # Update invulnerability
        if self.invulnerable_time > 0:
            self.invulnerable_time -= 1
            
    def move(self, dx, dy, win_width, win_height):
        if dx != 0:  # Horizontal movement
            self.x = np.clip(self.x + dx, 0, win_width - self.size)
        if dy < 0 and self.on_ground:  # Jump
            self.jump_velocity = dy
            self.on_ground = False
        if dy > 0 and not self.on_ground:  # Slide (quick fall)
            self.jump_velocity = dy * 2
            
    def take_damage(self, damage):
        if self.invulnerable_time <= 0:
            self.health -= damage
            self.invulnerable_time = 60  # 1 second at 60 FPS
            return True
        return False
        
    def draw(self, frame):
        # Flash when invulnerable
        if self.invulnerable_time > 0 and self.invulnerable_time % 10 < 5:
            color = (100, 100, 100)
        else:
            color = self.color
            
        cv2.rectangle(frame, 
                     (int(self.x), int(self.y)), 
                     (int(self.x + self.size), int(self.y + self.size)), 
                     color, -1)

class Enemy:
    def __init__(self, x, y, enemy_type="basic"):
        self.x = x
        self.y = y
        self.size = 25
        self.type = enemy_type
        self.speed = random.uniform(1, 3)
        self.health = 1
        
        if enemy_type == "basic":
            self.color = (0, 0, 255)  # Red
            self.damage = 10
        elif enemy_type == "fast":
            self.color = (255, 0, 255)  # Magenta
            self.damage = 15
            self.speed = random.uniform(3, 5)
        elif enemy_type == "heavy":
            self.color = (128, 0, 128)  # Purple
            self.damage = 25
            self.size = 35
            self.speed = random.uniform(0.5, 1.5)
            self.health = 2
            
    def update(self, player_x):
        # Move towards player
        if self.x < player_x:
            self.x += self.speed
        else:
            self.x -= self.speed
            
    def draw(self, frame):
        cv2.rectangle(frame,
                     (int(self.x), int(self.y)),
                     (int(self.x + self.size), int(self.y + self.size)),
                     self.color, -1)

class Collectible:
    def __init__(self, x, y, item_type="coin"):
        self.x = x
        self.y = y
        self.size = 20
        self.type = item_type
        self.collected = False
        self.bounce_offset = 0
        
        if item_type == "coin":
            self.color = (0, 255, 255)  # Yellow
            self.value = 10
        elif item_type == "health":
            self.color = (0, 255, 0)  # Green
            self.value = 25
        elif item_type == "power":
            self.color = (255, 255, 0)  # Cyan
            self.value = 50
            
    def update(self):
        self.bounce_offset = math.sin(time.time() * 5) * 5
        
    def draw(self, frame):
        if not self.collected:
            y_pos = int(self.y + self.bounce_offset)
            cv2.circle(frame, (int(self.x), y_pos), self.size//2, self.color, -1)

class Game:
    def __init__(self, win_size):
        self.win_size = win_size
        self.player = Player(win_size[0]//2, win_size[1] - 150)
        self.enemies = []
        self.collectibles = []
        self.score = 0
        self.level = 1
        self.enemy_spawn_timer = 0
        self.collectible_spawn_timer = 0
        self.game_over = False
        self.bg_color = (20, 20, 40)
        
    def spawn_enemy(self):
        enemy_types = ["basic", "fast", "heavy"]
        weights = [0.6, 0.3, 0.1]  # Probability weights
        enemy_type = random.choices(enemy_types, weights=weights)[0]
        
        x = random.choice([0, self.win_size[0]])  # Spawn from sides
        y = self.win_size[1] - 100 - 25  # Ground level
        self.enemies.append(Enemy(x, y, enemy_type))
        
    def spawn_collectible(self):
        item_types = ["coin", "health", "power"]
        weights = [0.7, 0.2, 0.1]
        item_type = random.choices(item_types, weights=weights)[0]
        
        x = random.randint(50, self.win_size[0] - 50)
        y = random.randint(100, self.win_size[1] - 200)
        self.collectibles.append(Collectible(x, y, item_type))
        
    def check_collisions(self):
        # Player-Enemy collisions
        for enemy in self.enemies[:]:
            if (abs(self.player.x - enemy.x) < (self.player.size + enemy.size) // 2 and
                abs(self.player.y - enemy.y) < (self.player.size + enemy.size) // 2):
                if self.player.take_damage(enemy.damage):
                    self.enemies.remove(enemy)
                    
        # Player-Collectible collisions
        for collectible in self.collectibles[:]:
            if (abs(self.player.x - collectible.x) < (self.player.size + collectible.size) // 2 and
                abs(self.player.y - collectible.y) < (self.player.size + collectible.size) // 2):
                
                collectible.collected = True
                if collectible.type == "coin":
                    self.score += collectible.value
                elif collectible.type == "health":
                    self.player.health = min(self.player.max_health, 
                                           self.player.health + collectible.value)
                elif collectible.type == "power":
                    self.score += collectible.value
                    
                self.collectibles.remove(collectible)
                
    def update(self):
        if self.game_over:
            return
            
        self.player.update(self.win_size[1])
        
        # Check game over
        if self.player.health <= 0:
            self.game_over = True
            return
            
        # Update enemies
        for enemy in self.enemies:
            enemy.update(self.player.x)
            
        # Update collectibles
        for collectible in self.collectibles:
            collectible.update()
            
        # Remove off-screen enemies
        self.enemies = [e for e in self.enemies if -50 < e.x < self.win_size[0] + 50]
        
        # Spawn enemies
        spawn_rate = max(30, 120 - self.level * 10)  # Faster spawning as level increases
        if self.enemy_spawn_timer <= 0:
            self.spawn_enemy()
            self.enemy_spawn_timer = spawn_rate
        else:
            self.enemy_spawn_timer -= 1
            
        # Spawn collectibles
        if self.collectible_spawn_timer <= 0:
            self.spawn_collectible()
            self.collectible_spawn_timer = random.randint(180, 300)  # 3-5 seconds
        else:
            self.collectible_spawn_timer -= 1
            
        # Check collisions
        self.check_collisions()
        
        # Level progression
        if self.score >= self.level * 100:
            self.level += 1
            
    def draw(self, frame):
        # Background
        frame[:] = self.bg_color
        
        # Ground
        cv2.rectangle(frame, (0, self.win_size[1] - 100), 
                     (self.win_size[0], self.win_size[1]), (100, 100, 100), -1)
        
        # Draw game objects
        self.player.draw(frame)
        
        for enemy in self.enemies:
            enemy.draw(frame)
            
        for collectible in self.collectibles:
            collectible.draw(frame)
            
        # HUD
        self.draw_hud(frame)
        
    def draw_hud(self, frame):
        # Health bar
        bar_width = 200
        bar_height = 20
        health_ratio = max(0, self.player.health / self.player.max_health)
        
        # Background
        cv2.rectangle(frame, (10, 10), (10 + bar_width, 10 + bar_height), 
                     (50, 50, 50), -1)
        # Health
        cv2.rectangle(frame, (10, 10), 
                     (10 + int(bar_width * health_ratio), 10 + bar_height), 
                     (0, 255, 0), -1)
        
        # Text info
        cv2.putText(frame, f"Health: {self.player.health}/{self.player.max_health}", 
                   (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(frame, f"Score: {self.score}", 
                   (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(frame, f"Level: {self.level}", 
                   (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Controls
        cv2.putText(frame, "Left/Right: Move | Jump: Jump | Slide: Quick Fall", 
                   (10, self.win_size[1] - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 
                   (200, 200, 200), 1)
        
        if self.game_over:
            # Game over screen
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), self.win_size, (0, 0, 0), -1)
            cv2.addWeighted(frame, 0.3, overlay, 0.7, 0, frame)
            
            cv2.putText(frame, "GAME OVER", 
                       (self.win_size[0]//2 - 100, self.win_size[1]//2), 
                       cv2.FONT_HERSHEY_DUPLEX, 1.5, (0, 0, 255), 2)
            cv2.putText(frame, f"Final Score: {self.score}", 
                       (self.win_size[0]//2 - 80, self.win_size[1]//2 + 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(frame, "Press ESC to exit", 
                       (self.win_size[0]//2 - 90, self.win_size[1]//2 + 80), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1)

# Extract only x, y from 5 keypoints of index finger
def extract_keypoints(results):
    index_finger_indices = [0, 5, 6, 7, 8]
    if results.multi_hand_landmarks:
        hand = results.multi_hand_landmarks[0]
        keypoints = []
        for idx in index_finger_indices:
            lm = hand.landmark[idx]
            keypoints.extend([lm.x, lm.y])
        return np.array(keypoints)
    return None

# Gesture to direction mapping
gesture_to_direction = {
    "swipe_left": (-20, 0),
    "swipe_right": (20, 0),
    "jump": (0, -25),
    "slide": (0, 20)
}

# Game setup
win_size = (800, 600)
game = Game(win_size)

# Start camera
print("[Enhanced Game] Survive as long as possible!")
print("Collect coins (yellow) for points")
print("Collect health (green) to restore health")
print("Avoid enemies (red/purple/magenta)")

gesture_cooldown = 0.1
last_gesture_time = time.time()
cap = cv2.VideoCapture(0)

# Set low resolution for speed
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

clock = time.time()
frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Process hand gesture
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    keypoints = extract_keypoints(results)
    if keypoints is not None:
        keypoints_scaled = scaler.transform([keypoints])
        input_tensor = torch.tensor(keypoints_scaled, dtype=torch.float32)

        with torch.no_grad():
            output = optimized_model(input_tensor)
            predicted_class = torch.argmax(output, dim=1).item()
            gesture = classes[predicted_class]

        # Update player position if valid gesture
        if gesture in gesture_to_direction and (time.time() - last_gesture_time) > gesture_cooldown:
            dx, dy = gesture_to_direction[gesture]
            game.player.move(dx, dy, win_size[0], win_size[1])
            last_gesture_time = time.time()

    # Update game
    game.update()
    
    # Create game frame
    game_frame = np.zeros((win_size[1], win_size[0], 3), dtype=np.uint8)
    game.draw(game_frame)

    # Show game window
    cv2.imshow("Enhanced Gesture Game", game_frame)
    
    # Frame rate control
    frame_count += 1
    if time.time() - clock >= 1.0:
        clock = time.time()
        frame_count = 0

    if cv2.waitKey(1) & 0xFF == 27:  # ESC key
        break

cap.release()
cv2.destroyAllWindows()