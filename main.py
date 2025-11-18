import cv2
import mediapipe as mp
import numpy as np
import time
import math

# Two Player Rock Paper Scissors with Hand Detection
# Play against your friend with automatic timer and score tracking
# Each player gets their own side of the screen!

# Initialize MediaPipe for dual hand tracking
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False, 
    max_num_hands=2,  # Track both players' hands
    min_detection_confidence=0.7, 
    min_tracking_confidence=0.7
)

# Game colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREEN = (0, 255, 0)
RED = (0, 0, 255)
BLUE = (255, 0, 0)
YELLOW = (0, 255, 255)
ORANGE = (0, 165, 255)
PURPLE = (255, 0, 255)

class TwoPlayerRockPaperScissors:
    """Game manager for two-player Rock Paper Scissors"""
    
    def __init__(self):
        self.player1_score = 0  # Left side player
        self.player2_score = 0  # Right side player
        self.game_state = "waiting"  # waiting, countdown, playing, result
        self.countdown_start = 0
        self.result_start = 0
        self.play_time = 5  # Seconds to make gesture
        self.play_start = 0
        self.player1_choice = ""
        self.player2_choice = ""
        self.result = ""
        self.choices = ["ROCK", "PAPER", "SCISSORS"]
        self.round_number = 0
        
    def determine_winner(self, p1_choice, p2_choice):
        """Classic Rock Paper Scissors logic for two players"""
        if p1_choice == p2_choice:
            return "TIE!"
        elif (p1_choice == "ROCK" and p2_choice == "SCISSORS") or \
             (p1_choice == "PAPER" and p2_choice == "ROCK") or \
             (p1_choice == "SCISSORS" and p2_choice == "PAPER"):
            return "PLAYER 1 WINS!"
        elif (p2_choice == "ROCK" and p1_choice == "SCISSORS") or \
             (p2_choice == "PAPER" and p1_choice == "ROCK") or \
             (p2_choice == "SCISSORS" and p1_choice == "PAPER"):
            return "PLAYER 2 WINS!"
        else:
            return "INVALID!"
    
    def start_countdown(self):
        """Begin round countdown"""
        self.game_state = "countdown"
        self.countdown_start = time.time()
        self.round_number += 1
    
    def update_countdown(self):
        """Update countdown timer and check if finished"""
        elapsed = time.time() - self.countdown_start
        if elapsed >= 3.0:  # 3 second countdown
            self.game_state = "playing"
            self.play_start = time.time()
            return True
        return False
    
    def update_play_time(self):
        """Update play timer and return remaining time"""
        elapsed = time.time() - self.play_start
        remaining = self.play_time - elapsed
        if remaining <= 0:
            self.game_state = "result"
            self.evaluate_round()
            self.result_start = time.time()
            return 0
        return remaining
    
    def evaluate_round(self):
        """Evaluate the round and update scores"""
        # Use detected gestures or default to ROCK
        p1 = self.player1_choice if self.player1_choice in self.choices else "ROCK"
        p2 = self.player2_choice if self.player2_choice in self.choices else "ROCK"
        
        self.player1_choice = p1
        self.player2_choice = p2
        self.result = self.determine_winner(p1, p2)
        
        # Update scores based on result
        if self.result == "PLAYER 1 WINS!":
            self.player1_score += 1
        elif self.result == "PLAYER 2 WINS!":
            self.player2_score += 1
    
    def show_result_complete(self):
        """Check if result display time is finished"""
        return time.time() - self.result_start >= 4.0
    
    def reset_round(self):
        """Reset for next round"""
        self.game_state = "waiting"
        self.player1_choice = ""
        self.player2_choice = ""
        self.result = ""

def detect_hand_gesture(landmarks):
    """Analyze hand landmarks to determine gesture"""
    if not landmarks:
        return "NONE"
    
    # Get finger tip and pip joint positions
    thumb_tip = landmarks[4]
    index_tip = landmarks[8]
    middle_tip = landmarks[12]
    ring_tip = landmarks[16]
    pinky_tip = landmarks[20]
    
    index_pip = landmarks[6]
    middle_pip = landmarks[10]
    ring_pip = landmarks[14]
    pinky_pip = landmarks[18]
    
    # Check which fingers are extended (tip above pip)
    fingers_extended = []
    
    # Check each finger
    fingers_extended.append(index_tip[1] < index_pip[1])  # Index
    fingers_extended.append(middle_tip[1] < middle_pip[1])  # Middle
    fingers_extended.append(ring_tip[1] < ring_pip[1])  # Ring
    fingers_extended.append(pinky_tip[1] < pinky_pip[1])  # Pinky
    
    # Count extended fingers
    extended_count = sum(fingers_extended)
    
    # Classify gesture based on finger count
    if extended_count == 0:
        return "ROCK"  # Closed fist
    elif extended_count >= 3:
        return "PAPER"  # Open hand
    elif extended_count == 2 and fingers_extended[0] and fingers_extended[1]:
        return "SCISSORS"  # Index and middle fingers
    else:
        return "ROCK"  # Default to rock for unclear gestures

def separate_hands_by_position(hand_landmarks_list, frame_width):
    """Separate detected hands based on screen position"""
    if not hand_landmarks_list:
        return None, None
    
    left_hand = None
    right_hand = None
    
    for hand_landmarks in hand_landmarks_list:
        # Use wrist position to determine left/right
        wrist_x = hand_landmarks.landmark[0].x * frame_width
        
        # Left half = Player 1, Right half = Player 2
        if wrist_x < frame_width / 2:
            left_hand = hand_landmarks
        else:
            right_hand = hand_landmarks
    
    return left_hand, right_hand

def draw_player_zones(frame):
    """Draw visual zones for each player"""
    h, w = frame.shape[:2]
    
    # Center dividing line
    cv2.line(frame, (w//2, 0), (w//2, h), WHITE, 3)
    
    # Player labels
    cv2.putText(frame, "PLAYER 1", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, GREEN, 3)
    cv2.putText(frame, "PLAYER 2", (w - 200, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, BLUE, 3)
    
    # Zone borders
    cv2.rectangle(frame, (20, 20), (w//2 - 20, 80), GREEN, 3)
    cv2.rectangle(frame, (w//2 + 20, 20), (w - 20, 80), BLUE, 3)

def draw_countdown_animation(frame, game):
    """Draw animated countdown before round starts"""
    h, w = frame.shape[:2]
    elapsed = time.time() - game.countdown_start
    countdown_num = 3 - int(elapsed)
    
    if countdown_num > 0:
        # Large countdown number
        cv2.putText(frame, str(countdown_num), (w//2 - 60, h//2), 
                    cv2.FONT_HERSHEY_SIMPLEX, 6, ORANGE, 12)
        
        # Round indicator
        cv2.putText(frame, f"ROUND {game.round_number}", (w//2 - 100, h//2 + 100), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, WHITE, 3)
    else:
        cv2.putText(frame, "SHOW YOUR HANDS!", (w//2 - 200, h//2), 
                    cv2.FONT_HERSHEY_SIMPLEX, 2, RED, 5)

def draw_play_timer(frame, remaining_time):
    """Draw visual timer bar during play phase"""
    h, w = frame.shape[:2]
    
    # Timer bar dimensions
    bar_width = 400
    bar_height = 20
    bar_x = w//2 - bar_width//2
    bar_y = h - 100
    
    # Background (empty) bar
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), RED, -1)
    
    # Progress (filled) bar
    progress = remaining_time / 5.0
    progress_width = int(bar_width * progress)
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + progress_width, bar_y + bar_height), GREEN, -1)
    
    # Timer text
    cv2.putText(frame, f"TIME: {remaining_time:.1f}s", (w//2 - 80, bar_y - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, WHITE, 2)

def draw_round_result(frame, game):
    """Display round results with choices and winner"""
    h, w = frame.shape[:2]
    
    # Result background panel
    cv2.rectangle(frame, (w//2 - 300, h//2 - 150), (w//2 + 300, h//2 + 150), (30, 30, 30), -1)
    cv2.rectangle(frame, (w//2 - 300, h//2 - 150), (w//2 + 300, h//2 + 150), WHITE, 3)
    
    # Player choices
    cv2.putText(frame, f"PLAYER 1: {game.player1_choice}", (w//2 - 280, h//2 - 100), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, GREEN, 3)
    cv2.putText(frame, f"PLAYER 2: {game.player2_choice}", (w//2 - 280, h//2 - 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, BLUE, 3)
    
    # Winner announcement
    result_color = GREEN if "PLAYER 1" in game.result else BLUE if "PLAYER 2" in game.result else YELLOW
    cv2.putText(frame, game.result, (w//2 - 150, h//2 + 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 2, result_color, 4)
    
    # Current score
    cv2.putText(frame, f"SCORE: {game.player1_score} - {game.player2_score}", 
                (w//2 - 120, h//2 + 120), cv2.FONT_HERSHEY_SIMPLEX, 1.2, WHITE, 3)

def main():
    """Main game loop"""
    # Initialize camera
    cap = cv2.VideoCapture(0)
    game = TwoPlayerRockPaperScissors()
    
    # Set camera resolution for better performance
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1200)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 800)
    
    print("🎮 TWO PLAYER ROCK PAPER SCISSORS")
    print("=== SETUP ===")
    print("👈 PLAYER 1 = Stand on left side of camera")
    print("👉 PLAYER 2 = Stand on right side of camera")
    print("=== CONTROLS ===")
    print("SPACE = Start new round")
    print("R = Reset entire game")
    print("Q = Quit game")
    print("🎯 Get ready to play!")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Flip frame for mirror effect
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        
        # Process frame for hand detection
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)
        
        # Draw player zones
        draw_player_zones(frame)
        
        # Initialize gesture tracking
        player1_gesture = "NONE"
        player2_gesture = "NONE"
        
        # Process detected hands
        if results.multi_hand_landmarks:
            left_hand, right_hand = separate_hands_by_position(results.multi_hand_landmarks, w)
            
            # Process Player 1 (left side)
            if left_hand:
                landmarks = [(int(l.x * w), int(l.y * h)) for l in left_hand.landmark]
                
                # Draw hand points
                for point in landmarks:
                    cv2.circle(frame, point, 4, GREEN, -1)
                
                player1_gesture = detect_hand_gesture(landmarks)
                if game.game_state == "playing":
                    game.player1_choice = player1_gesture
                
                # Display current gesture
                cv2.putText(frame, f"GESTURE: {player1_gesture}", (20, h - 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, GREEN, 2)
            
            # Process Player 2 (right side)
            if right_hand:
                landmarks = [(int(l.x * w), int(l.y * h)) for l in right_hand.landmark]
                
                # Draw hand points
                for point in landmarks:
                    cv2.circle(frame, point, 4, BLUE, -1)
                
                player2_gesture = detect_hand_gesture(landmarks)
                if game.game_state == "playing":
                    game.player2_choice = player2_gesture
                
                # Display current gesture
                cv2.putText(frame, f"GESTURE: {player2_gesture}", (w - 300, h - 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, BLUE, 2)
        
        # Game state management
        if game.game_state == "waiting":
            cv2.putText(frame, "PRESS SPACE TO START ROUND", (w//2 - 250, h//2), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, WHITE, 3)
            
            # Show current score
            cv2.putText(frame, f"SCORE: P1 {game.player1_score} - {game.player2_score} P2", 
                        (w//2 - 200, h//2 + 50), cv2.FONT_HERSHEY_SIMPLEX, 1, YELLOW, 2)
        
        elif game.game_state == "countdown":
            draw_countdown_animation(frame, game)
            game.update_countdown()
        
        elif game.game_state == "playing":
            remaining = game.update_play_time()
            if remaining > 0:
                draw_play_timer(frame, remaining)
                cv2.putText(frame, "MAKE YOUR GESTURES NOW!", (w//2 - 220, 150), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, RED, 3)
        
        elif game.game_state == "result":
            draw_round_result(frame, game)
            if game.show_result_complete():
                game.reset_round()
        
        # Instructions
        cv2.putText(frame, "ROCK=Fist  PAPER=Open  SCISSORS=2fingers", 
                    (w//2 - 200, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, WHITE, 2)
        
        cv2.imshow('Two Player Rock Paper Scissors ⚡👥', frame)
        
        # Handle keyboard input
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):  # Quit
            break
        elif key == ord(' '):  # Start round
            if game.game_state == "waiting":
                game.start_countdown()
        elif key == ord('r'):  # Reset game
            game = TwoPlayerRockPaperScissors()
            print("🔄 Game reset!")
    
    cap.release()
    cv2.destroyAllWindows()
    print(f"🎮 Final Score: Player 1: {game.player1_score}, Player 2: {game.player2_score}")
    
    # Announce overall winner
    if game.player1_score > game.player2_score:
        print("🏆 PLAYER 1 WINS THE GAME! 🎉")
    elif game.player2_score > game.player1_score:
        print("🏆 PLAYER 2 WINS THE GAME! 🎉")
    else:
        print("🤝 IT'S A TIE GAME!")

if __name__ == "__main__":
    main()