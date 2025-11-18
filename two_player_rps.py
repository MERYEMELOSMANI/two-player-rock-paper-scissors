"""
Two Player Rock Paper Scissors with Hand Detection
Created by: Meryem El Osmani

A multiplayer rock paper scissors game where two players can compete using hand gestures!
Perfect for playing with friends or siblings.
"""

import cv2
import mediapipe as mp
import numpy as np
import time
import math

# Initialize MediaPipe for detecting multiple hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False, 
    max_num_hands=2,  # Allow detecting 2 hands at once
    min_detection_confidence=0.7, 
    min_tracking_confidence=0.7
)

# Colors for the UI
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREEN = (0, 255, 0)
RED = (0, 0, 255)
BLUE = (255, 0, 0)
YELLOW = (0, 255, 255)
ORANGE = (0, 165, 255)
PURPLE = (255, 0, 255)

class TwoPlayerGame:
    """Handles all the game logic for two player rock paper scissors"""
    
    def __init__(self):
        self.player1_score = 0  # Left side player
        self.player2_score = 0  # Right side player
        self.game_state = "waiting"  # waiting, countdown, playing, result
        self.countdown_start = 0
        self.result_start = 0
        self.play_time = 5  # Players have 5 seconds to make their gesture
        self.play_start = 0
        self.player1_choice = ""
        self.player2_choice = ""
        self.result = ""
        self.choices = ["ROCK", "PAPER", "SCISSORS"]
        self.round_number = 0
        
    def determine_winner(self, p1_choice, p2_choice):
        """Figure out who wins based on classic rock paper scissors rules"""
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
        """Start the pre-game countdown"""
        self.game_state = "countdown"
        self.countdown_start = time.time()
        self.round_number += 1
    
    def update_countdown(self):
        """Check if countdown is finished and move to playing phase"""
        elapsed = time.time() - self.countdown_start
        if elapsed >= 3.0:  # 3 second countdown
            self.game_state = "playing"
            self.play_start = time.time()
            return True
        return False
    
    def update_play_time(self):
        """Track how much time players have left to make their gesture"""
        elapsed = time.time() - self.play_start
        remaining = self.play_time - elapsed
        if remaining <= 0:
            self.game_state = "result"
            self.evaluate_round()
            self.result_start = time.time()
            return 0
        return remaining
    
    def evaluate_round(self):
        """Evaluate the round result after time runs out"""
        # Use detected choices, default to ROCK if nothing detected
        p1 = self.player1_choice if self.player1_choice in self.choices else "ROCK"
        p2 = self.player2_choice if self.player2_choice in self.choices else "ROCK"
        
        self.player1_choice = p1
        self.player2_choice = p2
        self.result = self.determine_winner(p1, p2)
        
        # Update the scores
        if self.result == "PLAYER 1 WINS!":
            self.player1_score += 1
        elif self.result == "PLAYER 2 WINS!":
            self.player2_score += 1
    
    def show_result_complete(self):
        """Check if we've shown the result long enough"""
        return time.time() - self.result_start >= 4.0  # Show for 4 seconds
    
    def reset_round(self):
        """Reset everything for the next round"""
        self.game_state = "waiting"
        self.player1_choice = ""
        self.player2_choice = ""
        self.result = ""

def detect_hand_gesture(landmarks):
    """
    Detect rock, paper, or scissors from hand landmarks
    Same logic as single player but optimized for reliability
    """
    if not landmarks:
        return "NONE"
    
    # Get finger positions (using MediaPipe landmark indices)
    thumb_tip = landmarks[4]
    index_tip = landmarks[8]
    middle_tip = landmarks[12]
    ring_tip = landmarks[16]
    pinky_tip = landmarks[20]
    
    index_pip = landmarks[6]
    middle_pip = landmarks[10]
    ring_pip = landmarks[14]
    pinky_pip = landmarks[18]
    
    # Check if fingers are extended (tip above pip joint)
    fingers_up = []
    
    # Index finger
    fingers_up.append(index_tip[1] < index_pip[1])
    # Middle finger  
    fingers_up.append(middle_tip[1] < middle_pip[1])
    # Ring finger
    fingers_up.append(ring_tip[1] < ring_pip[1])
    # Pinky
    fingers_up.append(pinky_tip[1] < pinky_pip[1])
    
    # Count extended fingers
    fingers_count = sum(fingers_up)
    
    # Classify the gesture
    if fingers_count == 0:
        return "ROCK"
    elif fingers_count >= 3:
        return "PAPER"
    elif fingers_count == 2 and fingers_up[0] and fingers_up[1]:
        return "SCISSORS"
    else:
        return "ROCK"  # Default to rock for unclear gestures

def separate_hands(hand_landmarks_list, frame_width):
    """
    Separate detected hands into left and right based on position
    This was tricky to get right - had to handle edge cases
    """
    if not hand_landmarks_list:
        return None, None
    
    left_hand = None
    right_hand = None
    
    for hand_landmarks in hand_landmarks_list:
        # Use wrist position to determine left vs right
        wrist_x = hand_landmarks.landmark[0].x * frame_width
        
        # Left side = Player 1, Right side = Player 2
        if wrist_x < frame_width / 2:
            left_hand = hand_landmarks
        else:
            right_hand = hand_landmarks
    
    return left_hand, right_hand

def draw_player_zones(frame):
    """Draw visual zones to show where each player should position their hand"""
    h, w = frame.shape[:2]
    
    # Center dividing line
    cv2.line(frame, (w//2, 0), (w//2, h), WHITE, 3)
    
    # Player labels and zones
    cv2.putText(frame, "PLAYER 1", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, GREEN, 3)
    cv2.putText(frame, "PLAYER 2", (w - 200, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, BLUE, 3)
    
    # Zone boundaries
    cv2.rectangle(frame, (20, 20), (w//2 - 20, 80), GREEN, 3)
    cv2.rectangle(frame, (w//2 + 20, 20), (w - 20, 80), BLUE, 3)

def draw_countdown(frame, game):
    """Draw the countdown animation before each round"""
    h, w = frame.shape[:2]
    elapsed = time.time() - game.countdown_start
    countdown_num = 3 - int(elapsed)
    
    if countdown_num > 0:
        # Big countdown number
        cv2.putText(frame, str(countdown_num), (w//2 - 60, h//2), 
                    cv2.FONT_HERSHEY_SIMPLEX, 6, ORANGE, 12)
        
        # Round info
        cv2.putText(frame, f"ROUND {game.round_number}", (w//2 - 100, h//2 + 100), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, WHITE, 3)
    else:
        cv2.putText(frame, "SHOW YOUR HANDS!", (w//2 - 200, h//2), 
                    cv2.FONT_HERSHEY_SIMPLEX, 2, RED, 5)

def draw_play_timer(frame, remaining_time):
    """Draw the timer bar showing how much time players have left"""
    h, w = frame.shape[:2]
    
    # Timer bar dimensions
    bar_width = 400
    bar_height = 20
    bar_x = w//2 - bar_width//2
    bar_y = h - 100
    
    # Background (red when time running out)
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), RED, -1)
    
    # Progress bar (green, shrinks as time runs out)
    progress = remaining_time / 5.0  # 5 seconds total
    progress_width = int(bar_width * progress)
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + progress_width, bar_y + bar_height), GREEN, -1)
    
    # Timer text
    cv2.putText(frame, f"TIME: {remaining_time:.1f}s", (w//2 - 80, bar_y - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, WHITE, 2)

def draw_result(frame, game):
    """Display the results of the round"""
    h, w = frame.shape[:2]
    
    # Result background box
    cv2.rectangle(frame, (w//2 - 300, h//2 - 150), (w//2 + 300, h//2 + 150), (30, 30, 30), -1)
    cv2.rectangle(frame, (w//2 - 300, h//2 - 150), (w//2 + 300, h//2 + 150), WHITE, 3)
    
    # Player choices
    cv2.putText(frame, f"PLAYER 1: {game.player1_choice}", (w//2 - 280, h//2 - 100), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, GREEN, 3)
    cv2.putText(frame, f"PLAYER 2: {game.player2_choice}", (w//2 - 280, h//2 - 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, BLUE, 3)
    
    # Round result
    result_color = GREEN if "PLAYER 1" in game.result else BLUE if "PLAYER 2" in game.result else YELLOW
    cv2.putText(frame, game.result, (w//2 - 150, h//2 + 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 2, result_color, 4)
    
    # Current score
    cv2.putText(frame, f"SCORE: {game.player1_score} - {game.player2_score}", 
                (w//2 - 120, h//2 + 120), cv2.FONT_HERSHEY_SIMPLEX, 1.2, WHITE, 3)

def main():
    """Main game loop"""
    cap = cv2.VideoCapture(0)
    game = TwoPlayerGame()
    
    # Set camera resolution for better performance
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1200)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 800)
    
    print("🎮 TWO PLAYER ROCK PAPER SCISSORS")
    print("=== SETUP ===")
    print("👈 Player 1: Position your hand on the LEFT side")
    print("👉 Player 2: Position your hand on the RIGHT side")
    print("=== CONTROLS ===")
    print("SPACE = Start a new round")
    print("R = Reset the entire game")
    print("Q = Quit")
    print("Ready to compete! 🎯")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Camera error! Check your camera connection.")
            break
        
        frame = cv2.flip(frame, 1)  # Mirror the image
        h, w, _ = frame.shape
        
        # Process the frame for hand detection
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)
        
        # Draw the player zones
        draw_player_zones(frame)
        
        # Initialize gesture detection
        player1_gesture = "NONE"
        player2_gesture = "NONE"
        
        # Detect and process hands
        if results.multi_hand_landmarks:
            left_hand, right_hand = separate_hands(results.multi_hand_landmarks, w)
            
            # Process Player 1 (left side)
            if left_hand:
                landmarks = [(int(l.x * w), int(l.y * h)) for l in left_hand.landmark]
                
                # Draw hand landmarks in green for player 1
                for point in landmarks:
                    cv2.circle(frame, point, 4, GREEN, -1)
                
                player1_gesture = detect_hand_gesture(landmarks)
                if game.game_state == "playing":
                    game.player1_choice = player1_gesture
                
                # Show current gesture for player 1
                cv2.putText(frame, f"P1: {player1_gesture}", (20, h - 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, GREEN, 2)
            
            # Process Player 2 (right side)
            if right_hand:
                landmarks = [(int(l.x * w), int(l.y * h)) for l in right_hand.landmark]
                
                # Draw hand landmarks in blue for player 2
                for point in landmarks:
                    cv2.circle(frame, point, 4, BLUE, -1)
                
                player2_gesture = detect_hand_gesture(landmarks)
                if game.game_state == "playing":
                    game.player2_choice = player2_gesture
                
                # Show current gesture for player 2
                cv2.putText(frame, f"P2: {player2_gesture}", (w - 300, h - 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, BLUE, 2)
        
        # Handle different game states
        if game.game_state == "waiting":
            cv2.putText(frame, "PRESS SPACE TO START ROUND", (w//2 - 250, h//2), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, WHITE, 3)
            
            # Show current score
            cv2.putText(frame, f"SCORE: P1 {game.player1_score} - {game.player2_score} P2", 
                        (w//2 - 200, h//2 + 50), cv2.FONT_HERSHEY_SIMPLEX, 1, YELLOW, 2)
        
        elif game.game_state == "countdown":
            draw_countdown(frame, game)
            game.update_countdown()
        
        elif game.game_state == "playing":
            remaining = game.update_play_time()
            if remaining > 0:
                draw_play_timer(frame, remaining)
                cv2.putText(frame, "MAKE YOUR GESTURES NOW!", (w//2 - 220, 150), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, RED, 3)
        
        elif game.game_state == "result":
            draw_result(frame, game)
            if game.show_result_complete():
                game.reset_round()
        
        # Game instructions at bottom
        cv2.putText(frame, "ROCK=Fist  PAPER=Open Hand  SCISSORS=2 Fingers", 
                    (w//2 - 200, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, WHITE, 2)
        
        cv2.imshow('Two Player Rock Paper Scissors', frame)
        
        # Handle keyboard controls
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord(' '):
            if game.game_state == "waiting":
                game.start_countdown()
        elif key == ord('r'):
            game = TwoPlayerGame()
            print("🔄 Game reset! Starting fresh...")
    
    cap.release()
    cv2.destroyAllWindows()
    
    # Final results
    print(f"🎮 Final Score: Player 1: {game.player1_score} - Player 2: {game.player2_score}")
    
    if game.player1_score > game.player2_score:
        print("🏆 PLAYER 1 WINS THE MATCH! 🎉")
    elif game.player2_score > game.player1_score:
        print("🏆 PLAYER 2 WINS THE MATCH! 🎉")
    else:
        print("🤝 IT'S A TIE MATCH! Great game!")
    
    print("Thanks for playing! 🎮")

if __name__ == "__main__":
    main()