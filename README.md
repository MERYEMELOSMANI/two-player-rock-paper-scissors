# Two Player Rock Paper Scissors 👥⚡

A multiplayer computer vision game where two players compete in Rock Paper Scissors using hand gestures! Each player gets their own side of the screen with automatic timer and score tracking.

## Features ✨

- **Dual player support** - Play with a friend side by side
- **Automatic hand separation** - Left and right sides of screen
- **Timer-based rounds** - 5 seconds to make your gesture
- **Real-time score tracking** - See who's winning instantly
- **Visual countdown** - Exciting 3-2-1 countdown before each round
- **Clear player zones** - Distinct areas for each player

## How to Play 🎮

### Setup
- **Player 1** stands on the **LEFT** side of the camera
- **Player 2** stands on the **RIGHT** side of the camera
- Make sure both players are visible in their respective zones

### Gameplay
1. Press **SPACE** to start a new round
2. Watch the **3-2-1 countdown**
3. When "SHOW YOUR HANDS!" appears, both players have **5 seconds**
4. Make your gesture and hold it during the timer
5. Results appear automatically with score update
6. Get ready for the next round!

### Gestures
- 🪨 **ROCK** = Closed fist (all fingers down)
- 📄 **PAPER** = Open hand (all fingers extended)
- ✂️ **SCISSORS** = Index and middle finger extended

## Installation 🚀

### Requirements
- Python 3.7 or higher
- Webcam with good resolution
- Room for two players side by side
- Decent lighting for hand detection

### Setup Steps
```bash
# Clone or download this project
git clone [your-repo-url]
cd two-player-rock-paper-scissors

# Install dependencies
pip install -r requirements.txt

# Run the game
python main.py
```

## Controls 🕹️

- **SPACE** - Start a new round
- **R** - Reset entire game (scores back to 0-0)
- **Q** - Quit the game

## Game Features 🎯

### Visual Feedback
- **Green zone** for Player 1 (left side)
- **Blue zone** for Player 2 (right side)
- **Hand tracking dots** show detected hand positions
- **Live gesture recognition** displays current gesture
- **Progress timer bar** during play phase

### Scoring System
- Each round winner gets 1 point
- Ties give no points to either player
- Score displayed continuously
- Final winner announced at game end

### Timer System
- **3-second countdown** before each round
- **5-second play window** to make gestures
- **4-second result display** to see who won
- Automatic progression to next round

## Tips for Best Experience 📋

### Positioning
- Stand about 3-4 feet from the camera
- Make sure you're in your designated zone
- Keep your hand clearly visible
- Avoid overlapping into the other player's area

### Gestures
- Make **clear, distinct** gestures
- Hold your gesture **steady** during the timer
- **Face your palm** toward the camera for better detection
- Practice the gestures before starting

### Environment
- Use **good lighting** - avoid shadows on hands
- **Plain background** works better than busy patterns
- Make sure **both players fit** comfortably in frame
- **Remove distractions** from the background

## Technical Details 🔧

### Built With
- **OpenCV** - Video capture and image processing
- **MediaPipe** - Advanced hand landmark detection
- **NumPy** - Mathematical operations and arrays
- **Python** - Game logic and user interface

### How It Works
1. Camera captures video feed
2. MediaPipe detects up to 2 hands simultaneously
3. Hand positions determine which player owns each hand
4. Finger positions are analyzed to classify gestures
5. Game logic manages rounds, timing, and scoring
6. Results are calculated and displayed automatically

## Troubleshooting 🛠️

### Common Issues

**Hands not detected?**
- Improve lighting in the room
- Make sure hands are fully visible
- Check camera quality and position
- Try different hand positions

**Wrong player assignment?**
- Make sure players are clearly on left/right sides
- Avoid crossing into the other player's zone
- Try repositioning further apart

**Gestures not recognized?**
- Make clearer, more distinct gestures
- Hold gestures steady for longer
- Ensure fingers are clearly extended or closed
- Practice the three gestures beforehand

**Game running slowly?**
- Close other resource-intensive applications
- Try lowering camera resolution
- Ensure good computer performance

## Game Variations 🎲

### Tournament Mode
- Play best-of-5 or best-of-10 rounds
- Track multiple game wins
- Create brackets for multiple player pairs

### Speed Mode
- Reduce timer to 3 seconds for faster gameplay
- Quick succession rounds
- Test reaction time and decision making

### Practice Mode
- Single player vs computer
- Focus on gesture accuracy
- Learn the timing system

## Future Enhancements 🌟

Ideas for expanding this project:
- **Sound effects** for countdown and results
- **Different game modes** (best of X, sudden death)
- **Gesture difficulty levels** (more complex hand positions)
- **Replay system** to review close rounds
- **Statistics tracking** (win rates, favorite gestures)
- **Tournament brackets** for multiple players
- **Custom player names** and avatars

## Contributing 🤝

This is a fun learning project! Feel free to:
- Fork and add your own features
- Improve the gesture recognition
- Add visual enhancements
- Create new game modes
- Share your improvements

## License 📄

Open source project - use it for learning, teaching, or just having fun with friends!

---

**Ready to challenge your friend? 🏆**

*Get your hands ready and may the best gesture win! 🎉*