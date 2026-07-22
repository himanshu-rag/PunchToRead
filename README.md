# PunchToRead: The Future of Touchless Interaction 🖐️✨

Welcome to **PunchToRead**, a project born out of a simple question: *What if our computers felt as magical and intuitive as the sci-fi interfaces we grew up watching in movies?* 

This isn't just another Python script. It's a complete, futuristic dashboard powered by raw computer vision, Google's MediaPipe, and OpenCV. It entirely removes the need for a physical mouse, keyboard, or even a touch screen. With PunchToRead, your hands become the ultimate controller. By simply waving, pointing, and pinching in thin air, you can seamlessly navigate complex menus, read dynamic news articles, paint masterpieces on a multi-layered digital canvas, and sculpt 3D spatial models.

It’s fast. It’s fluid. And honestly? It feels like magic. 🪄

---

## 🆕 Latest Major Upgrades (Release Notes)

### 🧊 1. 3D Spatial Sculptor Engine (`3D_MODE`)
- **3-Finger Activation:** Hold up **3 fingers** (Index, Middle, Ring) at the Main Menu to enter the 3D Spatial Sculptor.
- **Fist Camera Rotation:** Close your hand into a **fist** (`0 non-thumb fingers`) and drag in thin air to orbit and rotate the 3D viewport camera (Pitch & Yaw control).
- **3D Spatial Path Drawing:** Point your index finger to paint floating 3D neon trails in physical space.
- **Primitive Shape Placement:** Place 3D **CUBES**, **PYRAMIDS**, and **CYLINDERS** directly into the 3D space in front of your camera.
- **Auto-Extrusion Engine:** Automatically extrude any 2D canvas drawing from Drawing Mode into a full 3D polygonal prism object with one click!

### ⚡ 2. Ultra-Low Latency & Fast Hover Response
- **60 FPS 1-Euro Adaptive Filter:** All interaction modes (`NEWS`, `DRAW`, `3D_MODE`) use 1-Euro adaptive filtered coordinates (`raw_lm_list[8]`) for instantaneous, zero-lag fingertip tracking.
- **Animated Progress Line Hover:** Pointing at any button smoothly fills an animated glowing progress line (`HOVER_TRIGGER_FRAMES = 8`, ~0.25s), giving fast visual feedback before activating.
- **UI Safety Guard Zones (`in_ui_margin`):** Automatic drawing suppression near top toolbars (`y < 75px`) and side panels (`x < 160px`). Zero trailing lines are drawn when reaching for buttons!
- **Main Menu Cooldown:** A 25-frame transition lock when returning to `MAIN_MENU` prevents accidental mode re-entry when clicking `< BACK`.

### 💥 3. Supernova Particle Shatter Effect
- Clearing the canvas triggers a physics-based particle explosion where drawn pixels shatter away across the screen.

---

## 🚀 The Core Philosophy

When I started building this, I knew that traditional gesture controls often feel clunky or unresponsive. Nobody wants to hold their arm out for five seconds just to click a button. I wanted this to feel instantaneous and premium.

To achieve that "Apple-like" level of polish, we implemented:
- **Asynchronous AI Tracking**: The heavy lifting of hand-tracking runs on a completely separate thread using MediaPipe’s `LIVE_STREAM` mode. This guarantees buttery-smooth framerates, ensuring the UI never lags behind your physical hand.
- **Glassmorphism Aesthetics**: The entire UI is built on elegant, frosted-glass panels with perfectly rounded corners. It dynamically blurs the real-time camera feed underneath it, giving it a stunning macOS-inspired look.
- **Magnetic Parallax**: As you move your hand across the screen, the UI panels subtly track your wrist position and tilt towards you. It creates a magnetic, tactile parallax effect that makes the interface feel alive and physically present in the room with you.

---

## 📰 Mode 1: The News Dashboard

If you hold up just **one finger** at the Main Menu, you'll dive straight into News Mode. Here, we track both of your hands simultaneously.

### Flick-to-Swipe Navigation
Forget tedious scrolling. Inside News Mode, you can literally flick your hand left or right in the air. The system calculates your hand's velocity heuristics in real-time, instantly swiping between topics like AI Breakthroughs, Geopolitics, or Tech Jobs. It’s snappy, satisfying, and heavily filtered to prevent accidental jitters.

### The "Punch-to-Read" Mechanic
This is where the app gets its name. When you point at a topic you want to dive into, the system locks on (thanks to some clever debouncing algorithms). When you're ready, simply close your hand into a fist—a literal **"punch"**. 

The UI instantly reacts. The normal dashboard shatters away, the background dynamically dims into an immersive Dark Mode, and a massive, centralized article drops into view. We even dynamically generate and color-grade mock article images using OpenCV civil generative filtering, so every reading experience feels visually distinct and fresh.

---

## 🎨 Mode 2: Mid-Air 2D Drawing

Holding **two fingers** (peace sign) at the main menu unlocks Drawing Mode.

### Dynamic 3D Depth Brush
The camera constantly analyzes the physical Z-axis distance of your hand relative to the lens.
- Push your hand **closer** to the camera for thick, bold lines.
- Pull your hand **further back** for ultra-thin precision details.

### Smart Shapes (Hold-to-Snap)
Draw a rough circle, rectangle, or triangle and hold your finger still at the end of the stroke for half a second—the app automatically vector-snaps it into a mathematically perfect shape.

### Advanced Project Management
- **Multi-Layer Support**: Create up to 5 distinct drawing layers with visibility toggles.
- **Memory Undo System**: Hover over `UNDO` to pop previous strokes off the active layer stack.
- **High-Res Export**: Click `SAVE` to composite and export `drawing_export.png` directly to disk.

---

## 🧊 Mode 3: 3D Spatial Sculptor

Holding **three fingers** at the main menu unlocks 3D Mode.

- **Fist Drag:** Orbit and rotate the 3D viewport camera in 3D space.
- **Index Point:** Paint floating 3D spatial stroke trails.
- **Primitives Menu:** Add 3D Cubes, Pyramids, and Cylinders.
- **EXTRUDE 2D:** Convert your active 2D drawing into a 3D polygonal prism!

---

## 🛠️ Installation & Requirements

### Requirements
- **Python 3.8+**
- **OpenCV** (`opencv-python`)
- **MediaPipe** (`mediapipe`)
- **NumPy** (`numpy`)

### Setup Instructions
```bash
git clone https://github.com/himanshu-rag/PunchToRead.git
cd PunchToRead
pip install -r requirements.txt
python PunchToRead.py
```

---

## 🕹️ How to Use (Controls)

### Main Menu
- **1 Finger (Index):** Enter News Dashboard Mode
- **2 Fingers (Peace Sign):** Enter 2D Drawing Mode
- **3 Fingers (Index, Middle, Ring):** Enter 3D Sculptor Mode

### Navigation & Buttons
- **Hover Over Button:** Point index finger over any button (`< BACK`, `CLEAR`, `RESET CAM`, color swatches). The glowing progress line fills up smoothly (~0.25s) and activates the button.
- **Fist Gesture:** Close hand into a fist in 3D Mode to rotate camera, or in News Mode to "Punch-to-Read".

---

PunchToRead is a completely reimagined way to interact with our digital world. Dive in, wave your hands, and experience the future. 🚀
