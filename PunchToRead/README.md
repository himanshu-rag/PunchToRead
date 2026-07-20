# PunchToRead: Gesture-Controlled News Dashboard

**PunchToRead** is a futuristic, touchless news dashboard powered by computer vision. Built using Python, OpenCV, and Google’s MediaPipe, this application allows users to seamlessly navigate, select, and read detailed news articles using only their hand gestures. Designed to feel like a sci-fi interface, it entirely removes the need for a mouse or keyboard.

### Key Features and Interactions

At its core, the program tracks both the left and right hands simultaneously in real-time. By raising specific fingers to the camera, users are presented with ten distinct news categories. The right hand maps to topics like AI breakthroughs, Geopolitics, and Startup Funding, while the left hand covers areas such as Metro Weather, Financial Budgets, and Multinational Tech Jobs. 

**1. Hover-to-Select**
If you point exactly one finger towards the camera, the system locks onto your choice. A centralized HUD element highlights the current "Hovered" topic so you always know what you are selecting. State-of-the-art debouncing algorithms ensure that selections are incredibly stable, ignoring camera flickers or shaky hands. 

**2. The "Punch-to-Read" Mechanic**
The hallmark feature of this application is its title mechanic: the Punch to Read. Once a topic is selected via hover, simply close that hovering hand into a fist (zero fingers). The software immediately detects this deliberate "punch" gesture, clears the normal UI, and plunges the user into a massive, centralized Article Reading Mode. 

**3. Dynamic Visuals**
Inside the Reading Mode, users are treated to a beautifully wrapped, multi-paragraph mock news article. But the magic lies in the visuals. The dashboard automatically loads one of 20 high-quality, AI-generated images dynamically matched to the news topic. If an image is reused, civil generative OpenCV filtering applies distinct color grading and tinting so the visual experience always feels fresh and unique. 

**4. Two-Handed Exit**
To exit the immersive reading article, users must deliberately hold up both hands completely open (all 10 fingers). Upon doing so for a fraction of a second, the article vanishes and control is gracefully returned to the main dashboard. 

### Recent Updates (Version 2.0)
- **Apple-like Minimalistic UI**: Elegant, frosted glass panels (glassmorphism) and perfectly rounded corners, giving the interface a sleek, premium, macOS-inspired aesthetic.
- **Immersive Dark Mode**: Reading mode dynamically dims the background feed, creating a highly focused dark mode reading environment.
- **Massive Performance Boost**: We completely decoupled the heavy AI hand-tracking from the camera feed by utilizing MediaPipe's asynchronous `LIVE_STREAM` mode.

### Recent Updates (Version 3.0: The Gestural Update)
- **New Main Menu**: Upon launch, you're greeted with a sleek Main Menu! Hold 1 finger for News Mode, or 2 fingers for the all-new Drawing Mode.
- **Mid-Air Drawing Mode**: A fully interactive drawing canvas! Hold your Index finger to draw with a glowing cursor, add your Middle finger (Peace sign) to hover/move without drawing, and open your whole hand to clear the canvas.
- **Flick-to-Swipe Navigation**: Inside News Mode, you can now seamlessly swipe left or right with your hand to change topics instantly. Powered by velocity-tracking heuristics and an anti-jitter moving average filter.
- **3D Pinch-to-Exit**: Closing content or exiting modes is now done effortlessly via a natural 3-finger pinch gesture (Thumb, Index, Middle), utilizing 3D Euclidean distance calculations for extreme precision.
- **Magnetic Parallax**: The frosted glass UI panels now subtlely track your wrist position, giving a magnetic, tactile parallax effect that brings the interface to life.

PunchToRead is a robust, dynamic, and incredibly satisfying dive into the future of human-computer interaction.
