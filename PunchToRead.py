import cv2
import time
import math
import glob
import os
import random
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20)
]

def draw_landmarks(image, hand_landmarks):
    h, w, c = image.shape
    # Draw connections
    for connection in HAND_CONNECTIONS:
        p1 = hand_landmarks[connection[0]]
        p2 = hand_landmarks[connection[1]]
        x1, y1 = int(p1.x * w), int(p1.y * h)
        x2, y2 = int(p2.x * w), int(p2.y * h)
        cv2.line(image, (x1, y1), (x2, y2), (255, 255, 255), 2)
        
    # Draw points
    for landmark in hand_landmarks:
        x, y = int(landmark.x * w), int(landmark.y * h)
        cv2.circle(image, (x, y), 5, (0, 0, 255), cv2.FILLED)

def main():
    # Setup Tasks API options
    BaseOptions = mp.tasks.BaseOptions
    HandLandmarker = mp.tasks.vision.HandLandmarker
    HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    # The single selected topic (locked in when exactly 1 finger is raised)
    selected_topic_global = None
    
    # Handedness that made the selection (to require the same hand to punch)
    selected_topic_hand = None
    
    # Is the user currently punching?
    show_content_mode = False
    
    # Debounce frames
    hover_frames = 0
    punch_frames = 0
    exit_frames = 0
    last_hovered_topic = None
    
    # Pre-loading Generated AI Images
    art_dir = "./resources"
    image_paths = {
        'Thumb: AI news': glob.glob(f"{art_dir}/ai_news_*.png"),
        'Index: Geo political': glob.glob(f"{art_dir}/geo_politics_*.png"),
        'Middle: India news': glob.glob(f"{art_dir}/india_news_*.png"),
        'Ring: Indian frauds': glob.glob(f"{art_dir}/indian_frauds_*.png"),
        'Pinky: AI & Startups': glob.glob(f"{art_dir}/ai_startups_*.png"),
        'Thumb: India budget': glob.glob(f"{art_dir}/india_budget_*.png"),
        'Index: MNC Jobs': glob.glob(f"{art_dir}/mnc_jobs_*.png"),
        'Middle: Claude AI': glob.glob(f"{art_dir}/claude_ai_*.png"),
        'Ring: Weather/AQI': glob.glob(f"{art_dir}/weather_aqi_*.png"),
        'Pinky: Mobile Tech': glob.glob(f"{art_dir}/mobile_tech_*.png")
    }
    loaded_images = {}
    for k, paths_list in image_paths.items():
        loaded_images[k] = []
        for p in paths_list:
            img = cv2.imread(p)
            if img is not None:
                img = cv2.resize(img, (320, 320))
                loaded_images[k].append(img)
                
    current_display_image = None
    
    # Mock content library
    mock_content = {
        'Thumb: AI news': "OpenAI drops new GPT-5 model focusing on advanced reasoning. In parallel, Meta has released Llama 4 to aggressive open-source adoption. The AI sector is experiencing a massive boom globally with trillion-dollar infrastructure investments spanning data centers and semiconductor manufacturing. Experts believe we are approaching a critical inflection point in machine intelligence.",
        'Index: Geo political': "Global summits conclude with sweeping new trade agreements aiming to stabilize international markets. Meanwhile, cross-border tensions have noticeably eased in eastern Europe following a series of diplomatic breakthroughs. Analysts predict these developments will lead to a 15% increase in global exports over the next fiscal year as supply chains regularize.",
        'Middle: India news': "Sensex reaches an extraordinary all-time high amidst remarkably strong domestic economic indicators and booming foreign direct investments. The manufacturing sector showed a 12% YoY growth, heavily driven by the 'Make in India' initiative. Retail inflation has also surprisingly cooled down, offering substantial relief to the middle class.",
        'Ring: Indian frauds': "The Enforcement Directorate (ED) has successfully uncovered a massive Rs 500 crore cryptocurrency scam operating primarily from tier-2 cities. Several dummy corporations were used to launder money under the guise of fake IT service exports. Three high-profile political figures have been summoned for questioning regarding their alleged involvement.",
        'Pinky: AI & Startups': "A leading Indian AI startup has secured a massive $50M Series B funding round to build localized, multilingual foundational models for rural sectors. The funding was led by top Silicon Valley venture capitalists. This milestone represents a paradigm shift for Indian tech, pivoting from service-oriented models to core DeepTech product innovation.",
        'Thumb: India budget': "The Finance Minister has officially announced major tax relief aimed directly at the middle class, alongside a staggering 20% boost in capital infrastructure spending. The new budget heavily prioritizes green energy initiatives and sustainable agriculture. Subsidies for electric vehicle manufacturing have also been extended until 2030.",
        'Index: MNC Jobs': "Multinational tech giants have collectively announced massive hiring drives in Bengaluru, Pune, and Hyderabad for the upcoming 2026 fiscal year. Over 100,000 new roles are expected to be created, heavily emphasizing prompt engineering, cybersecurity, and cloud architecture. Average starting salaries have reportedly increased by 15%.",
        'Middle: Claude AI': "Anthropic's newly released Claude 3.5 Opus is reportedly beginning to replace traditional Tier-1 customer support agents entirely across Fortune 500 companies. The model exhibits near-human empathy and reasoning out-of-the-box. Ethical debates are raging regarding workforce displacement, though companies cite a 300% increase in resolution speed.",
        'Ring: Weather/AQI': "The Air Quality Index (AQI) in Delhi NCR has once again reached the 'severe+' category, prompting emergency measures including school closures and heavy vehicle bans. Conversely, the IMD has predicted unseasonably heavy rainfall and potential cyclonic warnings across the southern peninsula over the next 48 hours.",
        'Pinky: Mobile Tech': "Apple and Samsung have jointly showcased functional prototypes of fully foldable tablets equipped with transparent micro-LED displays at the latest keynote. These devices utilize novel polymer batteries that charge fully in under 3 minutes. Consumer release is eagerly anticipated for late 2027."
    }
    
    def put_wrapped_text(img, text, position, font, font_scale, color, thickness, max_width):
        words = text.split(' ')
        lines = []
        current_line = words[0]
        for word in words[1:]:
            (w, h), _ = cv2.getTextSize(current_line + " " + word, font, font_scale, thickness)
            if w < max_width:
                current_line += " " + word
            else:
                lines.append(current_line)
                current_line = word
        lines.append(current_line)
        
        y = position[1]
        for line in lines:
            cv2.putText(img, line, (position[0], y), font, font_scale, color, thickness)
            y += int(h * 1.5)

    # Initialize webcam
    cap = cv2.VideoCapture(0)
    # Give the camera a moment to warm up
    time.sleep(1)
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path='hand_landmarker.task'),
        running_mode=VisionRunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.7,
        min_hand_presence_confidence=0.7,
        min_tracking_confidence=0.7)

    with HandLandmarker.create_from_options(options) as landmarker:
        p_time = 0
        start_time_ms = time.time() * 1000
        last_timestamp_ms = 0
        
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                break
            
            # Flip the image horizontally for a selfie-view display
            image = cv2.flip(image, 1)
            
            # Convert BGR to RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
            
            # Calculate strictly monotonically increasing timestamps for VIDEO mode
            current_timestamp_ms = int(time.time() * 1000 - start_time_ms)
            if current_timestamp_ms <= last_timestamp_ms:
                current_timestamp_ms = last_timestamp_ms + 1
            last_timestamp_ms = current_timestamp_ms
            
            # Process the image
            results = landmarker.detect_for_video(mp_image, current_timestamp_ms)
            
            # Store active fingers and news
            active_fingers_texts = []
            active_news_right = [] # Physical Right Hand (Mediapipe 'Left')
            active_news_left = []  # Physical Left Hand (Mediapipe 'Right')
            
            if results.hand_landmarks:
                for hand_idx, hand_landmarks in enumerate(results.hand_landmarks):
                    # Draw custom landmarks 
                    draw_landmarks(image, hand_landmarks)
                    
                    # Determine handedness (Left or Right)
                    handedness_list = results.handedness[hand_idx]
                    handedness_label = handedness_list[0].category_name
                    
                    tip_ids = [4, 8, 12, 16, 20]
                    finger_names = ['T', 'I', 'M', 'R', 'P']
                    
                    # Logic for Thumb
                    # A more robust detection: determine root of palm (0) and tip of thumb (4)
                    # and compare distance to index base (5) or pinky base (17)
                    import math
                    
                    def get_distance(p1, p2):
                        return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)
                    
                    # The thumb tip is landmark 4.
                    # Base of the index finger is landmark 5. 
                    # Base of the pinky is 17.
                    # If thumb tip is further from pinky base than the thumb IP joint (3), it's extended.
                    thumb_tip = hand_landmarks[4]
                    thumb_ip = hand_landmarks[3]
                    pinky_base = hand_landmarks[17]
                    
                    dist_tip_to_pinky_base = get_distance(thumb_tip, pinky_base)
                    dist_ip_to_pinky_base = get_distance(thumb_ip, pinky_base)
                    
                    # If the tip is significantly further away from the other side of the palm 
                    # than the second joint of the thumb, it's open.
                    
                    hand_fingers = []
                    
                    if dist_tip_to_pinky_base > dist_ip_to_pinky_base + 0.02:
                        active_fingers_texts.append('T')
                        hand_fingers.append('T')
                            
                    # Logic for other 4 Fingers
                    for id in range(1, 5):
                        if hand_landmarks[tip_ids[id]].y < hand_landmarks[tip_ids[id] - 2].y:
                            active_fingers_texts.append(finger_names[id])
                            hand_fingers.append(finger_names[id])
                            
                    # Track active news for both side panels
                    if hand_fingers:
                        if handedness_label == 'Left':  # Physical Right Hand
                            news_mapping_right = {
                                'T': 'Thumb: AI news',
                                'I': 'Index: Geo political',
                                'M': 'Middle: India news',
                                'R': 'Ring: Indian frauds',
                                'P': 'Pinky: AI & Startups'
                            }
                            for finger in hand_fingers:
                                news_text = news_mapping_right.get(finger, '')
                                if news_text:
                                    active_news_right.append(news_text)
                                    
                        elif handedness_label == 'Right':  # Physical Left Hand
                            news_mapping_left = {
                                'T': 'Thumb: India budget',
                                'I': 'Index: MNC Jobs',
                                'M': 'Middle: Claude AI',
                                'R': 'Ring: Weather/AQI',
                                'P': 'Pinky: Mobile Tech'
                            }
                            for finger in hand_fingers:
                                news_text = news_mapping_left.get(finger, '')
                                if news_text:
                                    active_news_left.append(news_text)
                        
                    # Central Hover Selection Logic with Debounce
                    if len(hand_fingers) == 1:
                        finger = hand_fingers[0]
                        current_hover_topic = news_mapping_right.get(finger) if handedness_label == 'Left' else news_mapping_left.get(finger)
                        
                        if current_hover_topic == last_hovered_topic:
                            hover_frames += 1
                        else:
                            hover_frames = 1
                            last_hovered_topic = current_hover_topic
                            
                        # Lock in selection after actively holding for 10 frames
                        if hover_frames >= 10:
                            selected_topic_global = current_hover_topic
                            selected_topic_hand = handedness_label
                    else:
                        hover_frames = 0
                            
                    # Punch to Read Logic with Debounce
                    # If the selected topic's hand is currently making a fist (0 fingers)
                    if selected_topic_global and len(hand_fingers) == 0 and handedness_label == selected_topic_hand:
                        punch_frames += 1
                        if punch_frames >= 10 and not show_content_mode:
                            show_content_mode = True
                            
                            # Pick a random image on entry!
                            if selected_topic_global in loaded_images and len(loaded_images[selected_topic_global]) > 0:
                                imgs = loaded_images[selected_topic_global]
                                chosen = random.choice(imgs)
                                
                                # If we only generated 1 image for this topic (rate limits), dynamically apply a color filter so it looks different
                                if len(imgs) == 1:
                                    hsv = cv2.cvtColor(chosen, cv2.COLOR_BGR2HSV)
                                    h, s, v = cv2.split(hsv)
                                    h = (h + random.randint(30, 150)) % 180 # Rotate hue
                                    v = cv2.add(v, random.randint(-40, 40)) # Tweak brightness
                                    current_display_image = cv2.cvtColor(cv2.merge((h, s, v)), cv2.COLOR_HSV2BGR)
                                else:
                                    current_display_image = chosen
                            else:
                                current_display_image = None
                    elif len(hand_fingers) > 0:
                        punch_frames = 0
                            
            total_fingers = len(active_fingers_texts)
            
            # Calculate FPS
            c_time = time.time()
            fps = 1 / (c_time - p_time) if (c_time - p_time) > 0 else 0
            p_time = c_time
            
            h, w, c = image.shape
            
            # --- CONTENT READING MODE (PUNCH) ---
            if show_content_mode and selected_topic_global:
                # Exit condition: 10 fingers held for 15 consecutive frames
                if total_fingers == 10:
                    exit_frames += 1
                    if exit_frames >= 15:
                        show_content_mode = False
                        selected_topic_global = None
                        selected_topic_hand = None
                        exit_frames = 0
                        hover_frames = 0
                        continue # Skip drawing to exit immediately this frame
                else:
                    exit_frames = 0

                # Dim the background slightly
                overlay = image.copy()
                cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), cv2.FILLED)
                cv2.addWeighted(overlay, 0.6, image, 0.4, 0, image)
                
                # Big central white box for reading
                box_w, box_h = 1000, 480
                cx, cy = int(w/2), int(h/2)
                x1, y1 = cx - int(box_w/2), cy - int(box_h/2)
                x2, y2 = cx + int(box_w/2), cy + int(box_h/2)
                
                cv2.rectangle(image, (x1, y1), (x2, y2), (255, 255, 255), cv2.FILLED)
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 4) # Green reading border
                
                # Title
                cv2.putText(image, f"IN-DEPTH: {selected_topic_global}", (x1 + 30, y1 + 60), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
                            
                # Separator line
                cv2.line(image, (x1 + 30, y1 + 80), (x2 - 30, y1 + 80), (200, 200, 200), 2)
                
                # --- LIVE VISUALIZATION / IMAGES ---
                vis_w, vis_h = 320, 320
                vis_x = x2 - vis_w - 40
                vis_y = y1 + 100
                cv2.rectangle(image, (vis_x, vis_y), (vis_x + vis_w, vis_y + vis_h), (240, 240, 240), cv2.FILLED)
                cv2.rectangle(image, (vis_x, vis_y), (vis_x + vis_w, vis_y + vis_h), (100, 100, 100), 2)
                
                if current_display_image is not None:
                    # Draw actual topic image
                    image[vis_y:vis_y+320, vis_x:vis_x+320] = current_display_image
                else:
                    # Fallback to animated chart
                    cv2.putText(image, "Live Context Data", (vis_x + 10, vis_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 2)
                    t = time.time()
                    bar_heights = [
                        int(80 + 50 * math.sin(t * 2)), 
                        int(120 + 40 * math.cos(t * 3)), 
                        int(90 + 60 * math.sin(t * 1.5)), 
                        int(140 + 30 * math.cos(t))
                    ]
                    
                    for i, bh in enumerate(bar_heights):
                        bx = vis_x + 30 + i * 70
                        by = vis_y + vis_h - 10 - bh
                        color_shift = int(127 + 128 * math.sin(t * 4 + i))
                        cv2.rectangle(image, (bx, by), (bx + 50, vis_y + vis_h - 10), (color_shift, 150, 255 - color_shift), cv2.FILLED)
                        cv2.rectangle(image, (bx, by), (bx + 50, vis_y + vis_h - 10), (0, 0, 0), 2)

                # Wrapped Content
                content = mock_content.get(selected_topic_global, "Content not found for this topic.")
                # Dynamically set width so it stops right before the visualization starts
                text_max_width = box_w - vis_w - 90
                put_wrapped_text(image, content, (x1 + 30, y1 + 140), cv2.FONT_HERSHEY_SIMPLEX, 
                                 0.8, (0, 0, 0), 2, text_max_width)
                                 
                # Instructions to exit
                cv2.putText(image, "SHOW BOTH HANDS FULLY OPEN (10 FINGERS) TO EXIT", (cx - 380, y2 - 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 3)
                            
            # --- NORMAL UI MODE ---
            else:
                # Display FPS
                cv2.putText(image, f'FPS: {int(fps)}', (1100, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                            1, (255, 0, 0), 2)
                            
                # Top-Center Display for Overall Finger Count
                cv2.rectangle(image, (int(w/2) - 150, 20), (int(w/2) + 150, 90), (0, 0, 0), cv2.FILLED)
                cv2.putText(image, f'Total Fingers: {total_fingers}', (int(w/2) - 130, 65), cv2.FONT_HERSHEY_SIMPLEX, 
                            0.9, (0, 255, 0), 3)
                
                # --- Right Hand News Box (Instant Tracking) ---
                if active_news_right:
                    box_width_R = 300
                    box_height_R = max(70, 40 * len(active_news_right) + 30)
                    
                    cv2.rectangle(image, (w - box_width_R - 20, 110), (w - 20, 110 + box_height_R), (255, 255, 255), cv2.FILLED)
                    cv2.rectangle(image, (w - box_width_R - 20, 110), (w - 20, 110 + box_height_R), (0, 0, 0), 2)
                    
                    cv2.putText(image, f'Right Hand Topics:', (w - box_width_R + 10, 135), cv2.FONT_HERSHEY_SIMPLEX, 
                                0.6, (0, 0, 255), 2)
                    
                    y_pos = 165
                    for news in active_news_right:
                        cv2.putText(image, news, (w - box_width_R + 10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 
                                    0.6, (0, 0, 0), 2)
                        y_pos += 35
                else:
                    box_width_R = 300
                    cv2.rectangle(image, (w - box_width_R - 20, 110), (w - 20, 180), (255, 255, 255), cv2.FILLED)
                    cv2.rectangle(image, (w - box_width_R - 20, 110), (w - 20, 180), (0, 0, 0), 2)
                    
                    cv2.putText(image, f'Right Hand Topics:', (w - box_width_R + 10, 135), cv2.FONT_HERSHEY_SIMPLEX, 
                                0.6, (0, 0, 0), 2)
                    cv2.putText(image, 'None', (w - box_width_R + 10, 165), cv2.FONT_HERSHEY_SIMPLEX, 
                                0.6, (150, 150, 150), 1)

                # --- Left Hand News Box (Instant Tracking) ---
                if active_news_left:
                    box_width_L = 300
                    box_height_L = max(70, 40 * len(active_news_left) + 30)
                    
                    cv2.rectangle(image, (20, 110), (20 + box_width_L, 110 + box_height_L), (255, 255, 255), cv2.FILLED)
                    cv2.rectangle(image, (20, 110), (20 + box_width_L, 110 + box_height_L), (0, 0, 0), 2)
                    
                    cv2.putText(image, f'Left Hand Topics:', (30, 135), cv2.FONT_HERSHEY_SIMPLEX, 
                                0.6, (0, 0, 255), 2)
                    
                    y_pos = 165
                    for news in active_news_left:
                        cv2.putText(image, news, (30, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 
                                    0.6, (0, 0, 0), 2)
                        y_pos += 35
                else:
                    box_width_L = 300
                    cv2.rectangle(image, (20, 110), (20 + box_width_L, 180), (255, 255, 255), cv2.FILLED)
                    cv2.rectangle(image, (20, 110), (20 + box_width_L, 180), (0, 0, 0), 2)
                    
                    cv2.putText(image, f'Left Hand Topics:', (30, 135), cv2.FONT_HERSHEY_SIMPLEX, 
                                0.6, (0, 0, 0), 2)
                    cv2.putText(image, 'None', (30, 165), cv2.FONT_HERSHEY_SIMPLEX, 
                                0.6, (150, 150, 150), 1)

                # --- Central Global "Selected News" Box ---
                cv2.rectangle(image, (int(w/2) - 300, h - 80), (int(w/2) + 300, h - 20), (255, 255, 255), cv2.FILLED)
                cv2.rectangle(image, (int(w/2) - 300, h - 80), (int(w/2) + 300, h - 20), (0, 0, 0), 3) # Thicker Border
                
                cv2.putText(image, 'SELECTED NEWS:', (int(w/2) - 280, h - 45), cv2.FONT_HERSHEY_SIMPLEX, 
                            0.7, (0, 0, 255), 2)
                            
                if selected_topic_global:
                    cv2.putText(image, selected_topic_global, (int(w/2) - 100, h - 45), cv2.FONT_HERSHEY_SIMPLEX, 
                                0.8, (0, 150, 0), 2)
                else:
                    cv2.putText(image, 'Raise exactly 1 finger to Select', (int(w/2) - 100, h - 45), cv2.FONT_HERSHEY_SIMPLEX, 
                                0.7, (150, 150, 150), 1)
            
            # Show the final image
            cv2.imshow('Precise Finger Counter', image)
            
            # Break loop on 'q' or ESC
            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
