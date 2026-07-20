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

def get_distance(p1, p2):
    return math.hypot(p1.x - p2.x, p1.y - p2.y)

def get_distance3d(p1, p2):
    return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2 + (p1.z - p2.z)**2)

def draw_rounded_rect(img, top_left, bottom_right, color, thickness=cv2.FILLED, radius=15):
    """Draws a rounded rectangle using OpenCV."""
    x1, y1 = top_left
    x2, y2 = bottom_right
    
    if thickness >= 0:
        # Draw straight lines
        cv2.line(img, (x1 + radius, y1), (x2 - radius, y1), color, thickness, cv2.LINE_AA)
        cv2.line(img, (x1 + radius, y2), (x2 - radius, y2), color, thickness, cv2.LINE_AA)
        cv2.line(img, (x1, y1 + radius), (x1, y2 - radius), color, thickness, cv2.LINE_AA)
        cv2.line(img, (x2, y1 + radius), (x2, y2 - radius), color, thickness, cv2.LINE_AA)
        
        # Draw arcs
        cv2.ellipse(img, (x1 + radius, y1 + radius), (radius, radius), 180, 0, 90, color, thickness, cv2.LINE_AA)
        cv2.ellipse(img, (x2 - radius, y1 + radius), (radius, radius), 270, 0, 90, color, thickness, cv2.LINE_AA)
        cv2.ellipse(img, (x2 - radius, y2 - radius), (radius, radius), 0, 0, 90, color, thickness, cv2.LINE_AA)
        cv2.ellipse(img, (x1 + radius, y2 - radius), (radius, radius), 90, 0, 90, color, thickness, cv2.LINE_AA)
    else:
        # Fill the center body
        cv2.rectangle(img, (x1 + radius, y1), (x2 - radius, y2), color, cv2.FILLED)
        cv2.rectangle(img, (x1, y1 + radius), (x2, y2 - radius), color, cv2.FILLED)
        # Fill the corners
        cv2.ellipse(img, (x1 + radius, y1 + radius), (radius, radius), 180, 0, 90, color, cv2.FILLED, cv2.LINE_AA)
        cv2.ellipse(img, (x2 - radius, y1 + radius), (radius, radius), 270, 0, 90, color, cv2.FILLED, cv2.LINE_AA)
        cv2.ellipse(img, (x2 - radius, y2 - radius), (radius, radius), 0, 0, 90, color, cv2.FILLED, cv2.LINE_AA)
        cv2.ellipse(img, (x1 + radius, y2 - radius), (radius, radius), 90, 0, 90, color, cv2.FILLED, cv2.LINE_AA)

def draw_glass_panel(img, top_left, bottom_right, radius=15, alpha=0.6, bg_color=(30, 30, 30), border_color=(100, 100, 100)):
    x1, y1 = top_left
    x2, y2 = bottom_right
    h, w = img.shape[:2]
    
    valid_x1, valid_y1 = max(0, x1), max(0, y1)
    valid_x2, valid_y2 = min(w, x2), min(h, y2)
    
    if valid_x2 <= valid_x1 or valid_y2 <= valid_y1: return
    
    roi = img[valid_y1:valid_y2, valid_x1:valid_x2]
    panel_w, panel_h = x2 - x1, y2 - y1
    
    mask = np.zeros((panel_h, panel_w), dtype=np.uint8)
    draw_rounded_rect(mask, (0, 0), (panel_w, panel_h), 255, thickness=cv2.FILLED, radius=radius)
    
    mx1, mx2 = valid_x1 - x1, valid_x2 - x1
    my1, my2 = valid_y1 - y1, valid_y2 - y1
    valid_mask = mask[my1:my2, mx1:mx2]
    
    colored_roi = roi.copy()
    idx = (valid_mask == 255)
    colored_roi[idx] = bg_color
    
    cv2.addWeighted(colored_roi, alpha, roi, 1 - alpha, 0, roi)
    draw_rounded_rect(img, top_left, bottom_right, border_color, thickness=1, radius=radius)

def draw_landmarks(image, hand_landmarks):
    h, w, c = image.shape
    # Draw subtle connections
    for connection in HAND_CONNECTIONS:
        p1 = hand_landmarks[connection[0]]
        p2 = hand_landmarks[connection[1]]
        x1, y1 = int(p1.x * w), int(p1.y * h)
        x2, y2 = int(p2.x * w), int(p2.y * h)
        cv2.line(image, (x1, y1), (x2, y2), (255, 255, 255), 1, cv2.LINE_AA)
        
    # Draw subtle elegant dots
    for landmark in hand_landmarks:
        x, y = int(landmark.x * w), int(landmark.y * h)
        cv2.circle(image, (x, y), 3, (220, 220, 220), cv2.FILLED, cv2.LINE_AA)

wrapped_content_cache = {}

def get_wrapped_lines(text, font, font_scale, thickness, max_width):
    cache_key = (text, font, font_scale, thickness, max_width)
    if cache_key in wrapped_content_cache:
        return wrapped_content_cache[cache_key]
    
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
    
    (_, h), _ = cv2.getTextSize("Ay", font, font_scale, thickness)
    wrapped_content_cache[cache_key] = (lines, h)
    return lines, h

def put_wrapped_text(img, text, position, font, font_scale, color, thickness, max_width):
    lines, h = get_wrapped_lines(text, font, font_scale, thickness, max_width)
    y = position[1]
    for line in lines:
        cv2.putText(img, line, (position[0], y), font, font_scale, color, thickness, cv2.LINE_AA)
        y += int(h * 1.6)

def main():
    # Setup Tasks API options
    BaseOptions = mp.tasks.BaseOptions
    HandLandmarker = mp.tasks.vision.HandLandmarker
    HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    # State variables
    app_mode = 'MAIN_MENU'
    menu_selection_choice = None
    menu_selection_frames = 0
    canvas = None
    prev_draw_x, prev_draw_y = 0, 0
    
    selected_topic_global = None
    selected_topic_hand = None
    
    class SmoothLM:
        def __init__(self, x, y, z):
            self.x = x; self.y = y; self.z = z
            
    hand_ema = {}
    EMA_ALPHA = 0.7  # 0.7 for high responsiveness, low lag
    
    hover_frames = 0
    punch_frames = 0
    exit_frames = 0
    last_hovered_topic = None
    
    swipe_cooldown_frames = 0
    transition_frames = 0
    transition_direction = 1
    
    wrist_x_history = []
    current_wrist_x = 0.5
    
    TOPICS_ORDER = [
        'Thumb: AI news', 'Index: Geo political', 'Middle: India news', 
        'Ring: Indian frauds', 'Pinky: AI & Startups', 'Thumb: India budget', 
        'Index: MNC Jobs', 'Middle: Claude AI', 'Ring: Weather/AQI', 'Pinky: Mobile Tech'
    ]
    
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
    
    def update_display_image():
        nonlocal current_display_image
        if selected_topic_global in loaded_images and len(loaded_images[selected_topic_global]) > 0:
            imgs = loaded_images[selected_topic_global]
            chosen = random.choice(imgs)
            if len(imgs) == 1:
                hsv = cv2.cvtColor(chosen, cv2.COLOR_BGR2HSV)
                h_c, s_c, v_c = cv2.split(hsv)
                h_c = (h_c + random.randint(30, 150)) % 180
                v_c = cv2.add(v_c, random.randint(-40, 40))
                current_display_image = cv2.cvtColor(cv2.merge((h_c, s_c, v_c)), cv2.COLOR_HSV2BGR)
            else:
                current_display_image = chosen
        else:
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

    news_mapping_right = {
        'T': 'Thumb: AI news',
        'I': 'Index: Geo political',
        'M': 'Middle: India news',
        'R': 'Ring: Indian frauds',
        'P': 'Pinky: AI & Startups'
    }
    
    news_mapping_left = {
        'T': 'Thumb: India budget',
        'I': 'Index: MNC Jobs',
        'M': 'Middle: Claude AI',
        'R': 'Ring: Weather/AQI',
        'P': 'Pinky: Mobile Tech'
    }

    # Initialize webcam
    cap = cv2.VideoCapture(0)
    time.sleep(1)
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    shared_state = {'results': None}
    
    def result_callback(result, output_image, timestamp_ms):
        shared_state['results'] = result

    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path='hand_landmarker.task'),
        running_mode=VisionRunningMode.LIVE_STREAM,
        result_callback=result_callback,
        num_hands=2,
        min_hand_detection_confidence=0.7,
        min_hand_presence_confidence=0.7,
        min_tracking_confidence=0.7)

    with HandLandmarker.create_from_options(options) as landmarker:
        p_time = 0
        fps_history = []
        start_time_ms = time.time() * 1000
        last_timestamp_ms = 0
        
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                break
            
            image = cv2.flip(image, 1)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
            
            current_timestamp_ms = int(time.time() * 1000 - start_time_ms)
            if current_timestamp_ms <= last_timestamp_ms:
                current_timestamp_ms = last_timestamp_ms + 1
            last_timestamp_ms = current_timestamp_ms
            
            try:
                landmarker.detect_async(mp_image, current_timestamp_ms)
            except Exception as e:
                pass # Skip frame if timestamp collision
                
            results = shared_state['results']
            
            active_fingers_texts = []
            active_news_right = []
            active_news_left = []
            pinch_detected = False
            
            active_handedness = set()
            h, w, c = image.shape
            
            if results and results.hand_landmarks:
                for hand_idx, raw_hand_landmarks in enumerate(results.hand_landmarks):
                    handedness_label = results.handedness[hand_idx][0].category_name
                    active_handedness.add(handedness_label)
                    
                    if handedness_label not in hand_ema:
                        hand_ema[handedness_label] = [(lm.x, lm.y, lm.z) for lm in raw_hand_landmarks]
                        
                    hand_landmarks = []
                    for i, lm in enumerate(raw_hand_landmarks):
                        prev_x, prev_y, prev_z = hand_ema[handedness_label][i]
                        new_x = prev_x + EMA_ALPHA * (lm.x - prev_x)
                        new_y = prev_y + EMA_ALPHA * (lm.y - prev_y)
                        new_z = prev_z + EMA_ALPHA * (lm.z - prev_z)
                        hand_ema[handedness_label][i] = (new_x, new_y, new_z)
                        hand_landmarks.append(SmoothLM(new_x, new_y, new_z))
                        
                    if app_mode != 'CONTENT_MODE':
                        draw_landmarks(image, hand_landmarks)
                    
                    tip_ids = [4, 8, 12, 16, 20]
                    finger_names = ['T', 'I', 'M', 'R', 'P']
                    
                    thumb_tip = hand_landmarks[4]
                    thumb_ip = hand_landmarks[3]
                    pinky_base = hand_landmarks[17]
                    index_tip = hand_landmarks[8]
                    middle_tip = hand_landmarks[12]
                    
                    # Accurate 3D pinch detection
                    if get_distance3d(thumb_tip, index_tip) < 0.05 and get_distance3d(thumb_tip, middle_tip) < 0.05 and get_distance3d(index_tip, middle_tip) < 0.05:
                        pinch_detected = True
                    
                    dist_tip_to_pinky_base = get_distance3d(thumb_tip, pinky_base)
                    dist_ip_to_pinky_base = get_distance3d(thumb_ip, pinky_base)
                    
                    hand_fingers = []
                    if dist_tip_to_pinky_base > dist_ip_to_pinky_base + 0.01:
                        active_fingers_texts.append('T')
                        hand_fingers.append('T')
                            
                    wrist = hand_landmarks[0]
                    for id in range(1, 5):
                        tip = hand_landmarks[tip_ids[id]]
                        pip = hand_landmarks[tip_ids[id] - 2]
                        if get_distance3d(wrist, tip) > get_distance3d(wrist, pip):
                            active_fingers_texts.append(finger_names[id])
                            hand_fingers.append(finger_names[id])
                            
                    # MODE LOGIC
                    if app_mode == 'MAIN_MENU':
                        if len(hand_fingers) == 1 and 'I' in hand_fingers:
                            if menu_selection_choice == 'NEWS_MENU':
                                menu_selection_frames += 1
                                if menu_selection_frames >= 15:
                                    app_mode = 'NEWS_MENU'
                                    menu_selection_frames = 0
                            else:
                                menu_selection_choice = 'NEWS_MENU'
                                menu_selection_frames = 1
                        elif len(hand_fingers) == 2 and 'I' in hand_fingers and 'M' in hand_fingers:
                            if menu_selection_choice == 'DRAW_MODE':
                                menu_selection_frames += 1
                                if menu_selection_frames >= 15:
                                    app_mode = 'DRAW_MODE'
                                    menu_selection_frames = 0
                                    if canvas is None:
                                        canvas = np.zeros_like(image)
                            else:
                                menu_selection_choice = 'DRAW_MODE'
                                menu_selection_frames = 1
                        else:
                            menu_selection_frames = 0
                            
                    elif app_mode == 'NEWS_MENU':
                        if pinch_detected:
                            exit_frames += 1
                            if exit_frames > 15:
                                app_mode = 'MAIN_MENU'
                                exit_frames = 0
                        else:
                            exit_frames = 0
                            
                        if handedness_label == 'Left':  # Physical Right Hand
                            for finger in hand_fingers:
                                news_text = news_mapping_right.get(finger, '')
                                if news_text:
                                    active_news_right.append(news_text)
                        elif handedness_label == 'Right':  # Physical Left Hand
                            for finger in hand_fingers:
                                news_text = news_mapping_left.get(finger, '')
                                if news_text:
                                    active_news_left.append(news_text)
                        
                        if len(hand_fingers) == 1:
                            finger = hand_fingers[0]
                            current_hover_topic = news_mapping_right.get(finger) if handedness_label == 'Left' else news_mapping_left.get(finger)
                            
                            if current_hover_topic == last_hovered_topic:
                                hover_frames += 1
                            else:
                                hover_frames = 1
                                last_hovered_topic = current_hover_topic
                                
                            if hover_frames >= 10:
                                selected_topic_global = current_hover_topic
                                selected_topic_hand = handedness_label
                        else:
                            hover_frames = 0
                                
                        if selected_topic_global and len(hand_fingers) == 0 and handedness_label == selected_topic_hand:
                            punch_frames += 1
                            if punch_frames >= 10:
                                app_mode = 'CONTENT_MODE'
                                update_display_image()
                                wrist_x_history.clear()
                                punch_frames = 0
                        elif len(hand_fingers) > 0:
                            punch_frames = 0
                            
                    elif app_mode == 'CONTENT_MODE':
                        if pinch_detected:
                            exit_frames += 1
                            if exit_frames >= 15:
                                app_mode = 'NEWS_MENU'
                                selected_topic_global = None
                                selected_topic_hand = None
                                exit_frames = 0
                                hover_frames = 0
                        else:
                            exit_frames = 0
                            
                        current_wrist_x = hand_landmarks[0].x
                        if swipe_cooldown_frames > 0:
                            swipe_cooldown_frames -= 1
                            wrist_x_history.clear()
                        else:
                            wrist_x = hand_landmarks[0].x
                            wrist_x_history.append(wrist_x)
                            
                            if len(wrist_x_history) > 6:
                                wrist_x_history.pop(0)
                                
                            if len(wrist_x_history) == 6:
                                dx = wrist_x_history[-1] - wrist_x_history[0]
                                if dx < -0.10: # Fast swipe left
                                    current_idx = TOPICS_ORDER.index(selected_topic_global)
                                    selected_topic_global = TOPICS_ORDER[(current_idx + 1) % len(TOPICS_ORDER)]
                                    swipe_cooldown_frames = 60
                                    wrist_x_history.clear()
                                    update_display_image()
                                    transition_frames = 15
                                    transition_direction = 1
                                elif dx > 0.10: # Fast swipe right
                                    current_idx = TOPICS_ORDER.index(selected_topic_global)
                                    selected_topic_global = TOPICS_ORDER[(current_idx - 1) % len(TOPICS_ORDER)]
                                    swipe_cooldown_frames = 60
                                    wrist_x_history.clear()
                                    update_display_image()
                                    transition_frames = 15
                                    transition_direction = -1
                                    
                    elif app_mode == 'DRAW_MODE':
                        if pinch_detected:
                            exit_frames += 1
                            if exit_frames > 15:
                                app_mode = 'MAIN_MENU'
                                exit_frames = 0
                                prev_draw_x, prev_draw_y = 0, 0
                        else:
                            exit_frames = 0
                            
                        if len(hand_fingers) == 5:
                            hover_frames += 1
                            if hover_frames > 15:
                                if canvas is not None:
                                    canvas.fill(0)
                                hover_frames = 0
                        else:
                            hover_frames = 0
                            
                        cx, cy = int(index_tip.x * w), int(index_tip.y * h)
                        
                        if len(hand_fingers) == 1 and 'I' in hand_fingers:
                            if prev_draw_x != 0 and prev_draw_y != 0 and canvas is not None:
                                cv2.line(canvas, (prev_draw_x, prev_draw_y), (cx, cy), (255, 100, 255), 8, cv2.LINE_AA)
                            prev_draw_x, prev_draw_y = cx, cy
                        elif len(hand_fingers) == 2 and 'I' in hand_fingers and 'M' in hand_fingers:
                            prev_draw_x, prev_draw_y = 0, 0
                        else:
                            prev_draw_x, prev_draw_y = 0, 0

            # Clean up EMA for lost hands
            if not results or not results.hand_landmarks:
                hand_ema.clear()
                if app_mode == 'DRAW_MODE':
                    prev_draw_x, prev_draw_y = 0, 0
            else:
                for h_label in list(hand_ema.keys()):
                    if h_label not in active_handedness:
                        del hand_ema[h_label]
            
            c_time = time.time()
            current_fps = 1 / (c_time - p_time) if (c_time - p_time) > 0 else 0
            p_time = c_time
            
            fps_history.append(current_fps)
            if len(fps_history) > 15:
                fps_history.pop(0)
            fps = sum(fps_history) / len(fps_history)
            
            # --- RENDERING BASED ON MODE ---
            if app_mode == 'MAIN_MENU':
                cx, cy = int(w/2), int(h/2)
                draw_glass_panel(image, (cx - 300, cy - 200), (cx + 300, cy + 200), radius=25, alpha=0.7)
                cv2.putText(image, "MAIN MENU", (cx - 110, cy - 120), cv2.FONT_HERSHEY_DUPLEX, 1.2, (255, 255, 255), 2, cv2.LINE_AA)
                
                # Option 1
                color_1 = (255, 100, 255) if menu_selection_choice == 'NEWS_MENU' else (150, 150, 150)
                cv2.putText(image, "1 Finger: News Mode", (cx - 200, cy), cv2.FONT_HERSHEY_DUPLEX, 0.8, color_1, 2 if menu_selection_choice == 'NEWS_MENU' else 1, cv2.LINE_AA)
                
                # Option 2
                color_2 = (255, 100, 255) if menu_selection_choice == 'DRAW_MODE' else (150, 150, 150)
                cv2.putText(image, "2 Fingers: Draw Mode", (cx - 200, cy + 60), cv2.FONT_HERSHEY_DUPLEX, 0.8, color_2, 2 if menu_selection_choice == 'DRAW_MODE' else 1, cv2.LINE_AA)
                
                # Progress bar for selection
                if menu_selection_frames > 0:
                    bar_w = int((menu_selection_frames / 15.0) * 400)
                    cv2.rectangle(image, (cx - 200, cy + 130), (cx - 200 + bar_w, cy + 135), (255, 255, 255), cv2.FILLED)

            elif app_mode == 'NEWS_MENU':
                cv2.putText(image, f'FPS: {int(fps)}', (w - 120, 40), cv2.FONT_HERSHEY_DUPLEX, 0.6, (150, 150, 150), 1, cv2.LINE_AA)
                
                # Exit instruction
                cv2.putText(image, "Pinch to exit to Main Menu", (w - 300, h - 30), cv2.FONT_HERSHEY_DUPLEX, 0.6, (120, 120, 120), 1, cv2.LINE_AA)
                            
                # --- Right Hand Panel ---
                if active_news_right:
                    box_width_R = 300
                    box_height_R = max(70, 40 * len(active_news_right) + 40)
                    x1, y1 = w - box_width_R - 30, 80
                    x2, y2 = w - 30, 80 + box_height_R
                    
                    draw_glass_panel(image, (x1, y1), (x2, y2), radius=20, alpha=0.7)
                    cv2.putText(image, f'Right Hand', (x1 + 20, y1 + 30), cv2.FONT_HERSHEY_DUPLEX, 0.55, (160, 160, 160), 1, cv2.LINE_AA)
                    
                    y_pos = y1 + 65
                    for news in active_news_right:
                        is_active = (selected_topic_global == news)
                        color = (255, 255, 255) if is_active else (180, 180, 180)
                        thickness = 2 if is_active else 1
                        cv2.putText(image, news, (x1 + 20, y_pos), cv2.FONT_HERSHEY_DUPLEX, 0.55, color, thickness, cv2.LINE_AA)
                        y_pos += 35

                # --- Left Hand Panel ---
                if active_news_left:
                    box_width_L = 300
                    box_height_L = max(70, 40 * len(active_news_left) + 40)
                    x1, y1 = 30, 80
                    x2, y2 = 30 + box_width_L, 80 + box_height_L
                    
                    draw_glass_panel(image, (x1, y1), (x2, y2), radius=20, alpha=0.7)
                    cv2.putText(image, f'Left Hand', (x1 + 20, y1 + 30), cv2.FONT_HERSHEY_DUPLEX, 0.55, (160, 160, 160), 1, cv2.LINE_AA)
                    
                    y_pos = y1 + 65
                    for news in active_news_left:
                        is_active = (selected_topic_global == news)
                        color = (255, 255, 255) if is_active else (180, 180, 180)
                        thickness = 2 if is_active else 1
                        cv2.putText(image, news, (x1 + 20, y_pos), cv2.FONT_HERSHEY_DUPLEX, 0.55, color, thickness, cv2.LINE_AA)
                        y_pos += 35

                # --- Floating Action Pill ---
                pill_w = 400
                pill_h = 50
                px1, py1 = int(w/2) - int(pill_w/2), h - 80
                px2, py2 = int(w/2) + int(pill_w/2), h - 30
                
                if selected_topic_global:
                    draw_glass_panel(image, (px1, py1), (px2, py2), radius=25, alpha=0.85, bg_color=(35, 35, 45), border_color=(180, 180, 180))
                    text = selected_topic_global
                    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, 0.65, 1)
                    cv2.putText(image, text, (int(w/2) - int(tw/2), py1 + 34), cv2.FONT_HERSHEY_DUPLEX, 0.65, (255, 255, 255), 1, cv2.LINE_AA)
                else:
                    draw_glass_panel(image, (px1, py1), (px2, py2), radius=25, alpha=0.5, bg_color=(20, 20, 20), border_color=(80, 80, 80))
                    text = "Select a topic"
                    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, 0.6, 1)
                    cv2.putText(image, text, (int(w/2) - int(tw/2), py1 + 34), cv2.FONT_HERSHEY_DUPLEX, 0.6, (150, 150, 150), 1, cv2.LINE_AA)
                    
            elif app_mode == 'CONTENT_MODE':
                # Dim the background for deep focus (Dark Mode look)
                overlay = image.copy()
                cv2.rectangle(overlay, (0, 0), (w, h), (10, 10, 15), cv2.FILLED)
                cv2.addWeighted(overlay, 0.85, image, 0.15, 0, image)
                
                # Central frosted glass modal
                box_w, box_h = 1000, 520
                
                offset_x = 0
                if transition_frames > 0:
                    t = transition_frames / 15.0
                    offset_x = int((t ** 3) * w * transition_direction)
                    transition_frames -= 1
                    
                parallax_offset = int((current_wrist_x - 0.5) * 80)
                
                cx, cy = int(w/2) + offset_x + parallax_offset, int(h/2)
                x1, y1 = cx - int(box_w/2), cy - int(box_h/2)
                x2, y2 = cx + int(box_w/2), cy + int(box_h/2)
                
                draw_glass_panel(image, (x1, y1), (x2, y2), radius=25, alpha=0.8, bg_color=(25, 25, 30), border_color=(60, 60, 70))
                
                # Clean, minimal Title
                cv2.putText(image, selected_topic_global.upper(), (x1 + 40, y1 + 70), cv2.FONT_HERSHEY_DUPLEX, 1.2, (240, 240, 240), 2, cv2.LINE_AA)
                            
                # Subtle Separator
                cv2.line(image, (x1 + 40, y1 + 100), (x2 - 40, y1 + 100), (80, 80, 80), 1, cv2.LINE_AA)
                
                # --- VISUALIZATION / IMAGES ---
                vis_w, vis_h = 320, 320
                vis_x = x2 - vis_w - 40
                vis_y = y1 + 140
                
                if current_display_image is not None:
                    img_to_show = cv2.resize(current_display_image, (vis_w, vis_h))
                    mask = np.zeros((vis_h, vis_w), dtype=np.uint8)
                    draw_rounded_rect(mask, (0, 0), (vis_w, vis_h), 255, thickness=cv2.FILLED, radius=20)
                    
                    valid_x1, valid_y1 = max(0, vis_x), max(0, vis_y)
                    valid_x2, valid_y2 = min(w, vis_x + vis_w), min(h, vis_y + vis_h)
                    
                    if valid_x2 > valid_x1 and valid_y2 > valid_y1:
                        mx1, mx2 = valid_x1 - vis_x, valid_x2 - vis_x
                        my1, my2 = valid_y1 - vis_y, valid_y2 - vis_y
                        
                        roi = image[valid_y1:valid_y2, valid_x1:valid_x2]
                        valid_mask = mask[my1:my2, mx1:mx2]
                        valid_img = img_to_show[my1:my2, mx1:mx2]
                        
                        idx = (valid_mask == 255)
                        roi[idx] = valid_img[idx]
                    
                    draw_rounded_rect(image, (vis_x, vis_y), (vis_x+vis_w, vis_y+vis_h), (80, 80, 80), thickness=1, radius=20)
                else:
                    draw_glass_panel(image, (vis_x, vis_y), (vis_x + vis_w, vis_y + vis_h), radius=20, alpha=0.4, bg_color=(40, 40, 45))
                    cv2.putText(image, "Visualization", (vis_x + 85, vis_y + 160), cv2.FONT_HERSHEY_DUPLEX, 0.7, (150,150,150), 1, cv2.LINE_AA)

                # Wrapped Content Text
                content = mock_content.get(selected_topic_global, "Content not found for this topic.")
                text_max_width = box_w - vis_w - 100
                put_wrapped_text(image, content, (x1 + 40, y1 + 160), cv2.FONT_HERSHEY_DUPLEX, 0.65, (200, 200, 200), 1, text_max_width)
                                 
                # Instructions to exit
                exit_txt = "Pinch to close | Swipe L/R for Next/Prev"
                (tw, th), _ = cv2.getTextSize(exit_txt, cv2.FONT_HERSHEY_DUPLEX, 0.6, 1)
                cv2.putText(image, exit_txt, (cx - int(tw/2), y2 - 25), cv2.FONT_HERSHEY_DUPLEX, 0.6, (120, 120, 120), 1, cv2.LINE_AA)
                
            elif app_mode == 'DRAW_MODE':
                # Dim background
                overlay = image.copy()
                cv2.rectangle(overlay, (0, 0), (w, h), (15, 15, 20), cv2.FILLED)
                cv2.addWeighted(overlay, 0.6, image, 0.4, 0, image)
                
                # Overlay Canvas
                if canvas is not None:
                    gray_canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
                    _, mask = cv2.threshold(gray_canvas, 1, 255, cv2.THRESH_BINARY)
                    image[mask == 255] = canvas[mask == 255]
                
                # Draw cursor if index is tracking
                if results and results.hand_landmarks:
                    for h_idx, hlm in enumerate(results.hand_landmarks):
                        # Use EMA smoothed landmarks to map to screen for cursor
                        h_label = results.handedness[h_idx][0].category_name
                        if h_label in hand_ema:
                            idx_tip_coords = hand_ema[h_label][8]
                            cx, cy = int(idx_tip_coords[0] * w), int(idx_tip_coords[1] * h)
                            cv2.circle(image, (cx, cy), 8, (255, 100, 255), cv2.FILLED, cv2.LINE_AA)
                            cv2.circle(image, (cx, cy), 15, (255, 100, 255), 2, cv2.LINE_AA)
                
                # Instructions Pill
                pill_w, pill_h = 700, 50
                px1, py1 = int(w/2) - int(pill_w/2), h - 80
                px2, py2 = px1 + pill_w, h - 30
                draw_glass_panel(image, (px1, py1), (px2, py2), radius=25, alpha=0.7)
                txt = "Index: Draw | Peace: Hover | Open Palm: Clear | Pinch: Exit"
                (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_DUPLEX, 0.6, 1)
                cv2.putText(image, txt, (int(w/2) - int(tw/2), py1 + 32), cv2.FONT_HERSHEY_DUPLEX, 0.6, (200, 200, 200), 1, cv2.LINE_AA)
            
            cv2.imshow('Precise Finger Counter', image)
            
            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
