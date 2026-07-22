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

def ease_out_cubic(t):
    t = min(1.0, max(0.0, t))
    return 1.0 - (1.0 - t) ** 3

def draw_rounded_rect(img, top_left, bottom_right, color, thickness=cv2.FILLED, radius=15):
    x1, y1 = top_left
    x2, y2 = bottom_right
    if thickness >= 0:
        cv2.line(img, (x1 + radius, y1), (x2 - radius, y1), color, thickness, cv2.LINE_AA)
        cv2.line(img, (x1 + radius, y2), (x2 - radius, y2), color, thickness, cv2.LINE_AA)
        cv2.line(img, (x1, y1 + radius), (x1, y2 - radius), color, thickness, cv2.LINE_AA)
        cv2.line(img, (x2, y1 + radius), (x2, y2 - radius), color, thickness, cv2.LINE_AA)
        cv2.ellipse(img, (x1 + radius, y1 + radius), (radius, radius), 180, 0, 90, color, thickness, cv2.LINE_AA)
        cv2.ellipse(img, (x2 - radius, y1 + radius), (radius, radius), 270, 0, 90, color, thickness, cv2.LINE_AA)
        cv2.ellipse(img, (x2 - radius, y2 - radius), (radius, radius), 0, 0, 90, color, thickness, cv2.LINE_AA)
        cv2.ellipse(img, (x1 + radius, y2 - radius), (radius, radius), 90, 0, 90, color, thickness, cv2.LINE_AA)
    else:
        cv2.rectangle(img, (x1 + radius, y1), (x2 - radius, y2), color, cv2.FILLED)
        cv2.rectangle(img, (x1, y1 + radius), (x2, y2 - radius), color, cv2.FILLED)
        cv2.ellipse(img, (x1 + radius, y1 + radius), (radius, radius), 180, 0, 90, color, cv2.FILLED, cv2.LINE_AA)
        cv2.ellipse(img, (x2 - radius, y1 + radius), (radius, radius), 270, 0, 90, color, cv2.FILLED, cv2.LINE_AA)
        cv2.ellipse(img, (x2 - radius, y2 - radius), (radius, radius), 0, 0, 90, color, cv2.FILLED, cv2.LINE_AA)
        cv2.ellipse(img, (x1 + radius, y2 - radius), (radius, radius), 90, 0, 90, color, cv2.FILLED, cv2.LINE_AA)

def draw_glass_panel(img, top_left, bottom_right, radius=15, alpha=0.55, bg_color=(40, 40, 40), border_color=(120, 120, 120)):
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
    roi_h, roi_w = roi.shape[:2]
    if roi_w > 0 and roi_h > 0:
        small_roi = cv2.resize(roi, (max(1, roi_w // 4), max(1, roi_h // 4)), interpolation=cv2.INTER_LINEAR)
        small_blurred = cv2.GaussianBlur(small_roi, (15, 15), 0)
        blurred = cv2.resize(small_blurred, (roi_w, roi_h), interpolation=cv2.INTER_LINEAR)
    else:
        blurred = roi
    tint = np.full_like(roi, bg_color, dtype=np.uint8)
    frosted = cv2.addWeighted(tint, alpha, blurred, 1 - alpha, 0)
    mask_inv = cv2.bitwise_not(valid_mask)
    bg_keep = cv2.bitwise_and(roi, roi, mask=mask_inv)
    fg_glass = cv2.bitwise_and(frosted, frosted, mask=valid_mask)
    roi[:] = cv2.add(bg_keep, fg_glass)
    draw_rounded_rect(img, top_left, bottom_right, border_color, thickness=1, radius=radius)

def draw_landmarks(image, hand_landmarks):
    h, w, c = image.shape
    for connection in HAND_CONNECTIONS:
        p1 = hand_landmarks[connection[0]]
        p2 = hand_landmarks[connection[1]]
        x1, y1 = int(p1.x * w), int(p1.y * h)
        x2, y2 = int(p2.x * w), int(p2.y * h)
        cv2.line(image, (x1, y1), (x2, y2), (255, 255, 255), 1, cv2.LINE_AA)
    for landmark in hand_landmarks:
        x, y = int(landmark.x * w), int(landmark.y * h)
        cv2.circle(image, (x, y), 3, (220, 220, 220), cv2.FILLED, cv2.LINE_AA)

wrapped_content_cache = {}
def get_wrapped_lines(text, font, font_scale, thickness, max_width):
    cache_key = (text, font, font_scale, thickness, max_width)
    if cache_key in wrapped_content_cache: return wrapped_content_cache[cache_key]
    words = text.split(' ')
    lines = []
    current_line = words[0]
    for word in words[1:]:
        (w, h), _ = cv2.getTextSize(current_line + " " + word, font, font_scale, thickness)
        if w < max_width: current_line += " " + word
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
    def generate_palette():
        # Curated 14 modern colors (BGR)
        return [
            # Column 0
            [(255, 255, 255), (180, 180, 180), (100, 100, 100), (30, 30, 30), 
             (200, 100, 100), (100, 200, 100), (100, 100, 200)],
            # Column 1
            [(100, 200, 255), (255, 200, 100), (255, 100, 200), (150, 255, 150),
             (200, 150, 255), (255, 150, 100), (50, 150, 255)]
        ]
        
    color_palette = generate_palette()
    
    BaseOptions = mp.tasks.BaseOptions
    HandLandmarker = mp.tasks.vision.HandLandmarker
    HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    app_mode = 'MAIN_MENU'
    ticker_offset = 0.0
    article_frame_count = 0
    menu_selection_choice = None
    menu_selection_frames = 0
    layers = []
    active_layer_idx = 0
    MAX_LAYERS = 5
    layer_history = {i: [] for i in range(MAX_LAYERS)}
    was_drawing = False
    required_draw_fingers = 1
    current_strokes = [[] for _ in range(4)]
    stroke_hold_frames = 0
    prev_draw_pos = [(0, 0) for _ in range(4)]
    
    current_draw_color = (255, 200, 100) # Light blue in BGR
    current_draw_tool = 'NEON'
    ui_hover_target = None
    ui_hover_frames = 0
    
    mode_enter_frame = 0
    prev_rendered_mode = 'MAIN_MENU'
    mirror_mode = False
    grid_mode = True
    glow_particles = []
    
    selected_topic_global = None
    selected_topic_hand = None
    
    class SmoothLM:
        def __init__(self, x, y, z):
            self.x = x; self.y = y; self.z = z
            
    hand_ema = {}
    EMA_ALPHA = 0.7
    
    hover_frames = 0
    punch_frames = 0
    exit_frames = 0
    last_hovered_topic = None
    
    swipe_cooldown_frames = 0
    transition_frames = 0
    transition_direction = 1
    swipe_arrow_frames = 0
    swipe_arrow_dir = 1
    
    wrist_x_history = []
    current_wrist_x = 0.5
    
    TOPICS_ORDER = [
        'Thumb: AI news', 'Index: Geo political', 'Middle: India news', 
        'Ring: Indian frauds', 'Pinky: AI & Startups', 'Thumb: India budget', 
        'Index: MNC Jobs', 'Middle: Claude AI', 'Ring: Weather/AQI', 'Pinky: Mobile Tech'
    ]
    
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
            else: current_display_image = chosen
        else: current_display_image = None
            
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

    news_mapping_right = { 'T': 'Thumb: AI news', 'I': 'Index: Geo political', 'M': 'Middle: India news', 'R': 'Ring: Indian frauds', 'P': 'Pinky: AI & Startups' }
    news_mapping_left = { 'T': 'Thumb: India budget', 'I': 'Index: MNC Jobs', 'M': 'Middle: Claude AI', 'R': 'Ring: Weather/AQI', 'P': 'Pinky: Mobile Tech' }

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
            if not success: break
            
            image = cv2.flip(image, 1)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
            
            current_timestamp_ms = int(time.time() * 1000 - start_time_ms)
            if current_timestamp_ms <= last_timestamp_ms:
                current_timestamp_ms = last_timestamp_ms + 1
            last_timestamp_ms = current_timestamp_ms
            
            try: landmarker.detect_async(mp_image, current_timestamp_ms)
            except Exception as e: pass
                
            results = shared_state['results']
            
            active_fingers_texts = []
            active_news_right = []
            active_news_left = []
            
            
            active_handedness = set()
            h, w, c = image.shape
            
            if results and results.hand_landmarks:
                for hand_idx, raw_hand_landmarks in enumerate(results.hand_landmarks):
                    handedness_label = results.handedness[hand_idx][0].category_name
                    active_handedness.add(handedness_label)
                    
                    if handedness_label not in hand_ema:
                        hand_ema[handedness_label] = [(lm.x, lm.y, lm.z) for lm in raw_hand_landmarks]
                        
                    raw_lm_list = []
                    hand_landmarks = []
                    for i, lm in enumerate(raw_hand_landmarks):
                        # Raw landmark for pixel-perfect drawing
                        raw_lm_list.append(SmoothLM(lm.x, lm.y, lm.z))
                        # EMA-smoothed landmark for stable gesture detection
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
                    
                    # Palm detection is checked after fingers are parsed
                    
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
                        mcp = hand_landmarks[tip_ids[id] - 3]
                        v1x, v1y, v1z = pip.x - mcp.x, pip.y - mcp.y, pip.z - mcp.z
                        v2x, v2y, v2z = tip.x - pip.x, tip.y - pip.y, tip.z - pip.z
                        if (v1x*v2x + v1y*v2y + v1z*v2z) > 0:
                            active_fingers_texts.append(finger_names[id])
                            hand_fingers.append(finger_names[id])
                            

                            
                    if app_mode == 'MAIN_MENU':
                        if len(hand_fingers) == 1:
                            if menu_selection_choice == 'NEWS_MENU':
                                menu_selection_frames += 1
                                if menu_selection_frames >= 15:
                                    app_mode = 'NEWS_MENU'
                                    menu_selection_frames = 0
                            else:
                                menu_selection_choice = 'NEWS_MENU'
                                menu_selection_frames = 1
                        elif len(hand_fingers) == 2:
                            if menu_selection_choice == 'DRAW_MODE':
                                menu_selection_frames += 1
                                if menu_selection_frames >= 15:
                                    app_mode = 'DRAW_MODE'
                                    menu_selection_frames = 0
                                    if len(layers) == 0:
                                        layers.append({'name': 'Layer 1', 'canvas': np.zeros_like(image), 'visible': True})
                                        active_layer_idx = 0
                            else:
                                menu_selection_choice = 'DRAW_MODE'
                                menu_selection_frames = 1
                        else:
                            menu_selection_frames = 0
                            
                    elif app_mode == 'NEWS_MENU':
                        hx, hy = int(hand_landmarks[8].x * w), int(hand_landmarks[8].y * h)
                        if 20 < hx < 120 and 20 < hy < 60:
                            exit_frames += 1
                            if exit_frames > 20:
                                app_mode = 'MAIN_MENU'
                                exit_frames = 0
                        else: exit_frames = 0
                            
                        if handedness_label == 'Left':
                            for finger in hand_fingers:
                                news_text = news_mapping_right.get(finger, '')
                                if news_text: active_news_right.append(news_text)
                        elif handedness_label == 'Right':
                            for finger in hand_fingers:
                                news_text = news_mapping_left.get(finger, '')
                                if news_text: active_news_left.append(news_text)
                        
                        if len(hand_fingers) == 1:
                            finger = hand_fingers[0]
                            current_hover_topic = news_mapping_right.get(finger) if handedness_label == 'Left' else news_mapping_left.get(finger)
                            if current_hover_topic == last_hovered_topic: hover_frames += 1
                            else:
                                hover_frames = 1
                                last_hovered_topic = current_hover_topic
                                
                            if hover_frames >= 10:
                                selected_topic_global = current_hover_topic
                                selected_topic_hand = handedness_label
                        else: hover_frames = 0
                                
                        if selected_topic_global and len(hand_fingers) == 0 and handedness_label == selected_topic_hand:
                            punch_frames += 1
                            if punch_frames >= 10:
                                app_mode = 'CONTENT_MODE'
                                article_frame_count = 0
                                update_display_image()
                                wrist_x_history.clear()
                                punch_frames = 0
                        elif len(hand_fingers) > 0: punch_frames = 0
                            
                    elif app_mode == 'CONTENT_MODE':
                        hx, hy = int(hand_landmarks[8].x * w), int(hand_landmarks[8].y * h)
                        if 20 < hx < 120 and 20 < hy < 60:
                            exit_frames += 1
                            if exit_frames > 20:
                                app_mode = 'MAIN_MENU'
                                selected_topic_global = None
                                selected_topic_hand = None
                                exit_frames = 0
                                hover_frames = 0
                        else: exit_frames = 0
                            
                        current_wrist_x = hand_landmarks[0].x
                        if swipe_cooldown_frames > 0:
                            swipe_cooldown_frames -= 1
                            wrist_x_history.clear()
                        else:
                            finger_x = hand_landmarks[8].x
                            wrist_x_history.append(finger_x)
                            
                            if len(wrist_x_history) > 4: wrist_x_history.pop(0)
                                
                            if len(wrist_x_history) == 4:
                                dx = wrist_x_history[-1] - wrist_x_history[0]
                                if dx < -0.07: # Swipe left
                                    current_idx = TOPICS_ORDER.index(selected_topic_global)
                                    selected_topic_global = TOPICS_ORDER[(current_idx + 1) % len(TOPICS_ORDER)]
                                    swipe_cooldown_frames = 25
                                    wrist_x_history.clear()
                                    update_display_image()
                                    article_frame_count = 0
                                    transition_frames = 15
                                    transition_direction = 1
                                    swipe_arrow_frames = 15
                                    swipe_arrow_dir = 1
                                elif dx > 0.07: # Swipe right
                                    current_idx = TOPICS_ORDER.index(selected_topic_global)
                                    selected_topic_global = TOPICS_ORDER[(current_idx - 1) % len(TOPICS_ORDER)]
                                    swipe_cooldown_frames = 25
                                    wrist_x_history.clear()
                                    update_display_image()
                                    article_frame_count = 0
                                    transition_frames = 15
                                    transition_direction = -1
                                    swipe_arrow_frames = 15
                                    swipe_arrow_dir = -1
                                    
                    elif app_mode == 'DRAW_MODE':
                        
                        def recognize_shape(pts):
                            if len(pts) < 15: return None
                            points = np.array([[p[0], p[1]] for p in pts], dtype=np.int32)
                            dist_start_end = math.hypot(pts[0][0] - pts[-1][0], pts[0][1] - pts[-1][1])
                            
                            x, y, w, h = cv2.boundingRect(points)
                            diag = math.hypot(w, h)
                            if diag < 20: return None # too small
                            
                            # Is it a closed shape?
                            is_closed = dist_start_end < (diag * 0.6)
                            
                            if is_closed:
                                # Use Convex Hull to eliminate natural hand jitters that inflate perimeter
                                hull = cv2.convexHull(points)
                                area = cv2.contourArea(hull)
                                perimeter = cv2.arcLength(hull, True)
                                if perimeter == 0: return None
                                
                                circularity = 4 * math.pi * area / (perimeter * perimeter)
                                epsilon = 0.05 * perimeter
                                approx = cv2.approxPolyDP(hull, epsilon, True)
                                
                                if circularity > 0.8:
                                    (cx, cy), radius = cv2.minEnclosingCircle(hull)
                                    return ('CIRCLE', (int(cx), int(cy)), int(radius))
                                elif len(approx) == 4:
                                    return ('RECTANGLE', approx)
                                elif len(approx) == 3:
                                    return ('TRIANGLE', approx)
                                elif len(approx) > 4: # Fallback to circle
                                    (cx, cy), radius = cv2.minEnclosingCircle(hull)
                                    return ('CIRCLE', (int(cx), int(cy)), int(radius))
                            else:
                                arc_len = cv2.arcLength(points, False)
                                if arc_len > 0 and dist_start_end / arc_len > 0.8:
                                    return ('LINE', (pts[0][0], pts[0][1]), (pts[-1][0], pts[-1][1]))
                            return None
                            
                        def draw_recognized_shape(canvas, shape_data, color, thickness, tool):
                            if not shape_data: return
                            stype = shape_data[0]
                            if stype == 'CIRCLE':
                                _, center, radius = shape_data
                                if tool == 'NEON':
                                    halo_color = (max(0, color[0]-50), max(0, color[1]-50), max(0, color[2]-50))
                                    cv2.circle(canvas, center, radius, halo_color, max(1, thickness*4), cv2.LINE_AA)
                                    cv2.circle(canvas, center, radius, color, max(1, thickness*2), cv2.LINE_AA)
                                    cv2.circle(canvas, center, radius, (255, 255, 255), max(1, thickness//2), cv2.LINE_AA)
                                else:
                                    cv2.circle(canvas, center, radius, color, max(2, thickness), cv2.LINE_AA)
                            elif stype in ['RECTANGLE', 'TRIANGLE']:
                                approx = shape_data[1]
                                if tool == 'NEON':
                                    halo_color = (max(0, color[0]-50), max(0, color[1]-50), max(0, color[2]-50))
                                    cv2.drawContours(canvas, [approx], 0, halo_color, max(1, thickness*4), cv2.LINE_AA)
                                    cv2.drawContours(canvas, [approx], 0, color, max(1, thickness*2), cv2.LINE_AA)
                                    cv2.drawContours(canvas, [approx], 0, (255, 255, 255), max(1, thickness//2), cv2.LINE_AA)
                                else:
                                    cv2.drawContours(canvas, [approx], 0, color, max(2, thickness), cv2.LINE_AA)
                            elif stype == 'LINE':
                                _, pt1, pt2 = shape_data
                                if tool == 'NEON':
                                    halo_color = (max(0, color[0]-50), max(0, color[1]-50), max(0, color[2]-50))
                                    cv2.line(canvas, pt1, pt2, halo_color, max(1, thickness*4), cv2.LINE_AA)
                                    cv2.line(canvas, pt1, pt2, color, max(1, thickness*2), cv2.LINE_AA)
                                    cv2.line(canvas, pt1, pt2, (255, 255, 255), max(1, thickness//2), cv2.LINE_AA)
                                else:
                                    cv2.line(canvas, pt1, pt2, color, max(2, thickness), cv2.LINE_AA)
                        
                        canvas = layers[active_layer_idx]['canvas'] if layers else None
                        if required_draw_fingers == 5:
                            pip_raw = raw_lm_list[6]
                            tip_raw = raw_lm_list[8]
                            dx = tip_raw.x - pip_raw.x
                            dy = tip_raw.y - pip_raw.y
                            cx, cy = int((tip_raw.x + dx * 1.5) * w), int((tip_raw.y + dy * 1.5) * h)
                            cv2.circle(image, (cx, cy), 3, (0, 0, 255), -1) # Red dot for pen tip
                        else:
                            # Use first detected extended fingertip — RAW for precision
                            tip_ids_order = [8, 12, 16, 20]
                            finger_name_order = ['I', 'M', 'R', 'P']
                            raw_cursor = raw_lm_list[8]  # default index
                            for fni, fn in enumerate(finger_name_order):
                                if fn in hand_fingers:
                                    raw_cursor = raw_lm_list[tip_ids_order[fni]]
                                    break
                            cx, cy = int(raw_cursor.x * w), int(raw_cursor.y * h)
                        
                        hover_targets = []
                        # Color Panel Targets (Right Side)
                        cp_w, cp_h = 280, 420
                        cp_x1, cp_y1 = w - 310, int(h/2) - int(cp_h/2)
                        
                        # ── Color swatches in top bar ──
                        swatch_colors_flat = color_palette[0] + color_palette[1]
                        n_show = 8
                        sw_gap = 30
                        sw_x_start = w - n_show * sw_gap - 20
                        for si in range(n_show):
                            sx = sw_x_start + si * sw_gap + 12
                            sy = 28  # bar_y=0 + bar_h//2
                            ci, ri = si // 7, si % 7
                            hover_targets.append({'name': f'COLOR_{ci}_{ri}', 'box': (sx-14, 8, sx+14, 48), 'color': swatch_colors_flat[si % len(swatch_colors_flat)]})

                        # ── Tool icons in top bar ──
                        tool_list = ['PEN', 'MARKER', 'NEON', 'CALLIGRAPHY', 'SPRAY', 'ERASER']
                        t_total = len(tool_list) * 46
                        t_start = w//2 - t_total//2
                        for ti, tool in enumerate(tool_list):
                            tx = t_start + ti * 46 + 23
                            hover_targets.append({'name': tool, 'box': (tx-22, 4, tx+22, 52)})

                        # ── Back button ──
                        hover_targets.append({'name': 'ACTION_BACK', 'box': (18, 14, 94, 42)})

                        # ── Bottom-left HUD (fingers) ──
                        hud_x, hud_y = 18, h - 78
                        hover_targets.append({'name': 'ACTION_FINGERS_MINUS', 'box': (hud_x+8,  hud_y+30, hud_x+38,  hud_y+52)})
                        hover_targets.append({'name': 'ACTION_FINGERS_PLUS',  'box': (hud_x+142, hud_y+30, hud_x+172, hud_y+52)})
                        hover_targets.append({'name': 'ACTION_MIRROR',        'box': (hud_x+110, hud_y+4,  hud_x+178, hud_y+26)})
                        hover_targets.append({'name': 'ACTION_GRID',          'box': (hud_x+44,  hud_y+30, hud_x+136, hud_y+52)})

                        # ── Bottom-right HUD (undo/clear/save) ──
                        br_x, br_y = w - 178, h - 78
                        for bi, bname in enumerate(['ACTION_UNDO', 'ACTION_CLEAR', 'ACTION_SAVE']):
                            bbx = br_x + 8 + bi * 52
                            hover_targets.append({'name': bname, 'box': (bbx, br_y+8, bbx+46, br_y+52)})
                        
                        hovering_ui = False
                        current_target = None
                        
                        # First determine drawing state, THEN check UI overlap
                        n = len(hand_fingers)
                        is_drawing = False
                        is_hovering = False
                        if required_draw_fingers == 5:  # PEN mode
                            if 'M' not in hand_fingers: is_drawing = True
                            else: is_hovering = True
                        else:
                            if n == required_draw_fingers: is_drawing = True
                            elif n > 0 and n != required_draw_fingers: is_hovering = True
                        
                        # Only check UI hover targets when NOT actively drawing
                        if not is_drawing:
                            for t in hover_targets:
                                x1, y1, x2, y2 = t['box']
                                if cx >= x1 and cx <= x2 and cy >= y1 and cy <= y2:
                                    current_target = t['name']
                                    hovering_ui = True
                                    break
                                    
                        if current_target == ui_hover_target and current_target is not None:
                            ui_hover_frames += 1
                            if ui_hover_frames >= 20: # 1 second click
                                if current_target.startswith('COLOR_'):
                                    parts = current_target.split('_')
                                    current_draw_color = color_palette[int(parts[1])][int(parts[2])]
                                elif current_target == 'ACTION_CLEAR':
                                    if canvas is not None:
                                        layer_history[active_layer_idx].append(canvas.copy())
                                        canvas.fill(0)
                                elif current_target == 'ACTION_UNDO':
                                    hl = layer_history[active_layer_idx]
                                    if len(hl) > 0 and canvas is not None:
                                        np.copyto(canvas, hl.pop())
                                elif current_target == 'ACTION_MIRROR':
                                    mirror_mode = not mirror_mode
                                elif current_target == 'ACTION_GRID':
                                    grid_mode = not grid_mode
                                elif current_target == 'ACTION_BACK':
                                    app_mode = 'MAIN_MENU'
                                    prev_draw_x, prev_draw_y = 0, 0
                                    ui_hover_frames = 0
                                    ui_hover_target = None
                                elif current_target == 'ACTION_SAVE':
                                    export_img = np.full_like(image, 255)
                                    for ld in layers:
                                        if ld['visible']:
                                            lc = ld['canvas']
                                            gray = cv2.cvtColor(lc, cv2.COLOR_BGR2GRAY)
                                            _, lmask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
                                            export_img[lmask == 255] = lc[lmask == 255]
                                    cv2.imwrite('drawing_export.png', export_img)
                                elif current_target == 'ACTION_FINGERS_MINUS':
                                    required_draw_fingers = required_draw_fingers - 1 if required_draw_fingers > 1 else 5
                                elif current_target == 'ACTION_FINGERS_PLUS':
                                    required_draw_fingers = (required_draw_fingers % 5) + 1
                                elif current_target == 'ACTION_NEW_LAYER':
                                    if len(layers) < MAX_LAYERS:
                                        layers.append({'name': f'Layer {len(layers)+1}', 'canvas': np.zeros_like(image), 'visible': True})
                                        active_layer_idx = len(layers) - 1
                                elif current_target.startswith('LAYER_SELECT_'):
                                    li = int(current_target.split('_')[-1])
                                    if li < len(layers): active_layer_idx = li
                                elif current_target.startswith('LAYER_VIS_'):
                                    li = int(current_target.split('_')[-1])
                                    if li < len(layers): layers[li]['visible'] = not layers[li]['visible']
                                else:
                                    current_draw_tool = current_target
                                ui_hover_frames = 0
                        else:
                            ui_hover_target = current_target
                            ui_hover_frames = 1 if current_target else 0
                            

                        if is_drawing:
                            if not was_drawing and canvas is not None:
                                hl = layer_history[active_layer_idx]
                                hl.append(canvas.copy())
                                if len(hl) > 10: hl.pop(0)
                                current_strokes = [[] for _ in range(4)]
                                stroke_hold_frames = 0
                            was_drawing = True
                            
                            wrist_lm = hand_landmarks[0]
                            mcp_lm = hand_landmarks[9]
                            hand_size = math.hypot(wrist_lm.x - mcp_lm.x, wrist_lm.y - mcp_lm.y)
                            depth_mult = max(0.3, min(2.5, hand_size * 10))
                            
                            active_tips = [8] # Index
                            if required_draw_fingers >= 2: active_tips.append(12) # Middle
                            if required_draw_fingers >= 3: active_tips.append(16) # Ring
                            if required_draw_fingers >= 4: active_tips.append(20) # Pinky
                            
                            for fi, tip_idx in enumerate(active_tips):
                                if required_draw_fingers == 5:
                                    fx, fy = cx, cy
                                else:
                                    fx, fy = int(raw_lm_list[tip_idx].x * w), int(raw_lm_list[tip_idx].y * h)
                                current_strokes[fi].append((fx, fy))
                                
                                px, py = prev_draw_pos[fi]
                                if px != 0 and py != 0 and canvas is not None:
                                    if len(current_strokes[fi]) > 0:
                                        dist_moved = math.hypot(fx - px, fy - py)
                                        if fi == 0: # Only track hold for index
                                            if dist_moved < 15.0: stroke_hold_frames += 1
                                            else: stroke_hold_frames = 0
                                            
                                        if stroke_hold_frames >= 15 and len(current_strokes[fi]) > 15:
                                            shape_data = recognize_shape(current_strokes[fi])
                                            if shape_data:
                                                if fi == 0: np.copyto(canvas, layer_history[active_layer_idx][-1])
                                                draw_recognized_shape(canvas, shape_data, current_draw_color, int(4 * depth_mult), current_draw_tool)
                                                if fi == len(active_tips) - 1:
                                                    current_strokes = [[] for _ in range(4)]
                                                    stroke_hold_frames = 0
                                                    
                                    if px != 0 and py != 0 and canvas is not None and len(current_strokes[fi]) > 0:
                                        if current_draw_tool == 'PEN':
                                            t = max(1, int(2 * depth_mult))
                                            cv2.line(canvas, (px, py), (fx, fy), current_draw_color, t, cv2.LINE_AA)
                                        elif current_draw_tool == 'MARKER':
                                            t = max(2, int(20 * depth_mult))
                                            cv2.line(canvas, (px, py), (fx, fy), current_draw_color, t, cv2.LINE_AA)
                                        elif current_draw_tool == 'NEON':
                                            t1 = max(4, int(16 * depth_mult))
                                            t2 = max(2, int(8 * depth_mult))
                                            t3 = max(1, int(2 * depth_mult))
                                            halo_color = (max(0, current_draw_color[0]-50), max(0, current_draw_color[1]-50), max(0, current_draw_color[2]-50))
                                            cv2.line(canvas, (px, py), (fx, fy), halo_color, t1, cv2.LINE_AA)
                                            cv2.line(canvas, (px, py), (fx, fy), current_draw_color, t2, cv2.LINE_AA)
                                            cv2.line(canvas, (px, py), (fx, fy), (255, 255, 255), t3, cv2.LINE_AA)
                                        elif current_draw_tool == 'CALLIGRAPHY':
                                            offset = max(1, int(8 * depth_mult))
                                            pts = np.array([
                                                [px - offset, py + offset],
                                                [px + offset, py - offset],
                                                [fx + offset, fy - offset],
                                                [fx - offset, fy + offset]
                                            ], np.int32)
                                            cv2.fillPoly(canvas, [pts], current_draw_color, cv2.LINE_AA)
                                        elif current_draw_tool == 'SPRAY':
                                            dist = math.hypot(fx - px, fy - py)
                                            steps = max(1, int(dist / 3))
                                            sr = int(18 * depth_mult)
                                            for i in range(steps):
                                                t_step = i / steps
                                                sx = int(px + t_step * (fx - px))
                                                sy = int(py + t_step * (fy - py))
                                                for _ in range(int(4 * depth_mult) + 1):
                                                    rx = sx + random.randint(-sr, sr)
                                                    ry = sy + random.randint(-sr, sr)
                                                    cv2.circle(canvas, (rx, ry), random.randint(1, max(2, int(2*depth_mult))), current_draw_color, -1)
                                        elif current_draw_tool == 'ERASER':
                                            t = max(5, int(40 * depth_mult))
                                            cv2.line(canvas, (px, py), (fx, fy), (0, 0, 0), t, cv2.LINE_AA)
                                prev_draw_pos[fi] = (fx, fy)
                        elif is_hovering:
                            prev_draw_pos = [(0, 0) for _ in range(4)]
                            was_drawing = False
                            current_strokes = [[] for _ in range(4)]
                            stroke_hold_frames = 0
                            # Spawn glow particles on hover
                            for _ in range(3):
                                glow_particles.append({
                                    'x': cx + random.randint(-10, 10),
                                    'y': cy + random.randint(-10, 10),
                                    'vx': random.uniform(-1.5, 1.5),
                                    'vy': random.uniform(-2.5, -0.5),
                                    'life': random.randint(10, 20),
                                    'color': current_draw_color
                                })
                        else:
                            prev_draw_pos = [(0, 0) for _ in range(4)]
                            was_drawing = False
                            current_strokes = [[] for _ in range(4)]
                            stroke_hold_frames = 0

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
            if len(fps_history) > 15: fps_history.pop(0)
            fps = sum(fps_history) / len(fps_history)
            
            # Animation system
            if app_mode != prev_rendered_mode:
                mode_enter_frame = 0
                prev_rendered_mode = app_mode
            mode_enter_frame = min(mode_enter_frame + 1, 100)
            
            if app_mode == 'MAIN_MENU':
                at = ease_out_cubic(min(mode_enter_frame / 20.0, 1.0))
                mx, my = int(w/2), int(h/2)
                
                # Animated background dim
                ov = image.copy()
                cv2.rectangle(ov, (0, 0), (w, h), (10, 10, 15), cv2.FILLED)
                cv2.addWeighted(ov, 0.3 * at, image, 1.0 - 0.3 * at, 0, image)
                
                # Title slides down from above
                ty = my - 155 + int((1 - at) * -70)
                tc = (int(255*at),)*3
                cv2.putText(image, "PunchToRead", (mx - 148, ty), cv2.FONT_HERSHEY_SIMPLEX, 1.3, tc, 2, cv2.LINE_AA)
                sc = (int(100*at), int(100*at), int(120*at))
                cv2.putText(image, "gesture-controlled reading & drawing", (mx - 182, ty + 38), cv2.FONT_HERSHEY_SIMPLEX, 0.42, sc, 1, cv2.LINE_AA)
                
                # Mode selection cards
                cw, ch = 220, 170
                gap = 40
                
                # News Card (rises from below)
                c1_slide = int((1 - at) * 180)
                c1x = mx - cw - gap//2
                c1y = my - 20 + c1_slide
                news_sel = menu_selection_choice == 'NEWS_MENU'
                nb = (120, 90, 255) if news_sel else (55, 55, 70)
                draw_glass_panel(image, (c1x, c1y), (c1x+cw, c1y+ch), radius=18, alpha=0.75, bg_color=(28,28,36), border_color=nb)
                nc_hi = (int(240*at), int(240*at), int(248*at))
                nc_lo = (int(100*at), int(100*at), int(115*at))
                cv2.putText(image, "NEWS", (c1x+70, c1y+60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, nc_hi, 1+int(news_sel), cv2.LINE_AA)
                cv2.putText(image, "1 finger", (c1x+68, c1y+90), cv2.FONT_HERSHEY_SIMPLEX, 0.42, nc_lo, 1, cv2.LINE_AA)
                cv2.putText(image, "hold to select", (c1x+45, c1y+120), cv2.FONT_HERSHEY_SIMPLEX, 0.36, nc_lo, 1, cv2.LINE_AA)
                
                # Draw Card (rises with stagger)
                c2_at = ease_out_cubic(min(max(0, mode_enter_frame - 4) / 20.0, 1.0))
                c2_slide = int((1 - c2_at) * 180)
                c2x = mx + gap//2
                c2y = my - 20 + c2_slide
                draw_sel = menu_selection_choice == 'DRAW_MODE'
                db = (120, 90, 255) if draw_sel else (55, 55, 70)
                draw_glass_panel(image, (c2x, c2y), (c2x+cw, c2y+ch), radius=18, alpha=0.75, bg_color=(28,28,36), border_color=db)
                dc_hi = (int(240*c2_at), int(240*c2_at), int(248*c2_at))
                dc_lo = (int(100*c2_at), int(100*c2_at), int(115*c2_at))
                cv2.putText(image, "DRAW", (c2x+72, c2y+60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, dc_hi, 1+int(draw_sel), cv2.LINE_AA)
                cv2.putText(image, "2 fingers", (c2x+64, c2y+90), cv2.FONT_HERSHEY_SIMPLEX, 0.42, dc_lo, 1, cv2.LINE_AA)
                cv2.putText(image, "hold to select", (c2x+45, c2y+120), cv2.FONT_HERSHEY_SIMPLEX, 0.36, dc_lo, 1, cv2.LINE_AA)
                
                # Selection highlight fade-in (replaces flat bar/ring)
                if menu_selection_frames > 0:
                    prog = menu_selection_frames / 15.0
                    fill_c = (120, 90, 255)
                    ov_fill = image.copy()
                    if news_sel:
                        draw_rounded_rect(ov_fill, (c1x, c1y), (c1x+cw, c1y+ch), fill_c, cv2.FILLED, radius=18)
                    else:
                        draw_rounded_rect(ov_fill, (c2x, c2y), (c2x+cw, c2y+ch), fill_c, cv2.FILLED, radius=18)
                    cv2.addWeighted(ov_fill, 0.25 * prog, image, 1.0 - 0.25 * prog, 0, image)

            elif app_mode == 'NEWS_MENU':
                cv2.putText(image, f'{int(fps)} fps', (w - 80, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (70, 70, 90), 1, cv2.LINE_AA)
                
                # BACK button
                draw_rounded_rect(image, (20, 20), (110, 52), (38, 38, 48), radius=8)
                cv2.putText(image, "< BACK", (30, 41), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 200), 1, cv2.LINE_AA)
                if exit_frames > 0:
                    prog = min(exit_frames / 20.0, 1.0)
                    cv2.rectangle(image, (20, 48), (20 + int(90*prog), 52), (120, 90, 255), cv2.FILLED)

                    
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

                pill_w = 400
                pill_h = 50
                px1, py1 = int(w/2) - int(pill_w/2), h - 100
                px2, py2 = int(w/2) + int(pill_w/2), h - 50
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
                    
                # News Ticker Tape at bottom
                ticker_offset -= 4.0
                if ticker_offset < -2500: ticker_offset = w
                
                ticker_h = 40
                draw_glass_panel(image, (0, h - ticker_h), (w, h), radius=0, alpha=0.85, bg_color=(15, 15, 20), border_color=(40, 40, 50))
                ticker_text = "BREAKING NEWS  |  AI STARTUPS RAISE BILLIONS  |  GLOBAL MARKETS SURGE  |  NEW TECH INNOVATIONS  |  WEATHER WARNINGS ISSUED  |  MNC JOBS ON THE RISE  |  " * 3
                cv2.putText(image, ticker_text, (int(ticker_offset), h - 13), cv2.FONT_HERSHEY_DUPLEX, 0.6, (200, 220, 255), 1, cv2.LINE_AA)
                
                    
            elif app_mode == 'CONTENT_MODE':
                article_frame_count += 1
                overlay = image.copy()
                
                cv2.rectangle(overlay, (0, 0), (w, h), (10, 10, 15), cv2.FILLED)
                cv2.addWeighted(overlay, 0.85, image, 0.15, 0, image)
                
                # BACK button
                draw_rounded_rect(image, (20, 20), (110, 52), (38, 38, 48), radius=8)
                cv2.putText(image, "< BACK", (30, 41), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 200), 1, cv2.LINE_AA)
                if exit_frames > 0:
                    prog = min(exit_frames / 20.0, 1.0)
                    cv2.rectangle(image, (20, 48), (20 + int(90*prog), 52), (120, 90, 255), cv2.FILLED)

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
                cv2.putText(image, selected_topic_global.upper(), (x1 + 40, y1 + 70), cv2.FONT_HERSHEY_DUPLEX, 1.2, (240, 240, 240), 2, cv2.LINE_AA)
                cv2.line(image, (x1 + 40, y1 + 100), (x2 - 40, y1 + 100), (80, 80, 80), 1, cv2.LINE_AA)
                
                vis_w, vis_h = 320, 320
                vis_x = x2 - vis_w - 40
                vis_y = y1 + 140
                
                if current_display_image is not None:
                    # Ken Burns Effect (Slow Pan/Zoom)
                    scale = 1.0 + (article_frame_count * 0.0008)
                    if scale > 1.25: scale = 1.25
                    
                    orig_h, orig_w = current_display_image.shape[:2]
                    new_w = int(orig_w / scale)
                    new_h = int(orig_h / scale)
                    cx_img, cy_img = orig_w // 2, orig_h // 2
                    
                    # Slight Pan based on time
                    pan_x = int(math.sin(article_frame_count * 0.01) * (orig_w * 0.05))
                    
                    x1_img = max(0, cx_img - new_w // 2 + pan_x)
                    y1_img = max(0, cy_img - new_h // 2)
                    x2_img = min(orig_w, x1_img + new_w)
                    y2_img = min(orig_h, y1_img + new_h)
                    
                    # Ensure dimensions match
                    cropped = current_display_image[y1_img:y2_img, x1_img:x2_img]
                    if cropped.size > 0:
                        img_to_show = cv2.resize(cropped, (vis_w, vis_h))
                    else:
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

                content = mock_content.get(selected_topic_global, "Content not found for this topic.")
                text_max_width = box_w - vis_w - 100
                put_wrapped_text(image, content, (x1 + 40, y1 + 160), cv2.FONT_HERSHEY_DUPLEX, 0.65, (200, 200, 200), 1, text_max_width)
                                 
                exit_txt = "Pinch to close | Swipe L/R for Next/Prev"
                (tw, th), _ = cv2.getTextSize(exit_txt, cv2.FONT_HERSHEY_DUPLEX, 0.6, 1)
                cv2.putText(image, exit_txt, (cx - int(tw/2), y2 - 25), cv2.FONT_HERSHEY_DUPLEX, 0.6, (120, 120, 120), 1, cv2.LINE_AA)
                
                # Swipe direction arrow
                if swipe_arrow_frames > 0:
                    sa = swipe_arrow_frames / 15.0
                    ac = (int(200*sa), int(200*sa), int(255*sa))
                    acy = int(h/2)
                    if swipe_arrow_dir == 1:
                        acx = w - 60
                        pts = np.array([[acx, acy-25], [acx+30, acy], [acx, acy+25]], np.int32)
                    else:
                        acx = 60
                        pts = np.array([[acx, acy-25], [acx-30, acy], [acx, acy+25]], np.int32)
                    cv2.fillPoly(image, [pts], ac, cv2.LINE_AA)
                    swipe_arrow_frames -= 1
                
            elif app_mode == 'DRAW_MODE':
                anim_t = ease_out_cubic(min(mode_enter_frame / 18.0, 1.0))

                # ── Canvas composite ──
                for layer_data in layers:
                    if layer_data['visible']:
                        lc = layer_data['canvas']
                        gray_lc = cv2.cvtColor(lc, cv2.COLOR_BGR2GRAY)
                        _, lmask = cv2.threshold(gray_lc, 1, 255, cv2.THRESH_BINARY)
                        image[lmask == 255] = lc[lmask == 255]

                # ── Cyber Grid Background Overlay ──
                if grid_mode:
                    grid_step = 40
                    grid_color = (40, 35, 55)
                    # Draw subtle grid lines
                    for x_g in range(0, w, grid_step):
                        cv2.line(image, (x_g, 0), (x_g, h), grid_color, 1, cv2.LINE_AA)
                    for y_g in range(0, h, grid_step):
                        cv2.line(image, (0, y_g), (w, y_g), grid_color, 1, cv2.LINE_AA)
                    # Draw intersection dots for high-tech HUD look
                    for x_g in range(0, w, grid_step * 2):
                        for y_g in range(0, h, grid_step * 2):
                            cv2.circle(image, (x_g, y_g), 2, (90, 75, 120), -1, cv2.LINE_AA)

                # ── Mirror guide ──
                if mirror_mode:
                    cv2.line(image, (w//2, 0), (w//2, h), (120, 90, 255), 1, cv2.LINE_AA)

                # ─────────────────────────────────────────────────────────────
                # TOP BAR  (single unified control strip)
                # ─────────────────────────────────────────────────────────────
                bar_h = 56
                bar_alpha = 0.82
                bar_y = int(-bar_h + anim_t * (bar_h + 2))
                bar_bg = image[max(0,bar_y):max(0,bar_y)+bar_h, :].copy()
                if bar_bg.shape[0] > 0:
                    dark = np.full_like(bar_bg, (22, 22, 28), dtype=np.uint8)
                    cv2.addWeighted(dark, bar_alpha, bar_bg, 1-bar_alpha, 0, bar_bg)
                    image[max(0,bar_y):max(0,bar_y)+bar_h, :] = bar_bg
                # subtle bottom border
                cv2.line(image, (0, bar_y+bar_h-1), (w, bar_y+bar_h-1), (55, 55, 75), 1, cv2.LINE_AA)

                # ── BACK button (left side) ──
                bx, by = 18, bar_y + 14
                draw_rounded_rect(image, (bx, by), (bx+76, by+28), (35, 35, 45), radius=6)
                cv2.putText(image, "< BACK", (bx+8, by+19), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 175), 1, cv2.LINE_AA)
                if ui_hover_target == 'ACTION_BACK' and ui_hover_frames > 0:
                    prog = min(ui_hover_frames / 20.0, 1.0)
                    cv2.line(image, (bx, by+27), (bx + int(76*prog), by+27), (120, 90, 255), 2, cv2.LINE_AA)

                # ── Tool icons (center) ──
                tool_list  = ['PEN', 'MARKER', 'NEON', 'CALLIGRAPHY', 'SPRAY', 'ERASER']
                tool_icons = ['P', 'M', 'N', 'C', 'S', 'E']
                t_total = len(tool_list) * 46
                t_start = w//2 - t_total//2
                for ti, (tool, icon) in enumerate(zip(tool_list, tool_icons)):
                    tx = t_start + ti * 46 + 23
                    ty_center = bar_y + bar_h//2
                    active = (current_draw_tool == tool)
                    if active:
                        draw_rounded_rect(image, (tx-18, ty_center-14), (tx+18, ty_center+14), (55, 44, 100), radius=5)
                    ic = (220, 220, 240) if active else (90, 90, 110)
                    cv2.putText(image, icon, (tx-5, ty_center+6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, ic, 1, cv2.LINE_AA)
                    if ui_hover_target == tool and ui_hover_frames > 0:
                        angle = int((ui_hover_frames / 20.0) * 360)
                        cv2.ellipse(image, (tx, ty_center), (20, 20), -90, 0, angle, (120, 90, 255), 1, cv2.LINE_AA)

                # ── Color swatches (right side, 7 dots) ──
                swatch_colors = [row[0] for row in color_palette] + [row[1] for row in color_palette]
                swatch_colors = color_palette[0] + color_palette[1]  # 14 colors flat
                n_show = 8
                sw_gap = 30
                sw_x_start = w - n_show * sw_gap - 20
                for si in range(n_show):
                    scol = swatch_colors[si % len(swatch_colors)]
                    sx = sw_x_start + si * sw_gap + 12
                    sy = bar_y + bar_h//2
                    active_c = (current_draw_color == scol)
                    r = 10 if not active_c else 12
                    cv2.circle(image, (sx, sy), r, scol, cv2.FILLED, cv2.LINE_AA)
                    if active_c:
                        cv2.circle(image, (sx, sy), r+3, (220, 220, 240), 1, cv2.LINE_AA)
                    cn = f'COLOR_{si//7}_{si%7}'
                    if ui_hover_target == cn and ui_hover_frames > 0:
                        angle = int((ui_hover_frames / 20.0) * 360)
                        cv2.ellipse(image, (sx, sy), (16, 16), -90, 0, angle, (255, 255, 255), 1, cv2.LINE_AA)

                # ── Bottom-left HUD: fingers + mirror ──
                hud_x, hud_y = 18, h - 78
                hud_bg = image[hud_y:hud_y+60, hud_x:hud_x+180].copy()
                dark2 = np.full_like(hud_bg, (22, 22, 28), dtype=np.uint8)
                cv2.addWeighted(dark2, 0.78, hud_bg, 0.22, 0, hud_bg)
                image[hud_y:hud_y+60, hud_x:hud_x+180] = hud_bg
                draw_rounded_rect(image, (hud_x, hud_y), (hud_x+180, hud_y+60), (40, 40, 52), radius=8)

                f_lbl = "PEN" if required_draw_fingers == 5 else f"{required_draw_fingers}F"
                cv2.putText(image, f"DRAW: {f_lbl}", (hud_x+12, hud_y+22), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (100, 90, 160), 1, cv2.LINE_AA)

                # Minus / plus
                draw_rounded_rect(image, (hud_x+8, hud_y+30), (hud_x+38, hud_y+52), (35, 35, 48), radius=4)
                cv2.putText(image, "-", (hud_x+18, hud_y+46), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (120, 100, 200), 1, cv2.LINE_AA)
                if ui_hover_target == 'ACTION_FINGERS_MINUS' and ui_hover_frames > 0:
                    prog = min(ui_hover_frames / 20.0, 1.0)
                    cv2.line(image, (hud_x+8, hud_y+51), (hud_x+8+int(30*prog), hud_y+51), (120, 90, 255), 2, cv2.LINE_AA)

                grid_btn_col = (42, 42, 58) if grid_mode else (28, 28, 38)
                grid_txt_col = (140, 120, 220) if grid_mode else (80, 80, 100)
                draw_rounded_rect(image, (hud_x+44, hud_y+30), (hud_x+136, hud_y+52), grid_btn_col, radius=4)
                grid_lbl = "GRID: ON" if grid_mode else "GRID: OFF"
                cv2.putText(image, grid_lbl, (hud_x+54, hud_y+46), cv2.FONT_HERSHEY_SIMPLEX, 0.38, grid_txt_col, 1, cv2.LINE_AA)
                if ui_hover_target == 'ACTION_GRID' and ui_hover_frames > 0:
                    prog = min(ui_hover_frames / 20.0, 1.0)
                    cv2.line(image, (hud_x+44, hud_y+51), (hud_x+44+int(92*prog), hud_y+51), (120, 90, 255), 2, cv2.LINE_AA)

                draw_rounded_rect(image, (hud_x+142, hud_y+30), (hud_x+172, hud_y+52), (35, 35, 48), radius=4)
                cv2.putText(image, "+", (hud_x+150, hud_y+46), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (120, 100, 200), 1, cv2.LINE_AA)
                if ui_hover_target == 'ACTION_FINGERS_PLUS' and ui_hover_frames > 0:
                    prog = min(ui_hover_frames / 20.0, 1.0)
                    cv2.line(image, (hud_x+142, hud_y+51), (hud_x+142+int(30*prog), hud_y+51), (120, 90, 255), 2, cv2.LINE_AA)

                # Mirror dot indicator
                mir_col = (72, 199, 142) if mirror_mode else (55, 55, 75)
                cv2.circle(image, (hud_x+165, hud_y+14), 5, mir_col, cv2.FILLED, cv2.LINE_AA)
                cv2.putText(image, "MIR", (hud_x+125, hud_y+22), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (80, 80, 100), 1, cv2.LINE_AA)
                if ui_hover_target == 'ACTION_MIRROR' and ui_hover_frames > 0:
                    prog = min(ui_hover_frames / 20.0, 1.0)
                    cv2.ellipse(image, (hud_x+165, hud_y+14), (9, 9), -90, 0, int(360*prog), (72, 199, 142), 1, cv2.LINE_AA)

                # ── Undo / Clear / Save (bottom right) ──
                br_x, br_y = w - 178, h - 78
                br_bg = image[br_y:br_y+60, br_x:br_x+160].copy()
                dark3 = np.full_like(br_bg, (22, 22, 28), dtype=np.uint8)
                cv2.addWeighted(dark3, 0.78, br_bg, 0.22, 0, br_bg)
                image[br_y:br_y+60, br_x:br_x+160] = br_bg
                draw_rounded_rect(image, (br_x, br_y), (br_x+160, br_y+60), (40, 40, 52), radius=8)

                for bi, (bname, blbl, bcol) in enumerate([
                    ('ACTION_UNDO', 'UNDO', (90, 150, 230)),
                    ('ACTION_CLEAR', 'CLR',  (200, 80, 80)),
                    ('ACTION_SAVE', 'SAVE', (72, 199, 142)),
                ]):
                    bbx = br_x + 8 + bi * 52
                    bby = br_y + 8
                    draw_rounded_rect(image, (bbx, bby), (bbx+46, bby+44), (32, 32, 42), radius=5)
                    cv2.putText(image, blbl, (bbx+5, bby+28), cv2.FONT_HERSHEY_SIMPLEX, 0.35, bcol, 1, cv2.LINE_AA)
                    if ui_hover_target == bname and ui_hover_frames > 0:
                        prog = min(ui_hover_frames / 20.0, 1.0)
                        cv2.line(image, (bbx, bby+43), (bbx+int(46*prog), bby+43), bcol, 2, cv2.LINE_AA)

                # ── Cursor ──
                if results and results.hand_landmarks:
                    pulse = 0.5 + 0.5 * math.sin(time.time() * 6)
                    for h_idx, hlm in enumerate(results.hand_landmarks):
                        h_label = results.handedness[h_idx][0].category_name
                        if h_label in hand_ema:
                            idx_tip_coords = hand_ema[h_label][8]
                            cx_cur = int(idx_tip_coords[0] * w)
                            cy_cur = int(idx_tip_coords[1] * h)
                            c_color = current_draw_color if current_draw_tool != 'ERASER' else (200, 200, 210)
                            dim = tuple(max(0, int(c - 140)) for c in c_color)
                            cv2.circle(image, (cx_cur, cy_cur), int(14 + pulse*5), dim, 1, cv2.LINE_AA)
                            cv2.circle(image, (cx_cur, cy_cur), 3, (255, 255, 255), cv2.FILLED, cv2.LINE_AA)

                # ── fps (tiny, bottom center) ──
                cv2.putText(image, f'{int(fps)}fps', (w//2 - 15, h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (50, 50, 65), 1, cv2.LINE_AA)

                # Rebuild hover targets to match new layout
                btn_y_ref = 0  # unused sentinel

            cv2.imshow('Precise Finger Counter', image)
            
            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
