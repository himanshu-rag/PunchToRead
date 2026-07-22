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

def rotate_3d_point(x, y, z, rx, ry, rz):
    # Rotate around X-axis (Pitch)
    if rx != 0:
        rad = math.radians(rx)
        cos_a, sin_a = math.cos(rad), math.sin(rad)
        y, z = y * cos_a - z * sin_a, y * sin_a + z * cos_a
    # Rotate around Y-axis (Yaw)
    if ry != 0:
        rad = math.radians(ry)
        cos_a, sin_a = math.cos(rad), math.sin(rad)
        x, z = x * cos_a + z * sin_a, -x * sin_a + z * cos_a
    # Rotate around Z-axis (Roll)
    if rz != 0:
        rad = math.radians(rz)
        cos_a, sin_a = math.cos(rad), math.sin(rad)
        x, y = x * cos_a - y * sin_a, x * sin_a + y * cos_a
    return x, y, z

def project_point(x, y, z, rx, ry, zoom, center_x, center_y, distance=600.0, f=550.0):
    # Global camera rotation (around Y then X)
    x, y, z = rotate_3d_point(x, y, z, rx, ry, 0)
    
    # Zoom scaling
    x *= zoom
    y *= zoom
    z *= zoom
    
    # Apply depth offset and perspective divide
    adj_z = z + distance
    if adj_z <= 15.0:
        adj_z = 15.0
    px = int(center_x + (x * f) / adj_z)
    py = int(center_y + (y * f) / adj_z)
    return px, py, adj_z

def make_cube(size):
    d = size / 2.0
    vertices = np.array([
        [-d, -d, -d], [d, -d, -d], [d, d, -d], [-d, d, -d],
        [-d, -d, d],  [d, -d, d],  [d, d, d],  [-d, d, d]
    ], dtype=np.float32)
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7)
    ]
    faces = [
        [0, 1, 2, 3],
        [4, 5, 6, 7],
        [0, 1, 5, 4],
        [2, 3, 7, 6],
        [0, 3, 7, 4],
        [1, 2, 6, 5]
    ]
    return vertices, edges, faces

def make_pyramid(size):
    d = size / 2.0
    h_offset = d * 1.4
    vertices = np.array([
        [-d, d, -d], [d, d, -d], [d, d, d], [-d, d, d],
        [0, -h_offset, 0]
    ], dtype=np.float32)
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),
        (0, 4), (1, 4), (2, 4), (3, 4)
    ]
    faces = [
        [0, 1, 2, 3],
        [0, 1, 4],
        [1, 2, 4],
        [2, 3, 4],
        [3, 0, 4]
    ]
    return vertices, edges, faces

def make_cylinder(radius, height, segments=12):
    vertices = []
    edges = []
    faces = []
    h_half = height / 2.0
    for i in range(segments):
        theta = i * 2.0 * math.pi / segments
        cx = radius * math.cos(theta)
        cz = radius * math.sin(theta)
        vertices.append([cx, -h_half, cz])
        vertices.append([cx, h_half, cz])
        
    vertices = np.array(vertices, dtype=np.float32)
    for i in range(segments):
        next_i = (i + 1) % segments
        edges.append((i * 2, next_i * 2))
        edges.append((i * 2 + 1, next_i * 2 + 1))
        edges.append((i * 2, i * 2 + 1))
        faces.append([i * 2, next_i * 2, next_i * 2 + 1, i * 2 + 1])
        
    faces.append([i * 2 for i in range(segments)])
    faces.append([i * 2 + 1 for i in range(segments)])
    return vertices, edges, faces

def make_extruded_prism_from_canvas(canvas_bgr, depth=100.0):
    gray = cv2.cvtColor(canvas_bgr, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
        
    all_vertices = []
    all_edges = []
    all_faces = []
    current_idx_offset = 0
    
    for contour in contours:
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.006 * peri, True)
        if len(approx) < 3:
            continue
            
        pts_2d = approx.reshape(-1, 2)
        n_pts = len(pts_2d)
        
        m = cv2.moments(contour)
        if m["m00"] != 0:
            cx = int(m["m10"] / m["m00"])
            cy = int(m["m01"] / m["m00"])
        else:
            cx, cy = np.mean(pts_2d, axis=0)
            
        max_dim = max(1, np.max(pts_2d) - np.min(pts_2d))
        scale = 180.0 / max_dim
        
        for pt in pts_2d:
            vx_f = (pt[0] - cx) * scale
            vy_f = (pt[1] - cy) * scale
            vz_f = -depth / 2.0
            all_vertices.append([vx_f, vy_f, vz_f])
            
            vx_b = (pt[0] - cx) * scale
            vy_b = (pt[1] - cy) * scale
            vz_b = depth / 2.0
            all_vertices.append([vx_b, vy_b, vz_b])
            
        for i in range(n_pts):
            next_i = (i + 1) % n_pts
            all_edges.append((current_idx_offset + i * 2, current_idx_offset + next_i * 2))
            all_edges.append((current_idx_offset + i * 2 + 1, current_idx_offset + next_i * 2 + 1))
            all_edges.append((current_idx_offset + i * 2, current_idx_offset + i * 2 + 1))
            all_faces.append([
                current_idx_offset + i * 2,
                current_idx_offset + next_i * 2,
                current_idx_offset + next_i * 2 + 1,
                current_idx_offset + i * 2 + 1
            ])
            
        all_faces.append([current_idx_offset + i * 2 for i in range(n_pts)])
        all_faces.append([current_idx_offset + i * 2 + 1 for i in range(n_pts)])
        current_idx_offset += n_pts * 2
        
    if not all_vertices:
        return None
    return np.array(all_vertices, dtype=np.float32), all_edges, all_faces

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
    ui_hover_missed_frames = 0
    HOVER_TRIGGER_FRAMES = 8
    last_triggered_target = None
    exit_zone_triggered = False
    menu_cooldown_frames = 0
    
    mode_enter_frame = 0
    prev_rendered_mode = 'MAIN_MENU'
    mirror_mode = False
    grid_mode = True
    glow_particles = []
    shatter_particles = []
    
    # ── 3D Sculptor State Variables ──
    angle_x = -15.0
    angle_y = 45.0
    zoom_level = 1.0
    draw_points_3d = []
    current_stroke_3d = []
    placed_primitives_3d = []
    selected_primitive_type = 'CUBE'
    last_drag_pos = None
    extruded_prism = None
    
    selected_topic_global = None
    selected_topic_hand = None
    
    
    class OneEuroFilter:
        def __init__(self, min_cutoff=0.1, beta=0.007):
            self.min_cutoff = min_cutoff
            self.beta = beta
            self.x_prev = None
            self.dx_prev = 0.0
            
        def filter(self, x, dt=0.033):
            if self.x_prev is None:
                self.x_prev = x
                return x
            dx = (x - self.x_prev) / dt
            edx = self.dx_prev + 0.1 * (dx - self.dx_prev)
            self.dx_prev = edx
            cutoff = self.min_cutoff + self.beta * abs(edx)
            alpha = 1.0 / (1.0 + (1.0 / (2.0 * math.pi * cutoff * dt)))
            x_hat = self.x_prev + alpha * (x - self.x_prev)
            self.x_prev = x_hat
            return x_hat

    euro_filters = {}

    def draw_smooth_curve(img, p1, p2, color, thickness):
        # Quadratic Bezier Curve interpolation between previous & current point
        mx, my = (p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2
        steps = max(2, int(math.hypot(p2[0] - p1[0], p2[1] - p1[1]) / 3))
        prev_pt = p1
        for t in range(1, steps + 1):
            u = t / steps
            # Quadratic interpolation
            cx = int((1 - u)**2 * p1[0] + 2 * (1 - u) * u * mx + u**2 * p2[0])
            cy = int((1 - u)**2 * p1[1] + 2 * (1 - u) * u * my + u**2 * p2[1])
            cv2.line(img, prev_pt, (cx, cy), color, thickness, cv2.LINE_AA)
            prev_pt = (cx, cy)

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

    # ── Initialize Session Debug Log ──
    log_file_path = "session_debug.log"
    log_buffer = []
    try:
        with open(log_file_path, "w") as f:
            f.write(f"--- Session started at {time.strftime('%Y-%m-%d %H:%M:%S')} ---\n")
    except Exception as e:
        print(f"Warning: could not initialize log file: {e}")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Camera index 0 not available, trying index 1...")
        cap = cv2.VideoCapture(1)
    
    if not cap.isOpened():
        print("ERROR: Could not open any camera device (0 or 1). Please check webcam permissions!")
        return

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
                print("Error: cap.read() failed to grab frame!")
                break
            
            image = cv2.flip(image, 1)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
            
            current_timestamp_ms = int(time.time() * 1000 - start_time_ms)
            if current_timestamp_ms <= last_timestamp_ms:
                current_timestamp_ms = last_timestamp_ms + 1
            last_timestamp_ms = current_timestamp_ms
            
            try: landmarker.detect_async(mp_image, current_timestamp_ms)
            except Exception as e: pass
            
            is_drawing = False
            is_dragging_cam = False
                
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
                        fid_x = f"{handedness_label}_{i}_x"
                        fid_y = f"{handedness_label}_{i}_y"
                        fid_z = f"{handedness_label}_{i}_z"
                        if fid_x not in euro_filters:
                            euro_filters[fid_x] = OneEuroFilter(min_cutoff=0.05, beta=0.01)
                            euro_filters[fid_y] = OneEuroFilter(min_cutoff=0.05, beta=0.01)
                            euro_filters[fid_z] = OneEuroFilter(min_cutoff=0.05, beta=0.01)
                        
                        fx = euro_filters[fid_x].filter(lm.x)
                        fy = euro_filters[fid_y].filter(lm.y)
                        fz = euro_filters[fid_z].filter(lm.z)
                        
                        raw_lm_list.append(SmoothLM(fx, fy, fz))
                        
                        # EMA for gestures
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
                        # Robust finger extension: tip must be further from wrist than pip
                        d_tip_wrist = math.sqrt((tip.x - wrist.x)**2 + (tip.y - wrist.y)**2 + (tip.z - wrist.z)**2)
                        d_pip_wrist = math.sqrt((pip.x - wrist.x)**2 + (pip.y - wrist.y)**2 + (pip.z - wrist.z)**2)
                        # Also check dot product: MCP→PIP and PIP→TIP should point same direction
                        v1x, v1y, v1z = pip.x - mcp.x, pip.y - mcp.y, pip.z - mcp.z
                        v2x, v2y, v2z = tip.x - pip.x, tip.y - pip.y, tip.z - pip.z
                        dot = v1x*v2x + v1y*v2y + v1z*v2z
                        if d_tip_wrist > d_pip_wrist and dot > 0:
                            active_fingers_texts.append(finger_names[id])
                            hand_fingers.append(finger_names[id])
                            

                            
                    if app_mode == 'MAIN_MENU':
                        if menu_cooldown_frames > 0:
                            menu_cooldown_frames -= 1
                            menu_selection_frames = 0
                        else:
                            non_thumb = [f for f in hand_fingers if f != 'T']
                            n = len(non_thumb)
                            if n == 1:
                                if menu_selection_choice == 'NEWS_MENU':
                                    menu_selection_frames += 1
                                    if menu_selection_frames >= 15:
                                        app_mode = 'NEWS_MENU'
                                        menu_selection_frames = 0
                                else:
                                    menu_selection_choice = 'NEWS_MENU'
                                    menu_selection_frames = 1
                            elif n == 2:
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
                            elif n == 3:
                                if menu_selection_choice == '3D_MODE':
                                    menu_selection_frames += 1
                                    if menu_selection_frames >= 15:
                                        app_mode = '3D_MODE'
                                        menu_selection_frames = 0
                                        # Auto-extrude the active canvas drawing if it exists
                                        if len(layers) > 0 and active_layer_idx < len(layers):
                                            extruded_prism = make_extruded_prism_from_canvas(layers[active_layer_idx]['canvas'])
                                else:
                                    menu_selection_choice = '3D_MODE'
                                    menu_selection_frames = 1
                            else:
                                menu_selection_frames = 0
                            
                    elif app_mode == 'NEWS_MENU':
                        hx, hy = int(raw_lm_list[8].x * w), int(raw_lm_list[8].y * h)
                        if 0 <= hx <= 140 and 0 <= hy <= 65:
                            if not exit_zone_triggered:
                                app_mode = 'MAIN_MENU'
                                menu_cooldown_frames = 25
                                exit_zone_triggered = True
                            exit_frames = 20
                        else:
                            exit_frames = 0
                            exit_zone_triggered = False
                            
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
                                
                            if hover_frames >= 6:
                                selected_topic_global = current_hover_topic
                                selected_topic_hand = handedness_label
                        else: hover_frames = 0
                                
                        if selected_topic_global and len(hand_fingers) == 0 and handedness_label == selected_topic_hand:
                            punch_frames += 1
                            if punch_frames >= 6:
                                app_mode = 'CONTENT_MODE'
                                article_frame_count = 0
                                update_display_image()
                                wrist_x_history.clear()
                                punch_frames = 0
                        elif len(hand_fingers) > 0: punch_frames = 0
                            
                    elif app_mode == 'CONTENT_MODE':
                        hx, hy = int(raw_lm_list[8].x * w), int(raw_lm_list[8].y * h)
                        if 0 <= hx <= 140 and 0 <= hy <= 65:
                            if not exit_zone_triggered:
                                app_mode = 'MAIN_MENU'
                                selected_topic_global = None
                                selected_topic_hand = None
                                menu_cooldown_frames = 25
                                exit_zone_triggered = True
                            exit_frames = 20
                            hover_frames = 0
                        else:
                            exit_frames = 0
                            exit_zone_triggered = False
                            
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
                            ci, ri = si // 7, si % 7
                            hover_targets.append({'name': f'COLOR_{ci}_{ri}', 'box': (sx-18, 0, sx+18, 56), 'color': swatch_colors_flat[si % len(swatch_colors_flat)]})

                        # ── Tool icons in top bar ──
                        tool_list = ['PEN', 'MARKER', 'NEON', 'AURORA', 'CALLIGRAPHY', 'SPRAY', 'ERASER']
                        t_total = len(tool_list) * 46
                        t_start = w//2 - t_total//2
                        for ti, tool in enumerate(tool_list):
                            tx = t_start + ti * 46 + 23
                            hover_targets.append({'name': tool, 'box': (tx-25, 0, tx+25, 60)})

                        # ── Back button (generous top-left zone) ──
                        hover_targets.append({'name': 'ACTION_BACK', 'box': (0, 0, 120, 60)})

                        # ── Bottom-left HUD (fingers) ──
                        hud_x, hud_y = 18, h - 78
                        hover_targets.append({'name': 'ACTION_FINGERS_MINUS', 'box': (hud_x,  hud_y+20, hud_x+45,  hud_y+60)})
                        hover_targets.append({'name': 'ACTION_FINGERS_PLUS',  'box': (hud_x+135, hud_y+20, hud_x+180, hud_y+60)})
                        hover_targets.append({'name': 'ACTION_MIRROR',        'box': (hud_x+100, hud_y-5,  hud_x+185, hud_y+30)})
                        hover_targets.append({'name': 'ACTION_GRID',          'box': (hud_x+38,  hud_y+20, hud_x+140, hud_y+60)})

                        # ── Bottom-right HUD (undo/clear/save) ──
                        br_x, br_y = w - 178, h - 78
                        for bi, bname in enumerate(['ACTION_UNDO', 'ACTION_CLEAR', 'ACTION_SAVE']):
                            bbx = br_x + bi * 56
                            hover_targets.append({'name': bname, 'box': (bbx, br_y, bbx+54, br_y+60)})
                        
                        # Check UI targets collision FIRST
                        hovering_ui = False
                        current_target = None
                        for t in hover_targets:
                            x1, y1, x2, y2 = t['box']
                            if cx >= x1 and cx <= x2 and cy >= y1 and cy <= y2:
                                current_target = t['name']
                                hovering_ui = True
                                break
                                
                        is_drawing = False
                        is_hovering = False
                        index_up = 'I' in hand_fingers
                        middle_up = 'M' in hand_fingers
                        ring_up = 'R' in hand_fingers
                        pinky_up = 'P' in hand_fingers
                        
                        # UI Safety Margin: disable drawing whenever finger is in top/bottom UI bar zones
                        in_ui_margin = (cy < 65) or (cy > h - 85) or (cx < 140 and cy < 80)
                        
                        if hovering_ui or in_ui_margin:
                            is_hovering = True
                            is_drawing = False
                        else:
                            if required_draw_fingers == 5:  # PEN mode
                                if not middle_up: is_drawing = True
                            elif required_draw_fingers == 1:
                                if index_up: is_drawing = True  # Pointing at canvas = draw!
                            else:
                                non_thumb = [f for f in hand_fingers if f != 'T']
                                if len(non_thumb) == required_draw_fingers: is_drawing = True
                                    
                        if current_target is not None:
                            if current_target == ui_hover_target:
                                ui_hover_frames += 1
                                ui_hover_missed_frames = 0
                            else:
                                ui_hover_target = current_target
                                ui_hover_frames = 1
                                ui_hover_missed_frames = 0
                                
                            if ui_hover_frames >= HOVER_TRIGGER_FRAMES:
                                if current_target.startswith('COLOR_'):
                                    parts = current_target.split('_')
                                    current_draw_color = color_palette[int(parts[1])][int(parts[2])]
                                elif current_target == 'ACTION_CLEAR':
                                    if canvas is not None:
                                        layer_history[active_layer_idx].append(canvas.copy())
                                        # Supernova Particle Shatter: sample drawing pixels to explode into physics particles
                                        non_zero_y, non_zero_x = np.nonzero(cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY))
                                        if len(non_zero_x) > 0:
                                            step = max(1, len(non_zero_x) // 180)  # cap at ~180 particles for high FPS
                                            for idx_p in range(0, len(non_zero_x), step):
                                                px_x, px_y = non_zero_x[idx_p], non_zero_y[idx_p]
                                                p_color = tuple(int(c) for c in canvas[px_y, px_x])
                                                angle = random.uniform(0, 2 * math.pi)
                                                speed = random.uniform(3.0, 12.0)
                                                shatter_particles.append({
                                                    'x': float(px_x),
                                                    'y': float(px_y),
                                                    'vx': math.cos(angle) * speed,
                                                    'vy': math.sin(angle) * speed - 2.0, # initial upward pop
                                                    'life': random.randint(30, 55),
                                                    'max_life': 55,
                                                    'size': random.randint(2, 6),
                                                    'color': p_color
                                                })
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
                                    menu_cooldown_frames = 25
                                    prev_draw_x, prev_draw_y = 0, 0
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
                                ui_hover_target = None
                        else:
                            if ui_hover_target is not None:
                                ui_hover_missed_frames += 1
                                if ui_hover_missed_frames > 4:
                                    ui_hover_target = None
                                    ui_hover_frames = 0
                            else:
                                ui_hover_frames = 0
                            

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
                                                    
                                    # Actually draw the stroke
                                    dist_step = math.hypot(fx - px, fy - py)
                                    if dist_step < 200:  # skip teleport jumps
                                        speed_factor = max(0.5, min(1.8, 15.0 / (dist_step + 1.0)))
                                        
                                        if current_draw_tool == 'PEN':
                                            t = max(1, int(2 * depth_mult * speed_factor))
                                            draw_smooth_curve(canvas, (px, py), (fx, fy), current_draw_color, t)
                                        elif current_draw_tool == 'MARKER':
                                            t = max(2, int(18 * depth_mult))
                                            draw_smooth_curve(canvas, (px, py), (fx, fy), current_draw_color, t)
                                        elif current_draw_tool == 'NEON':
                                            t1 = max(4, int(16 * depth_mult))
                                            t2 = max(2, int(8 * depth_mult))
                                            t3 = max(1, int(2 * depth_mult))
                                            halo_color = (max(0, current_draw_color[0]-50), max(0, current_draw_color[1]-50), max(0, current_draw_color[2]-50))
                                            draw_smooth_curve(canvas, (px, py), (fx, fy), halo_color, t1)
                                            draw_smooth_curve(canvas, (px, py), (fx, fy), current_draw_color, t2)
                                            draw_smooth_curve(canvas, (px, py), (fx, fy), (255, 255, 255), t3)
                                            
                                            if random.random() < 0.4:
                                                glow_particles.append({
                                                    'x': fx + random.randint(-4, 4),
                                                    'y': fy + random.randint(-4, 4),
                                                    'vx': random.uniform(-1.0, 1.0),
                                                    'vy': random.uniform(-1.0, 1.0),
                                                    'life': random.randint(12, 25),
                                                    'color': current_draw_color
                                                })
                                        elif current_draw_tool == 'AURORA':
                                            # Rainbow Plasma Energy Brush
                                            hue = int((time.time() * 120 + len(current_strokes[fi]) * 5) % 180)
                                            hsv_pixel = np.uint8([[[hue, 255, 255]]])
                                            bgr_rainbow = tuple(int(c) for c in cv2.cvtColor(hsv_pixel, cv2.COLOR_HSV2BGR)[0][0])
                                            t_aurora = max(3, int(14 * depth_mult))
                                            draw_smooth_curve(canvas, (px, py), (fx, fy), bgr_rainbow, t_aurora)
                                            draw_smooth_curve(canvas, (px, py), (fx, fy), (255, 255, 255), max(1, t_aurora // 3))
                                            for _ in range(2):
                                                shatter_particles.append({
                                                    'x': float(fx),
                                                    'y': float(fy),
                                                    'vx': random.uniform(-2.5, 2.5),
                                                    'vy': random.uniform(-3.5, 0.5),
                                                    'life': random.randint(15, 30),
                                                    'max_life': 30,
                                                    'size': random.randint(2, 5),
                                                    'color': bgr_rainbow
                                                })
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
                        else:
                            # Only reset after several non-drawing frames (debounce)
                            if was_drawing:
                                was_drawing = False
                            prev_draw_pos = [(0, 0) for _ in range(4)]
                            current_strokes = [[] for _ in range(4)]
                            stroke_hold_frames = 0
                            
                    elif app_mode == '3D_MODE':
                        cx, cy = int(raw_lm_list[8].x * w), int(raw_lm_list[8].y * h)
                        hover_targets = []
                        # Back button (generous top-left zone)
                        hover_targets.append({'name': 'ACTION_BACK', 'box': (0, 0, 120, 60)})
                        
                        # Top toolbar: Reset Camera, Clear, Extrude
                        tb_w = 3 * 120
                        tb_x = w // 2 - tb_w // 2
                        hover_targets.append({'name': 'ACTION_3D_RESET_CAM', 'box': (tb_x - 10, 0, tb_x + 110, 56)})
                        hover_targets.append({'name': 'ACTION_3D_CLEAR', 'box': (tb_x + 110, 0, tb_x + 230, 56)})
                        hover_targets.append({'name': 'ACTION_3D_EXTRUDE', 'box': (tb_x + 230, 0, tb_x + 350, 56)})
                        
                        # Primitives Selection (generous side panel)
                        hover_targets.append({'name': 'SELECT_3D_CUBE', 'box': (0, 90, 140, 148)})
                        hover_targets.append({'name': 'SELECT_3D_PYRAMID', 'box': (0, 148, 140, 198)})
                        hover_targets.append({'name': 'SELECT_3D_CYLINDER', 'box': (0, 198, 140, 248)})
                        hover_targets.append({'name': 'ACTION_3D_ADD_SHAPE', 'box': (0, 255, 140, 310)})
                        
                        hovering_ui = False
                        current_target = None
                        
                        # Check UI targets collision FIRST
                        hovering_ui = False
                        current_target = None
                        for t in hover_targets:
                            x1, y1, x2, y2 = t['box']
                            if cx >= x1 and cx <= x2 and cy >= y1 and cy <= y2:
                                current_target = t['name']
                                hovering_ui = True
                                break
                                
                        is_drawing = False
                        is_dragging_cam = False
                        is_hovering = False
                        
                        # UI Safety Margin: disable drawing whenever finger is in top bar (cy < 75) or left panel (cx < 160)
                        in_ui_margin = (cy < 75) or (cx < 160)
                        
                        if hovering_ui or in_ui_margin:
                            is_hovering = True
                            is_drawing = False
                            current_stroke_3d = []
                            was_drawing = False
                        else:
                            non_thumb = [f for f in hand_fingers if f != 'T']
                            index_up = 'I' in hand_fingers
                            
                            if len(non_thumb) == 0:
                                is_dragging_cam = True # Fist = Rotate 3D Camera!
                            elif index_up:
                                is_drawing = True     # Pointing at 3D canvas = Draw 3D Path!
                                    
                        if current_target is not None:
                            if current_target == ui_hover_target:
                                ui_hover_frames += 1
                                ui_hover_missed_frames = 0
                            else:
                                ui_hover_target = current_target
                                ui_hover_frames = 1
                                ui_hover_missed_frames = 0
                                
                            if ui_hover_frames >= HOVER_TRIGGER_FRAMES:
                                if current_target == 'ACTION_BACK':
                                    app_mode = 'MAIN_MENU'
                                    menu_cooldown_frames = 25
                                elif current_target == 'ACTION_3D_RESET_CAM':
                                    angle_x = -15.0
                                    angle_y = 45.0
                                    zoom_level = 1.0
                                elif current_target == 'ACTION_3D_CLEAR':
                                    draw_points_3d = []
                                    placed_primitives_3d = []
                                    extruded_prism = None
                                elif current_target == 'ACTION_3D_EXTRUDE':
                                    if len(layers) > 0 and active_layer_idx < len(layers):
                                        extruded_prism = make_extruded_prism_from_canvas(layers[active_layer_idx]['canvas'])
                                elif current_target == 'SELECT_3D_CUBE':
                                    selected_primitive_type = 'CUBE'
                                elif current_target == 'SELECT_3D_PYRAMID':
                                    selected_primitive_type = 'PYRAMID'
                                elif current_target == 'SELECT_3D_CYLINDER':
                                    selected_primitive_type = 'CYLINDER'
                                elif current_target == 'ACTION_3D_ADD_SHAPE':
                                    # Add shape in front of camera
                                    z_cam = 0.0
                                    D = 600.0
                                    x_cam = (cx - w//2) * D / 550.0
                                    y_cam = (cy - h//2) * D / 550.0
                                    
                                    rad_x = math.radians(-angle_x)
                                    cos_x, sin_x = math.cos(rad_x), math.sin(rad_x)
                                    y1 = y_cam * cos_x
                                    z1 = y_cam * sin_x
                                    
                                    rad_y = math.radians(-angle_y)
                                    cos_y, sin_y = math.cos(rad_y), math.sin(rad_y)
                                    x_w = x_cam * cos_y + z1 * sin_y
                                    z_w = -x_cam * sin_y + z1 * cos_y
                                    
                                    placed_primitives_3d.append({
                                        'type': selected_primitive_type,
                                        'pos': (x_w, y1, z_w),
                                        'size': 75.0,
                                        'color': current_draw_color,
                                        'rot': [0.0, 0.0, 0.0]
                                    })
                                ui_hover_frames = 0
                                ui_hover_target = None
                        else:
                            if ui_hover_target is not None:
                                ui_hover_missed_frames += 1
                                if ui_hover_missed_frames > 4:
                                    ui_hover_target = None
                                    ui_hover_frames = 0
                            else:
                                ui_hover_frames = 0
                            
                        if is_dragging_cam:
                            if last_drag_pos is not None:
                                dx = cx - last_drag_pos[0]
                                dy = cy - last_drag_pos[1]
                                angle_y += dx * 0.45
                                angle_x = max(-80.0, min(80.0, angle_x + dy * 0.45))
                            last_drag_pos = (cx, cy)
                            current_stroke_3d = []
                            was_drawing = False
                        elif is_drawing:
                            if not was_drawing:
                                current_stroke_3d = []
                            was_drawing = True
                            last_drag_pos = None
                            
                            z_cam = raw_lm_list[8].z * 600.0
                            D = 600.0 + z_cam
                            x_cam = (cx - w//2) * D / (550.0 * zoom_level)
                            y_cam = (cy - h//2) * D / (550.0 * zoom_level)
                            
                            rad_x = math.radians(-angle_x)
                            cos_x, sin_x = math.cos(rad_x), math.sin(rad_x)
                            y1 = y_cam * cos_x - z_cam * sin_x
                            z1 = y_cam * sin_x + z_cam * cos_x
                            
                            rad_y = math.radians(-angle_y)
                            cos_y, sin_y = math.cos(rad_y), math.sin(rad_y)
                            x_w = x_cam * cos_y + z1 * sin_y
                            z_w = -x_cam * sin_y + z1 * cos_y
                            
                            if not current_stroke_3d or math.hypot(x_w - current_stroke_3d[-1][0], y1 - current_stroke_3d[-1][1]) > 4.0:
                                current_stroke_3d.append((x_w, y1, z_w))
                        else:
                            if was_drawing and len(current_stroke_3d) > 1:
                                draw_points_3d.append({
                                    'points': current_stroke_3d,
                                    'color': current_draw_color,
                                    'tool': current_draw_tool
                                })
                            current_stroke_3d = []
                            was_drawing = False
                            last_drag_pos = None

            if not results or not results.hand_landmarks:
                hand_ema.clear()
                if app_mode == 'DRAW_MODE':
                    prev_draw_x, prev_draw_y = 0, 0
            else:
                if random.random() < 0.05:
                    cx_val = cx if 'cx' in locals() else None
                    cy_val = cy if 'cy' in locals() else None
                    print(f"[DEBUG] Mode: {app_mode} | Fingers: {hand_fingers} | Draw: {is_drawing} | DragCam: {is_dragging_cam} | Cursor: ({cx_val}, {cy_val}) | Hover UI: {current_target if 'current_target' in locals() else None}")
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
                
                # Mode selection cards (3-card symmetrical layout)
                cw, ch = 190, 170
                gap = 35
                
                # News Card
                c1_slide = int((1 - at) * 180)
                c1x = mx - int(1.5 * cw) - gap
                c1y = my - 20 + c1_slide
                news_sel = menu_selection_choice == 'NEWS_MENU'
                nb = (120, 90, 255) if news_sel else (55, 55, 70)
                draw_glass_panel(image, (c1x, c1y), (c1x+cw, c1y+ch), radius=18, alpha=0.75, bg_color=(28,28,36), border_color=nb)
                nc_hi = (int(240*at), int(240*at), int(248*at))
                nc_lo = (int(100*at), int(100*at), int(115*at))
                cv2.putText(image, "NEWS", (c1x+55, c1y+60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, nc_hi, 1+int(news_sel), cv2.LINE_AA)
                cv2.putText(image, "1 finger", (c1x+53, c1y+90), cv2.FONT_HERSHEY_SIMPLEX, 0.42, nc_lo, 1, cv2.LINE_AA)
                cv2.putText(image, "hold to select", (c1x+30, c1y+120), cv2.FONT_HERSHEY_SIMPLEX, 0.36, nc_lo, 1, cv2.LINE_AA)
                
                # Draw Card
                c2_at = ease_out_cubic(min(max(0, mode_enter_frame - 4) / 20.0, 1.0))
                c2_slide = int((1 - c2_at) * 180)
                c2x = mx - int(0.5 * cw)
                c2y = my - 20 + c2_slide
                draw_sel = menu_selection_choice == 'DRAW_MODE'
                db = (120, 90, 255) if draw_sel else (55, 55, 70)
                draw_glass_panel(image, (c2x, c2y), (c2x+cw, c2y+ch), radius=18, alpha=0.75, bg_color=(28,28,36), border_color=db)
                dc_hi = (int(240*c2_at), int(240*c2_at), int(248*c2_at))
                dc_lo = (int(100*c2_at), int(100*c2_at), int(115*c2_at))
                cv2.putText(image, "DRAW", (c2x+58, c2y+60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, dc_hi, 1+int(draw_sel), cv2.LINE_AA)
                cv2.putText(image, "2 fingers", (c2x+50, c2y+90), cv2.FONT_HERSHEY_SIMPLEX, 0.42, dc_lo, 1, cv2.LINE_AA)
                cv2.putText(image, "hold to select", (c2x+30, c2y+120), cv2.FONT_HERSHEY_SIMPLEX, 0.36, dc_lo, 1, cv2.LINE_AA)
                
                # 3D Sculptor Card
                c3_at = ease_out_cubic(min(max(0, mode_enter_frame - 8) / 20.0, 1.0))
                c3_slide = int((1 - c3_at) * 180)
                c3x = mx + int(0.5 * cw) + gap
                c3y = my - 20 + c3_slide
                sculpt_sel = menu_selection_choice == '3D_MODE'
                sb = (120, 90, 255) if sculpt_sel else (55, 55, 70)
                draw_glass_panel(image, (c3x, c3y), (c3x+cw, c3y+ch), radius=18, alpha=0.75, bg_color=(28,28,36), border_color=sb)
                sc_hi = (int(240*c3_at), int(240*c3_at), int(248*c3_at))
                sc_lo = (int(100*c3_at), int(100*c3_at), int(115*c3_at))
                cv2.putText(image, "3D ART", (c3x+54, c3y+60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, sc_hi, 1+int(sculpt_sel), cv2.LINE_AA)
                cv2.putText(image, "3 fingers", (c3x+50, c3y+90), cv2.FONT_HERSHEY_SIMPLEX, 0.42, sc_lo, 1, cv2.LINE_AA)
                cv2.putText(image, "hold to select", (c3x+30, c3y+120), cv2.FONT_HERSHEY_SIMPLEX, 0.36, sc_lo, 1, cv2.LINE_AA)
                
                # Selection highlight fade-in
                if menu_selection_frames > 0:
                    prog = menu_selection_frames / 15.0
                    fill_c = (120, 90, 255)
                    ov_fill = image.copy()
                    if news_sel:
                        draw_rounded_rect(ov_fill, (c1x, c1y), (c1x+cw, c1y+ch), fill_c, cv2.FILLED, radius=18)
                    elif draw_sel:
                        draw_rounded_rect(ov_fill, (c2x, c2y), (c2x+cw, c2y+ch), fill_c, cv2.FILLED, radius=18)
                    elif sculpt_sel:
                        draw_rounded_rect(ov_fill, (c3x, c3y), (c3x+cw, c3y+ch), fill_c, cv2.FILLED, radius=18)
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
                    prog = min(ui_hover_frames / HOVER_TRIGGER_FRAMES, 1.0)
                    cv2.line(image, (bx, by+27), (bx + int(76*prog), by+27), (120, 90, 255), 2, cv2.LINE_AA)

                # ── Tool icons (center) ──
                tool_list  = ['PEN', 'MARKER', 'NEON', 'AURORA', 'CALLIGRAPHY', 'SPRAY', 'ERASER']
                tool_icons = ['P', 'M', 'N', 'A', 'C', 'S', 'E']
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
                        angle = int((ui_hover_frames / HOVER_TRIGGER_FRAMES) * 360)
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
                        angle = int((ui_hover_frames / HOVER_TRIGGER_FRAMES) * 360)
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
                    prog = min(ui_hover_frames / HOVER_TRIGGER_FRAMES, 1.0)
                    cv2.line(image, (hud_x+8, hud_y+51), (hud_x+8+int(30*prog), hud_y+51), (120, 90, 255), 2, cv2.LINE_AA)

                grid_btn_col = (42, 42, 58) if grid_mode else (28, 28, 38)
                grid_txt_col = (140, 120, 220) if grid_mode else (80, 80, 100)
                draw_rounded_rect(image, (hud_x+44, hud_y+30), (hud_x+136, hud_y+52), grid_btn_col, radius=4)
                grid_lbl = "GRID: ON" if grid_mode else "GRID: OFF"
                cv2.putText(image, grid_lbl, (hud_x+54, hud_y+46), cv2.FONT_HERSHEY_SIMPLEX, 0.38, grid_txt_col, 1, cv2.LINE_AA)
                if ui_hover_target == 'ACTION_GRID' and ui_hover_frames > 0:
                    prog = min(ui_hover_frames / HOVER_TRIGGER_FRAMES, 1.0)
                    cv2.line(image, (hud_x+44, hud_y+51), (hud_x+44+int(92*prog), hud_y+51), (120, 90, 255), 2, cv2.LINE_AA)

                draw_rounded_rect(image, (hud_x+142, hud_y+30), (hud_x+172, hud_y+52), (35, 35, 48), radius=4)
                cv2.putText(image, "+", (hud_x+150, hud_y+46), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (120, 100, 200), 1, cv2.LINE_AA)
                if ui_hover_target == 'ACTION_FINGERS_PLUS' and ui_hover_frames > 0:
                    prog = min(ui_hover_frames / HOVER_TRIGGER_FRAMES, 1.0)
                    cv2.line(image, (hud_x+142, hud_y+51), (hud_x+142+int(30*prog), hud_y+51), (120, 90, 255), 2, cv2.LINE_AA)

                # Mirror dot indicator
                mir_col = (72, 199, 142) if mirror_mode else (55, 55, 75)
                cv2.circle(image, (hud_x+165, hud_y+14), 5, mir_col, cv2.FILLED, cv2.LINE_AA)
                cv2.putText(image, "MIR", (hud_x+125, hud_y+22), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (80, 80, 100), 1, cv2.LINE_AA)
                if ui_hover_target == 'ACTION_MIRROR' and ui_hover_frames > 0:
                    prog = min(ui_hover_frames / HOVER_TRIGGER_FRAMES, 1.0)
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
                        prog = min(ui_hover_frames / HOVER_TRIGGER_FRAMES, 1.0)
                        cv2.line(image, (bbx, bby+43), (bbx+int(46*prog), bby+43), bcol, 2, cv2.LINE_AA)

                # ── Futuristic Holographic Cyber Cursor ──
                if results and results.hand_landmarks:
                    pulse = 0.5 + 0.5 * math.sin(time.time() * 8)
                    for h_idx, hlm in enumerate(results.hand_landmarks):
                        h_label = results.handedness[h_idx][0].category_name
                        if h_label in hand_ema:
                            idx_tip_coords = hand_ema[h_label][8]
                            cx_cur = int(idx_tip_coords[0] * w)
                            cy_cur = int(idx_tip_coords[1] * h)
                            c_color = current_draw_color if current_draw_tool != 'ERASER' else (200, 200, 210)
                            
                            # Outer target ring
                            r_outer = int(16 + pulse * 6) if is_drawing else int(12 + pulse * 3)
                            cv2.circle(image, (cx_cur, cy_cur), r_outer, c_color, 1, cv2.LINE_AA)
                            
                            # Crosshair marks for precision HUD aesthetic
                            cv2.line(image, (cx_cur - r_outer - 4, cy_cur), (cx_cur - r_outer + 2, cy_cur), c_color, 1, cv2.LINE_AA)
                            cv2.line(image, (cx_cur + r_outer - 2, cy_cur), (cx_cur + r_outer + 4, cy_cur), c_color, 1, cv2.LINE_AA)
                            cv2.line(image, (cx_cur, cy_cur - r_outer - 4), (cx_cur, cy_cur - r_outer + 2), c_color, 1, cv2.LINE_AA)
                            cv2.line(image, (cx_cur, cy_cur + r_outer - 2), (cx_cur, cy_cur + r_outer + 4), c_color, 1, cv2.LINE_AA)
                            
                            # Center precision core dot
                            cv2.circle(image, (cx_cur, cy_cur), 3, (255, 255, 255), cv2.FILLED, cv2.LINE_AA)

                # ── fps (tiny, bottom center) ──
                cv2.putText(image, f'{int(fps)}fps', (w//2 - 15, h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (50, 50, 65), 1, cv2.LINE_AA)

                # Rebuild hover targets to match new layout
                btn_y_ref = 0  # unused sentinel

            elif app_mode == '3D_MODE':
                anim_t = ease_out_cubic(min(mode_enter_frame / 18.0, 1.0))
                
                # 1. Holographic Floor Grid (XZ plane at Y = 100)
                grid_color = (48, 42, 60)
                for z_val in range(-200, 201, 40):
                    pt1 = project_point(-200, 100, z_val, angle_x, angle_y, zoom_level, w//2, h//2)
                    pt2 = project_point(200, 100, z_val, angle_x, angle_y, zoom_level, w//2, h//2)
                    cv2.line(image, (pt1[0], pt1[1]), (pt2[0], pt2[1]), grid_color, 1, cv2.LINE_AA)
                for x_val in range(-200, 201, 40):
                    pt1 = project_point(x_val, 100, -200, angle_x, angle_y, zoom_level, w//2, h//2)
                    pt2 = project_point(x_val, 100, 200, angle_x, angle_y, zoom_level, w//2, h//2)
                    cv2.line(image, (pt1[0], pt1[1]), (pt2[0], pt2[1]), grid_color, 1, cv2.LINE_AA)
                    
                # 2. Glowing Coordinate Axes at Grid Center
                orig = project_point(0, 100, 0, angle_x, angle_y, zoom_level, w//2, h//2)
                ax_x = project_point(80, 100, 0, angle_x, angle_y, zoom_level, w//2, h//2)
                ax_y = project_point(0, 20, 0, angle_x, angle_y, zoom_level, w//2, h//2) # Upward in 3D
                ax_z = project_point(0, 100, 80, angle_x, angle_y, zoom_level, w//2, h//2)
                
                # Draw lines
                cv2.line(image, (orig[0], orig[1]), (ax_x[0], ax_x[1]), (70, 70, 220), 2, cv2.LINE_AA) # X: Red
                cv2.line(image, (orig[0], orig[1]), (ax_y[0], ax_y[1]), (70, 220, 70), 2, cv2.LINE_AA) # Y: Green
                cv2.line(image, (orig[0], orig[1]), (ax_z[0], ax_z[1]), (220, 70, 70), 2, cv2.LINE_AA) # Z: Blue
                
                # Text labels
                cv2.putText(image, "X", (ax_x[0]+5, ax_x[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (70, 70, 220), 1, cv2.LINE_AA)
                cv2.putText(image, "Y", (ax_y[0], ax_y[1]-5), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (70, 220, 70), 1, cv2.LINE_AA)
                cv2.putText(image, "Z", (ax_z[0]+5, ax_z[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (220, 70, 70), 1, cv2.LINE_AA)
                
                # Setup projection queues
                render_queue_faces = []
                render_queue_lines = []
                
                # 3D Extruded Prism
                if extruded_prism is not None:
                    vertices, edges, faces = extruded_prism
                    rot_t = time.time() * 15.0
                    proj_verts = []
                    cam_verts = []
                    for v in vertices:
                        rx, ry, rz = rotate_3d_point(v[0], v[1], v[2], 0.0, rot_t, 0.0)
                        # Sit on grid floor
                        ry += 30.0
                        
                        screen_px, screen_py, adj_z = project_point(rx, ry, rz, angle_x, angle_y, zoom_level, w//2, h//2)
                        proj_verts.append((screen_px, screen_py))
                        
                        cx_c, cy_c, cz_c = rotate_3d_point(rx, ry, rz, angle_x, angle_y, 0)
                        cam_verts.append(cz_c)
                        
                    for edge in edges:
                        v1, v2 = proj_verts[edge[0]], proj_verts[edge[1]]
                        z_mid = (cam_verts[edge[0]] + cam_verts[edge[1]]) / 2.0
                        render_queue_lines.append({
                            'p1': v1, 'p2': v2, 'z': z_mid,
                            'color': (72, 199, 142), 'thickness': 2, 'neon': True
                        })
                    for face in faces:
                        pts = np.array([proj_verts[idx] for idx in face], dtype=np.int32)
                        z_avg = sum(cam_verts[idx] for idx in face) / len(face)
                        render_queue_faces.append({
                            'pts': pts, 'z': z_avg,
                            'color': (72, 199, 142), 'alpha': 0.16
                        })
                        
                # 3D Placed Primitives
                for prim in placed_primitives_3d:
                    ptype = prim['type']
                    pos_w = prim['pos']
                    pcol = prim['color']
                    psize = prim['size']
                    
                    if ptype == 'CUBE':
                        verts, edges, faces = make_cube(psize)
                    elif ptype == 'PYRAMID':
                        verts, edges, faces = make_pyramid(psize)
                    else:
                        verts, edges, faces = make_cylinder(psize/2.0, psize)
                        
                    proj_verts = []
                    cam_verts = []
                    for v in verts:
                        # Rotate locally
                        rx, ry, rz = rotate_3d_point(v[0], v[1], v[2], prim['rot'][0], prim['rot'][1], prim['rot'][2])
                        # Translate in world coords
                        wx, wy, wz = rx + pos_w[0], ry + pos_w[1], rz + pos_w[2]
                        
                        screen_px, screen_py, adj_z = project_point(wx, wy, wz, angle_x, angle_y, zoom_level, w//2, h//2)
                        proj_verts.append((screen_px, screen_py))
                        
                        cx_c, cy_c, cz_c = rotate_3d_point(wx, wy, wz, angle_x, angle_y, 0)
                        cam_verts.append(cz_c)
                        
                    for edge in edges:
                        v1, v2 = proj_verts[edge[0]], proj_verts[edge[1]]
                        z_mid = (cam_verts[edge[0]] + cam_verts[edge[1]]) / 2.0
                        render_queue_lines.append({
                            'p1': v1, 'p2': v2, 'z': z_mid,
                            'color': pcol, 'thickness': 2, 'neon': True
                        })
                    for face in faces:
                        pts = np.array([proj_verts[idx] for idx in face], dtype=np.int32)
                        z_avg = sum(cam_verts[idx] for idx in face) / len(face)
                        render_queue_faces.append({
                            'pts': pts, 'z': z_avg,
                            'color': pcol, 'alpha': 0.22
                        })
                        
                # 3D Freehand strokes
                for stroke in draw_points_3d:
                    pts_w = stroke['points']
                    scol = stroke['color']
                    stool = stroke['tool']
                    
                    proj_pts = []
                    cam_depths = []
                    for pt in pts_w:
                        screen_px, screen_py, adj_z = project_point(pt[0], pt[1], pt[2], angle_x, angle_y, zoom_level, w//2, h//2)
                        proj_pts.append((screen_px, screen_py))
                        cx_c, cy_c, cz_c = rotate_3d_point(pt[0], pt[1], pt[2], angle_x, angle_y, 0)
                        cam_depths.append(cz_c)
                        
                    for i in range(len(proj_pts) - 1):
                        p1, p2 = proj_pts[i], proj_pts[i+1]
                        z_mid = (cam_depths[i] + cam_depths[i+1]) / 2.0
                        
                        thick = 2
                        if stool == 'MARKER': thick = 6
                        is_neon = (stool == 'NEON')
                        
                        render_queue_lines.append({
                            'p1': p1, 'p2': p2, 'z': z_mid,
                            'color': scol, 'thickness': thick, 'neon': is_neon
                        })
                        
                # Current active stroke
                if len(current_stroke_3d) > 0:
                    proj_pts = []
                    cam_depths = []
                    for pt in current_stroke_3d:
                        screen_px, screen_py, adj_z = project_point(pt[0], pt[1], pt[2], angle_x, angle_y, zoom_level, w//2, h//2)
                        proj_pts.append((screen_px, screen_py))
                        cx_c, cy_c, cz_c = rotate_3d_point(pt[0], pt[1], pt[2], angle_x, angle_y, 0)
                        cam_depths.append(cz_c)
                        
                    for i in range(len(proj_pts) - 1):
                        p1, p2 = proj_pts[i], proj_pts[i+1]
                        z_mid = (cam_depths[i] + cam_depths[i+1]) / 2.0
                        render_queue_lines.append({
                            'p1': p1, 'p2': p2, 'z': z_mid,
                            'color': current_draw_color, 'thickness': 2, 'neon': (current_draw_tool == 'NEON')
                        })
                        
                # Depth buffer rendering
                render_queue_faces.sort(key=lambda f: f['z'], reverse=True)
                overlay = image.copy()
                for face in render_queue_faces:
                    cv2.fillPoly(overlay, [face['pts']], face['color'], cv2.LINE_AA)
                    cv2.addWeighted(overlay, face['alpha'], image, 1.0 - face['alpha'], 0, image)
                    np.copyto(overlay, image)
                    
                render_queue_lines.sort(key=lambda l: l['z'], reverse=True)
                for line in render_queue_lines:
                    p1, p2 = line['p1'], line['p2']
                    col = line['color']
                    thick = line['thickness']
                    if line['neon']:
                        cv2.line(image, p1, p2, (max(0, col[0]-60), max(0, col[1]-60), max(0, col[2]-60)), thick * 3, cv2.LINE_AA)
                        cv2.line(image, p1, p2, col, thick, cv2.LINE_AA)
                        cv2.line(image, p1, p2, (255, 255, 255), max(1, thick//2), cv2.LINE_AA)
                    else:
                        cv2.line(image, p1, p2, col, thick, cv2.LINE_AA)
                        
                # ── HUD / Menu Panels Rendering ──
                # Top bar panel
                bar_h = 56
                bar_y = int(-bar_h + anim_t * (bar_h + 2))
                bar_bg = image[max(0,bar_y):max(0,bar_y)+bar_h, :].copy()
                if bar_bg.shape[0] > 0:
                    dark = np.full_like(bar_bg, (22, 22, 28), dtype=np.uint8)
                    cv2.addWeighted(dark, 0.82, bar_bg, 0.18, 0, bar_bg)
                    image[max(0,bar_y):max(0,bar_y)+bar_h, :] = bar_bg
                cv2.line(image, (0, bar_y+bar_h-1), (w, bar_y+bar_h-1), (55, 55, 75), 1, cv2.LINE_AA)
                
                # Back button
                bx, by = 18, bar_y + 14
                draw_rounded_rect(image, (bx, by), (bx+76, by+28), (35, 35, 45), radius=6)
                cv2.putText(image, "< BACK", (bx+8, by+19), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 175), 1, cv2.LINE_AA)
                if ui_hover_target == 'ACTION_BACK' and ui_hover_frames > 0:
                    prog = min(ui_hover_frames / HOVER_TRIGGER_FRAMES, 1.0)
                    cv2.line(image, (bx, by+27), (bx + int(76*prog), by+27), (120, 90, 255), 2, cv2.LINE_AA)
                    
                # Center Actions: RESET CAM, CLEAR, EXTRUDE
                tb_w = 3 * 120
                tb_x = w // 2 - tb_w // 2
                for bi, (bname, blbl) in enumerate([
                    ('ACTION_3D_RESET_CAM', 'RESET CAM'),
                    ('ACTION_3D_CLEAR', 'CLEAR'),
                    ('ACTION_3D_EXTRUDE', 'EXTRUDE 2D')
                ]):
                    bbx = tb_x + bi * 120
                    bby = bar_y + 14
                    draw_rounded_rect(image, (bbx, bby), (bbx+100, bby+28), (35, 35, 45), radius=6)
                    cv2.putText(image, blbl, (bbx+10, bby+19), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (180, 180, 200), 1, cv2.LINE_AA)
                    if ui_hover_target == bname and ui_hover_frames > 0:
                        prog = min(ui_hover_frames / HOVER_TRIGGER_FRAMES, 1.0)
                        cv2.line(image, (bbx, bby+27), (bbx+int(100*prog), bby+27), (120, 90, 255), 2, cv2.LINE_AA)
                        
                # Left side primitives selector panel
                panel_y = int(120 - (1.0 - anim_t) * 150)
                draw_glass_panel(image, (12, panel_y - 28), (134, panel_y + 220), radius=10, alpha=0.75, bg_color=(20,20,25))
                cv2.putText(image, "3D PRIMITIVES", (18, panel_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (100, 90, 160), 1, cv2.LINE_AA)
                
                for bi, (bname, blbl) in enumerate([
                    ('SELECT_3D_CUBE', 'CUBE'),
                    ('SELECT_3D_PYRAMID', 'PYRAMID'),
                    ('SELECT_3D_CYLINDER', 'CYLINDER')
                ]):
                    bbx = 18
                    bby = panel_y + 8 + bi * 50
                    active = (selected_primitive_type == blbl)
                    bg_col = (55, 44, 100) if active else (30, 30, 40)
                    txt_col = (220, 220, 250) if active else (130, 130, 150)
                    draw_rounded_rect(image, (bbx, bby), (bbx+106, bby+36), bg_col, radius=5)
                    cv2.putText(image, blbl, (bbx+10, bby+22), cv2.FONT_HERSHEY_SIMPLEX, 0.38, txt_col, 1, cv2.LINE_AA)
                    if ui_hover_target == bname and ui_hover_frames > 0:
                        prog = min(ui_hover_frames / HOVER_TRIGGER_FRAMES, 1.0)
                        cv2.line(image, (bbx, bby+35), (bbx+int(106*prog), bby+35), (120, 90, 255), 2, cv2.LINE_AA)
                        
                # Place shape button
                bby = panel_y + 164
                draw_rounded_rect(image, (18, bby), (124, bby+36), (40, 60, 50), radius=5)
                cv2.putText(image, "+ PLACE SHAPE", (24, bby+22), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (120, 220, 150), 1, cv2.LINE_AA)
                if ui_hover_target == 'ACTION_3D_ADD_SHAPE' and ui_hover_frames > 0:
                    prog = min(ui_hover_frames / HOVER_TRIGGER_FRAMES, 1.0)
                    cv2.line(image, (18, bby+35), (18+int(106*prog), bby+35), (120, 90, 255), 2, cv2.LINE_AA)
                    
                # Bottom Status HUD
                hud_x, hud_y = 18, h - 84
                draw_glass_panel(image, (hud_x, hud_y), (hud_x+300, hud_y+68), radius=8, alpha=0.75, bg_color=(20,20,25))
                cv2.putText(image, f"CAM PITCH: {int(angle_x)}deg | YAW: {int(angle_y)}deg", (hud_x+12, hud_y+20), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (150,150,170), 1, cv2.LINE_AA)
                cv2.putText(image, f"SHAPES: {len(placed_primitives_3d)} | 3D PATHS: {len(draw_points_3d)}", (hud_x+12, hud_y+38), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (150,150,170), 1, cv2.LINE_AA)
                
                guideline_txt = "FIST = ROTATE CAM | INDEX = DRAW 3D"
                cv2.putText(image, guideline_txt, (hud_x+12, hud_y+55), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (100, 90, 160), 1, cv2.LINE_AA)
                
                # 3D Cursor Indicator
                if results and results.hand_landmarks:
                    pulse = 0.5 + 0.5 * math.sin(time.time() * 8)
                    for h_idx, hlm in enumerate(results.hand_landmarks):
                        h_label = results.handedness[h_idx][0].category_name
                        if h_label in hand_ema:
                            idx_tip_coords = hand_ema[h_label][8]
                            cx_cur = int(idx_tip_coords[0] * w)
                            cy_cur = int(idx_tip_coords[1] * h)
                            
                            c_color = current_draw_color if not is_dragging_cam else (150,150,170)
                            
                            # Outer dynamic target ring
                            r_outer = int(18 + pulse * 6) if is_drawing else int(12 + pulse * 3)
                            cv2.circle(image, (cx_cur, cy_cur), r_outer, c_color, 1, cv2.LINE_AA)
                            
                            # Draw coordinate markers around 3D cursor
                            cv2.putText(image, f"Z:{int(raw_lm_list[8].z * 600)}", (cx_cur+r_outer+6, cy_cur+4), cv2.FONT_HERSHEY_SIMPLEX, 0.3, c_color, 1, cv2.LINE_AA)
                            
                            # Center precision point
                            cv2.circle(image, (cx_cur, cy_cur), 3, (255, 255, 255), cv2.FILLED, cv2.LINE_AA)
                            
                cv2.putText(image, f'{int(fps)}fps', (w//2 - 15, h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (50, 50, 65), 1, cv2.LINE_AA)

            # ── Session Logging ──
            t_str = time.strftime('%H:%M:%S')
            cx_val = cx if 'cx' in locals() else None
            cy_val = cy if 'cy' in locals() else None
            has_hand = len(results.hand_landmarks) > 0 if (results and results.hand_landmarks) else False
            active_f = hand_fingers if (results and results.hand_landmarks) else []
            tgt = current_target if 'current_target' in locals() else None
            
            log_entry = f"[{t_str}] Mode: {app_mode} | Hand: {has_hand} | Fingers: {active_f} | Draw: {is_drawing} | DragCam: {is_dragging_cam} | Cursor: ({cx_val}, {cy_val}) | Hover UI: {tgt}\n"
            log_buffer.append(log_entry)
            
            if len(log_buffer) >= 30:
                try:
                    with open(log_file_path, "a") as f:
                        f.writelines(log_buffer)
                except Exception:
                    pass
                log_buffer.clear()

            cv2.imshow('Precise Finger Counter', image)
            
            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
