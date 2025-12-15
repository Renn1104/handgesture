import cv2
import mediapipe as mp
import numpy as np
from gtts import gTTS
import pygame
import os
import time

class IntroductionDetector:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.8,
            model_complexity=0
        )
        self.mp_draw = mp.solutions.drawing_utils
        
        self.landmark_drawing_spec = self.mp_draw.DrawingSpec(
            thickness=2, 
            circle_radius=2, 
            color=(0, 0, 0)
        )
        self.connection_drawing_spec = self.mp_draw.DrawingSpec(
            thickness=2, 
            circle_radius=1, 
            color=(182, 182, 182)
        )
        
        pygame.mixer.init()
        
        if not os.path.exists('audio'):
            os.makedirs('audio')
        
        self.phrases = {
            "open_palm": "perkenalkan",
            "two_fingers": "nama saya",
            "scissors": "rendy",
            "thumb_index": "nayogi",
            "metal": "pramudya",
            "shaka": "terima kasih"
        }
        
        self.generate_audio_files()
        
        self.last_trigger_time = 0
        self.trigger_cooldown = 2.0
        
        self.current_detected_gesture = None
        self.gesture_start_time = 0
        self.display_gesture = None
        self.audio_played = False
        
        self.frame_count = 0
        self.fps = 0
        self.fps_time = time.time()
        
    def generate_audio_files(self):
        print("Generating audio files...")
        for key, phrase in self.phrases.items():
            audio_path = f'audio/{key}_{phrase.replace(" ", "_")}.mp3'
            if not os.path.exists(audio_path):
                tts = gTTS(text=phrase, lang='id', slow=False)
                tts.save(audio_path)
                print(f"Generated: {audio_path}")
        print("Audio files ready!")
    
    def play_audio(self, phrase_key):
        phrase = self.phrases[phrase_key]
        audio_path = f'audio/{phrase_key}_{phrase.replace(" ", "_")}.mp3'
        
        if os.path.exists(audio_path):
            pygame.mixer.music.load(audio_path)
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy():
                pygame.time.Clock().tick(10)
    
    def is_open_palm(self, hand_landmarks):
        finger_tips = [8, 12, 16, 20]
        thumb_tip = 4
        
        fingers_up = 0
        
        if hand_landmarks.landmark[thumb_tip].x < hand_landmarks.landmark[thumb_tip - 1].x:
            fingers_up += 1
        
        for tip in finger_tips:
            if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[tip - 2].y:
                fingers_up += 1
        
        return fingers_up == 5
    
    def is_two_fingers(self, hand_landmarks):
        index_tip = hand_landmarks.landmark[8]
        index_pip = hand_landmarks.landmark[6]
        middle_tip = hand_landmarks.landmark[12]
        middle_pip = hand_landmarks.landmark[10]
        
        ring_tip = hand_landmarks.landmark[16]
        ring_pip = hand_landmarks.landmark[14]
        pinky_tip = hand_landmarks.landmark[20]
        pinky_pip = hand_landmarks.landmark[18]
        
        index_up = index_tip.y < index_pip.y
        middle_up = middle_tip.y < middle_pip.y
        
        ring_down = ring_tip.y > ring_pip.y
        pinky_down = pinky_tip.y > pinky_pip.y
        
        return index_up and middle_up and ring_down and pinky_down
    
    def is_scissors(self, hand_landmarks):
        index_tip = hand_landmarks.landmark[8]
        index_pip = hand_landmarks.landmark[6]
        middle_tip = hand_landmarks.landmark[12]
        middle_pip = hand_landmarks.landmark[10]
        ring_tip = hand_landmarks.landmark[16]
        ring_pip = hand_landmarks.landmark[14]
        
        pinky_tip = hand_landmarks.landmark[20]
        pinky_pip = hand_landmarks.landmark[18]
        thumb_tip = hand_landmarks.landmark[4]
        thumb_mcp = hand_landmarks.landmark[2]
        
        index_up = index_tip.y < index_pip.y
        middle_up = middle_tip.y < middle_pip.y
        ring_up = ring_tip.y < ring_pip.y
        
        pinky_down = pinky_tip.y > pinky_pip.y
        
        thumb_not_extended = thumb_tip.x > thumb_mcp.x - 0.05
        
        return index_up and middle_up and ring_up and pinky_down and thumb_not_extended
    
    def is_thumb_index(self, hand_landmarks):
        index_tip = hand_landmarks.landmark[8]
        index_base = hand_landmarks.landmark[6]
        thumb_tip = hand_landmarks.landmark[4]
        thumb_base = hand_landmarks.landmark[2]
        
        middle_tip = hand_landmarks.landmark[12]
        middle_base = hand_landmarks.landmark[10]
        ring_tip = hand_landmarks.landmark[16]
        ring_base = hand_landmarks.landmark[14]
        pinky_tip = hand_landmarks.landmark[20]
        pinky_base = hand_landmarks.landmark[18]
        
        index_up = index_tip.y < index_base.y
        
        thumb_out = abs(thumb_tip.x - thumb_base.x) > 0.05 or thumb_tip.y < thumb_base.y
        
        middle_down = middle_tip.y > middle_base.y
        ring_down = ring_tip.y > ring_base.y
        pinky_down = pinky_tip.y > pinky_base.y
        
        return index_up and thumb_out and middle_down and ring_down and pinky_down
    
    def is_metal(self, hand_landmarks):
        index_tip = hand_landmarks.landmark[8]
        index_base = hand_landmarks.landmark[6]
        thumb_tip = hand_landmarks.landmark[4]
        thumb_base = hand_landmarks.landmark[2]
        pinky_tip = hand_landmarks.landmark[20]
        pinky_base = hand_landmarks.landmark[18]
        
        middle_tip = hand_landmarks.landmark[12]
        middle_base = hand_landmarks.landmark[10]
        ring_tip = hand_landmarks.landmark[16]
        ring_base = hand_landmarks.landmark[14]
        
        index_up = index_tip.y < index_base.y
        
        thumb_out = abs(thumb_tip.x - thumb_base.x) > 0.05
        
        pinky_up = pinky_tip.y < pinky_base.y
        
        middle_down = middle_tip.y > middle_base.y
        ring_down = ring_tip.y > ring_base.y
        
        return index_up and thumb_out and pinky_up and middle_down and ring_down
    
    def is_shaka(self, hand_landmarks):
        thumb_tip = hand_landmarks.landmark[4]
        thumb_base = hand_landmarks.landmark[2]
        pinky_tip = hand_landmarks.landmark[20]
        pinky_base = hand_landmarks.landmark[18]
        
        index_tip = hand_landmarks.landmark[8]
        index_base = hand_landmarks.landmark[6]
        middle_tip = hand_landmarks.landmark[12]
        middle_base = hand_landmarks.landmark[10]
        ring_tip = hand_landmarks.landmark[16]
        ring_base = hand_landmarks.landmark[14]
        
        thumb_out = thumb_tip.x < thumb_base.x or thumb_tip.x > thumb_base.x
        pinky_up = pinky_tip.y < pinky_base.y
        
        index_down = index_tip.y > index_base.y
        middle_down = middle_tip.y > middle_base.y
        ring_down = ring_tip.y > ring_base.y
        
        return thumb_out and pinky_up and index_down and middle_down and ring_down
    
    def detect_gesture(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb_frame.flags.writeable = False
        results = self.hands.process(rgb_frame)
        rgb_frame.flags.writeable = True
        
        detected_gesture = None
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_draw.draw_landmarks(
                    frame, 
                    hand_landmarks, 
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_draw.DrawingSpec(thickness=4, circle_radius=2, color=(255, 255, 255)),
                    self.mp_draw.DrawingSpec(thickness=3, circle_radius=1, color=(217, 217, 217))
                )
                
                self.mp_draw.draw_landmarks(
                    frame, 
                    hand_landmarks, 
                    self.mp_hands.HAND_CONNECTIONS,
                    None,
                    self.connection_drawing_spec
                )
                
                for landmark in hand_landmarks.landmark:
                    x = int(landmark.x * frame.shape[1])
                    y = int(landmark.y * frame.shape[0])
                    square_size = 3
                    cv2.rectangle(frame, 
                                (x - square_size, y - square_size), 
                                (x + square_size, y + square_size), 
                                (0, 0, 0), -1)
                
                if self.is_metal(hand_landmarks):
                    detected_gesture = "metal"
                elif self.is_shaka(hand_landmarks):
                    detected_gesture = "shaka"
                elif self.is_thumb_index(hand_landmarks):
                    detected_gesture = "thumb_index"
                elif self.is_scissors(hand_landmarks):
                    detected_gesture = "scissors"
                elif self.is_two_fingers(hand_landmarks):
                    detected_gesture = "two_fingers"
                elif self.is_open_palm(hand_landmarks):
                    detected_gesture = "open_palm"
                    
                break
        
        return detected_gesture
    
    def run(self):
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)
        
        print("\n" + "="*50)
        print("INTRODUCTION DETECTOR - STARTED")
        print("="*50)
        print("\nGesture Mapping:")
        for key, phrase in self.phrases.items():
            print(f"  {key} fingers -> {phrase}")
        print("\nPress 'q' to quit")
        print("Press 'r' to reset")
        print("Press 's' to speak current phrase")
        print("="*50 + "\n")
        print("Deteksi dimulai!\n")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            
            self.frame_count += 1
            if time.time() - self.fps_time > 1:
                self.fps = self.frame_count
                self.frame_count = 0
                self.fps_time = time.time()
            
            gesture = self.detect_gesture(frame)
            
            current_time = time.time()
            if gesture and gesture in self.phrases:
                self.display_gesture = gesture
                
                if gesture != self.current_detected_gesture:
                    self.current_detected_gesture = gesture
                    self.gesture_start_time = current_time
                    self.audio_played = False
                else:
                    hold_time = current_time - self.gesture_start_time
                    
                    if hold_time >= 2.0 and not self.audio_played:
                        gesture_names = {
                            "open_palm": "Open Palm ✋",
                            "two_fingers": "Two Fingers ✌",
                            "scissors": "Scissors ✌",
                            "thumb_index": "Thumb + Index 👍",
                            "metal": "Metal 🤘",
                            "shaka": "Shaka 🤙"
                        }
                        print(f"Detected: {gesture_names.get(gesture, gesture)} -> '{self.phrases[gesture]}'")
                        self.play_audio(gesture)
                        self.audio_played = True
            else:
                self.current_detected_gesture = None
                self.gesture_start_time = 0
                self.display_gesture = None
                self.audio_played = False
            
            self.draw_ui(frame, gesture)
            
            cv2.imshow('Introduction Detector', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                self.current_detected_gesture = None
                self.gesture_start_time = 0
                self.display_gesture = None
                self.audio_played = False
                print("Reset!")
            elif key == ord('s') and gesture in self.phrases:
                self.play_audio(gesture)
        
        cap.release()
        cv2.destroyAllWindows()
    
    def draw_ui(self, frame, gesture):
        height, width, _ = frame.shape
        
        if self.display_gesture and self.display_gesture in self.phrases:
            detected_text = self.phrases[self.display_gesture]
            text_size = cv2.getTextSize(detected_text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)[0]
            text_x = (width - text_size[0]) // 2
            text_y = height - 50
            
            overlay = frame.copy()
            cv2.rectangle(overlay, (text_x - 20, text_y - text_size[1] - 20), 
                         (text_x + text_size[0] + 20, text_y + 10), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
            
            cv2.putText(frame, detected_text, (text_x, text_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

if __name__ == "__main__":
    detector = IntroductionDetector()
    detector.run()
