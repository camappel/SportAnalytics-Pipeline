import models
import competency
from gui import *
import pose_estimation
import pygame
import numpy as np
import csv
import os

# Configuration for video processing
cfg = {
    "SCREEN_SIZE": (1920, 1080),
    "SOURCE_SIZE": None,
    'PATH_VIDEO_OPEN': 'video cut 3.mp4',
    'PATH_VIDEO_SAVE': 'video combinedscreen 3.mp4',
    'PATH_CSV_SAVE': 'metrics_data.csv',
    'SIZE_VIDEO_SAVE': (1920, 1080),
    "FPS_CAP": 30,
    'fps': 30,
    "WEBCAM_FLIPPED": True,
    'dummy_left_default': 0.19,
    'dummy_right_default': 0.26,
    'dummy_top_default': 0.38,
    'dummy_bottom_default': 0.75,
    'dummy_real_height': 1.3,
    'right_foot': False
}

# Create a enhanced pipeline model controller with CSV output
class EnhancedPipelineModel(models.Model):
    def __init__(self, cfg):
        self.cfg = cfg
        self.source = models.OpenVideoFileModel(cfg)
        self.dummy = models.DummyModel(cfg, [])
        self.poser = pose_estimation.PoseModel(cfg)
        self.saver = models.SaveVideoFileModel(cfg)
        self.hit_detected = False
        
        # Create CSV file and write header
        self.csv_file = open(cfg['PATH_CSV_SAVE'], 'w', newline='')
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow([
            'Frame', 
            'Time (s)', 
            'Pose Found', 
            'Distance to Dummy', 
            'Hit Detected',
            'Head Position X',
            'Head Position Y',
            'Left Shoulder X',
            'Left Shoulder Y',
            'Right Shoulder X',
            'Right Shoulder Y',
            'Left Hip X',
            'Left Hip Y',
            'Right Hip X',
            'Right Hip Y',
            'Left Knee X',
            'Left Knee Y',
            'Right Knee X',
            'Right Knee Y',
            'Left Ankle X',
            'Left Ankle Y',
            'Right Ankle X',
            'Right Ankle Y'
        ])
        self.frame_count = 0
        
    def process(self, data):
        # Process source video
        data = self.source.process(data)
        if not data['healthy']:
            return data
            
        # Process dummy boundaries
        data = self.dummy.process(data)
        
        # Process pose estimation
        data = self.poser.process(data)
        self.frame_count += 1
        
        # Calculate metrics and record to CSV
        pose_found = data['pose_found']
        if pose_found:
            # Calculate distance to dummy
            data['pose_furthest'] = data['pose'].furthest_shoulder()
            data['dis_to_dummy'] = data['pose_furthest'] - data['dummy_right']
            
            # Detect hit
            if data['dis_to_dummy'] < 0 and not self.hit_detected:
                self.hit_detected = True
                print("Hit detected!")
                
            # Extract key landmarks for CSV
            pose = data['pose']
            
            # Get landmark coordinates
            head_pos = pose.pos_local('nose')
            l_shoulder = pose.pos_local('shoulder_l')
            r_shoulder = pose.pos_local('shoulder_r')
            l_hip = pose.pos_local('hip_l')
            r_hip = pose.pos_local('hip_r')
            l_knee = pose.pos_local('knee_l')
            r_knee = pose.pos_local('knee_r')
            l_ankle = pose.pos_local('ankle_l')
            r_ankle = pose.pos_local('ankle_r')
            
            # Write frame data to CSV
            self.csv_writer.writerow([
                self.frame_count,                             # Frame number
                round(self.frame_count / cfg['fps'], 3),      # Time in seconds
                pose_found,                                   # Pose detection status
                round(data['dis_to_dummy'], 4),              # Distance to dummy
                self.hit_detected,                           # Hit detection status
                round(head_pos.x, 4), round(head_pos.y, 4),  # Head position
                round(l_shoulder.x, 4), round(l_shoulder.y, 4), # Left shoulder
                round(r_shoulder.x, 4), round(r_shoulder.y, 4), # Right shoulder
                round(l_hip.x, 4), round(l_hip.y, 4),        # Left hip
                round(r_hip.x, 4), round(r_hip.y, 4),        # Right hip
                round(l_knee.x, 4), round(l_knee.y, 4),      # Left knee
                round(r_knee.x, 4), round(r_knee.y, 4),      # Right knee
                round(l_ankle.x, 4), round(l_ankle.y, 4),    # Left ankle
                round(r_ankle.x, 4), round(r_ankle.y, 4),    # Right ankle
            ])
        else:
            # Write frame data with NaN for pose information
            self.csv_writer.writerow([
                self.frame_count,                             # Frame number
                round(self.frame_count / cfg['fps'], 3),      # Time in seconds
                pose_found,                                   # Pose detection status
                'NaN',                                        # Distance to dummy
                self.hit_detected,                            # Hit detection status
            ] + ['NaN'] * 18)  # Fill remaining columns with NaN
                
        return data
    
    def draw(self, data, surface):
        # Draw original video
        self.source.draw(data, surface)
        
        # Draw dummy boundaries
        self.dummy.draw(data, surface)
        
        # Draw pose overlay
        if data['pose_found']:
            self.poser.draw(data, surface)
            
            # Display hit status
            if self.hit_detected:
                draw_text(surface, (50, 50), "HIT DETECTED", (255, 0, 0), size=40)
                
            # Draw useful metrics
            if 'dis_to_dummy' in data:
                draw_text(surface, (50, 100), f"Distance: {data['dis_to_dummy']:.2f}", (0, 255, 0), size=30)
        else:
            draw_text(surface, (20, 20), 'No pose found', (255, 255, 255))
        
        # Draw frame counter
        draw_text(surface, (50, 150), f"Frame: {self.frame_count}", (255, 255, 255), size=30)
        
        # Save the frame
        self.saver.draw(data, surface)
    
    def close(self):
        self.source.close()
        self.saver.close()
        self.csv_file.close()
        print(f"CSV metrics data saved to: {os.path.abspath(self.cfg['PATH_CSV_SAVE'])}")

def process_video():
    # Initialize data dictionary
    data = {
        'healthy': True,
        'rect_target': pygame.Rect(0, 0, cfg['SIZE_VIDEO_SAVE'][0], cfg['SIZE_VIDEO_SAVE'][1]),
        'sx': cfg['SIZE_VIDEO_SAVE'][0],
        'sy': cfg['SIZE_VIDEO_SAVE'][1],
        'dt': 1/cfg['fps'],
        't': 0,
        'multiplier_line_width': min(16*cfg['SIZE_VIDEO_SAVE'][1], 9*cfg['SIZE_VIDEO_SAVE'][0]) * .0003
    }
    
    # Initialize pipeline model
    pipeline = EnhancedPipelineModel(cfg)
    
    # Initialize screen surface
    surface = pygame.Surface(cfg['SIZE_VIDEO_SAVE'])
    
    print(f"Processing video: {cfg['PATH_VIDEO_OPEN']}")
    print(f"Output video: {cfg['PATH_VIDEO_SAVE']}")
    print(f"Output CSV: {cfg['PATH_CSV_SAVE']}")
    
    # Main processing loop
    frame_count = 0
    while True:
        # Process frame
        data = pipeline.process(data)
        
        # Break if video ended or error occurred
        if not data['healthy']:
            break
        
        # Draw the frame
        pipeline.draw(data, surface)
        
        frame_count += 1
        if frame_count % 30 == 0:
            print(f"Processed {frame_count} frames")
    
    # Clean up
    pipeline.close()
    print(f"Processing complete. Processed {frame_count} frames.")

if __name__ == '__main__':
    pygame.init()
    process_video()
    pygame.quit()