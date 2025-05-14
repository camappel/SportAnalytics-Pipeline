import cv2

from mediapipe import solutions
import pygame
import numpy as np

ASPECT_RATIO = 1280/720

from gui import draw_text

connections_visible = (
     (0,7),
     (7,8),
     (8,0),
     (16,14),
     (14,12),
     (12,11),
     (11,13),
     (13,15),
     (12,24),
     (11,23),
     (23,24),
     (24,26),
     (26,28),
     (28,30),
     (30,32),
     (32,28),
     (23,25),
     (25,27),
     (27,29),
     (29,31),
     (31,27)
)

connections_visible_funky = (
     ((0, 0), (11, 12)),
)










class PoseModel():
     def __init__(model, cfg):
          model.cfg = cfg
          model.process_pose = solutions.pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
          #model.default_drawer = mp.solutions.drawing_utils.draw_landmarks


     def process(model, data):
          frame = data["image"]
          data['pose_results'] = model.process_pose.process(frame)
          data['pose'] = Pose(data['pose_results'].pose_landmarks)
          #model.default_drawer(frame, pose.pose_landmarks, mp.solutions.pose.POSE_CONNECTIONS)
          data['pose_found'] = not data['pose_results'].pose_landmarks is None
          return data
     
     def draw(model, data, surface):

          sx,sy = data['sx'], data['sy']
          line_width = max(1, int(data['multiplier_line_width']))

          if not data['pose_found']:
               draw_text(surface, (20,20), 'No pose found', (255,255,255))
               return

          dest = data['webcam_rect']

          def tos(x, y): #to screen
               return dest.left + x*dest.width, dest.top + y*dest.height
          
          for (start_i, end_i) in connections_visible:
               start = data['pose_results'].pose_landmarks.landmark[start_i]
               end   = data['pose_results'].pose_landmarks.landmark[end_i]

               start_pos = tos(start.y, start.x)
               end_pos = tos(end.y, end.x)

               pygame.draw.line(surface, (0,156,222), start_pos, end_pos, line_width)
          
          for (start_is, end_is) in connections_visible_funky:

               start1 = data['pose_results'].pose_landmarks.landmark[start_is[0]]
               end1   = data['pose_results'].pose_landmarks.landmark[end_is[0]]

               start_pos1 = tos(start1.y, start1.x)
               end_pos1 = tos(end1.y, end1.x)

               start2 = data['pose_results'].pose_landmarks.landmark[start_is[1]]
               end2   = data['pose_results'].pose_landmarks.landmark[end_is[1]]

               start_pos2 = tos(start2.y, start2.x)
               end_pos2 = tos(end2.y, end2.x)

               start_pos = ((start_pos1[0] + start_pos2[0])/2, (start_pos1[1] + start_pos2[1])/2) 
               end_pos = ((end_pos1[0] + end_pos2[0])/2, (end_pos1[1] + end_pos2[1])/2)

               pygame.draw.line(surface, (0,156,222), start_pos, end_pos, line_width)
               



     def update(model, dt, events):
          return True
     
     def close(model):
          return
          


class Pose:

     def __init__(self, pose_landmarks):
          self.data = pose_landmarks
          self.key = {
               'nose': 0,
               'ear_l': 7,
               'ear_r': 8,
               'shoulder_l': 11,
               'shoulder_r': 12,
               'elbow_l': 13,
               'elbow_r': 14,
               'wrist_l': 15,
               'wrist_r': 16,
               'hip_l': 23,
               'hip_r': 24,
               'knee_l': 25,
               'knee_r': 26,
               'ankle_l': 27,
               'ankle_r': 28,
               'heel_l': 29,
               'heel_r': 30,
               'toe_l': 31,
               'toe_r': 32
               }
          
     def rect_bounding(self):


          labels_core = ['shoulder_l', 'shoulder_r', 'hip_l', 'hip_r']

          xs = [self.pos_local(label).y for label in labels_core]
          ys = [self.pos_local(label).x for label in labels_core]

          x_mean = sum(xs)/len(xs)
          y_mean = sum(ys)/len(ys)

          width = 0.2
          height = 0.5

          left  = x_mean-width/2
          right = x_mean+width/2
          top   = y_mean-height/2
          bot   = y_mean+height/2

          clip = lambda p: max(0, min(1, p))
          
          left = clip(left); right=clip(right); top=clip(top); bot=clip(bot)

          return left, top,  (right-left), (bot-top)

     

     def pos_local(self, label):
          return self.data.landmark[self.key[label]]

     def between(self, label1, label2):
          # `pos_local` is called inside `between` to get positions
          mark1 = self.pos_local(label1)
          mark2 = self.pos_local(label2)
          return (((mark1.x - mark2.x))**2 + (ASPECT_RATIO * (mark1.y - mark2.y))**2)**0.5

     def angle_between(self, label1, label2, label3, use_z=False):
          # Get positions of the marks based on labels
          mark1 = self.pos_local(label1)
          mark2 = self.pos_local(label2)
          mark3 = self.pos_local(label3)

          # Create vectors from mark2 to mark1 and from mark2 to mark3
          if use_z:
               v1 = np.array([mark1.x - mark2.x, mark1.y - mark2.y, mark1.z - mark2.z])
               v2 = np.array([mark3.x - mark2.x, mark3.y - mark2.y, mark3.z - mark2.z])
          else:
               v1 = np.array([mark1.x - mark2.x, mark1.y - mark2.y])
               v2 = np.array([mark3.x - mark2.x, mark3.y - mark2.y])

          # Calculate the magnitudes of the vectors
          mag_v1 = np.linalg.norm(v1)
          mag_v2 = np.linalg.norm(v2)

          # Calculate the dot product between v1 and v2
          dot_product = np.dot(v1, v2)

          # Calculate the angle in radians and convert to degrees
          angle_radians = np.arccos(np.clip(dot_product / (mag_v1 * mag_v2), -1.0, 1.0))
          return np.degrees(angle_radians)
     


     def angle_to_axis(self, label1, label2, axis='y', use_z=False):
          # Get positions of the marks based on labels
          mark1 = self.pos_local(label1)
          mark2 = self.pos_local(label2)

          # Create the vector from mark1 to mark2
          if use_z:
               vector = np.array([mark2.x - mark1.x, mark2.y - mark1.y, mark2.z - mark1.z])
          else:
               vector = np.array([mark2.x - mark1.x, mark2.y - mark1.y])

          # Define the unit vector for the chosen axis
          if axis == 'x':
               axis_vector = np.array([1, 0, 0] if use_z else [1, 0])
          elif axis == 'y':
               axis_vector = np.array([0, 1, 0] if use_z else [0, 1])
          elif axis == 'z':
               if not use_z:
                    raise ValueError("z-axis can only be used if use_z=True.")
               axis_vector = np.array([0, 0, 1])
          else:
               raise ValueError("Invalid axis. Choose 'x', 'y', or 'z'.")

          # Calculate the magnitudes of the vector and axis_vector
          mag_vector = np.linalg.norm(vector)
          mag_axis_vector = np.linalg.norm(axis_vector)

          # Calculate the dot product between vector and axis_vector
          dot_product = np.dot(vector, axis_vector)

          # Calculate the angle in radians and convert to degrees
          angle_radians = np.arccos(np.clip(dot_product / (mag_vector * mag_axis_vector), -1.0, 1.0))
          return np.degrees(angle_radians)

     def chain_length(self, *chain):
          total_length = 0
          # Pass labels directly to `between`, which now handles `pos_local` calls
          for (label_start, label_end) in zip(chain[:-1], chain[1:]):
               total_length += self.between(label_start, label_end)
          return total_length
     
     def foot_length(self):
          return (
                self.chain_length('heel_l', 'toe_l')
               +self.chain_length('heel_r', 'toe_r') )/2
     
     def furthest_shoulder(self):
          return min(self.pos_local('shoulder_l').y, self.pos_local('shoulder_r').y)

     def height(self):

          return (
                self.chain_length('nose', 'shoulder_l', 'hip_l', 'knee_l', 'ankle_l', 'heel_l')
               +self.chain_length('nose', 'shoulder_r', 'hip_r', 'knee_r', 'ankle_r', 'heel_r') )/2
     
     def hip_height(self):
          return ( self.chain_length('heel_l', 'ankle_l', 'knee_l', 'hip_l') 
                  +self.chain_length('heel_r', 'ankle_r', 'knee_r', 'hip_r') )/2 
     
     def knee_height(self):
          return ( self.chain_length('heel_l', 'knee_l') 
                  +self.chain_length('heel_r', 'knee_r') )/2 
     
     def shoulder_height(self):

          return (
                self.chain_length('shoulder_l', 'hip_l', 'knee_l', 'ankle_l', 'heel_l')
               +self.chain_length('shoulder_r', 'hip_r', 'knee_r', 'ankle_r', 'heel_r') )/2
     
     def hip_bound(self):
          femur_length = (self.chain_length('knee_l', 'hip_l')
                         +self.chain_length('knee_r', 'hip_r') )/2
          torso_length = (self.chain_length('shoulder_l', 'hip_l') 
                         +self.chain_length('shoulder_l', 'hip_l'))/2
          return (self.hip_height() - femur_length * 1/2,
                  self.hip_height() + torso_length * 1/3)





if __name__=='__main__':
     mp_drawing = mp.solutions.drawing_utils
     mp_pose = mp.solutions.pose
     pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)






     cap = cv2.VideoCapture(0)
     while cap.isOpened():
          # read frame
          _, frame = cap.read()
          try:
               # resize the frame for portrait video
               # frame = cv2.resize(frame, (350, 600))
               # convert to RGB
               frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
               
               # process the frame for pose detection
               pose_results = pose.process(frame_rgb)
               # print(pose_results.pose_landmarks)
               #print()
               print(pose_results.pose_landmarks)
               #print()
               raise
               
               # draw skeleton on the frame
               mp_drawing.draw_landmarks(frame, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
               # display the frame
               cv2.imshow('Output', frame)
          except:
               break
     
          if cv2.waitKey(1) == ord('q'):
               break
               
     cap.release()
     cv2.destroyAllWindows()