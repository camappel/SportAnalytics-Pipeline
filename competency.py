from models import Model, ModelController
from gui import Text, draw_text
import pygame
import numpy as np


BLACK = (0,0,0)
GRAY = (100,100,100)
RED = (255,0,0)
YELLOW = (255,255,0)
GREEN = (0,255,0)

class NormalScoreMixin:
    def calculate_score(self, x, mu, sd):
        z = abs(x - mu) / sd
        return max(0, 10 * (1 - z / 3))

def color_from_score(s):
    # Clamp the value of s to the range [0, 1]
    s = max(0, min(s, 1))
    
    if s <= 0.5:
        # Interpolate between RED and YELLOW
        factor = s / 0.5  # Scale s to [0, 1] for this range
        r = int(RED[0] + factor * (YELLOW[0] - RED[0]))
        g = int(RED[1] + factor * (YELLOW[1] - RED[1]))
        b = int(RED[2] + factor * (YELLOW[2] - RED[2]))
    else:
        # Interpolate between YELLOW and GREEN
        factor = (s - 0.5) / 0.5  # Scale s to [0, 1] for this range
        r = int(YELLOW[0] + factor * (GREEN[0] - YELLOW[0]))
        g = int(YELLOW[1] + factor * (GREEN[1] - YELLOW[1]))
        b = int(YELLOW[2] + factor * (GREEN[2] - YELLOW[2]))

    return (r, g, b)


STABLE_MAX = 30*1.5 - 1
def stable_next(prev, val_next):
    val_prev, n_prev = prev

    if n_prev == 0:
        return (val_next, 1)
    
    n_prev = min(STABLE_MAX, n_prev)
    
    return (val_prev * n_prev + val_next) / (n_prev + 1), (n_prev + 1)


# Define two separate lists for the different scoring methods
labels_scores_basic = [
    'score_headheight',
    'score_toescross',
    'score_hug',    
]

labels_scores_full = [
    'score_headheight',
    'score_toescross',
    'score_deceleration',
    'score_hipfoot',
    'score_gaze',
    'score_hug',    
]

class CompetencyModel(ModelController):
    def __init__(control, cfg, components):
        control.reset_hit = True
        control.height_measurements = []  # Store height measurements
        control.average_height = None  # Store the fixed average height
        super().__init__(cfg, components)

    def process(control, data):
        # Set the reset_threshhold to 1
        data['reset_threshhold'] = 1

        if control.reset_hit:
            data['hit_yet'] = False
            data['hit_yet_right'] = False
            data['time_close'] = 0
            data['time_close_decelerating'] = 0
            data['dummy_shoulder_stable'] = (0,0)
            data['dummy_waist_stable'] = (0,0)
            data['dummy_thigh_stable'] = (0,0)
            data['rect_person'] = (0,0,1,1)
            data['score_headheight']   = {'score': 0, 'raw': 0, 'longname': 'Head Height', 'name': 'Head'}
            data['score_toescross']    = {'score': 0, 'raw': 0, 'longname': 'Toes Close',  'name': 'Toes'}
            data['score_deceleration'] = {'score': 0, 'raw': 0, 'longname': 'Deceleration', 'name': 'Decel.'}
            data['score_hipfoot']      = {'score': 0, 'raw': 0, 'longname': 'Center Ahead', 'name': 'Head'}
            data['score_gaze']         = {'score': 0, 'raw': 0, 'longname': 'Gaze Up', 'name': 'Gaze'}
            data['score_hug']          = {'score': 0, 'raw': 0, 'longname': 'Hugging Arms', 'name': 'Hug'}
            data['score_safety']       = {'score': 0, 'raw': 0, 'longname': 'TEK Score', 'name': 'TEK'}
            data['score_performance']  = {'score': 0, 'raw': 0, 'longname': 'Speed Score', 'name': 'Speed'}
            control.height_measurements = []  # Reset height measurements
            control.average_height = None  # Reset average height

        if not data['pose_found']:
            return data
        
        # Resetting

        if control.reset_hit:
            control.reset_hit = False
            data['list_poses_furthest_previous'] = [data['pose_furthest']]

            

        elif data['pose_furthest'] > data['reset_threshhold'] and data['reset_threshhold']<1:
            control.reset_person()
        elif data['hit_yet'] == False and data['dis_to_dummy'] <0:
            data['hit_yet'] = True  
            # print("Hit Yet: True")
        elif data['hit_yet_right'] == False and data['dis_to_dummy_right'] < 0:
            data['hit_yet_right'] = True
            # print("Hit Yet Right: True")
        
        
        data['pose_height']      = data['pose'].height()
        # print (f"Pose Height: {data['pose_height']}")
        data['pose_hip_height']  = data['pose'].hip_height()
        data['pose_knee_height'] = data['pose'].knee_height()
        data['pose_hip_top']     = data['pose'].hip_bound()[1]
        data['pose_hip_bottom']  = data['pose'].hip_bound()[0]
        data['pose_furthest']    = data['pose'].furthest_shoulder()
        data['pose_foot_length'] = data['pose'].foot_length()
        data['rect_person']      = data['pose'].rect_bounding()
        
        data['list_poses_furthest_previous'].append(data['pose_furthest'])
        if len(data['list_poses_furthest_previous'])>20:
            data['list_poses_furthest_previous'] = data['list_poses_furthest_previous'][1:]

        new_hip_height =  data['dummy_bottom'] - data['pose_hip_height']

        data['dummy_hip_height'] = (data['dummy_hip_height']*0.5 + new_hip_height*0.5)
        data['dummy_hip_top'] = data['dummy_bottom'] - data['pose_hip_top']
        data['dummy_hip_bottom'] = data['dummy_bottom'] - data['pose_hip_bottom']
        data['dummy_shoulder_height'] = data['dummy_bottom'] - data['pose'].shoulder_height()
        data['dummy_sternum_height'] = data['dummy_shoulder_height'] * 2/3 + data['dummy_hip_height'] * 1/3
        data['dummy_knee_height'] =  data['dummy_bottom'] - data['pose_knee_height']

        # Calculate the distance to the left dummy
        data['dis_to_dummy'] = data['pose_furthest'] - data['dummy_left']
        data['dis_to_dummy_right']= data['pose_furthest'] - data['dummy_right']
        
        # Print statements for debugging
        # print(f"Pose Furthest: {data['pose_furthest']}")
        # print(f"Dummy Left: {data['dummy_left']}")
        # print(f"Distance to Dummy: {data['dis_to_dummy']}")

        # Stabilised dummy values
        data['dummy_shoulder_stable'] = stable_next(data['dummy_shoulder_stable'], data['dummy_shoulder_height'])
        data['dummy_waist_stable']    = stable_next(data['dummy_waist_stable'],    data['dummy_hip_top'])
        data['dummy_thigh_stable']    = stable_next(data['dummy_thigh_stable'],    data['dummy_hip_bottom'])

        # Measure height and store it if average height is not yet fixed
        if control.average_height is None:
            current_height = data['pose'].height()
            control.height_measurements.append(current_height)
            if len(control.height_measurements) > 10:
                control.height_measurements.pop(0)

            # Calculate average height once after 10 measurements
            if len(control.height_measurements) == 10:
                control.average_height = sum(control.height_measurements) / len(control.height_measurements)
                print(f"Average Height: {control.average_height}")  # Print the average height

        # Use the fixed average height
        if control.average_height is not None:
            data['average_height'] = control.average_height

        # Velocity and acceleration calculation

        dis = np.array(data['list_poses_furthest_previous']) * data['dummy_zoom_factor_y']
        if len(dis)<=1:
            data['vel'] = 0
        else:
            m = len(dis)//2
            data['vel'] = (dis[:m].mean() - dis[m:].mean()).item() / (data['dt'] * m) * data['webcam_rect'].width / data['webcam_rect'].height

        if len(dis)<=2:
            data['acc'] = 0
        else:
            m = len(dis)//2
            q1 = len(dis)//4
            q2 = (len(dis)*3)//4
            data['acc'] = -(dis[:m].mean() + dis[m:].mean() - 2*dis[q1:q2].mean()
                           ).item() / ((len(dis)/4*data['dt'])**2 ) * data['webcam_rect'].width / data['webcam_rect'].height

        
    
        
        return(super().process(data))
    
    def reset_person(control):
        control.reset_hit = True
        control.height_measurements = []  # Reset height measurements
        control.average_height = None  # Reset average height




class Competency(Model):

    def process(model, data):

        return data
    
    def reset_data(model):
        pass

    def draw(model, data, surface):
        return 

class CompetencyHeadHeight(Competency):

    def process(model, data): 

        if data['hit_yet']:
            return data

        model.head_pos = data['pose'].pos_local('nose')
        
        # Use the average height instead of a fixed height
        if 'average_height' in data:
            dummy_height = data['average_height']
        else:
            dummy_height = 1.7  # Fallback to 1.7 if average height is not available

        dummy_top, dummy_bottom = data['dummy_top'], data['dummy_bottom']
        
        # Calculate positions using the average height
        shoulder_pos = dummy_top + dummy_height * 0.18  # 18% from top
        waist_pos = dummy_top + dummy_height * 0.50     # 50% from top (hip level)
        knee_pos = dummy_top + dummy_height * 0.72      # 72% from top
        
        nose_pos = data['pose'].pos_local('nose').x
        
        # Normal distribution for smoother scoring
        # Using Gaussian function: f(x) = a * exp(-(x-b)²/(2*c²))
        # where b is the mean (waist_pos), c is standard deviation, a is max score (10)
        
        # Set standard deviation as fraction of the distance from shoulder to knee
        sd = (knee_pos - shoulder_pos) / 5  # Adjust divisor to control curve steepness
        
        # Calculate score using Gaussian distribution
        gaussian = lambda x, mu, sd: 10 * np.exp(-((x - mu) ** 2) / (2 * sd ** 2))
        score = gaussian(nose_pos, waist_pos, sd)
        
        score = max(min(score, 10), 0)
        data['score_headheight']['score'] = int(score)

        # Debugging print
        if score > 10:
            print(f"Head Height Score exceeded 10: {score}")

        return data

class CompetencyToesCross(Competency):

    def process(model, data):

        print(f"Hit Yet: {data['hit_yet']}")

        if data['hit_yet']:
            return data

        # Get the y-coordinate of the toes
        model.toe_y = min(data['pose'].pos_local('toe_l').y, data['pose'].pos_local('toe_r').y)
        # print(f"toe.y coords: {model.toe_y}")
        foot_length = data['pose_foot_length']
        
        # Calculate distances to dummy_left and dummy_right
        distance_to_dummy_left = data['dummy_left']
        distance_to_dummy_right = data['dummy_right']

        # Determine which dummy is closer
        if abs(model.toe_y - distance_to_dummy_left) < abs(model.toe_y - distance_to_dummy_right):
            closest_dummy_distance = distance_to_dummy_left
            # print(f"Closest Dummy: Left, Distance: {closest_dummy_distance}")
        else:
            closest_dummy_distance = distance_to_dummy_right
            # print(f"Closest Dummy: Right, Distance: {closest_dummy_distance}")

        ideal_position = closest_dummy_distance + foot_length*1

        # Check if the toe y-coordinate is lower than the distance to the closest dummy
        if model.toe_y < closest_dummy_distance:
            # Score is zero if the toe y-coordinate is lower than the closest dummy distance
            data['score_toescross']['score'] = 0
            return data

        # Determine the target position based on the closest dummy
        mean = ideal_position
        std_dev = foot_length  # Standard deviation can be set to the foot length

        # Gaussian function: f(x) = a * exp(-(x-b)²/(2*c²))
        # where a is the max score (10), b is the mean (ideal position), c is the standard deviation
        score = 10 * np.exp(-((model.toe_y - mean) ** 2) / (2 * (std_dev ** 2)))

        # Clamp the score to be between 0 and 10
        score = max(min(score, 10), 0)
        data['score_toescross']['score'] = int(score)
        
        if model.toe_y < closest_dummy_distance:
            # Score is zero if the toe y-coordinate is lower than the closest dummy distance
            data['score_toescross']['score'] = 0
            return data

        # Debugging print
        if score > 10:
            print(f"Foot Close Score exceeded 10: {score}")

        return data
    


class CompetencyDeceleration(Competency):

    def process(model, data):

        if data['hit_yet']:
            return data

        data['time_close'] += data['dt']

        if data['vel']>0.5:
            
            if data['acc'] < 0:
                data['time_close_decelerating'] += data['dt']   

        #print(data['vel'], data['acc'])
        


        # Decay both values over time, so only deceleration in about the last 3 seconds matters, but does not affect the ratio short-term
        data['time_close'] *= 0.99
        data['time_close_decelerating'] *= 0.99

        score = lerp(0, 0.02, 0, 10, data['time_close_decelerating']/data['time_close'] if data['time_close']>0 else 0)
        score = max(min(score, 10), 0)
        data['score_deceleration']['score'] = int(score)

        #model.score = data['time_close_decelerating']/data['time_close'] if data['time_close']>0 else 0
        #print(data['time_close_decelerating']/data['time_close'] if data['time_close']>0 else 0)
        return data


    def simple_draw(model, data, surface):

        sx,sy = data['sx'], data['sy']



class CompetencyHipFoot(Competency):

    def process(model, data): 

        if data['hit_yet']:
            return data

        angle_hipfoot_l = data['pose'].angle_to_axis('hip_l', 'ankle_l', 'y')
        angle_hipfoot_r = data['pose'].angle_to_axis('hip_r', 'ankle_r', 'y')
        

        score = min( lerp(90, 85, 0, 10, angle_hipfoot_l),
                           lerp(90, 85, 0, 10, angle_hipfoot_r))
        score = max(min(score, 10), 0)
        data['score_hipfoot']['score'] = score
        
        data['score_hipfoot']['raw'] = max(angle_hipfoot_l, angle_hipfoot_r)

        return data


class CompetencyGaze(Competency):

    def process(model, data): 

        if data['hit_yet']:
            return data

        angle_gaze_l = data['pose'].angle_to_axis('ear_l', 'nose', 'y')
        angle_gaze_r = data['pose'].angle_to_axis('ear_r', 'nose', 'y')


        score = max( lerp(100, 130, 0, 10, angle_gaze_l),
                           lerp(100,130, 0, 10, angle_gaze_r))
        score = max(min(score, 10), 0)
        data['score_gaze']['score'] = score
        data['score_gaze']['raw'] = max(angle_gaze_r, angle_gaze_l)
        

        return data



class CompetencyHug(Competency):

    def process(model, data): 

        if data['hit_yet_right']:
            return data

        # print("Processing CompetencyHug")  # Print when CompetencyHug is processed

        angle_brachium_l = data['pose'].angle_to_axis('shoulder_l', 'elbow_l', 'x')
        angle_brachium_r = data['pose'].angle_to_axis('shoulder_r', 'elbow_r', 'x')

        score = min( lerp(15, 40, 0, 10, angle_brachium_l),
                    lerp(15, 40, 0, 10, angle_brachium_r))
        score = max(min(score, 10), 0)
        data['score_hug']['score'] = score

        # print(f"Hug Score: {score}")

        # Debugging print
        if score > 10:
            print(f"Hug Score exceeded 10: {score}")

        return data


class CompetencySafety(Competency):

    def process(model, data):
        if data['hit_yet']:
            return data
        
        # Get speed score (0-100)
        speed_score = data['score_performance']['score']
        
        # Calculate speed penalty factor (exponential)
        speed_penalty = 2 ** (speed_score / 175)  # Exponential penalty based on speed score
        multiplier = 2 ** (speed_score / 50)  # Exponential multiplier based on speed score
        
        # Calculate basic TEK score (3 metrics)
        basic_score = sum(data[label]['score'] for label in labels_scores_basic) / len(labels_scores_basic) * 10
        
        # Apply speed penalty to missing points
        if speed_score == 0:
            final_basic_score = basic_score  # No penalty applied if speed score is zero
        else:
            # Apply penalty directly to the base score and then multiply
            final_basic_score = basic_score * (1 - (speed_penalty - 1)) * multiplier  # Adjusting the base score by the penalty and multiplier
        
        # Ensure minimum score for basic_score
        final_basic_score = max(final_basic_score, basic_score * 0.1)  # Ensure score doesn't go below 10% of basic score
        data['score_safety']['score'] = int(final_basic_score)  # Store the final basic score
        
        # Calculate full TEK score (6 metrics)
        full_score = sum(data[label]['score'] for label in labels_scores_full) / len(labels_scores_full) * 10
        
        # Apply speed penalty to full score
        full_score = 100 - ((100 - full_score) * speed_penalty)
        data['score_safety_full'] = {'score': int(full_score), 'longname': 'Full TEK Score', 'name': 'Full TEK'}
        
        return data
    

    
class CompetencySpeed(Competency):
    def __init__(model, cfg):
        model.v = 0
        super().__init__(cfg)

    def process(model, data):
        if data['hit_yet']:
            return data

        v = data['vel']
        model.v = v

        # New scoring logic:
        # 0-2 m/s (walking):  gradual increase
        # 2-4 m/s (jogging): linear increase from 0 to 50
        # 4-8 m/s (running): linear increase from 50 to 100
        # >8 m/s: 100 points
        if v < 2:
            score = v * 5  # Gradual increase from 0 to 10
        elif v < 4:
            score = 10 + (v - 2) * 20  # Continue to increase to 50
        elif v < 8:
            score = 50 + (v - 4) * 12.5  # Continue to increase to 100
        else:
            score = 100

        score = max(min(score, 100), 0)  # Clamp between 0-100
        data['score_performance']['raw'] = v       # Store raw velocity
        data['score_performance']['score'] = score # Store processed score
        return data
        

        
class CompetencyDrawCombined(Competency):
# for keypad 2 

    def draw(model, data, surface):
        sx, sy = data['sx'], data['sy']
        line_width = max(1, int(data['multiplier_line_width']))
        cam_rect = data['webcam_rect']

        # --- FURTHEST SHOULDER INDICATOR ---
        pygame.draw.rect(surface, (255,100,0), (
            cam_rect.left + cam_rect.width * data['pose_furthest'], 0,
            line_width, sy))
        draw_text(surface, (cam_rect.width * data['pose_furthest'] + 5, 5),
                  'furthest shoulder', (255,255,0))

        # --- RIGHT PANEL ---
        pygame.draw.rect(surface, (255,255,255), (sx*0.78, 0, sx*0.22, sy))
        draw_text(surface, (sx*0.8, sy*0.02), 'Hit occurred' if data['hit_yet'] else 'No hit yet',
                  BLACK if data['hit_yet'] else GRAY, size=20)

        real_dis = data['dis_to_dummy'] * data['dummy_zoom_factor_y'] * cam_rect.width / cam_rect.height
        draw_text(surface, (sx*0.8, sy*0.1), 
                  'Distance: {:>4.2f}m\nVelocity: {:>4.1f}m/s\nAcceleration: {:>4.1f}m/s²'.format(
                      real_dis, 
                      0 if -0.1 < data['vel'] <= 0 else data['vel'],
                      0 if -0.1 < data['acc'] <= 0 else data['acc'],
                  ), (0,0,0))

        # --- HEAD HEIGHT ---
        draw_text(surface, (sx*0.8, sy*0.25), 'Head height', BLACK, size=20)
        draw_text(surface, (sx*0.92, sy*0.29), '{}/10'.format(data['score_headheight']['score']),
                  BLACK if data['hit_yet'] else GRAY, size=35, justify='right',
                  shadow_color=color_from_score(data['score_headheight']['score']/10) if data['hit_yet'] else None)

        # --- TOES CROSS ---
        draw_text(surface, (sx*0.8, sy*0.38), 'Front foot close', BLACK, size=20)
        draw_text(surface, (sx*0.8, sy*0.41), 'to target', BLACK, size=20)
        draw_text(surface, (sx*0.92, sy*0.45), '{}/10'.format(data['score_toescross']['score']),
                  BLACK if data['hit_yet'] else GRAY, size=35, justify='right',
                  shadow_color=color_from_score(data['score_toescross']['score']/10) if data['hit_yet'] else None)

        # # --- DECELERATION ---
        # draw_text(surface, (sx*0.8, sy*0.55), 'Has decelerated', BLACK, size=20)
        # draw_text(surface, (sx*0.92, sy*0.59), '{}/10'.format(data['score_deceleration']['score']),
        #           BLACK if data['hit_yet'] else GRAY, size=35, justify='right',
        #           shadow_color=color_from_score(data['score_deceleration']['score']/10) if data['hit_yet'] else None)
        
        draw_text(surface, (sx*0.8, sy*0.55), 'Arm Wrap', BLACK, size=20)
        draw_text(surface, (sx*0.92, sy*0.59), '{:.0f}/10'.format(data['score_hug']['score']), BLACK if data['hit_yet'] else GRAY, size=35,
                  justify='right',
                  shadow_color=color_from_score(data['score_hug']['score']/10) if data['hit_yet'] else None)

        # --- TEK SCORE ---
        draw_text(surface, (sx*0.8, sy*0.7), 'TEK Score', BLACK, size=25)
        draw_text(surface, (sx*0.98, sy*0.75), '{}'.format(data['score_safety']['score']),
                  BLACK if data['hit_yet'] else GRAY, size=40, justify='right',
                  shadow_color=color_from_score(data['score_safety']['score']/200) if data['hit_yet'] else None)

        # --- SPEED SCORE ---
        draw_text(surface, (sx*0.8, sy*0.85), 'SPEED m/s', BLACK, size=25)
        draw_text(surface, (sx*0.98, sy*0.90), '{:.1f}   '.format(data['score_performance']['raw']), BLACK if data['hit_yet'] else GRAY, size=40,
                  justify='right',
                  shadow_color=color_from_score(data['score_performance']['score']/6) if data['hit_yet'] else None)

 # --- DRAW THE DUMMY ---
        if not data['pose_found']:
            return

        # Convert normalized coordinates to screen-space
        def tos(x, y):
            return cam_rect.left + x * cam_rect.width, cam_rect.top + y * cam_rect.height

        # Dummy box sides
        left, _ = tos(data['dummy_left'], 0)
        right, _ = tos(data['dummy_right'], 0)
        width = right - left

        # Fixed heights based on assumed height of 1.7m
        # Standard body proportions as percentages of total height:
        # - Shoulder height: ~82% of total height
        # - Hip height: ~50% of total height
        # - Knee height: ~28% of total height
        
        # Calculate positions using fixed proportions of dummy height
        dummy_bottom = data['dummy_bottom']
        dummy_zoom_factor_y = data['dummy_zoom_factor_y']

        # virtual_player_height = 1.55 if data['dummy_real_height'] == 1.1 else 1.75
        virtual_player_height = data['average_height'] if 'average_height' in data else 1.7
        dummy_top = dummy_bottom - (virtual_player_height / dummy_zoom_factor_y)
        dummy_height = dummy_bottom - dummy_top

        shoulder_pos = dummy_top + dummy_height * 0.18  # 18% from top
        sternum_pos = dummy_top + dummy_height * 0.33   # 33% from top
        waist_pos = dummy_top + dummy_height * 0.50     # 50% from top (hip level)
        thigh_pos = dummy_top + dummy_height * 0.62     # 62% from top
        knee_pos = dummy_top + dummy_height * 0.72      # 72% from top

        # Convert to screen coordinates
        _, y_shoulder = tos(data['dummy_left'], shoulder_pos)
        _, y_sternum = tos(data['dummy_left'], sternum_pos)
        _, y_waist = tos(data['dummy_left'], waist_pos)
        _, y_thigh = tos(data['dummy_left'], thigh_pos)
        _, y_knee = tos(data['dummy_left'], knee_pos)

        # Interpolate two inner zone lines
        zone_height = y_knee - y_waist
        y1 = y_waist + zone_height / 3
        y2 = y_waist + 2 * zone_height / 3

        zone_height2 = y_waist - y_sternum
        ys1 = y_sternum + zone_height2 / 3
        ys2 = y_sternum + 2 * zone_height2 / 3

        def draw_transparent_rect(y_start, y_end, color, alpha=80):
            y_top = min(y_start, y_end)
            height = abs(y_end - y_start)
            if height < 1:  # avoid invalid size
                return
            s = pygame.Surface((width, height), pygame.SRCALPHA)
            s.fill((*color, alpha))
            surface.blit(s, (left, y_top))


        draw_transparent_rect(y_knee, y2, (255, 0, 0))       # Red
        draw_transparent_rect(y2, y1, (255, 165, 0))         # Amber
        draw_transparent_rect(y1, y_waist, (0, 255, 0))      # Green

        # Same order check for safety
        draw_transparent_rect(y_waist, ys2, (0, 255, 0))        # Red
        draw_transparent_rect(ys2, ys1, (255, 165, 0))          # Amber
        draw_transparent_rect(ys1, y_sternum, (255, 0, 0))      # Green
    

class CompetencyDrawSummary(Competency):
    # Keypad 4: Basic metric summary showing head height, foot position, and hug score 

    def draw(model, data, surface):
        sx, sy = data['sx'], data['sy']
        
        # HEAD HEIGHT SCORE
        draw_text(surface, (sx*0.05, sy*0.28), 'HEAD HEIGHT', BLACK, size=40, justify='left')
        draw_text(surface, (sx*0.05, sy*0.32), '{}/10'.format(data['score_headheight']['score']), BLACK if data['hit_yet'] else GRAY, size=100,
                  justify='left',
                  shadow_color=color_from_score(data['score_headheight']['score']/10) if data['hit_yet'] else None)   

        # FOOT POSITION SCORE
        
        draw_text(surface, (sx*0.4, sy*0.28), 'FOOT CLOSE', BLACK, size=40, justify='left')
        draw_text(surface, (sx*0.4, sy*0.32), '{}/10'.format(data['score_toescross']['score']), BLACK if data['hit_yet'] else GRAY, size=100,
                  justify='left',
                  shadow_color=color_from_score(data['score_toescross']['score']/10) if data['hit_yet'] else None)

        # HUG SCORE
        draw_text(surface, (sx*0.95, sy*0.28), 'Arm Wrap', BLACK, size=40, justify='right')
        draw_text(surface, (sx*0.95, sy*0.32), '{:.0f}/10'.format(data['score_hug']['score']), BLACK if data['hit_yet'] else GRAY, size=100,
                  justify='right',
                  shadow_color=color_from_score(data['score_hug']['score']/10) if data['hit_yet'] else None)
        # draw_text(surface, (sx*0.95, sy*0.66), '{:.0f}'.format(data['score_hug']['raw']), GRAY, size=60, justify='right')

        # SAFETY SCORE
        draw_text(surface, (sx*0.25, sy*0.65), 'TEK SCORE', BLACK, size=60, justify='center')
        draw_text(surface, (sx*0.4, sy*0.70), '{}'.format(data['score_safety']['score']), BLACK if data['hit_yet'] else GRAY, size=220,
                  justify='right',
                  shadow_color=color_from_score(data['score_safety']['score']/100) if data['hit_yet'] else None)
        
        # SPEED SCORE
        draw_text(surface, (sx*0.75, sy*0.65), 'SPEED m/s', BLACK, size=60, justify='center')
        draw_text(surface, (sx*0.92, sy*0.70), '{:.1f}    '.format(data['score_performance']['raw']), BLACK if data['hit_yet'] else GRAY, size=180,
                  justify='right',
                  shadow_color=color_from_score(data['score_performance']['score']/8) if data['hit_yet'] else None)

class CompetencyDrawAll(Competency):
# keypad 5 all metrics and tek score based on all metrics

    def draw(model, data, surface):

        sx,sy = data['sx'], data['sy']


        # HEAD HEIGHT SCORE
        draw_text(surface, (sx*0.05, sy*0.28), 'HEAD HEIGHT', BLACK, size=40, justify='left')
        draw_text(surface, (sx*0.05, sy*0.32), '{}/10'.format(data['score_headheight']['score']), BLACK if data['hit_yet'] else GRAY, size=100,
                  justify='left',
                  shadow_color=color_from_score(data['score_headheight']['score']/10) if data['hit_yet'] else None)        



        # FOOT POSITION SCORE
        draw_text(surface, (sx*0.4, sy*0.28), 'FOOT CLOSE', BLACK, size=40, justify='left')
        draw_text(surface, (sx*0.4, sy*0.32), '{}/10'.format(data['score_toescross']['score']), BLACK if data['hit_yet'] else GRAY, size=100,
                  justify='left',
                  shadow_color=color_from_score(data['score_toescross']['score']/10) if data['hit_yet'] else None)


        # DECELERATION SCORE
        draw_text(surface, (sx*0.95, sy*0.28), 'DECELERATE', BLACK, size=40, justify='right')
        draw_text(surface, (sx*0.95, sy*0.32), '{}/10'.format(data['score_deceleration']['score']), BLACK if data['hit_yet'] else GRAY, size=100,
                  justify='right',
                  shadow_color=color_from_score(data['score_deceleration']['score']/10) if data['hit_yet'] else None)


        # HIPFRONT SCORE
        draw_text(surface, (sx*0.05, sy*0.48), 'HIPFOOT', BLACK, size=40, justify='left')
        draw_text(surface, (sx*0.05, sy*0.52), '{:.0f}/10'.format(data['score_hipfoot']['score']), BLACK if data['hit_yet'] else GRAY, size=100,
                  justify='left',
                  shadow_color=color_from_score(data['score_hipfoot']['score']/10) if data['hit_yet'] else None)
        draw_text(surface, (sx*0.05, sy*0.66), '{:.0f}'.format(data['score_hipfoot']['raw']), GRAY, size=60, justify='left')

        # GAZE SCORE
        draw_text(surface, (sx*0.4, sy*0.48), 'GAZE', BLACK, size=40, justify='left')
        draw_text(surface, (sx*0.4, sy*0.52), '{:.0f}/10'.format(data['score_gaze']['score']), BLACK if data['hit_yet'] else GRAY, size=100,
                  justify='left',
                  shadow_color=color_from_score(data['score_gaze']['score']/10) if data['hit_yet'] else None)
        draw_text(surface, (sx*0.4, sy*0.66), '{:.0f}'.format(data['score_gaze']['raw']), GRAY, size=60, justify='left')

        # HUG SCORE
        draw_text(surface, (sx*0.95, sy*0.48), 'Arm Wrap', BLACK, size=40, justify='right')
        draw_text(surface, (sx*0.95, sy*0.52), '{:.0f}/10'.format(data['score_hug']['score']), BLACK if data['hit_yet'] else GRAY, size=100,
                  justify='right',
                  shadow_color=color_from_score(data['score_hug']['score']/10) if data['hit_yet'] else None)
        draw_text(surface, (sx*0.95, sy*0.66), '{:.0f}'.format(data['score_hug']['raw']), GRAY, size=60, justify='right')



        # SPEED SCORE
        draw_text(surface, (sx*0.75, sy*0.80), 'SPEED m/s', BLACK, size=50, justify='center')
        draw_text(surface, (sx*0.92, sy*0.82), '{:.1f}    '.format(data['score_performance']['raw']), BLACK if data['hit_yet'] else GRAY, size=120,
                  justify='right',
                  shadow_color=color_from_score(data['score_performance']['score']/8) if data['hit_yet'] else None)
        

        # Replace the two TEK scores with just the full TEK score
        draw_text(surface, (sx*0.25, sy*0.80), 'FULL TEK', BLACK, size=50, justify='center')
        draw_text(surface, (sx*0.3, sy*0.82), '{}'.format(data['score_safety_full']['score']), 
                 BLACK if data['hit_yet'] else GRAY, size=120, justify='right',
                 shadow_color=color_from_score(data['score_safety_full']['score']/100) if data['hit_yet'] else None)


class CompetencyDrawBest(Competency):


    def draw(model, data, surface):

        sx,sy = data['sx'], data['sy']

        # Use labels_scores_full to show best/worst from all 6 metrics
        label_max = max(labels_scores_full, key=lambda l: data[l]['score'])
        label_min = min(labels_scores_full, key=lambda l: data[l]['score'])
        
        # LOWER SCORE
        draw_text(surface, (sx*0.05, sy*0.28), data[label_min]['longname'].upper(), BLACK, size=50, justify='left')
        draw_text(surface, (sx*0.05, sy*0.35), '{:.0f}/10'.format(data[label_min]['score']), BLACK if data['hit_yet'] else GRAY, size=140,
                  justify='left',
                  shadow_color=color_from_score(data[label_min]['score']/10) if data['hit_yet'] else None)        

        # higher SCORE
        draw_text(surface, (sx*0.55, sy*0.28), data[label_max]['longname'].upper(), BLACK, size=50, justify='left')
        draw_text(surface, (sx*0.55, sy*0.35), '{:.0f}/10'.format(data[label_max]['score']), BLACK if data['hit_yet'] else GRAY, size=140,
                  justify='left',
                  shadow_color=color_from_score(data[label_max]['score']/10) if data['hit_yet'] else None)

        # SAFETY SCORE
        draw_text(surface, (sx*0.25, sy*0.65), 'TEK SCORE', BLACK, size=60, justify='center')
        draw_text(surface, (sx*0.4, sy*0.70), '{}'.format(data['score_safety']['score']), BLACK if data['hit_yet'] else GRAY, size=220,
                  justify='right',
                  shadow_color=color_from_score(data['score_safety']['score']/100) if data['hit_yet'] else None)
        
        # SPEED SCORE
        draw_text(surface, (sx*0.75, sy*0.65), 'SPEED m/s', BLACK, size=60, justify='center')
        draw_text(surface, (sx*0.92, sy*0.70), '{:.1f}    '.format(data['score_performance']['raw']), BLACK if data['hit_yet'] else GRAY, size=180,
                  justify='right',
                  shadow_color=color_from_score(data['score_performance']['score']/8) if data['hit_yet'] else None)
        
       



class CompetencyDrawRightLeft(Competency):

    def __init__(model, cfg):
        super().__init__(cfg)

        model.reset_lr(sound=False)


        pygame.mixer.init()


        # Load a sound file
        model.sounds = [pygame.mixer.Sound(path) for path in ['sound_right.wav', 'sound_left.wav']]



    def reset_lr(model, sound=True):
        model.option = np.random.random()>0.5 
        model.direction = ['right', 'left'][model.option]

        if sound:
            model.sounds[model.option].play()


    def draw(model, data, surface):

        sx,sy = data['sx'], data['sy']

        draw_text(surface, (sx*0.5, sy*0.5), 
                  'RIGHT' if model.direction=='right' else 'LEFT', 
                  BLACK, size=250, justify='center')






def lerp(x1, x2, y1, y2, x):
    return y1 + (y2 - y1) * (x - x1) / (x2 - x1)

