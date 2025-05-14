import pygame
import numpy as np
import cv2

from gui import draw_text

class Model:
    def __init__(model, cfg):
        model.cfg = cfg

    def update(model, dt, events):
        return True
    
    def process(model, data):
        return data
    
    def draw(model, data, surface):
        pass

    def close(model):
        pass


class ModelController(Model):

    def __init__(control, cfg, components, to_draw=lambda i: True):
        control.cfg = cfg
        control.components = components
        control.to_draw = to_draw
        
    def process(control, data: dict):
        
        for comp in control.components:
            data = comp.process(data)
            if not data['healthy']:
                control.close()
                quit()
        return data
    
    def draw(control, data, surface):
        for i, component in enumerate(control.components):
            if control.to_draw(i):
                component.draw(data, surface)
    
    def update(control, dt, events):

        for comp in control.components:
            data = comp.update(dt, events)
        return True

    def close(control):

        for comp in control.components:
            comp.close()




class TestModel(Model):
    def process(model, data):
        data["image"][50:100,50:100,:] = 255
        return data

    def close(model):
        pass
                

class ScreenFillModel(Model):

    def __init__(model, cfg, color):
        model.color = color

    def draw(model, data, surface):
        surface.fill(model.color)




        
# WebcamModel class commented out as it depends on missing webcam module
"""
class WebcamModel(webcam.Webcam):
    def process(webcam, data):
        healthy,data["image"] = webcam.get()
        data["healthy"] &= healthy 
        data['webcam_rect'] = data['rect_target']
        return data
    
    def draw(webcam, data, surface):
        sx    = data['sx']
        sy    = data['sy']
        rect  = data['rect_target']
        frame = data["image"]

        surfarray = pygame.surfarray.pixels3d(surface)

        frame_resized, newx, newy = resize_image_cv2(frame, sx, sy)



        surfarray[rect.left:rect.left + newx,
                  rect.top:rect.top + newy] = frame_resized
        del surfarray

        data['webcam_rect'] = pygame.Rect(rect.left, rect.top, newx, newy)   
    
    def update(webcam, dt, events):

        uncaught_events = []

        for event in events:

            if event.type==pygame.KEYDOWN:
                if event.key == pygame.K_LEFTBRACKET:
                    webcam.cycle_camera()
                elif event.key == pygame.K_RIGHTBRACKET:
                    webcam.cycle_camera(back=True)



            uncaught_events.append(event)
        
        return True
"""




class PersonView(Model):

    def draw(webcam, data, surface):
        sx = data['sx']
        sy = data['sy']
        frame = data["image"]

        rect_person = data['rect_person']
        left   = int(rect_person[0] * sx)
        top    = int(rect_person[1] * sy)
        width  = int(rect_person[2] * sx)
        height = int(rect_person[3] * sy)
        print(rect_person)
        #print(rect_target)

        frame_cropped = frame[ left:left + width,  top:top + height,]

        #frame_resized, newx, newy = resize_image_cv2(frame_cropped, sx, sy)


        surfarray = pygame.surfarray.pixels3d(surface)

        surfarray[:frame_cropped.shape[0],:frame_cropped.shape[1]] = frame_cropped
        newx = frame_cropped.shape[0]
        newy = frame_cropped.shape[1]

        #surfarray[:sx//2, :sy//3] = 0

        #surfarray[rect_target.left:rect_target.left + newx,
        #        rect_target.top:rect_target.top + newy] = frame_resized

        del surfarray

        #data['webcam_rect'] = pygame.Rect(rect_target.left, rect_target.top, newx, newy)
        #data['webcam_rect'] = 




class OpenVideoFileModel(Model):

    def __init__(model, cfg):
        model.cfg = cfg
        model.cap = cv2.VideoCapture(cfg['PATH_VIDEO_OPEN'])
        if not model.cap.isOpened():
            print("Error: Could not open video at path", cfg['PATH_VIDEO_OPEN'])

        model.fps = model.cap.get(cv2.CAP_PROP_FPS)
        print('Overwrite config fps {} with {}'.format(cfg['fps'], model.fps))
        cfg['fps'] = model.fps
        # Initialize _background attribute to avoid AttributeError in __del__
        model._background = None

    def get(model):
        ret, frame = model.cap.read()
        if ret:
            reshaped_frame = np.transpose(frame, (1,0,2))
            reshaped_frame = cv2.cvtColor(reshaped_frame, cv2.COLOR_RGB2BGR)
            return ret, reshaped_frame
        else:
            return ret, frame

    def process(model, data):
        healthy, data["image"] = model.get()
        data["healthy"] &= healthy 
        data['webcam_rect'] = data['rect_target']
        return data

    def draw(model, data, surface):
        sx    = data['sx']
        sy    = data['sy']
        rect  = data['rect_target']
        frame = data["image"]

        surfarray = pygame.surfarray.pixels3d(surface)

        frame_resized, newx, newy = resize_image_cv2(frame, sx, sy)

        surfarray[rect.left:rect.left + newx,
                  rect.top:rect.top + newy] = frame_resized
        del surfarray

        data['webcam_rect'] = pygame.Rect(rect.left, rect.top, newx, newy)   

    def update(model, dt, events):
        return True
    
    def close(model):
        model.cap.release()


class SaveVideoFileModel(Model):

    def __init__(self, cfg):
        fourcc = cv2.VideoWriter_fourcc(*'H264')  # You can use other codecs like 'MJPG', 'X264', etc.
        self.out = cv2.VideoWriter(cfg['PATH_VIDEO_SAVE'], 
                              fourcc, 
                              cfg['fps'], 
                              cfg['SIZE_VIDEO_SAVE'])  
        if not self.out.isOpened:
            print('Destination is not open at ', cfg['PATH_VIDEO_SAVE'])
        
    def draw(self, data, surface):
        surfarray = pygame.surfarray.pixels3d(surface)
        

        img_array = pygame.surfarray.array3d(surface)
        # Transpose the array to match the shape expected by OpenCV (height, width, channels)
        img_array = np.transpose(surfarray, (1, 0, 2))
        # Convert RGB to BGR
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

        self.out.write(img_array)
        del surfarray

    def close(self):
        print('Closing save file')
        self.out.release()



class DummyModel(Model):
    def __init__(dummy, cfg, components):
        dummy.cfg = cfg
        dummy.components = components

        dummy.left = cfg['dummy_left_default']
        dummy.right = cfg['dummy_right_default']
        dummy.top = cfg['dummy_top_default']
        dummy.bottom = cfg['dummy_bottom_default']
        dummy.real_height = cfg['dummy_real_height']
        

    def process(dummy, data):
        data['dummy_left'] = dummy.left
        data['dummy_right'] = dummy.right
        data['dummy_top'] = dummy.top
        data['dummy_bottom'] = dummy.bottom
        data['dummy_real_height'] = dummy.real_height
        data['dummy_zoom_factor_y'] = data['dummy_real_height']/(data['dummy_bottom'] - data['dummy_top'])
        
        

        return data

    def change_left(dummy, new_left):
        dummy.left = new_left

    def change_right(dummy, new_right):
        dummy.right = new_right

    def change_top(dummy, new_top):
        dummy.top = new_top

    def change_bottom(dummy, new_bottom):
        dummy.bottom = new_bottom
    
    def change_real_height(dummy, new_height):
        dummy.real_height = new_height*3

    def change_height_solid(dummy,new_height):
        dummy.real_height = new_height

    def draw(dummy, data, surface):
        dest = data['webcam_rect']
        
        # Define line widths
        left_line_width = max(4, int(data['multiplier_line_width']) + 2)  # Slightly thicker left side
        other_line_width = max(1, int(data['multiplier_line_width']))  # Standard width for other sides

        def tos(x, y):  # to screen
            return dest.left + x * dest.width, dest.top + y * dest.height
        
        left, top = tos(dummy.left, dummy.top)
        right, bottom = tos(dummy.right, dummy.bottom)
        
        # Use a more professional shade of purple
        professional_purple = (199,36,177)  # A softer, more professional purple

        # Draw each side of the rectangle separately
        pygame.draw.line(surface, professional_purple, (left, top), (left, bottom), left_line_width)  # Left side
        pygame.draw.line(surface, professional_purple, (left, top), (right, top), other_line_width)  # Top side
        pygame.draw.line(surface, professional_purple, (right, top), (right, bottom), other_line_width)  # Right side
        pygame.draw.line(surface, professional_purple, (left, bottom), (right, bottom), other_line_width)  # Bottom side

        # Annotate the left side with "Hit Side Left" rotated 90 degrees
        vertical_text = "Hit Side Right"
        font_size = 30  # Smaller font size
        font = pygame.font.SysFont(None, font_size)
        text_color = professional_purple  # Use the same professional purple color

        # Render the text and rotate it
        text_surface = font.render(vertical_text, True, text_color)
        rotated_text_surface = pygame.transform.rotate(text_surface, 90)

        # Calculate the position to place the rotated text inside the rectangle
        text_x = right + 5  # Slightly inside the left side
        text_y = (top + bottom) / 2 - rotated_text_surface.get_height() / 2  # Centered vertically

        # Blit the rotated text onto the surface
        surface.blit(rotated_text_surface, (text_x, text_y))

        draw_text(surface, (right + 4, top + 4), '{:.2f}m'.format(dummy.real_height), professional_purple)
    


    def update(model, data):
        pass









import numpy as np 
import cv2
def resize_image_cv2(image: np.ndarray, dest_width: int, dest_height: int) -> np.ndarray:
    # Get the current size of the image
    orig_height, orig_width = image.shape[:2]
    
    # Calculate the aspect ratio
    aspect_ratio = orig_width / orig_height
    
    # Calculate the new size while maintaining the aspect ratio
    '''
    if orig_width > orig_height:
        new_width = dest_width
        new_height = int(dest_width / aspect_ratio)
    else:
        new_height = dest_height
        new_width = int(dest_height * aspect_ratio)
        '''

    new_width = min(dest_width, int(dest_height / aspect_ratio))
    new_height = min(dest_height, int(dest_width * aspect_ratio))
    
    # Make sure the new size fits within the destination dimensions
    '''
    if new_width > dest_width:
        new_width = dest_width
        new_height = int(dest_width / aspect_ratio)
    if new_height > dest_height:
        new_height = dest_height
        new_width = int(dest_height * aspect_ratio)
        '''
    
    # Resize the image
    resized_image = cv2.resize(image, (new_height, new_width), interpolation=cv2.INTER_AREA)
    
    return resized_image, new_width, new_height