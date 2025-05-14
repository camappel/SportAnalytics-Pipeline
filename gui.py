import pygame
import sys

# Initialize Pygame
pygame.init()

# Constants
SCREEN_WIDTH, SCREEN_HEIGHT = 800, 600
WHITE = (255, 255, 255)
GRAY = (50,50,50)
LIGHT_GRAY = (100, 100, 100)
COLOR_THEME_MAIN = (0, 149, 214)
COLOUR_THEME_ALT = (0, 73, 104)
BLUE = (0, 0, 255)
DARK_BLUE = (0, 0, 200)



# Font setup
pygame.font.init()
font = pygame.font.SysFont(None, 36)
font_small = pygame.font.SysFont(None, 12)

# Screen Base Class
class Screen:
    def __init__(self):
        self.next_screen = self

    def handle_events(self, events):
        for event in events:
            
            self.handle_event(event)

    
    def handle_event(self, event):
        for widget in self.widgets:
            widget.handle_event(event)

    def update(self):
        pass
        

    def draw(self, screen, dt):
        screen.fill(WHITE)
        for widget in self.widgets:
            widget.draw(screen)

    def change_screen(self, new_screen):
        print('Empty change_screen call: is this screen part of a collection?')
        pass

    def quit(self):
        pygame.quit()
        sys.exit()


class ScreenCollection(Screen):

    def __init__(self, *screens):

        self.screens = screens
        self.current_screen = self.screens[0]

        # Let children talk to parent to change screen
        for screen in self.screens:
            screen.change_screen = self.change_screen

    
    def handle_event(self, event):
        self.current_screen.handle_event(event)
    
    def update(self):        
        self.current_screen.update()
    
    def draw(self, surface, dt):
        self.current_screen.draw(surface, dt)

    def change_screen(self, new_screen):
        self.current_screen = new_screen

    


class Widget:
    def __init__(self):
        pass

    def change_rect(self, new_rect):
        self.rect = new_rect

    def draw(self, surface):
        pass

    def update(self, dt, events):
        for event in events:
            self.handle_event(event)

    def handle_event(self, event):
        pass

    def relativify(self):
        return WidgetRelatively(self, self.rect.left/100, self.rect.top/100, self.rect.right/100, self.rect.bottom/100)


class WidgetRelatively(Widget):

    def __init__(self, widget, left=None, top=None, right=None, bottom=None, width=None, height=None, centre_x=None, centre_y=None):
        # Note all the positions in this object should be in [0,1] which is relative to screen size
        self.widget = widget
        self.change_position(left, top, right, bottom, width, height, centre_x, centre_y)
        self.last_surface_size = (0,0)

    def change_rect(self, new_rect):
        self.change_position(*new_rect)

    def change_position(self, left=None, top=None, right=None, bottom=None, width=None, height=None, centre_x=None, centre_y=None):

        self.left = left; self.right = right; self.top = top; self.bottom=bottom
        self.width = width; self.height = height
        self.centre_x = centre_x; self.centre_y = centre_y

        # Multiple modes corresponding to different sets of parameters specified
        # There is a priority to modes (boundary, then rect, then centre), when overspecified the lower priority values are overwritten
        if left is None:
            if centre_x is None:
                print('Insufficient position data given')
                self.mode = 'insufficient'
                return
                    
            else:
                if width is None: self.mode = 'centre_stiff'

                else: self.mode = 'centre_stretch'
        
        else:
            if right is None:
                if width is None: self.mode = 'corner_stiff'
                
                else: self.mode = 'rect'
            
            else: self.mode = 'boundary'

        # Specify other parameters by those given
        if self.mode == 'boundary':
            self.width    = right - left
            self.height   = bottom - top
            self.centre_x = (right + left)/2
            self.centre_y = (top + bottom)/2
        elif self.mode == 'rect':
            self.right    = left + width
            self.bottom   = top + height
            self.centre_x = left + width/2
            self.centre_y = top + height/2
        elif self.mode == 'corner_stiff':
            pass
        elif self.mode == 'centre_stiff':
            pass
        elif self.mode == 'centre_stretch':
            self.left   = centre_x - width/2
            self.right  = centre_x + width/2
            self.top    = centre_y - height/2
            self.bottom = centre_y + height/2
    
    def update_widget_position(self, new_size):
        sx, sy = new_size
        widget = self.widget
        if self.mode in ('boundary', 'rect', 'centre_stretch'):
            widget.rect = pygame.Rect(self.left*sx, self.top*sy, self.width*sx, self.height*sy)
        elif self.mode in ('corner_stiff', 'centre_stiff'):
            widget.rect = pygame.Rect(self.left*sx, self.top*sy, widget.rect.width*sx, widget.rect.height*sy)


    def draw(self, surface):
        if surface.get_size() != self.last_surface_size:
            self.update_widget_position(surface.get_size())
        
        self.last_surface_size = surface.get_size()

        self.widget.draw(surface)
    
    def handle_event(self, event):
        self.widget.handle_event(event)


class Panel(Widget):
    def __init__(self, x, y, width, height, color):
        self.rect = pygame.Rect(x, y, width, height)
        self.color = color

    def draw(self, screen):
        pygame.draw.rect(screen, self.color, self.rect)




def draw_text(surface, topleft, text, color, size=16, justify='left', scale_by_screen_size=True, shadow_color=None):

    if scale_by_screen_size:
        sx, sy = surface.get_size()
        #print(sx, sy, min(16*sy, 9*sx) * .0001)
        size *= min(16*sy, 9*sx) * .00015
        size = int(size)
    

    font = pygame.font.SysFont(None, size)
    lines = text.split('\n')
    x, y = topleft
    blit_x, blit_y = x, y # If center is false, these will be the same as x and y initially, and blit_y moves down with new lines.

    if justify == 'center':
        # Calculate total height of the block and adjust the y position to vertically center the text
        total_height = len(lines) * font.render(lines[0], True, color).get_height()
        blit_y -= total_height // 2

    for line in lines:
        text_surface = font.render(line, True, color)
        
        text_width = text_surface.get_width()
        blit_x = x - text_width//2 if justify=='center' else x
        blit_x = x - text_width  if justify=='right' else blit_x
        

        if shadow_color:
            shadow_offset = size//35+1
            text_surface_shadow = font.render(line, True, shadow_color)
            surface.blit(text_surface_shadow, (blit_x+shadow_offset, blit_y+shadow_offset))


        surface.blit(text_surface, (blit_x, blit_y))
        blit_y += size  # Move y down for the next line


class KeyCatcher(Widget):
    def __init__(self, key, action):
        self.key = key
        self.action = action

    def handle_event(self, event):
        if event.type == pygame.KEYDOWN:

            if event.key == self.key:
                self.action()


class KeysCatcher(Widget):
    def __init__(self, keys, action):

        self.keys = keys
        self.action = action

    def handle_event(self, event):
        if event.type == pygame.KEYDOWN:

            if event.key in self.keys:
                self.action()


class MousePressCatcher(Widget):
    def __init__(self, button, action):
        self.button = button
        self.action = action

    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == self.button:
                self.action()


class Text(Widget):
    def __init__(self, x, y, width, height, text, color=WHITE, size=24, justify='left', scale_by_screen_size=True):
        self.font = pygame.font.SysFont(None, size)
        self.rect = pygame.Rect(x, y, width, height)

        self.text = text
        self.color = color
        self.size = size
        self.justify = justify
        self.scale_by_screen_size = scale_by_screen_size

    def change_text(self, new_text):
        self.text = new_text

    def draw(self, screen):
        
        draw_text(screen, 
                  self.rect.center if self.justify=='center' else self.rect.topleft, 
                  self.text, self.color, size=self.size, justify=self.justify, scale_by_screen_size=self.scale_by_screen_size)


    def handle_event(self, event):
        pass


class ScreenFill(Widget):

    def __init__(self, color):
        self.color = color

    def draw(self, screen):
        screen.fill(self.color)



class ImageDisplay(Widget):
    def __init__(self, x, y, width, height, image_path):
        self.rect = pygame.Rect(x, y, width, height)
        self.image = None
        self.original_image = None  # Store the original image to avoid rescaling the already scaled image
        self.previous_size = self.rect.size  # Store the initial size of the rect
        
        # Try to load the image from the provided path
        try:
            self.original_image = pygame.image.load(image_path)
            self.image = pygame.transform.scale(self.original_image, self.rect.size)
            print(f'~~~Success: Image loaded of size {self.original_image.get_size()}')
        except pygame.error as e:
            print(f"~~~Warning: Failed to load image at path: {image_path} \n Error given: {e}")
        except FileNotFoundError as e:
            print(f"~~~Warning: Failed to find image at path: {image_path}\n Error given: {e}")
        

        
    def draw(self, screen):
        # Check if the size of the rect has changed
        if self.rect.size != self.previous_size:
            # Rescale the image to match the new size
            if self.original_image:
                self.image = pygame.transform.scale(self.original_image, self.rect.size)
            # Update the stored size to the current size
            self.previous_size = self.rect.size
        
        
        # Draw the image if it was successfully loaded
        if self.image:
            
            
            screen.blit(self.image, self.rect.topleft)




# Button Class
class Button(Widget):
    def __init__(self, x, y, width, height, text, action, color=COLOR_THEME_MAIN, hover_color=COLOUR_THEME_ALT, click_color=COLOR_THEME_MAIN,  text_color = WHITE, text_size=35, scale_text_by_screen_size=True):
        self.rect = pygame.Rect(x, y, width, height)
        self.text = text
        self.text_size = text_size
        self.text_color = text_color
        self.scale_text_by_screen_size = scale_text_by_screen_size
        self.color = color
        self.hover_color = hover_color
        self.click_color = click_color
        self.current_color = color
        self.clicked = False
        self.action = action

        #self.text_object = Text(x, y, width, height, text, color=DARK_BLUE, size=text_size, scale_by_screen_size=scale_text_by_screen_size)

    def draw(self, screen):
        

        # Draw a shadow using surface blitting
        rect_shadow = pygame.Rect(self.rect.left + 3, self.rect.top + 5, self.rect.width, self.rect.height)
        transparent_surface = pygame.Surface(rect_shadow.size, pygame.SRCALPHA)
        transparent_surface.fill((0, 0, 0, 100))  # self.current_color[:3] extracts RGB, alpha is the transparency level
        screen.blit(transparent_surface, rect_shadow.topleft)

        pygame.draw.rect(screen, self.current_color, self.rect)


        #draw_text(screen, 
        #          self.rect.center if self.justify=='center' else self.rect.topleft, 
        #          self.text, self.color, size=self.text_size, justify='center', scale_by_screen_size=self.scale_text_by_screen_size)

           
        draw_text(screen, self.rect.center, self.text, self.text_color, size=self.text_size, justify='center', scale_by_screen_size=self.scale_text_by_screen_size)

    def handle_event(self, event):
        if event.type == pygame.MOUSEMOTION:
            if self.rect.collidepoint(event.pos):
                if not self.clicked:
                    self.current_color = self.hover_color
            else:
                self.current_color = self.color
        elif event.type == pygame.MOUSEBUTTONDOWN:
            
            if event.button == 1 and self.rect.collidepoint(event.pos):
                self.current_color = self.click_color
                self.clicked = True
        elif event.type == pygame.MOUSEBUTTONUP:
            if event.button == 1 and self.clicked:
                if self.rect.collidepoint(event.pos):
                    self.action()
                self.current_color = self.color
                self.clicked = False

# Slider Class
class Slider(Widget):
    def __init__(self, x, y, width, height, handle_radius, start_pos = 0.5, on_move = lambda x: None):
        self.rect = pygame.Rect(x, y, width, height)
        self.handle_radius = handle_radius
        self.handle_pos = start_pos
        self.dragging = False

        self.on_move = on_move

    def draw(self, screen):
        pygame.draw.rect(screen, GRAY, self.rect)
        pygame.draw.rect(screen, LIGHT_GRAY, (self.rect.left + 2, self.rect.top + 2, self.rect.width-2, self.rect.height-2))
        handle_color = COLOUR_THEME_ALT if self.dragging else COLOR_THEME_MAIN
        pygame.draw.circle(screen, handle_color, (self.local_to_screen(self.handle_pos), self.rect.top + self.rect.height // 2), self.handle_radius)

    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:  # Left mouse button
                mx, my = event.pos
                if (self.rect.left - self.handle_radius <= mx <= self.rect.left + self.rect.width + self.handle_radius and
                        self.rect.top - self.handle_radius <= my <= self.rect.top + self.rect.height + self.handle_radius):
                    self.dragging = True
                    
                    mx, my = event.pos
                    
                    self.handle_pos = max(0, min(1, self.screen_to_local(mx)))
                    self.on_move(self.handle_pos)
        elif event.type == pygame.MOUSEBUTTONUP:
            if event.button == 1:  # Left mouse button
                self.dragging = False
        elif event.type == pygame.MOUSEMOTION:
            if self.dragging:
                mx, my = event.pos
                self.handle_pos = max(0, min(1, self.screen_to_local(mx)))
                self.on_move(self.handle_pos)

    def get_value(self):
        return self.handle_pos
    
    def screen_to_local(self, x):
        return (x - self.rect.left) / self.rect.width
    
    def local_to_screen(self, pos):
        return self.rect.left + pos*self.rect.width


class ModelControllerWidget(Widget):
    def __init__(self, model, rect, data_source=None):
        self.model = model

        self.change_rect(rect)
        if not data_source is None:
            self.data = data_source
        else:
            self.data = {
                'healthy': True, 
                'rect_target': self.rect,
                'dummy_hip_height': 0.2, 
                'hit_yet': False,
                'dis_to_dummy': -1,
                'pose_furthest': -1,
                'list_poses_furthest_previous': [],
                'time_close': 0,
                'time_close_decelerating': 0,
                'vel': 0,
                'acc': 0}

    def change_rect(self, new_rect):
        self.rect = new_rect

    def draw(self, surface , dt=0.03):

        sx,sy = self.rect.width, self.rect.height
        self.data['sx'] = sx
        self.data['sy'] = sy
        self.data['dt'] = dt
        self.data['t']  = pygame.time.get_ticks()
        self.data['multiplier_line_width'] = min(16*sy, 9*sx) * .0003
        self.data['reset_threshhold'] = 1
        
        self.data = self.model.process(self.data)

        self.model.draw(self.data, surface)
        


    def handle_events(self, dt, events):
        self.model.update(dt, events)

    def set_reset_threshhold(self, thresh):
        self.data['reset_threshhold'] = thresh
    


def loop(screen):

    # Initialize screen
    surface = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.RESIZABLE)
    pygame.display.set_caption('TakleTek')

    clock = pygame.time.Clock()
    running = True
    while running:
        uncaught_events = []
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                running = False
                return
            else:
                uncaught_events.append(event)
        
        screen.handle_events(uncaught_events)

        screen.update()

        # Draw current screen
        
        dt = clock.tick(20)
        screen.draw(surface, dt)
        pygame.display.flip()

        # Cap the frame rate



def scale_rect(rect, sx, sy):
    return pygame.Rect(rect.left*sx, rect.top*sy, rect.width*sx, rect.height*sy)


if __name__=='__main__':

    # Instantiate Screens
    main_menu = MainMenu()
    settings_screen = SettingsScreen()
    another_screen = AnotherScreen()

    screen_collection = ScreenCollection(main_menu,
                                        settings_screen,
                                        another_screen)

    # Main application loop
    
