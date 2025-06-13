from kivy.core.window import Window
from kivy.uix.widget import Widget
from kivy.uix.image import Image
from kivy.graphics import Color, Rectangle, StencilPop, StencilUse, StencilUnUse, StencilPush
from kivy.graphics import PushMatrix, PopMatrix, Translate, Scale

from abc import abstractmethod

class ImageAnnotator(Widget):

    def __init__(self, zoom_min=0.5, zoom_max=1.0):
        super().__init__()
        self.zoom_min, self.zoom_max, self.zoom = zoom_min, zoom_max, 1.0
        self.bind(pos=lambda *args: self.update_transforms(), size=lambda *args: self.update_transforms())
        Window.bind(mouse_pos=self.mouse_pos)

    def mouse_pos(self, window, pos):
        if not self.collide_point(*pos):
            return False
        x, y = self.window2image(*pos)
        self.on_cursor_moved((x, y))
        
    def on_touch_down(self, touch):
        if not self.collide_point(touch.x, touch.y):
            return False
        elif touch.is_mouse_scrolling:
            scale = 0.1 if touch.button == 'scrolldown' else -0.1
            self.zoom = max(self.zoom_min, min(self.zoom + scale, self.zoom_max))
            self.update_transforms()
            x, y = self.window2image(*touch.pos)
            self.on_cursor_moved((x, y))
        elif touch.button in ["left", "right", "middle"]:
            x, y = self.window2image(*touch.pos)
            self.on_click((x, y), touch.button)
        
        return True
   
    @abstractmethod
    def on_cursor_moved(self, position):
        pass
    
    @abstractmethod
    def on_click(self, position, button):
        pass
    
    @abstractmethod
    def on_draw(self):
        pass
    
    @abstractmethod
    def new_image(self):
        pass
   
    def set_image(self, image_file):
        self.texture = Image(source=image_file).texture
        # self.zoom = 1.0
        self.update_transforms()
        self.new_image()

    def draw(self):
        x, y, w, h  = self.rect
        pw, ph = self.texture.size
        self.canvas.clear()
        with self.canvas:
            StencilPush()
            Rectangle(pos=self.pos, size=self.size)
            StencilUse()
            Color(0.2, 0.2, 0.2, 1)
            Rectangle(pos=self.pos, size=self.size)
            Color(1, 1, 1, 1)
            Rectangle(pos=(x, y), size=(w, h), texture=self.texture)
            
            PushMatrix()
            Translate(x, y, 0.0)
            Scale(w / pw, h / ph, 0.0)
            Translate(0, ph, 0.0)
            Scale(1.0, -1.0, 0.0)
            self.on_draw()
            PopMatrix()
            
            StencilUnUse()
            Rectangle(pos=self.pos, size=self.size)
            StencilPop()
            
    def update_transforms(self):
        self.rect = self.calculate_rect()
        self.draw()
    
    def calculate_rect(self):
        
        w, h = self.texture.size
        x2, y2 = self.pos
        w2, h2 = self.size
        
        aspect_ratio1 = w / h
        aspect_ratio2 = w2 / h2
        
        if aspect_ratio1 > aspect_ratio2:
            scaling_factor = w2 / w
        else:
            scaling_factor = h2 / h
            
        new_w = w * scaling_factor * self.zoom
        new_h = h * scaling_factor * self.zoom
        
        new_x = x2 + (w2 - new_w) / 2
        new_y = y2 + (h2 - new_h) / 2
        
        return new_x, new_y, new_w, new_h

    def window2image(self, x, y):
        rx, ry, rw, rh = self.rect
        pw, ph = self.texture.size
        x = pw * (x - rx) / rw
        y = ph * (1 - (y - ry) / rh)
        return x, y
    