import os
import json
os.environ['KIVY_NO_ARGS'] = '1'

from kivy.config import Config
Config.set('input', 'mouse', 'mouse,multitouch_on_demand')
Config.set('graphics', 'fullscreen', 'auto')

from kivy.core.window import Window
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.label import Label
from kivy.uix.floatlayout import FloatLayout
from kivy.graphics import Color, Ellipse, Line
from kivy.uix.button import Button

from src import ImageAnnotator

from glob import glob
import json
import argparse
import numpy as np

from PIL import Image

ESCAPE_KEYCODE = 41
BACKSPACE_KEYCODE = 42
RIGHT_KEYCODE = 79
LEFT_KEYCODE = 80
UP_KEYCODE = 81
DOWN_KEYCODE = 82
S_KEYCODE = 22

MOUSE_BUTTON_MAP = {
    "left": "visible",
    "right": "occluded",
}

OPPOSITE_TAG = {
    "visible" : "occluded",
    "occluded" : "visible",
}

TAG_COLOR_VALUE_MAP = {
    "visible": 1.0,
    "occluded": 0.3,
}

def position_on_line(p, a, b):
    p, a, b = tuple(map(np.array, (p, a, b)))
    ap, ab = p - a, b - a
    abl = np.linalg.norm(ab)
    abu = ab / abl
    return tuple(a + abu * np.clip(np.dot(ap, abu), 0, abl))

class Skeleton():
    def __init__(self, position, tag, hue, node_radius) -> None:
        self.color_map = {k: (hue, 0.5, TAG_COLOR_VALUE_MAP[k]) for k in TAG_COLOR_VALUE_MAP}
        self.node_radius = 0.5 * node_radius
        self.node_size = (node_radius,  node_radius)
        self.nodes = [position, position]
        self.tags = [tag, tag]
        self.edges = [(0, 1), (1, 2), (1, 3)]
        self.transitions = [[], [], []]
            
    @property
    def current_node(self):
        return len(self.nodes)-1
    
    @property
    def base_node(self):
        index = [e[1] for e in self.edges].index(self.current_node)
        return self.edges[index][0]
    
    @property
    def is_interpolating(self):
        return self.tags[self.base_node] != self.tags[self.current_node]
        
    @property
    def must_stop(self):
        return len(self.nodes) == 5
    
    @property
    def can_stop(self):
        return self.current_node > 1 and not self.is_interpolating
    
    def add_point(self, position, tag):
        if not self.is_interpolating:
            self.tags[self.current_node] = tag
            if not self.is_interpolating:
                self.nodes[self.current_node] = position
                self.nodes.append(self.nodes[self.current_node])
                if len(self.nodes) < 5:
                    self.tags.append(self.tags[self.base_node])
            else:
                edge_index = self.edges.index((self.base_node, self.current_node))
                self.transitions[edge_index].append(position)
        else:
            self.nodes.append(position)
            if len(self.nodes) < 5:
                self.tags.append(self.tags[self.base_node])
        
    def finish(self):
        self.nodes = self.nodes[:-1]
        self.tags = self.tags[:len(self.nodes)]
        while len(self.nodes) < 4:
            self.nodes.append(None)
            self.tags.append("missing")
        
    def update_cursor_position(self, position):
        if self.is_interpolating:
            a = self.nodes[self.base_node]
            b = self.nodes[self.current_node]
            edge_index = self.edges.index((self.base_node, self.current_node))
            self.transitions[edge_index][0] = position_on_line(position, a, b)
        else:
            self.nodes[self.current_node] = position
        
    def try_add_transition(self, position):
        for i, ((start, end), transitions) in enumerate(zip(self.edges, self.transitions)):
            if end >= len(self.nodes) or self.tags[end] == "missing":
                return False
            point = position_on_line(position, self.nodes[start], self.nodes[end])
            if np.linalg.norm(np.array(position) - np.array(point)) < 2 * self.node_radius:
                start_pos = np.array(self.nodes[start])
                if np.linalg.norm(point - start_pos) < 2 * self.node_radius:
                    self.tags[start] = OPPOSITE_TAG[self.tags[start]]
                    self._clean_tags()
                    return True
                if np.linalg.norm(point - np.array(self.nodes[end])) < 2 * self.node_radius:
                    self.tags[end] = OPPOSITE_TAG[self.tags[end]]
                    self._clean_tags()
                    return True
                for j, t in enumerate(map(np.array, transitions)):
                    if np.linalg.norm(point - t) < 2 * self.node_radius:
                        return True
                for j, t in enumerate(map(np.array, transitions + [self.nodes[end]])):
                    if np.linalg.norm(t - start_pos) > np.linalg.norm(point - start_pos):
                        self.transitions[i].insert(j, list(point))
                        self._clean_tags()
                        return True
        return False
        
    def _clean_tags(self):
        for (start, end), transitions in zip(self.edges, self.transitions):
            if end >= len(self.nodes) or self.tags[end] == "missing":
                break
            else:
                self.tags[end] = OPPOSITE_TAG[self.tags[start]] if len(transitions) % 2 == 1 else self.tags[start]

    def draw(self):
        for (start, end), transitions in zip(self.edges, self.transitions):
            if end >= len(self.nodes) or self.tags[end] == "missing":
                break
            curent_tag = self.tags[start]
            tags = list(self.color_map.keys())
            start_node, end_node = self.nodes[start], self.nodes[end]
            nodes = [start_node] + transitions + [end_node]
            for start, end in zip(nodes[:-1], nodes[1:]):
                Color(*self.color_map[curent_tag], mode='hsv')
                Line(points=[start, end], width=0.5 * self.node_radius)
                curent_tag = tags[(tags.index(curent_tag) + 1)%2]
        for node, tag in zip(self.nodes, self.tags):
            if tag != "missing":
                Color(*self.color_map[tag], mode='hsv')
                Ellipse(pos=(node[0] - self.node_radius, node[1] - self.node_radius), size=self.node_size)

    def set_data(self, data):
        self.nodes = data['nodes']
        self.tags = data['tags']
        self.edges = data['edges']
        
        # VERSION UPGRADE HANDLING
        transition_sets = data['transitions']
        self.transitions = []
        for transitions in transition_sets:
            if transitions == None or len(transitions) == 0:
                self.transitions.append([])
            elif type(transitions[0]) is not list:
                self.transitions.append([transitions])
            else:
                self.transitions.append(transitions)
    
    def get_data(self):
        return {'nodes': self.nodes, 'tags': self.tags, 'edges': self.edges, 'transitions': self.transitions}
    
class SkeletonAnnotator(ImageAnnotator):
    def __init__(self, allow_editing):
        super().__init__()
        self.allow_editing = allow_editing
        self.skeletons = []
        self.current_skeleton = None
        self.node_radius = 6.0
    
    @property
    def is_busy(self):
        return self.current_skeleton != None
    
    def on_cursor_moved(self, position):
        if self.current_skeleton != None:
            self.current_skeleton.update_cursor_position(position)
            self.draw()
    
    def on_click(self, position, button):
        if not self.allow_editing:
            return
        elif self.current_skeleton == None:
            for skeleton in self.skeletons:
                if skeleton.try_add_transition(position):
                    self.draw()
                    return
            if button in MOUSE_BUTTON_MAP:
                hue = (1 + len(self.skeletons)) * 2 / 7.0
                self.current_skeleton = Skeleton(position, MOUSE_BUTTON_MAP[button], hue, self.node_radius)
                self.draw()
        else:
            if button in MOUSE_BUTTON_MAP:
                self.current_skeleton.add_point(position, MOUSE_BUTTON_MAP[button])
                self.draw()
            if (button == "middle" and self.current_skeleton.can_stop) or self.current_skeleton.must_stop:
                self.current_skeleton.finish()
                self.skeletons.append(self.current_skeleton)
                self.current_skeleton = None
                self.draw()
    
    def on_draw(self):
        for skeleton in self.skeletons:
            skeleton.draw()
        if self.current_skeleton != None:
            self.current_skeleton.draw()
    
    def delete_last(self):
        if not self.allow_editing:
            return
        elif self.current_skeleton != None:
            self.current_skeleton = None
        elif len(self.skeletons) > 0:
            self.skeletons = self.skeletons[:-1]
        self.draw()
    
    def set_data(self, data):
        for skeleton_data in data:
            hue = (1 + len(self.skeletons)) * 2 / 7.0
            skeleton = Skeleton((0, 0), "missing", hue, self.node_radius)
            skeleton.set_data(skeleton_data)
            self.skeletons.append(skeleton)
        self.draw()
    
    def get_data(self):
        return [skeleton.get_data() for skeleton in self.skeletons]
    
    def reset(self):
        self.skeletons = []
        self.current_skeleton = None
        
        
def texture_to_numpy(texture) -> np.ndarray:
    """Return texture data as an (H, W, C) NumPy array in RGB(A)"""
    if texture is None:
        return None

    w, h = texture.size
    # Kivy stores pixels bottom-left origin, RGBA by default
    raw = texture.pixels  # bytes-like, length = w * h * channels
    channels = len(raw) // (w * h)
    arr = np.frombuffer(raw, dtype=np.uint8)
    arr = arr.reshape((h, w, channels))
    # arr = np.flipud(arr)  # flip vertically to top-left origin
    return arr


def blit_numpy_to_texture(array, texture):
    if array is None or texture is None:
        return

    height, width = array.shape[:2]
    channels = array.shape[2] if array.ndim == 3 else 1

    # Flip vertically to match Kivy's coordinate system
    # flipped = np.flipud(array)

    # Convert to bytes
    raw_data = array.tobytes()

    # Determine color format
    colorfmt = {1: 'luminance', 3: 'rgb', 4: 'rgba'}.get(channels)
    if not colorfmt:
        raise ValueError("Unsupported number of channels in array")

    texture.blit_buffer(raw_data, colorfmt=colorfmt, bufferfmt='ubyte')
    

class SegmentationAnnotator(ImageAnnotator):
    def __init__(self, allow_editing):
        super().__init__(zoom_min=1.0)
        self.allow_editing = allow_editing
        self.paint_val = 0
    
    def revert(self):
        blit_numpy_to_texture(self.original_mask, self.texture)
    
    def on_click(self, position, button):
        if not self.allow_editing:
            return
        
        x, y = map(int, position)
        w, h = self.texture.size
        
        if 0 <= x < w and 0 <= y < h:
            img = texture_to_numpy(self.texture)
            
            if button == "right":
                self.paint_val = img[y, x, 0]
            elif button == "left":
                val = img[y, x, 0]
                img = np.where(img == val, self.paint_val, img)
            
            blit_numpy_to_texture(img, self.texture)
        
            self.draw()
    
    def new_image(self):
        self.original_mask = texture_to_numpy(self.texture)
        
        
    def get_data(self):
        return texture_to_numpy(self.texture)[..., 0]



class AnnotationApp(App):

    def __init__(self, image_files, annotation_files, segmentation_files, allow_editing):
        super().__init__()
        self.image_files = image_files
        self.annotation_files = annotation_files
        self.segmentation_files = segmentation_files
        self.allow_editing = allow_editing
        self.index = 0
        
    def build(self):
        self.root = FloatLayout()
        
        self.layout = BoxLayout()
        self.skel_annotator = SkeletonAnnotator(self.allow_editing)
        self.seg_annotator = SegmentationAnnotator(self.allow_editing)
        self.layout.add_widget(self.skel_annotator)
        self.layout.add_widget(self.seg_annotator)
        self.root.add_widget(self.layout)
        
        self.text = Label(size_hint=(0.3, 0.04), pos_hint={'left': 0.98, 'top': 0.98})
        self.revert_button = Button(text="Revert Mask", size_hint=(0.1, 0.04), pos_hint={'right': 0.98, 'top': 0.98})
        self.revert_button.bind(on_press=lambda instance: self.seg_annotator.revert())
        self.revert_button.on_press()
        self.root.add_widget(self.revert_button)
        self.root.add_widget(self.text)
        
        Window.bind(on_key_down=self.key_down)
        Window.bind(on_request_close=self.on_request_close)
        return self.root
    
    def on_start(self):
        self.load()

    def key_down(self, instance, keyboard, keycode, text, modifiers):
        if self.allow_editing and keycode == BACKSPACE_KEYCODE:
            self.skel_annotator.delete_last()
        elif not self.skel_annotator.is_busy and keycode == LEFT_KEYCODE and self.index > 0:
            self.save()
            self.index -= 1
            self.load()
        elif not self.skel_annotator.is_busy and keycode == RIGHT_KEYCODE and self.index < len(self.image_files) - 1:
            self.save()
            self.index += 1
            self.load()
        elif keycode == S_KEYCODE:
            self.cache_image()
            
    def on_request_close(self, *args, **kwargs):
        if self.skel_annotator.is_busy:
            return True
        else:
            self.save()
            return False

    def cache_image(self):
        if os.path.exists("cached-images.json"):
            with open("cached-images.json", "r") as f:
                file_list = json.load(f)
        else:
            file_list = []
        
        if self.image_files[self.index] not in file_list:
            file_list.append(self.image_files[self.index])
        
        with open("cached-images.json", "w") as f:
            json.dump(file_list, f)

    def load(self):
        
        self.text.text = self.annotation_files[self.index]
        
        self.seg_annotator.set_image(self.segmentation_files[self.index])
        
        self.skel_annotator.set_image(self.image_files[self.index])
        self.skel_annotator.reset()
        if os.path.exists(self.annotation_files[self.index]):
            with open(self.annotation_files[self.index], 'r') as file:
                data = json.load(file)
            self.skel_annotator.set_data(data)
            
    def save(self):
        if self.allow_editing:
            data = self.skel_annotator.get_data()
            with open(self.annotation_files[self.index], 'w') as file:
                json.dump(data, file)
                
            seg = self.seg_annotator.get_data()
            Image.fromarray(seg).save(self.segmentation_files[self.index])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--image-glob', type=str,
        help='glob pattern for finding images.'
    )
    parser.add_argument(
        '--visualise-only', action="store_true",
        help='blocks annotation editing.'
    )
    parser.add_argument(
        '--cached-only', action="store_true",
        help='only shows files which have been cached.'
    )
    args = parser.parse_args()


    # images = sorted(glob(args.image_glob, recursive=True))
    # if args.cached_only:
    #     if not os.path.exists("cached-images.json"):
    #         print("No cached files!")
    #         quit()
    #     with open("cached-images.json", "r") as f:
    #         file_list = json.load(f)
    #     images = [i for i in images if i in file_list]
        

    with open("samples_to_check.json") as file:
        images = json.load(file)    
    images = [i.replace("/", os.path.sep).replace("data\\", "") for i in images]
    
    assert len(images) > 0, 'No images found!'
    
    annotations = [os.path.sep.join(i.split(os.path.sep)[:-1] + ["toolposes.json"]) for i in images]
    segmentations = [os.path.sep.join(i.split(os.path.sep)[:-1] + ["instrument_instances.png"]) for i in images]

    AnnotationApp(images, annotations, segmentations, not args.visualise_only).run()
    