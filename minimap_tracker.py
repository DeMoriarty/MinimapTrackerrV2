import cv2
import numpy as np
import util
import json
import itertools

from icon_tracker import IconTracker
from rect_tracker import RectTracker


class MinimapTracker:
    def __init__(self, targets, icon_radius=11, *args, **kwargs):

        # Icon radius
        self.icon_radius = icon_radius

        # Create trackers for every target
        self.targets = targets
        self.trackers = [ IconTracker(i, team_color=k, radius=self.icon_radius, *args, **kwargs) for k,v in self.targets.items() for i in v ]
        
        # Load minimap Image
        self._minimap_image_path = kwargs['map_image'] if 'map_image' in kwargs.keys() else 'minimap.png'
        self.minimap_image = cv2.imread(self._minimap_image_path)
        
        # Map position
        self.map_pos = kwargs['map_pos'] if 'map_pos' in kwargs.keys() else None
        
        # Map size
        self.map_size = kwargs['map_size'] if 'map_size' in kwargs.keys() else None
        
        # Map border (for padding)
        self.map_border = kwargs['map_border'] if 'map_border' in kwargs.keys() else 10
        
        # Rect tracker
        self.rect_tracker = RectTracker()
        self.rect_pos = None
        self.rect_path = dict()
        
        # Counter, increment by 1 each time track() is called.
        self.counter = 0
        
    def track(self, frame, counter = None):
        assert frame is not None, 'Invalid input image'
        if counter:
            self.counter = counter
        if not self.locate_minimap(frame):
            return None
        
        maparea = frame[self.map_pos[0]:self.map_pos[2],
                        self.map_pos[1]:self.map_pos[3],:]
        # Track and Remove rectangle
#        top_left, bot_right = self.rect_tracker.track(util.grayscale(maparea))
#        self.rect_pos = (*top_left, *bot_right)
#        self.rect_path[counter] =(round(self.rect_pos[0][0]/maparea.shape[1], 4),
#                                  round(self.rect_pos[0][1]/maparea.shape[0], 4),
#                                  round(self.rect_pos[1][0]/maparea.shape[1], 4),
#                                  round(self.rect_pos[1][1]/maparea.shape[0], 4))
#        maparea = self.rect_tracker.remove_rect(maparea)
        
        # Pad the maparea
        padded_map = self.pad(maparea, self.map_border)
        gray = util.grayscale(padded_map)
        circle_map = np.zeros_like(gray)
        circles = cv2.HoughCircles(image = gray,
                                   method = cv2.HOUGH_GRADIENT,
                                   dp = 1,
                                   minDist = 8,
                                   param1 = 20,
                                   param2 = 9.0, # 8.0 to 9.0
                                   minRadius = self.icon_radius-1,
                                   maxRadius = self.icon_radius+1)
        if circles is None:
            return None
        if len(circles) > 0:
            circles = np.uint16(np.around(circles))
            for i in circles[0, :]:
                cv2.circle(img = circle_map,
                           center = tuple(i[:2]),
                           radius = i[2]+1,
                           color = (255,255,255),
                           thickness = -1)
        circle_mask = np.uint8(circle_map / 255) * 255
        masked_map = cv2.bitwise_and(gray, gray, mask = circle_mask)
#        cv2.imshow('circle_mask', masked_map)
        
        show_map = padded_map.copy()
        for i in self.trackers:
            pos = i.track(show_map, padded_map, masked_map, self.counter)
            cv2.circle(img = show_map,
                       center = pos, 
                       radius = round(i.radius)+1, 
                       color = (0, 255, 0), 
                       thickness = 1)

        ### Detect Collision and BITE (test)
        for i,j in itertools.combinations(self.trackers, 2):
            a = i.loc
            b = j.loc
            dist = util.distance(a, b)
            if dist <= 2 * i.radius:
                if self.trackers.index(i) < self.trackers.index(j):                    
                    i.bite(j)
                else:
                    j.bite(i)
            else:
                i.icon_mask = i.icon_mask_og.copy()
                j.icon_mask = j.icon_mask_og.copy()

        self.counter += 1        
#        cv2.imshow('show_map', show_map)
    
    def locate_minimap(self, frame):
        '''
            Returns True if minimap is located.
            Returns False if no valid match.
        '''
        if not self.map_pos:
            pos = [0,self.minimap_image.shape[0],(0,0)]
            gray_frame = util.grayscale(frame)
            for i in range(self.minimap_image.shape[0], 120, -2):
                resized_minimap = cv2.resize(util.grayscale(self.minimap_image), (i,i))
                res = cv2.matchTemplate(gray_frame , resized_minimap, cv2.TM_CCOEFF_NORMED)
                t_pos = tuple([j.tolist()[0] for j in np.where(res == res.max())])
                if res.max() > pos[0]:
                    pos = [res.max(),i, t_pos]
            if pos[0] >= 0.5:
                self.map_pos = (pos[2][0], pos[2][1], pos[2][0]+pos[1], pos[2][1]+pos[1])
                self.map_size = (pos[1], pos[1])
                return True
            return False
        else:
            return True
    
    
    @staticmethod
    def pad(image, padding):
        '''
            padding = (top, left, bottom, right)
            padding = all
            padding = (top_bottom, left_right)
        '''
        if type(padding) == int:
            top, left, bottom, right = padding, padding, padding, padding
        elif len(padding) == 1:
            top, left, bottom, right = padding, padding, padding, padding
        elif len(padding) == 2:
            top, left, bottom, right = padding[0], padding[1], padding[0], padding[1]
        elif len(padding) == 4:
            tpp, left, bottom, right = padding
        else:
            raise ValueError
        
        if len(image.shape) == 2:
            empty = np.zeros((image.shape[0]+top+bottom, image.shape[1]+left+right), dtype=np.uint8)
            empty[top:-bottom, left:-right] = image.copy()
        elif len(image.shape) == 3:
            empty = np.zeros((image.shape[0]+top+bottom, image.shape[1]+left+right, image.shape[2]), dtype=np.uint8)
            empty[top:-bottom, left:-right,:] = image.copy()
        return empty
    
    def save(self, savepath, path = True, rect = True):
        with open(savepath, 'w') as f:
            json.dump(self.json(path = path, rect = rect), f)
        print('successfully saved!')
    
    def json(self, path = True, rect = True):
        output = dict()
        if path:
            for tracker in self.trackers:
                name = tracker.name
                path = tracker.path
                path = { i : [round((j[0]-self.map_border)/self.map_size[0], 4), round((j[1]-self.map_border)/self.map_size[1], 4)] for i,j in path.items() }
                output[name] = path
        if rect:
            output['rect'] = self.rect_path
        return output

