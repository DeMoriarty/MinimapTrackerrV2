import cv2
import util
import os
import numpy as np

UNDEFINED = 'undefined'
REDTEAM = 'red'
BLUETEAM = 'blue'
SELFGREEN = 'green'
SELFYELLOW = 'yellow'

class IconTracker:
    def __init__(self, name, radius, team_color = UNDEFINED, *args, **kwargs):
        # BLUETEAM, REDTEAM, SELFGREEN, SELFYELLOW or UNDEFINED
        
        assert team_color in [REDTEAM, BLUETEAM, SELFGREEN, SELFYELLOW, UNDEFINED], f'team_color: {team_color}'
        self.team_color = team_color
        
        self.conf_thresh = kwargs['conf_thresh'] if 'conf_thresh' in kwargs.keys() else 0.8
        assert 0 <= self.conf_thresh < 1, f"confidence threshold {self.conf_thresh} not in range [0.0, 1.0)"
        
        assert type(name) is str, 'name is not str'        
        self.name = util.regularize(name)
        
        assert type(name) is str, 'icon_folder is not str'
        self.icon_folder = kwargs['icon_folder'] if 'icon_folder' in kwargs.keys() else 'champion_icons'
        assert os.path.exists(self.icon_folder), f'icon_folder: {self.icon_folder} does not exists'
        
        self.filename = f'{self.icon_folder}/{self.name}.jpg'
        assert os.path.exists(self.filename), f'icon_path: {self.filename} does not exists'
        
        self.icon = cv2.imread(self.filename, cv2.IMREAD_UNCHANGED)
        
        # Icon radius
        assert type(radius) is int or type(radius) is float, 'radius is not int or float'
        self.radius = radius

        # Local border
        self.border = kwargs['border'] if 'border' in kwargs.keys() else 5#self.radius - 1
        assert type(self.border) is int, f'border: {type(self.border)} is not int'
#        assert self.border >= self.radius - 1, f'border: {self.border} < {self.radius - 1}'

        # draw red or blue ring around champion icon
        og_radius = round(self.icon.shape[0]/2)
        if team_color == REDTEAM:
            cv2.circle(self.icon, (og_radius, og_radius), radius = og_radius, color = (73, 70, 183, 255), thickness = 4, lineType = cv2.LINE_AA)
        elif team_color == BLUETEAM\
            or team_color == SELFGREEN\
            or team_color == SELFYELLOW:              
            cv2.circle(self.icon, (og_radius, og_radius), radius = og_radius, color = (154, 117, 31, 255), thickness = 4, lineType = cv2.LINE_AA)
        
        self.icon = cv2.resize(self.icon,  ( int(self.radius * 2), int(self.radius * 2)),  interpolation = cv2.INTER_AREA)  
        self.gray_icon = util.grayscale(self.icon)
        
        # Alpha mask
        if self.icon.shape[2] == 4:
            self.icon_mask = self.icon[:,:,3]
        else:
            self.icon_mask = np.ones_like(self.icon[0]) * 255
        self.icon_mask_og = self.icon_mask.copy()
                
        # Init localtion
        self.loc = None
        self.path = dict()
        
        # Init previous location
        self.prev_loc = None
        
        # Init global location
        self.global_loc = None
        self.global_path = []
        
        # Init local location
        self.local_loc = None
        self.local_path = []
        
        # Init teleport location
        self.teleport_loc = None
        
        # counter
        self.counter = 0
        
    def track(self, showmap, maparea, image, counter):
        if counter:
            self.counter = counter
        
        #  Global observation
        assert image is not None, f'Invalid input image'
        res = cv2.matchTemplate(image, self.gray_icon, cv2.TM_CCOEFF_NORMED)
        min_max_loc = cv2.minMaxLoc(res)
        if min_max_loc[1] > 0.8:
            loc = min_max_loc[3]
            loc = tuple([round(i+self.radius) for i in loc])
            if loc[0] < self.border + self.radius or loc[1] < self.border + self.radius:
                loc = (-100, -100)
                print('loc out of box')
        else:
            loc = (-100, -100)
        self.global_loc = loc

        # Local observation
        if self.prev_loc:
            local_rect = ( self.prev_loc[1] - self.radius - self.border,
                           self.prev_loc[1] + self.radius + self.border,
                           self.prev_loc[0] - self.radius - self.border,
                           self.prev_loc[0] + self.radius + self.border)
            
            # ROI
            local_area = maparea[local_rect[0]: local_rect[1],
                                 local_rect[2]: local_rect[3],:]
            
            ## TEMPORARY
            show_area =  showmap[local_rect[0]: local_rect[1],
                                 local_rect[2]: local_rect[3],:]
            
            local_area_gray = util.grayscale(local_area)
            # OPENCV: R * 0.3 + g * 0.587 + b * 0.114
            # CUSTOM: R * 0.4 + g * 0.35 + b * 0.25
            
            # Template matcing
            res = cv2.matchTemplate(image = local_area_gray,
                                    templ = util.grayscale(self.icon),
                                    method = cv2.TM_CCORR_NORMED,
                                    mask = self.icon_mask)
            
            ### IDEA : Apply -(2 * (i - x) ^ 2+ 2 * (j - y) ^ 2) / res.shape[0] * res.shape[1] + 1 on res,
            ### Where i, j are indices of res
            ### x, y is predicted location
            ### res.shape is shape of matrix res
            
            local_mml = cv2.minMaxLoc(res)
            
            # get the index location local maximum
            # maximum for cross correlation and cross coeff
            # minimum for square difference
            local_loc = local_mml[3][0] + self.radius, local_mml[3][1] + self.radius
              
            # Rendering
            cv2.circle(img = show_area,
                       center = local_loc,  
                       radius = self.radius+1,
                       color = (0, 0, 255),
                       thickness = 2)
            self.local_loc = local_loc[0] + local_rect[2], local_loc[1] + local_rect[0]
        
        # Final location
        # Distance of global and local location
        lg_dist = util.distance(self.global_loc, self.local_loc)
        
        # Distance of current and previous location
        cp_dist = util.distance(self.local_loc, self.prev_loc)
        if cp_dist is None:
            self.loc = self.global_loc
        elif self.teleport_loc:
            self.loc = self.teleport_loc
            self.teleport_loc = None
        elif (lg_dist > 3 and cp_dist > 3):
            self.teleport_loc = self.global_loc
        else:
            self.loc = self.local_loc

        if self.loc and self.loc != (-100, -100):
            self.path[self.counter] = self.loc
        
        # set previous position
        if self.loc != (-100, -100):
            self.prev_loc = self.loc
        self.counter += 1
        return self.loc
    
    def bite(self, target):
        '''
            Cut the intersection from current object's mask if current object and target object are collided.
        '''
        offset = np.subtract(target.loc, self.loc)
        mask = self.icon_mask_og.copy()
        cv2.circle(img = mask,
                   center = (offset[0]+self.radius, offset[1]+self.radius),
                   radius = self.radius + 1,
                   color = (0,) ,
                   thickness = -1)
        self.icon_mask = mask
    
    def sd(self, pts):
        distances = []
        for i in range(len(pts) - 1):
            distances.append( util.distance(pts[i],pts[i+1]))
        
#        mn = np.mean(distances)
        std = np.std(distances)
        print('std', std)