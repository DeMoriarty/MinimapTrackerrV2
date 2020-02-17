import cv2
import numpy as np
import util
from itertools import product

class RectTracker():
    def __init__(self):
        self.height = 0
        self.width =  0
        self.top_left = (0, 0)
        self.bottom_right = (0, 0)
        self.counter = 0
    
    def track(self, image):        
        _, th = cv2.threshold(image, 180, 255, cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
        th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel)
        th = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel)
        
        lines = cv2.HoughLinesP(th, rho = 1, theta = np.pi/180, threshold=25, minLineLength=20, maxLineGap = 10)
        if np.any(lines):
            lines = lines.squeeze(axis=1)
            h_lines = [tuple(line.tolist()) for line in lines if abs(line[3] - line[1]) <= 1]
            h_lines = [ (line[0],line[1],line[2],line[1]) for line in h_lines]
            
            v_lines = [tuple(line.tolist()) for line in lines if abs(line[2] - line[0]) <= 1]
            v_lines = [ (line[0],line[1],line[0],line[3]) for line in v_lines]

            intersects = []
            for v_line, h_line in product(v_lines, h_lines):
                inter = self.intersection(v_line, h_line)
                intersects.append(inter)

            intersects = self.cluster(intersects, (5, 5), 2)
            if len(intersects) >= 1:
                xs = [i[0] for i in intersects]
                max_x = int(max(xs)) + 1
                min_x = int(min(xs)) + 1
                
                ys = [i[1] for i in intersects]
                max_y = int(max(ys)) + 1
                min_y = int(min(ys)) + 1
                
                if max_x <= self.width:
                    min_x = max_x - round(self.width)
                    
                if max_y <= self.height:
                    min_y = max_y - round(self.height)
                
                out_of_range = False
                
                if min_x >= image.shape[0] - self.width:
                    out_of_range = True
                
                if min_y >= image.shape[1] - self.height:
                    out_of_range = True
                
                if len(intersects) >= 3:
                    self.counter += 1
                    self.width = self.width * (self.counter - 1)/(self.counter) + (max_x - min_x) / self.counter
                    self.height = self.height * (self.counter - 1)/(self.counter) + (max_y - min_y) / self.counter
                    self.top_left = (min_x, min_y)
                    self.bottom_right = (round(min_x + self.width) -1 , round(min_y + self.height) -1)
                elif util.distance_p2p( self.top_left, (min_x, min_y)) <= 5.0:
                    self.top_left = (min_x, min_y)
                    self.bottom_right = (round(min_x + self.width) -1, round(min_y + self.height) -1)
                elif out_of_range:
                    self.top_left = (min_x, min_y)
                    self.bottom_right = (round(min_x + self.width) -1, round(min_y + self.height) -1)
            
        return self.top_left,self.bottom_right
    
    def intersection(self, line1, line2):
        line1 = ((line1[0], line1[1]),(line1[2],line1[3]))
        line2 = ((line2[0], line2[1]),(line2[2],line2[3]))
        
        xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
        ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])
    
        def det(a, b):
            return a[0] * b[1] - a[1] * b[0]
    
        div = det(xdiff, ydiff)
        if div == 0:
           return None
    
        d = (det(*line1), det(*line2))
        x = det(d, xdiff) / div
        y = det(d, ydiff) / div
        return int(x), int(y)
    
    def cluster(self, pts, threshold, iterations = 1):
        ### not percise
        pts = pts.copy()
        for i in range(len(pts)):
            for j in range(i+1, len(pts)):
                diff = np.abs(np.subtract(pts[i][:2],pts[j][:2]))
                if np.less(diff,threshold).all():
                    pts[i] = pts[j] = tuple((np.add(pts[i][:2],pts[j][:2])/2).astype(int))
        if iterations == 1:
            return list(set(pts))
        else:
            return self.cluster(list(set(pts)), threshold, iterations-1)
        
    def remove_rect(self, image):
        padded = self.pad(image, (0, 0, round(self.height)+2, round(self.width)+2))
        
        #left bar
        left_right = padded[self.top_left[1]:self.top_left[1]+round(self.height), self.top_left[0]+1:self.top_left[0]+ 2,].copy()
        left_left = padded[self.top_left[1]:self.top_left[1]+round(self.height), self.top_left[0] - 3:self.top_left[0] - 2,].copy()
        left_mid = cv2.addWeighted(left_right, 0.5, left_left, 0.5, 0)
        padded[self.top_left[1]:self.top_left[1]+round(self.height), self.top_left[0]:self.top_left[0]+ 1,] = left_right
        padded[self.top_left[1]:self.top_left[1]+round(self.height), self.top_left[0]-2:self.top_left[0] - 1,] = left_left
        padded[self.top_left[1]:self.top_left[1]+round(self.height), self.top_left[0]-1:self.top_left[0]] = left_mid
        
        #right bar
        right_right = padded[self.top_left[1]:self.top_left[1]+round(self.height), self.bottom_right[0]+2:self.bottom_right[0]+ 3,].copy()
        right_left = padded[self.top_left[1]:self.top_left[1]+round(self.height), self.bottom_right[0] - 2:self.bottom_right[0] - 1,].copy()
        right_mid = cv2.addWeighted(right_right, 0.5, right_left, 0.5, 0)
        padded[self.top_left[1]:self.top_left[1]+round(self.height), self.bottom_right[0]+1:self.bottom_right[0]+ 2,] = right_right
        padded[self.top_left[1]:self.top_left[1]+round(self.height), self.bottom_right[0]-1:self.bottom_right[0],] = right_left
        padded[self.top_left[1]:self.top_left[1]+round(self.height), self.bottom_right[0]:self.bottom_right[0] + 1] = right_mid
        
        #top bar
        top_top = padded[self.top_left[1]+1:self.top_left[1]+2, self.top_left[0]:self.top_left[0]+round(self.width)]
        top_bot = padded[self.top_left[1]-3:self.top_left[1]-2, self.top_left[0]:self.top_left[0]+round(self.width)]
        top_mid = cv2.addWeighted(top_top, 0.5, top_bot, 0.5, 0)
        padded[self.top_left[1]:self.top_left[1]+1, self.top_left[0]:self.top_left[0]+round(self.width)] = top_top
        padded[self.top_left[1]-2:self.top_left[1]-1, self.top_left[0]:self.top_left[0]+round(self.width)] = top_bot
        padded[self.top_left[1]-1:self.top_left[1], self.top_left[0]:self.top_left[0]+round(self.width)] = top_mid
        
        #bottom bar
        bot_top = padded[self.bottom_right[1]+2:self.bottom_right[1]+3, self.top_left[0]:self.top_left[0]+round(self.width)]
        bot_bot = padded[self.bottom_right[1]-2:self.bottom_right[1]-1, self.top_left[0]:self.top_left[0]+round(self.width)]
        bot_mid = cv2.addWeighted(bot_top, 0.5, bot_bot, 0.5, 0)
        padded[self.bottom_right[1]+1:self.bottom_right[1]+2, self.top_left[0]:self.top_left[0]+round(self.width)] = bot_top
        padded[self.bottom_right[1]-1:self.bottom_right[1], self.top_left[0]:self.top_left[0]+round(self.width)] = bot_bot
        padded[self.bottom_right[1]:self.bottom_right[1]+1, self.top_left[0]:self.top_left[0]+round(self.width)] = bot_mid
        
        image = padded[:image.shape[0], :image.shape[1],]
        return image
        
    def pad(self, image, padding):
        '''
            padding = (top, left, bottom, right)
            padding = all
            padding = (top_bottom, left_right)
        '''
        if type(padding) == int:
            top, left, bottom, right = padding, padding, -padding, -padding
        elif len(padding) == 1:
            top, left, bottom, right = padding, padding, -padding, -padding
        elif len(padding) == 2:
            top, left, bottom, right = padding[0], padding[1], -padding[0], -padding[1]
        elif len(padding) == 4:
            top, left, bottom, right = padding[0], padding[1], -padding[2], -padding[3]
        else:
            raise ValueError
        
        
        if len(image.shape) == 2:
            empty = np.zeros((image.shape[0]+top-bottom, image.shape[1]+left-right), dtype=np.uint8)
#            empty[top:-bottom, left:-right] = image.copy()
        elif len(image.shape) == 3:
            empty = np.zeros((image.shape[0]+top-bottom, image.shape[1]+left-right, image.shape[2]), dtype=np.uint8)
        if bottom == 0:
            bottom = None
        if right == 0:
            right = None
        empty[top:bottom, left:right,] = image.copy()
        return empty
