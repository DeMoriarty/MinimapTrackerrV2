import cv2
import numpy as np
import os
import json
from minimap_tracker import MinimapTracker
from time import time
import traceback

def t2f(time):
    time = time.split(':')
    time = [int(i) for i in time]
    if len(time) == 2:
        return (time[0] * 60 + time[1]) * 30
    elif len(time) == 3:
        return (time[0] * 60 * 60 + time[1] * 60 + time[2]) * 30

with open('matches_2.json','r', encoding="utf-8") as f:
    data = json.load(f)

data = {k: v for (k, v) in data.items() if int(k) < 50}
data = sorted(list(data.items()), key = lambda x: int(x[0]))

for index, item in data[1:]:
    index = int(index)
    start = t2f(item['starts_at'])
    end = t2f(item['ends_at'])
    counter = t2f(item['game_starts_at'])
    url = item['url']
    video_id = url.replace('https://www.youtube.com/watch?v=','')
    video_path = f'videos/{video_id}.mp4'
    if not os.path.exists(video_path):
        print('Video does not exist')
        continue
    if os.path.exists(f'match_info/{video_id}.json'):
        print("This video is already processed")
        continue
    vid = cv2.VideoCapture(video_path)
    ok, frame = vid.read()
    mmtracker = MinimapTracker({'red':list(item['red_champions'].values()) ,
                                'blue':list(item['blue_champions'].values())},
                                icon_radius = 12,
                                map_size = frame.shape[:2],
                                map_pos = (0, 0, frame.shape[0], frame.shape[1]),
                                )
    while vid.isOpened():
        start_time = time()
        ok, frame = vid.read()
        if not ok or frame is None:
            break
        try:
            mmtracker.track(frame, counter)
        except:
#            traceback.print_exc()
            pass
        counter += 1
        if counter % 300 == 0:
            percent = counter / end
            print(f'{index}>>{int(percent * 100)}% {"#" * int(50 * percent)}{"-" * int(50 * (1 - percent))}\r', flush=True)
#            key = cv2.waitKey(1) & 0xFF
#            if key == ord('q'):
#                break
#            print( round(1/(time() - start_time), 2) )
    mmtracker.save(f'match_info/{video_id}.json', rect = False)
    vid.release()
    cv2.destroyAllWindows()
