import cv2, pafy
import numpy as np
#from soznlp.tools import timer
import json
import os
from itertools import product, combinations

def grayscale(img):
    '''
    convert image to grayscale
    parameters:
        img: numpy array with shape = (X, Y, C)
    return:
        numpy array with shape = (X, Y)
    '''
#    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return custom_grayscale(img, 0.15, 0.2, 0.65)

def redc(img):
    '''
    get R channel of RGB image
    parameters:
        img: numpy array with shape = (X, Y, C)
    return
        numpy array with shape = (X, Y)
    '''
    return img[:,:,2].copy()

def greenc(img):
    '''
    get G channel of RGB image
    parameters:
        img: numpy array with shape = (X, Y, C)
    return:
        numpy array with shape = (X, Y)
    '''
    return img[:,:,1].copy()

def bluec(img):
    '''
    get B channel of RGB image
    parameters:
        img: numpy array with shape = (X, Y, C)
    return:
        numpy array with shape = (X, Y)
    '''
    return img[:,:,0].copy()

def yellowc(img):
    '''
    input RGB image, return 0.336 * R + 0.664 * G
    parameters:
        img: numpy array with shape = (X, Y, C)
    return:
        numpy array with shape = (X, Y)
    '''
    return custom_grayscale(img, 0, 0.336, 0.664)

def custom_grayscale(img, alpha, beta, gamma):
    assert type(img) is np.ndarray, f'img: {type(imp)} is not numpy array'
    assert len(img.shape) == 3, f'img: invalid shape {img.shape}'
    for i in range(len(img.shape)):
        assert img.shape[i] > 0, f'img: invalid shape {img.shape}'
    assert type(alpha) is float, 'alpha: {type(alpha)} is not float'
    assert type(beta) is float, 'beta: {type(beta)} is not float'
    assert type(gamma) is float, 'gamma: {type(gamma)} is not float'
    assert 0 <= alpha <= 1, 'alpha: {alpha} is not in range [0, 1]'
    assert 0 <= beta <= 1, 'beta: {beta} is not in range[0, 1]'
    assert 0 <= gamma <= 1, 'gamma: {gamma} is not in range [0, 1]'
    assert alpha + beta + gamma - 1 - 1e-2 <= 0, 'alpha, beta and gamma are not sum up to 1. the sum is {alpha + beta + gamma}'
    return cv2.addWeighted(cv2.addWeighted(img[:,:,0], alpha, img[:,:,1], beta, 0), 1.0, img[:,:,2], gamma, 0)
    
## load video from youtube
def load_yt(url):
    '''
    load video stream from Youtube
    parameters:
        url: str, URL or 11 digit VideoID of the video
    return:
        pafy.Stream object
    '''
    print(f'Start loading from {url}')
    vpafy = pafy.new(url)
    best = vpafy.getbestvideo("any", False)
    print('Title:',best.title)
    print('Filesize:',round(best.get_filesize()/ (1000 * 1000), 2), 'MB')
    print('Resolution:',best.resolution)
    return best

def HSV(h, s, v):
    # (360, 100%, 100%) => (180, 255, 255)
    return int(h/2), int(s/100 * 255), int(v/100*255)

def cluster(pts, threshold, iterations = 1):
    ### not percise
    pts = pts.copy()
    for i in range(len(pts)):
        for j in range(i+1, len(pts)):
#            counter += 1
            diff = np.abs(np.subtract(pts[i][:2],pts[j][:2]))
            if np.less(diff,threshold).all():
                lenth = max(pts[i][2], pts[j][2])
                pts[i] = pts[j] = tuple((np.add(pts[i][:2],pts[j][:2])/2).astype(int))+(lenth,)
    if iterations == 1:
        return list(set(pts))
    else:
        return cluster(list(set(pts)), threshold, iterations-1)
    
def cluster_v2(pts, threshold):
    res = pts.copy()
    mark = False
    for pt1, pt2 in combinations(pts, 2):
        diff = np.abs(np.subtract(pt1[:2], pt2[:2]))
        if np.less(diff, threshold).all():
            length = max(pt1[2], pt2[2])
            res.remove(pt1)
            res.remove(pt2)
            res.append(tuple((np.add(pt1[:2],pt2[:2])/2).astype(int))+(length,))
            mark = True
            break
    res = list(set(res))
    if not mark:
        return res
    else:
        return cluster_v2(res, threshold)
    
def cluster_v3(pts, threshold):
    res = pts.copy()
    for pt1, pt2 in combinations(pts, 2):
        if not pt2 in res:
            continue
        diff = np.abs(np.subtract(pt1[:2], pt2[:2]))
        if np.less(diff, threshold).all():
            res.remove(pt2)
    return list(tuple(res))
    
champ_names = json.loads(open('champ_nicknames.json', 'r').read())

def regularize(name):
    name = name.capitalize()
    for k,  v in champ_names.items():
        if name in v:
            return k
    return name

def iterate_folder(path):
    if os.path.exists(path):
        folders = os.walk(path)
        folders = [(i[0].replace('\\','/'),i[2]) for i in folders if i[2]]
        pics = [ i[0]+'/'+j for i in folders for j in i[1]]
        return pics
    else:
        print("Path doesn't exist")
        return []
    
def labeled_files(folder ,label):
    files = [ (label, folder+'/'+i) for i in list(os.walk(folder))[0][2]]
    return files
    
def fig2data ( fig ):
    """
    @brief Convert a Matplotlib figure to a 4D numpy array with RGBA channels and return it
    @param fig a matplotlib figure
    @return a numpy 3D array of RGBA values
    """
    # draw the renderer
    fig.canvas.draw ( )
 
    # Get the RGBA buffer from the figure
    w,h = fig.canvas.get_width_height()
    buf = np.fromstring ( fig.canvas.tostring_argb(), dtype=np.uint8 )
    buf.shape = ( w, h,4 )
 
    # canvas.tostring_argb give pixmap in ARGB mode. Roll the ALPHA channel to have it in RGBA mode
    buf = np.roll ( buf, 3, axis = 2 )
    return buf

def line_in_box(line, box):
    if np.less_equal(line[:2], box[2:]).all() and np.greater_equal(line[:2],box[:2]).all() \
    or np.less_equal(line[2:],box[2:]).all() and np.greater_equal(line[2:], box[:2]).all():
        return True
    else:
        return False

def distance_p2p(pt1, pt2):
    return np.sqrt(np.sum((np.array(pt1) - np.array(pt2)) ** 2))

def distance(pt1, pt2):
    pt1 = np.array(pt1)
    pt2 = np.array(pt2)
    if len(pt1.shape) == 1 and len(pt2.shape) == 1:
        return np.sqrt(np.sum((pt2 - pt1) ** 2))
    elif len(pt1.shape) == 2 and len(pt2.shape) == 2:
        return np.sqrt(np.sum((pt2 - pt1) ** 2, axis=1))