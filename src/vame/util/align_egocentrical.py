"""
Variational Animal Motion Embedding 0.1 Toolbox
© K. Luxem & J. Kürsch & P. Bauer, Department of Cellular Neuroscience
Leibniz Institute for Neurobiology, Magdeburg, Germany

https://github.com/LINCellularNeuroscience/VAME
Licensed under GNU General Public License v3.0
"""

import os
import cv2 as cv
import numpy as np
import pandas as pd
import tqdm
from typing import Tuple, List
from datetime import datetime
from vame.logging.redirect_stream import StreamToLogger
from pathlib import Path
from vame.util.auxiliary import read_config

#Returns cropped image using rect tuple
def crop_and_flip(
    rect: Tuple,
    src: np.ndarray,
    points: List[np.ndarray],
    ref_index: Tuple[int, int]
) -> Tuple[np.ndarray, List[np.ndarray]]:
    """
    Crop and flip the image based on the given rectangle and points.

    Args:
        rect (Tuple): Rectangle coordinates (center, size, theta).
        src (np.ndarray): Source image.
        points (List[np.ndarray]): List of points.
        ref_index (Tuple[int, int]): Reference indices for alignment.

    Returns:
        Tuple[np.ndarray, List[np.ndarray]]: Cropped and flipped image, and shifted points.
    """
    #Read out rect structures and convert
    center, size, theta = rect

    center, size = tuple(map(int, center)), tuple(map(int, size))

    # center_lst = list(center)
    # center_lst[0] = center[0] - size[0]//2
    # center_lst[1] = center[1] - size[1]//2

    # center = tuple(center_lst)


    # center[0] -= size[0]//2
    # center[1] -= size[0]//2 # added this shift to change center to belly 2/28/2024

    #Get rotation matrix
    M = cv.getRotationMatrix2D(center, theta, 1)

    #shift DLC points
    x_diff = center[0] - size[0]//2
    y_diff = center[1] - size[1]//2

    # x_diff = center[0]
    # y_diff = center[1]

    dlc_points_shifted = []

    for i in points:
        point=cv.transform(np.array([[[i[0], i[1]]]]),M)[0][0]

        point[0] -= x_diff
        point[1] -= y_diff

        dlc_points_shifted.append(point)

    # Perform rotation on src image
    dst = cv.warpAffine(src.astype('float32'), M, src.shape[:2])
    out = cv.getRectSubPix(dst, size, center)

    #check if flipped correctly, otherwise flip again
    if dlc_points_shifted[ref_index[1]][0] >= dlc_points_shifted[ref_index[0]][0]:
        rect = ((size[0]//2,size[0]//2),size,180) #should second value be size[1]? Is this relevant to the flip? 3/5/24 KKL
        center, size, theta = rect
        center, size = tuple(map(int, center)), tuple(map(int, size))


        # center_lst = list(center)
        # center_lst[0] = center[0] - size[0]//2
        # center_lst[1] = center[1] - size[1]//2
        # center = tuple(center_lst)

        #Get rotation matrix
        M = cv.getRotationMatrix2D(center, theta, 1)


        #shift DLC points
        x_diff = center[0] - size[0]//2
        y_diff = center[1] - size[1]//2

        # x_diff = center[0]
        # y_diff = center[1]


        points = dlc_points_shifted
        dlc_points_shifted = []

        for i in points:
            point=cv.transform(np.array([[[i[0], i[1]]]]),M)[0][0]

            point[0] -= x_diff
            point[1] -= y_diff

            dlc_points_shifted.append(point)

        # Perform rotation on src image
        dst = cv.warpAffine(out.astype('float32'), M, out.shape[:2])
        out = cv.getRectSubPix(dst, size, center)

    return out, dlc_points_shifted


def nan_helper(y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Helper function to identify NaN values in an array.

    Args:
        y (np.ndarray): Input array.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Boolean mask for NaN values and function to interpolate them.
    """
    return np.isnan(y), lambda z: z.nonzero()[0]


def interpol(arr: np.ndarray) -> np.ndarray:
    """
    Interpolates NaN values in the given array.

    Args:
        arr (np.ndarray): Input array.

    Returns:
        np.ndarray: Array with interpolated NaN values.
    """

    y = np.transpose(arr)

    nans, x = nan_helper(y[0])
    y[0][nans]= np.interp(x(nans), x(~nans), y[0][~nans])
    nans, x = nan_helper(y[1])
    y[1][nans]= np.interp(x(nans), x(~nans), y[1][~nans])

    arr = np.transpose(y)

    return arr

def background(path_to_file: str, filename: str, video_format: str = '.mp4', num_frames: int = 1000) -> np.ndarray:
    """
    Compute the background image from a fixed camera.

    Args:
        path_to_file (str): Path to the file directory.
        filename (str): Name of the video file without the format.
        video_format (str, optional): Format of the video file. Defaults to '.mp4'.
        num_frames (int, optional): Number of frames to use for background computation. Defaults to 1000.

    Returns:
        np.ndarray: Background image.
    """
    import scipy.ndimage
    capture = cv.VideoCapture(os.path.join(path_to_file,'videos',filename+video_format))

    if not capture.isOpened():
        raise Exception("Unable to open video file: {0}".format(os.path.join(path_to_file,'videos',filename+video_format)))

    frame_count = int(capture.get(cv.CAP_PROP_FRAME_COUNT))
    ret, frame = capture.read()

    height, width, _ = frame.shape
    frames = np.zeros((height,width,num_frames))

    for i in tqdm.tqdm(range(num_frames), disable=not True, desc='Compute background image for video %s' %filename):
        rand = np.random.choice(frame_count, replace=False)
        capture.set(1,rand)
        ret, frame = capture.read()
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        frames[...,i] = gray

    print('Finishing up!')
    medFrame = np.median(frames,2)
    background = scipy.ndimage.median_filter(medFrame, (5,5))

    # np.save(path_to_file+'videos/'+'background/'+filename+'-background.npy',background)

    capture.release()
    return background


def align_mouse(
    path_to_file: str,
    filename: str,
    video_format: str,
    crop_size: Tuple[int, int],
    pose_list: List[np.ndarray],
    pose_ref_index: Tuple[int, int],
    confidence: float,
    pose_flip_ref: Tuple[int, int],
    bg: np.ndarray,
    frame_count: int,
    use_video: bool = True
) -> Tuple[List[np.ndarray],List[List[np.ndarray]], np.ndarray]:
    """
    Align the mouse in the video frames.

    Args:
        path_to_file (str): Path to the file directory.
        filename (str): Name of the video file without the format.
        video_format (str): Format of the video file.
        crop_size (Tuple[int, int]): Size to crop the video frames.
        pose_list (List[np.ndarray]): List of pose coordinates.
        pose_ref_index (Tuple[int, int]): Pose reference indices.
        confidence (float): Pose confidence threshold.
        pose_flip_ref (Tuple[int, int]): Reference indices for flipping.
        bg (np.ndarray): Background image.
        frame_count (int): Number of frames to align.
        use_video (bool, optional): bool if video should be cropped or DLC points only. Defaults to True.

    Returns:
        Tuple[List[np.ndarray], List[List[np.ndarray]], np.ndarray]: List of aligned images, list of aligned DLC points, and time series data.
    """

    images = []
    points = []

    for i in pose_list:
        for j in i:
            if j[2] <= confidence:
                j[0],j[1] = np.nan, np.nan


    for i in pose_list:
        i = interpol(i)

    if use_video:
        capture = cv.VideoCapture(os.path.join(path_to_file,'videos',filename+video_format))

        if not capture.isOpened():
            raise Exception("Unable to open video file: {0}".format(os.path.join(path_to_file,'videos',filename+video_format)))

    for idx in tqdm.tqdm(range(frame_count), disable=not True, desc='Align frames'):

        if use_video:
            #Read frame
            try:
                ret, frame = capture.read()
                frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
                frame = frame - bg
                frame[frame <= 0] = 0
            except:
                print("Couldn't find a frame in capture.read(). #Frame: %d" %idx)
                continue
        else:
            frame=np.zeros((1,1))

        #Read coordinates and add border
        pose_list_bordered = []

        for i in pose_list:
            pose_list_bordered.append((int(i[idx][0]+crop_size[0]),int(i[idx][1]+crop_size[1])))

        img = cv.copyMakeBorder(frame, crop_size[1], crop_size[1], crop_size[0], crop_size[0], cv.BORDER_CONSTANT, 0)


        coord_center = []
        punkte = []

        for i in pose_ref_index:
            coord = []

            coord.append(pose_list_bordered[i][0]) # changed from pose_list_bordered[i][0] 2/28/2024 PN
            coord.append(pose_list_bordered[i][1]) # changed from pose_list_bordered[i][1] 2/28/2024 PN

            punkte.append(coord)


        # coord_center.append(pose_list_bordered[5][0]-5)
        # coord_center.append(pose_list_bordered[5][0]+5)

        # coord_center = [coord_center]
        punkte = [punkte]

        # coord_center = np.asarray(coord_center)
        punkte = np.asarray(punkte)

        #calculate minimal rectangle around snout and tail
        rect = cv.minAreaRect(punkte)
        # rect_belly = cv.minAreaRect(coord_center)

        # center_belly, size_belly, theta_belly = rect_belly

        #change size in rect tuple structure to be equal to crop_size
        lst = list(rect)
        lst[1] = crop_size
        # lst[0] = center_belly
        rect = tuple(lst)

        center, size, theta = rect

        # lst2 = list(rect)
        # lst2[0][0] = center[0] - size[0]//2
        # lst2[0][1] = center[1] - size[1]//2

        # rect = tuple(lst2)

        # center[0] -= size[0]//2
        # center[1] -= size[0]//2 # added this shift to change center to belly 2/28/2024

        #crop image
        out, shifted_points = crop_and_flip(rect, img,pose_list_bordered,pose_flip_ref)

        if use_video: #for memory optimization, just save images when video is used.
            images.append(out)
        points.append(shifted_points)

    if use_video:
        capture.release()

    time_series = np.zeros((len(pose_list)*2,frame_count))
    for i in range(frame_count):
        idx = 0
        for j in range(len(pose_list)):
            time_series[idx:idx+2,i] = points[i][j]
            idx += 2

    return images, points, time_series


def play_aligned_video(a: List[np.ndarray], n: List[List[np.ndarray]], frame_count: int) -> None:
    """
    Play the aligned video.

    Args:
        a (List[np.ndarray]): List of aligned images.
        n (List[List[np.ndarray]]): List of aligned DLC points.
        frame_count (int): Number of frames in the video.
    """
    colors = [(255,0,0),(0,255,0),(0,0,255),(255,255,0),(255,0,255),(0,255,255),(0,0,0),(255,255,255)]

    for i in range(frame_count):
        # Capture frame-by-frame
        ret, frame = True,a[i]
        if ret == True:

          # Display the resulting frame
          frame = cv.cvtColor(frame.astype('uint8')*255, cv.COLOR_GRAY2BGR)
          im_color = cv.applyColorMap(frame, cv.COLORMAP_JET)

          for c,j in enumerate(n[i]):
              cv.circle(im_color,(j[0], j[1]), 5, colors[c], -1)

          cv.imshow('Frame',im_color)

          # Press Q on keyboard to  exit
          if cv.waitKey(25) & 0xFF == ord('q'):
            break

        # Break the loop
        else:
            break
    cv.destroyAllWindows()


def alignment(
    path_to_file: str,
    filename: str,
    pose_ref_index: List[int],
    video_format: str,
    crop_size: Tuple[int, int],
    confidence: float,
    use_video: bool = False,
    check_video: bool = False
) -> Tuple[np.ndarray, List[np.ndarray]]:
    """
    Perform alignment of egocentric data.

    Args:
        path_to_file (str): Path to the file directory.
        filename (str): Name of the video file without the format.
        pose_ref_index (List[int]): Pose reference indices.
        video_format (str): Format of the video file.
        crop_size (Tuple[int, int]): Size to crop the video frames.
        confidence (float): Pose confidence threshold.
        use_video (bool, optional): Whether to use video for alignment. Defaults to False.
        check_video (bool, optional): Whether to check the aligned video. Defaults to False.

    Returns:
        Tuple[np.ndarray, List[np.ndarray]]: Aligned time series data and list of aligned frames.
    """

    #read out data
    data = pd.read_csv(os.path.join(path_to_file,'videos','pose_estimation',filename+'.csv'), skiprows = 2)
    data_mat = pd.DataFrame.to_numpy(data)
    data_mat = data_mat[:,1:]

    # get the coordinates for alignment from data table
    pose_list = []

    for i in range(int(data_mat.shape[1]/3)):
        pose_list.append(data_mat[:,i*3:(i+1)*3])

    #list of reference coordinate indices for alignment
    #0: snout, 1: forehand_left, 2: forehand_right,
    #3: hindleft, 4: hindright, 5: tail

    pose_ref_index = pose_ref_index

    #list of 2 reference coordinate indices for avoiding flipping
    pose_flip_ref = pose_ref_index

    if use_video:
        #compute background
        bg = background(path_to_file,filename,video_format)
        capture = cv.VideoCapture(os.path.join(path_to_file,'videos',filename+video_format))
        if not capture.isOpened():
            raise Exception("Unable to open video file: {0}".format(os.path.join(path_to_file,'videos',filename+video_format)))

        frame_count = int(capture.get(cv.CAP_PROP_FRAME_COUNT))
        capture.release()
    else:
        bg = 0
        frame_count = len(data) # Change this to an abitrary number if you first want to test the code


    frames, n, time_series = align_mouse(path_to_file, filename, video_format, crop_size, pose_list, pose_ref_index,
                                         confidence, pose_flip_ref, bg, frame_count, use_video)

    if check_video:
        play_aligned_video(frames, n, frame_count)

    return time_series, frames


def egocentric_alignment(
    config: str,
    pose_ref_index: list = [5,6],
    crop_size: tuple = (300,300),
    use_video: bool = False,
    video_format: str = '.mp4',
    check_video: bool = False,
    save_logs: bool = False
) -> None:
    """Aligns egocentric data for VAME training

    Args:
        config (str): Path for the project config file.
        pose_ref_index (list, optional): Pose reference index to be used to align. Defaults to [5,6].
        crop_size (tuple, optional): Size to crop the video. Defaults to (300,300).
        use_video (bool, optional): Weather to use video to do the post alignment. Defaults to False. # TODO check what to put in this docstring
        video_format (str, optional): Video format, can be .mp4 or .avi. Defaults to '.mp4'.
        check_video (bool, optional): Weather to check the video. Defaults to False.

    Raises:
        ValueError: If the config.yaml indicates that the data is not egocentric.
    """

    # pose_ref_index changed in this script from [0,5] to                                                                                                                                         [5,6] on 2/7/2024 PN
    """ Happy aligning """
    #config parameters

    try:
        redirect_stream = StreamToLogger()
        config_file = Path(config).resolve()
        cfg = read_config(config_file)

        if save_logs:
            log_filename_datetime = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_path = Path(cfg['project_path']) / 'logs' / 'align_egocentrical' / f'egocentric_alignment-{log_filename_datetime}.log'
            redirect_stream.add_file_handler(log_path)

        path_to_file = cfg['project_path']
        filename = cfg['video_sets']
        confidence = cfg['pose_confidence']
        num_features = cfg['num_features']
        video_format=video_format
        crop_size=crop_size

        y_shifted_indices = np.arange(0, num_features, 2)
        x_shifted_indices = np.arange(1, num_features, 2)
        belly_Y_ind = pose_ref_index[0] * 2
        belly_X_ind = (pose_ref_index[0] * 2) + 1

        if cfg['egocentric_data']:
            raise ValueError("The config.yaml indicates that the data is egocentric. Please check the parameter egocentric_data")

        # call function and save into your VAME data folder
        for file in filename:
            print("Aligning data %s, Pose confidence value: %.2f" %(file, confidence))
            egocentric_time_series, frames = alignment(path_to_file, file, pose_ref_index, video_format, crop_size,
                                                    confidence, use_video=use_video, check_video=check_video)

            # Shifiting section added 2/29/2024 PN
            egocentric_time_series_shifted = egocentric_time_series
            belly_Y_shift = egocentric_time_series[belly_Y_ind,:]
            belly_X_shift = egocentric_time_series[belly_X_ind,:]

            egocentric_time_series_shifted[y_shifted_indices, :] -= belly_Y_shift
            egocentric_time_series_shifted[x_shifted_indices, :] -= belly_X_shift

            np.save(os.path.join(path_to_file,'data',file,file+'-PE-seq.npy'), egocentric_time_series_shifted) # save new shifted file
    #        np.save(os.path.join(path_to_file,'data/',file,"",file+'-PE-seq.npy', egocentric_time_series))

        print("Your data is now ine right format and you can call vame.create_trainset()")
    except Exception as e:
        print(f"An error occurred: {e}")
        raise e
    finally:
        if save_logs:
            redirect_stream.stop()

