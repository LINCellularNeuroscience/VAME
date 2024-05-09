# Running the first demo (demo.py)

The first demo is a simple example of how to use the library.

## 1. Downloading the necessary resources:

To run the demo you will need vame package installed, follow the installation guide [here](/README.md#Installation) .
Also you will need two files to properly run the demo:
- `video-1.mp4`: A video file that will be used as input, for the demo you can download from [this link](https://drive.google.com/file/d/1w6OW9cN_-S30B7rOANvSaR9c3O5KeF0c/view)
- `video-1.csv`: the pose estimation results for the video file. You can use the video-1.csv file that is in the examples folder [video](/examples/video-1.csv)

## 2. Setting the demo variables
To start the demo you must define 4 variables, being them:

```
working_directory = './' # The directory where the project will be saved
project = 'first_vame_project' # The name you want for the project
videos =  ['./video-1.mp4'] # A list of paths to the videos file
poses_estimations = ['./video-1.csv'] # A list of paths to the poses estimations files. **Important**: The name (without the extension) of the video file and the pose estimation file must be the same. E.g. `video-1.mp4` and `video-1.csv`
```


