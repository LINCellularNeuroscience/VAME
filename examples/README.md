# Running the first demo (demo.py)

The first demo is a simple example of how to use the library.

## 1. Downloading the necessary resources:

To run the demo you will need vame package installed, follow the installation guide [here](/README.md#Installation) .
Also you will need two files to properly run the demo:
- `video-1.mp4`: A video file that will be used as input, for the demo you can download from [this link](https://drive.google.com/file/d/1w6OW9cN_-S30B7rOANvSaR9c3O5KeF0c/view)
- `video-1.csv`: the pose estimation results for the video file. You can use the video-1.csv file that is in the examples folder [video](/examples/video-1.csv)

## 2. Setting the demo variables
To start the demo you must define 4 variables. In order to do that, open the `demo.py` file and edit the following:

**The values below are just examples. You must set the variables according to your needs.**
```python
# The directory where the project will be saved
working_directory = './'

# The name you want for the project
project = 'first_vame_project'

# A list of paths to the videos file
videos =  ['./video-1.mp4']

# A list of paths to the poses estimations files.
# Important: The name (without the extension) of the video file and the pose estimation file must be the same. E.g. `video-1.mp4` and `video-1.csv`
poses_estimations = ['./video-1.csv']
```

## 3. Running the demo
After setting the variables, you can run the demo by running the following code:

```python
python demo.py
```

The demo will create a project folder in the working directory with the name you defined in the `project` variable and a date suffix, e.g: `first_name-May-9-2024`.

In this folder you can find a config file called `config.yaml` where you can set the parameters for the VAME algorithm. The videos and poses estimations files will be copied to the project videos folder. If everything is ok, the workflow will run and the logs will be displayed in your terminal. The image below shows the VAME workflow.

![demo workflow](/Images/vame-workflow-diagram.jpg)