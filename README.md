# camera-calibration
You can estimate the camera intrinsic parameters using checkerboard images taken from various different view points. 
Refer to the brief explanation [here](https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html).

## Install 
```bash
conda create -n camera-calib python=3.10 -y
conda activate camera-calib
pip install opencv-python tqdm pprintpp
```

## Data 
Put ```.jpg``` images into ```img_dir```. ```img_dir``` should be the same as the one going into calibrate_camera.py.

## Run 
```python
python calibrate_camera.py --visualize --img_dir ./img_dir --output_dir ./output_dir --dims (8,5)
```

## Output 
You can find the output json file in ```output_dir```. 
