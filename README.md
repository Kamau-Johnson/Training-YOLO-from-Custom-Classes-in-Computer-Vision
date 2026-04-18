## 3.4 Custom Objects with YOLO
<p float="center">
    <img src="YOLO5.PNG" width="50%" /> 
</p

```
Getting Started
```
We'll start by importing the packages we'll need in this notebook.  They're mostly the same as the ones we've used in the previous notebooks.


```python
import pathlib
import random
import shutil
import sys
import xml.etree.ElementTree as ET

import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import torch
import ultralytics
import yaml
from IPython import display
from PIL import Image
from tqdm.notebook import tqdm
from ultralytics import YOLO
```

In case we want to reproduce this notebook in the future, we'll record the version information. 


```python
print("Platform:", sys.platform)
print("Python version:", sys.version)
print("---")
print("matplotlib version:", plt.matplotlib.__version__)
print("pandas version:", pd.__version__)
print("PIL version:", Image.__version__)
print("PyYAML version:", yaml.__version__)
print("torch version:", torch.__version__)
print("ultralytics version:", ultralytics.__version__)
```

    Platform: linux
    Python version: 3.11.0 (main, Nov 15 2022, 20:12:54) [GCC 10.2.1 20210110]
    ---
    matplotlib version: 3.9.2
    pandas version: 2.2.3
    PIL version: 10.2.0
    PyYAML version: 6.0.2
    torch version: 2.2.2+cu121
    ultralytics version: 8.3.27


This notebook will require a GPU to run in any reasonable amount of time.  Check that the device is `cuda`.


```python
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print(f"Using {device} device.")
```

    Using cuda device.


### Data Transformation

Much of the work in this lesson will be getting all of the annotation data in the format that YOLO expects it.  Once that's done, we can relax while the computer runs many epochs of training.

In a previous lesson, we arranged the Dhaka AI data set in the `data_images` directory.  Here's the structure:

<div class="alert alert-info" role="alert">
    <p><b>Change with respect to the video</b></p>
    <p>You may notice that the number of files in the <code>data_images/train/</code> subdirectories is much lower than what is shown in the video. The video uses the full dataset, which contains <b>3003 files</b> each in the <code>images</code> and <code>annotations</code> directories.</p>
    <p>For performance reasons, this lab uses a smaller subset of only <b>61 files</b> in each of those directories. The files from the <code>test</code> directory have also been removed, as they are not used in this particular lab. These changes allow the exercises to run much faster and do not affect the concepts being taught. The subsequent labs contain the right number of files.</p>
</div>


```python
!tree data_images --filelimit=10
```

    [01;34mdata_images[00m
    └── [01;34mtrain[00m
        ├── [01;34mannotations[00m [61 entries exceeds filelimit, not opening dir]
        └── [01;34mimages[00m [61 entries exceeds filelimit, not opening dir]
    
    3 directories, 0 files


**Task 3.4.1:** Set the directories for the training images and annotations as variables.


```python
training_dir = pathlib.Path("data_images", "train")
images_dir = training_dir / "images"
annotations_dir = training_dir / "annotations"

print("Images     :", images_dir)
print("Annotations:", annotations_dir)
```

    Images     : data_images/train/images
    Annotations: data_images/train/annotations


Let's remind ourselves what the annotation XML files look like.  This next cell will print out the first 25 lines of the first annotation file.


```python
!head -n 25 $annotations_dir/01.xml
```

    <annotation>
    	<folder>Images</folder>
    	<filename>02_Motijheel_280714_0005.jpg</filename>
    	<path>E:\Datasets\Dataset\Images\02_Motijheel_280714_0005.jpg</path>
    	<source>
    		<database>Unknown</database>
    	</source>
    	<size>
    		<width>1200</width>
    		<height>800</height>
    		<depth>3</depth>
    	</size>
    	<segmented>0</segmented>
    	<object>
    		<name>bus</name>
    		<pose>Unspecified</pose>
    		<truncated>1</truncated>
    		<difficult>0</difficult>
    		<bndbox>
    			<xmin>833</xmin>
    			<ymin>390</ymin>
    			<xmax>1087</xmax>
    			<ymax>800</ymax>
    		</bndbox>
    	</object>


Each detected object is described in an `<object>` tag.  The `<name>` tag gives us the class of the object, and the `<bndbox>` gives the upper-left and lower-right corners  of the bounding box.  This is a sensible and readable format.

Unfortunately, this is not the format that YOLO needs.  The YOLO format is a text file, with each object being a line of the format
```
class_index x_center y_center width height
```
where `class_index` is a number assigned to class.  The bounding box is centered at (`x_center`, `y_center`), with a size of `width`$\times$`height`.  All of these dimensions are given as a fraction of the image size, rather than in pixels.  These are called _normalized_ coordinates.

Let's start by assigning the class indices.  We happen to know that these are the classes in this data set:


```python
classes = [
    "ambulance",
    "army vehicle",
    "auto rickshaw",
    "bicycle",
    "bus",
    "car",
    "garbagevan",
    "human hauler",
    "minibus",
    "minivan",
    "motorbike",
    "pickup",
    "policecar",
    "rickshaw",
    "scooter",
    "suv",
    "taxi",
    "three wheelers (CNG)",
    "truck",
    "van",
    "wheelbarrow",
]
```

<div class="alert alert-info" role="alert">
A puzzle for you to consider: If we didn't know that these were the classes in this data set, how would you have found this list?
</div>

**Task 3.4.2:** Generate a dictionary that maps the classes names to their indices.


```python
class_mapping = {cls: idx for idx, cls in enumerate(classes)}


print(class_mapping)
```

    {'ambulance': 0, 'army vehicle': 1, 'auto rickshaw': 2, 'bicycle': 3, 'bus': 4, 'car': 5, 'garbagevan': 6, 'human hauler': 7, 'minibus': 8, 'minivan': 9, 'motorbike': 10, 'pickup': 11, 'policecar': 12, 'rickshaw': 13, 'scooter': 14, 'suv': 15, 'taxi': 16, 'three wheelers (CNG)': 17, 'truck': 18, 'van': 19, 'wheelbarrow': 20}


Let's work out the bounding box transformation "by hand" first.  We'll use the object we saw above as an example.


```python
width = 1200
height = 800
xmin = 833
ymin = 390
xmax = 1087
ymax = 800
```

**Task 3.4.3:** Compute the center of the bounding box.  We take it to be halfway between the min and max values.  We also need to divide by the width or height as appropriate, to measure it in a fraction of the image size.


```python
x_center = (xmax + xmin) / 2 / width
y_center = (ymax + ymin) / 2 / height

print(f"Bounding box center: ({x_center}, {y_center})")
```

    Bounding box center: (0.8, 0.74375)


<div class="alert alert-info" role="alert">
Sanity check: Consider the $y$ coordinates.  Does this value for center make sense?
</div>

**Task 3.4.4:** Compute the width and height of the bounding box.  Again, measure it as a fraction of the width or height of the whole image.


```python
bb_width = (xmax - xmin) / width
bb_height = (ymax - ymin) / height

print(f"Bounding box size: {bb_width:0.3f} ⨯ {bb_height:0.3f}")
```

    Bounding box size: 0.212 ⨯ 0.512


<div class="alert alert-info" role="alert">
Sanity check: Consider the height.  Does this value make sense?
</div>

**Task 3.4.5:** Encapsulate this code in a function.


```python
def xml_to_yolo_bbox(bbox, width, height):
    """Convert the XML bounding box coordinates into YOLO format.

    Input:  bbox    The bounding box, defined as [xmin, ymin, xmax, ymax],
                    measured in pixels.
            width   The image width in pixels.
            height  The image height in pixels.

    Output: [x_center, y_center, bb_width, bb_height], where the bounding
            box is centered at (x_center, y_center) and is of size
            bb_width x bb_height.  All values are measured as a fraction
            of the image size."""

    xmin, ymin, xmax, ymax = bbox
    x_center = (xmax + xmin) / 2 / width
    y_center = (ymax + ymin) / 2 / height
    bb_width = (xmax - xmin) / width
    bb_height = (ymax - ymin) / height

    return [x_center, y_center, bb_width, bb_height]


xml_to_yolo_bbox([xmin, ymin, xmax, ymax], width, height)
```




    [0.8, 0.74375, 0.21166666666666667, 0.5125]



**Task 3.4.6:** Write a function to parse all of the objects in an XML file.  Much of the code comes from a previous lesson.  We need to add code to look up the class index from the class name.


```python
def parse_annotations(f):
    """Parse all of the objects in a given XML file to YOLO format.

    Input:  f      The path of the file to parse.

    Output: A list of objects in YOLO format.
            Each object is a list [index, x_center, y_center, width, height]."""

    objects = []

    tree = ET.parse(f)
    root = tree.getroot()
    width = int(root.find("size").find("width").text)
    height = int(root.find("size").find("height").text)

    for obj in root.findall("object"):
        label = obj.find("name").text
        class_id = class_mapping[label]
        bndbox = obj.find("bndbox")
        xmin = int(bndbox.find("xmin").text)
        ymin = int(bndbox.find("ymin").text)
        xmax = int(bndbox.find("xmax").text)
        ymax = int(bndbox.find("ymax").text)
        yolo_bbox = xml_to_yolo_bbox([xmin, ymin, xmax, ymax], width, height)

        objects.append([class_id] + yolo_bbox)

    return objects


objects = parse_annotations(annotations_dir / "01.xml")
print("First object:", objects[0])
```

    First object: [4, 0.8, 0.74375, 0.21166666666666667, 0.5125]



```python
classes[4]
```




    'bus'



**Task 3.4.7:** Write a function that outputs the YOLO objects in the text format.  Each object should be on its own line, with spaces between the components.


```python
def write_label(objects, filename):
    """Write the annotations to a file in the YOLO text format.

    Input:  objects   A list of YOLO objects, each a list of numbers.
            filename  The path to write the text file."""

    with open(filename, "w") as f:
        for obj in objects:
            # Write the object out as space-separated values
            f.write(" ".join(str(x) for x in obj))
            # Write a newline
            f.write("\n")
```

We'll test this on the objects we've parsed.  Check that the first line agrees with the values from the previous task.


```python
write_label(objects, "yolo_test.txt")
!head -n 1 yolo_test.txt
```

    4 0.8 0.74375 0.21166666666666667 0.5125


### Preparing the Directory Structure

We're now set up to convert all of the annotations into the correct format.  But where should we put them?  YOLO has a distinct file layout it expects, so we'll have to put both the formatted annotations and the images in new locations.

First, let's look at the types of images that we have.

**Task 3.4.8:** Determine the types of images in the training set, from their file extensions.


```python
f = list(images_dir.glob("*"))[0]
```


```python
f
```




    PosixPath('data_images/train/images/01.jpg')




```python
f.suffix
```




    '.jpg'




```python
set(f.suffix for f in images_dir.glob("*"))
```




    {'.JPG', '.jpg', '.png'}




```python
extensions = set(f.suffix for f in images_dir.glob("*"))

print(extensions)
```

    {'.jpg', '.png', '.JPG'}


This isn't technically a problem&mdash;YOLO can read PNG files as well as JPEG files.  But as a format, PNG is inefficient for photographs, compared to JPEG.  Additionally, some of these files are mildly corrupted.  YOLO can open them, but it'll print out warnings when it does.

To address both of these issues, we'll convert all of the images to RGB JPEG files before feeding them into YOLO.  The following function implements the conversion.


```python
def convert_image(fin, fout):
    """Open the image at `fin`, convert to a RGB JPEG, and save at `fout`."""
    Image.open(fin).convert("RGB").save(fout, "JPEG")
```

**Task 3.4.9:** Convert the PNG image `193.png` to a JPEG.  Then display the original and converted versions to check by eye that the image is the same.


```python
test_image = images_dir / "193.png"
convert_image(test_image, "test_image.jpg")

display.display(
    Image.open(images_dir / "193.png"),
    Image.open(test_image)  # Add path to the test JPEG

)
```


    
![png](output_49_0.png)
    



    
![png](output_49_1.png)
    


For training, YOLO expects a directory structure like so:
```
data_yolo
├── images
│   ├── train
│   └── val
└── labels
    ├── train
    └── val
```
You may find this surprising.  (Your author certainly did!)  The top-level split is between the images and labels, not between the training and validation sets.

**Task 3.4.10:** Create the directory structure for YOLO training.


```python
yolo_base = pathlib.Path("data_yolo")

# It's important to clear out the directory, if it already
# exists.  We'll get a different train / validation split
# each time, so we need to make sure the old images are
# cleared out.
shutil.rmtree(yolo_base, ignore_errors=True)

(yolo_base / "images" / "train").mkdir(parents=True)
# Create the remaining directories.
(yolo_base / "images" / "val").mkdir(parents=True)
(yolo_base / "labels" / "train").mkdir(parents=True)
(yolo_base / "labels" / "val").mkdir(parents=True)
!tree $yolo_base
```

    [01;34mdata_yolo[00m
    ├── [01;34mimages[00m
    │   ├── [01;34mtrain[00m
    │   └── [01;34mval[00m
    └── [01;34mlabels[00m
        ├── [01;34mtrain[00m
        └── [01;34mval[00m
    
    6 directories, 0 files


With the directory structure in place, the code below will iterate through all of the images in the original `images_dir`.  It will randomly assign each image either to the training or validation split.  It will parse the annotations file and write it to the correct subdirectory in `data_yolo/labels`.  And it will convert the image into a JPEG and write it to the correct subdirectory in `data_yolo/images`.

Before we run this, and indeed before processing any large set of data, it's worth considering what might go wrong.  We don't want to get 95% of the way through this process and run into an error that forces us to restart.

In this case, the `parse_annotations` function causes us the most worry.  There's all sorts of things that can go wrong with XML parsing, and we haven't written any checks for that yet.

What should we do if we hit an error?  If there are only a few errors in a large data set, that's generally not worth the effort.  We should just skip any files we can't process.  We'll check at the end that we only got a few failures.

**Task 3.4.11:** Add error handling around the `parse_annotations` function call.


```python
train_frac = 0.8
images = list(images_dir.glob("*"))

for img in tqdm(images):
    split = "train" if random.random() < train_frac else "val"

    annotation = annotations_dir / f"{img.stem}.xml"
    # This might raise an error:
    try:
        parsed = parse_annotations(annotation)
    except Exception as e:
        print(f'Failed to parse "{img.stem}". Skipping.')
        print(e)
        continue
        
    parsed = parse_annotations(annotation)
    
    dest = yolo_base / "labels" / split / f"{img.stem}.txt"
    write_label(parsed, dest)

    dest = yolo_base / "images" / split / f"{img.stem}.jpg"
    convert_image(img, dest)
```


      0%|          | 0/61 [00:00<?, ?it/s]


<div class="alert alert-info" role="alert">
    <p><b>Note on Error Messages Shown in the Video</b></p>
    <p>If you are following the video, you might expect this code ☝️ to produce a few errors. However, your code will run without any, and this is the correct outcome.</p>
    <p>To improve lab performance, we have removed some files that include those files that caused the <code>XML parsing</code> and <code>division-by-zero</code> errors. While you won't encounter these errors yourself, the original lesson about the importance of handling exceptions is still a key takeaway.</p>
</div>

**Task 3.4.12:** Check that we ended up with the expected 80/20 split between training and validation.


```python
train_count = len(list((yolo_base / "images" / "train").glob("*")))
val_count = len(list((yolo_base / "images" / "val").glob("*")))
total_count = train_count + val_count

print(f"Training fraction:   {train_count/total_count:0.3f}")
print(f"Validation fraction: {val_count/total_count:0.3f}")
```

    Training fraction:   0.685
    Validation fraction: 0.315


<div class="alert alert-info" role="alert">
This technique of randomly assigning each item to a split won't give the exact ratio, but it should be pretty close.
</div>

### Training the Model

The data for training a YOLO model needs to be described in a YAML file.  YAML is a structured document format, somewhat like JSON.  (In fact, YAML is a superset of JSON, so any JSON document is also a valid YAML file.)  Python dictionaries, lists, strings, and numbers map naturally to YAML constructions, so we'll start by defining a dictionary with the necessary keys.

**Task 3.4.13:** Create a dictionary with the appropriate keys for a YOLO data set.


```python
metadata = {
    "path": str(
        yolo_base.absolute()
    ),  # It's easier to specify absolute paths with YOLO.
    
    "train": "images/train", # Training images, relative to above.
    
    "val": "images/val", # Validation images
    
    "names": classes, # Class names, as a list
    
    "nc": len(classes), # Number of classes
}

print(metadata)
```

    {'path': '/app/data_yolo', 'train': 'images/train', 'val': 'images/val', 'names': ['ambulance', 'army vehicle', 'auto rickshaw', 'bicycle', 'bus', 'car', 'garbagevan', 'human hauler', 'minibus', 'minivan', 'motorbike', 'pickup', 'policecar', 'rickshaw', 'scooter', 'suv', 'taxi', 'three wheelers (CNG)', 'truck', 'van', 'wheelbarrow'], 'nc': 21}


**Task 3.4.14:** Write the YAML file to disk.


```python
yolo_config = "data.yaml"
# Using yaml.safe_dump() protects you from some oddities in the YAML format.
# It takes the same arguments as json.dump().
yaml.safe_dump(metadata, open(yolo_config, 'w'))



!cat data.yaml
```

    names:
    - ambulance
    - army vehicle
    - auto rickshaw
    - bicycle
    - bus
    - car
    - garbagevan
    - human hauler
    - minibus
    - minivan
    - motorbike
    - pickup
    - policecar
    - rickshaw
    - scooter
    - suv
    - taxi
    - three wheelers (CNG)
    - truck
    - van
    - wheelbarrow
    nc: 21
    path: /app/data_yolo
    train: images/train
    val: images/val


In a previous lesson, we used to `YOLOv8s` (*s* for small) model for object detection.  This time, we'll use the `YOLOv8n` (*n* for nano) model.  The nano model is less than 30% of the size of the small model, but it still manages 80% of the small model's performance.  This'll cut down the training time without hurting performance too much.


```python
model = YOLO("yolov8n.pt")

print(model)
```

    YOLO(
      (model): DetectionModel(
        (model): Sequential(
          (0): Conv(
            (conv): Conv2d(3, 16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
            (bn): BatchNorm2d(16, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
            (act): SiLU(inplace=True)
          )
          (1): Conv(
            (conv): Conv2d(16, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
            (bn): BatchNorm2d(32, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
            (act): SiLU(inplace=True)
          )
          (2): C2f(
            (cv1): Conv(
              (conv): Conv2d(32, 32, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(32, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
              (act): SiLU(inplace=True)
            )
            (cv2): Conv(
              (conv): Conv2d(48, 32, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(32, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
              (act): SiLU(inplace=True)
            )
            (m): ModuleList(
              (0): Bottleneck(
                (cv1): Conv(
                  (conv): Conv2d(16, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                  (bn): BatchNorm2d(16, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
                  (act): SiLU(inplace=True)
                )
                (cv2): Conv(
                  (conv): Conv2d(16, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                  (bn): BatchNorm2d(16, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
                  (act): SiLU(inplace=True)
                )
              )
            )
          )
          (3): Conv(
            (conv): Conv2d(32, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
            (bn): BatchNorm2d(64, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
            (act): SiLU(inplace=True)
          )
          (4): C2f(
            (cv1): Conv(
              (conv): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(64, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
              (act): SiLU(inplace=True)
            )
            (cv2): Conv(
              (conv): Conv2d(128, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(64, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
              (act): SiLU(inplace=True)
            )
            (m): ModuleList(
              (0-1): 2 x Bottleneck(
                (cv1): Conv(
                  (conv): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                  (bn): BatchNorm2d(32, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
                  (act): SiLU(inplace=True)
                )
                (cv2): Conv(
                  (conv): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                  (bn): BatchNorm2d(32, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
                  (act): SiLU(inplace=True)
                )
              )
            )
          )
          (5): Conv(
            (conv): Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
            (bn): BatchNorm2d(128, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
            (act): SiLU(inplace=True)
          )
          (6): C2f(
            (cv1): Conv(
              (conv): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(128, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
              (act): SiLU(inplace=True)
            )
            (cv2): Conv(
              (conv): Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(128, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
              (act): SiLU(inplace=True)
            )
            (m): ModuleList(
              (0-1): 2 x Bottleneck(
                (cv1): Conv(
                  (conv): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                  (bn): BatchNorm2d(64, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
                  (act): SiLU(inplace=True)
                )
                (cv2): Conv(
                  (conv): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                  (bn): BatchNorm2d(64, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
                  (act): SiLU(inplace=True)
                )
              )
            )
          )
          (7): Conv(
            (conv): Conv2d(128, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
            (bn): BatchNorm2d(256, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
            (act): SiLU(inplace=True)
          )
          (8): C2f(
            (cv1): Conv(
              (conv): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(256, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
              (act): SiLU(inplace=True)
            )
            (cv2): Conv(
              (conv): Conv2d(384, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(256, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
              (act): SiLU(inplace=True)
            )
            (m): ModuleList(
              (0): Bottleneck(
                (cv1): Conv(
                  (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                  (bn): BatchNorm2d(128, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
                  (act): SiLU(inplace=True)
                )
                (cv2): Conv(
                  (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                  (bn): BatchNorm2d(128, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
                  (act): SiLU(inplace=True)
                )
              )
            )
          )
          (9): SPPF(
            (cv1): Conv(
              (conv): Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(128, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
              (act): SiLU(inplace=True)
            )
            (cv2): Conv(
              (conv): Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(256, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
              (act): SiLU(inplace=True)
            )
            (m): MaxPool2d(kernel_size=5, stride=1, padding=2, dilation=1, ceil_mode=False)
          )
          (10): Upsample(scale_factor=2.0, mode='nearest')
          (11): Concat()
          (12): C2f(
            (cv1): Conv(
              (conv): Conv2d(384, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(128, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
              (act): SiLU(inplace=True)
            )
            (cv2): Conv(
              (conv): Conv2d(192, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(128, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
              (act): SiLU(inplace=True)
            )
            (m): ModuleList(
              (0): Bottleneck(
                (cv1): Conv(
                  (conv): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                  (bn): BatchNorm2d(64, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
                  (act): SiLU(inplace=True)
                )
                (cv2): Conv(
                  (conv): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                  (bn): BatchNorm2d(64, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
                  (act): SiLU(inplace=True)
                )
              )
            )
          )
          (13): Upsample(scale_factor=2.0, mode='nearest')
          (14): Concat()
          (15): C2f(
            (cv1): Conv(
              (conv): Conv2d(192, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(64, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
              (act): SiLU(inplace=True)
            )
            (cv2): Conv(
              (conv): Conv2d(96, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(64, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
              (act): SiLU(inplace=True)
            )
            (m): ModuleList(
              (0): Bottleneck(
                (cv1): Conv(
                  (conv): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                  (bn): BatchNorm2d(32, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
                  (act): SiLU(inplace=True)
                )
                (cv2): Conv(
                  (conv): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                  (bn): BatchNorm2d(32, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
                  (act): SiLU(inplace=True)
                )
              )
            )
          )
          (16): Conv(
            (conv): Conv2d(64, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
            (bn): BatchNorm2d(64, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
            (act): SiLU(inplace=True)
          )
          (17): Concat()
          (18): C2f(
            (cv1): Conv(
              (conv): Conv2d(192, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(128, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
              (act): SiLU(inplace=True)
            )
            (cv2): Conv(
              (conv): Conv2d(192, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(128, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
              (act): SiLU(inplace=True)
            )
            (m): ModuleList(
              (0): Bottleneck(
                (cv1): Conv(
                  (conv): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                  (bn): BatchNorm2d(64, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
                  (act): SiLU(inplace=True)
                )
                (cv2): Conv(
                  (conv): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                  (bn): BatchNorm2d(64, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
                  (act): SiLU(inplace=True)
                )
              )
            )
          )
          (19): Conv(
            (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
            (bn): BatchNorm2d(128, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
            (act): SiLU(inplace=True)
          )
          (20): Concat()
          (21): C2f(
            (cv1): Conv(
              (conv): Conv2d(384, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(256, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
              (act): SiLU(inplace=True)
            )
            (cv2): Conv(
              (conv): Conv2d(384, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(256, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
              (act): SiLU(inplace=True)
            )
            (m): ModuleList(
              (0): Bottleneck(
                (cv1): Conv(
                  (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                  (bn): BatchNorm2d(128, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
                  (act): SiLU(inplace=True)
                )
                (cv2): Conv(
                  (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                  (bn): BatchNorm2d(128, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
                  (act): SiLU(inplace=True)
                )
              )
            )
          )
          (22): Detect(
            (cv2): ModuleList(
              (0): Sequential(
                (0): Conv(
                  (conv): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                  (bn): BatchNorm2d(64, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
                  (act): SiLU(inplace=True)
                )
                (1): Conv(
                  (conv): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                  (bn): BatchNorm2d(64, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
                  (act): SiLU(inplace=True)
                )
                (2): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1))
              )
              (1): Sequential(
                (0): Conv(
                  (conv): Conv2d(128, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                  (bn): BatchNorm2d(64, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
                  (act): SiLU(inplace=True)
                )
                (1): Conv(
                  (conv): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                  (bn): BatchNorm2d(64, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
                  (act): SiLU(inplace=True)
                )
                (2): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1))
              )
              (2): Sequential(
                (0): Conv(
                  (conv): Conv2d(256, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                  (bn): BatchNorm2d(64, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
                  (act): SiLU(inplace=True)
                )
                (1): Conv(
                  (conv): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                  (bn): BatchNorm2d(64, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
                  (act): SiLU(inplace=True)
                )
                (2): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1))
              )
            )
            (cv3): ModuleList(
              (0): Sequential(
                (0): Conv(
                  (conv): Conv2d(64, 80, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                  (bn): BatchNorm2d(80, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
                  (act): SiLU(inplace=True)
                )
                (1): Conv(
                  (conv): Conv2d(80, 80, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                  (bn): BatchNorm2d(80, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
                  (act): SiLU(inplace=True)
                )
                (2): Conv2d(80, 80, kernel_size=(1, 1), stride=(1, 1))
              )
              (1): Sequential(
                (0): Conv(
                  (conv): Conv2d(128, 80, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                  (bn): BatchNorm2d(80, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
                  (act): SiLU(inplace=True)
                )
                (1): Conv(
                  (conv): Conv2d(80, 80, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                  (bn): BatchNorm2d(80, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
                  (act): SiLU(inplace=True)
                )
                (2): Conv2d(80, 80, kernel_size=(1, 1), stride=(1, 1))
              )
              (2): Sequential(
                (0): Conv(
                  (conv): Conv2d(256, 80, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                  (bn): BatchNorm2d(80, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
                  (act): SiLU(inplace=True)
                )
                (1): Conv(
                  (conv): Conv2d(80, 80, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                  (bn): BatchNorm2d(80, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
                  (act): SiLU(inplace=True)
                )
                (2): Conv2d(80, 80, kernel_size=(1, 1), stride=(1, 1))
              )
            )
            (dfl): DFL(
              (conv): Conv2d(16, 1, kernel_size=(1, 1), stride=(1, 1), bias=False)
            )
          )
        )
      )
    )


**Task 3.4.15:** Load the pre-trained model

Load the model that is located at `runs/detect/train/weights/last.pt`.

<div class="alert alert-info" role="alert"> <strong>Regarding Model Training Times</strong>

This task involves training the model for more 30 epochs, which might take a long time and your lab might run out of resources. Instead of training it from scratch, load the pre-trained model using the cells below.

<b>Loading the model instead of training it is the only way to pass the activity.</b>
</div>


```python
model = YOLO("runs/detect/train/weights/last.pt")
```

<div class="alert alert-info" role="alert">
    <p><b>Changes with respect to the video</b></p>
<p>The instructor in the video is using <code>results.save_dir</code> to explore the results of the train process, but as we have loaded the model instead of training, we'll need to define a new variable <code>save_dir</code>.</p>
    <p>From now on, any reference that you see to <code>results.save_dir</code> should be replaced with <code>save_dir</code>.</p>
</div>


```python
# Don't change this
save_dir = Path("runs/detect/train")
```

### Evaluating the Model

During the training process, various diagnostic information is saved to disk.  This is easier to understand than all of that output above.  You can see the directory printed out at the end of the output.  (It will be `runs/detect/train` the first time you run that cell, but will change the next time.)  The directory is also available as a property of the `results` object returned by the `train` function.


```python
print(save_dir)
```

    runs/detect/train


We can use the `tree` command to display the contents of the save directory.


```python
!tree $save_dir
```

    [01;34mruns/detect/train[00m
    ├── F1_curve.png
    ├── PR_curve.png
    ├── P_curve.png
    ├── R_curve.png
    ├── args.yaml
    ├── confusion_matrix.png
    ├── confusion_matrix_normalized.png
    ├── [01;35mlabels.jpg[00m
    ├── [01;35mlabels_correlogram.jpg[00m
    ├── results.csv
    ├── results.png
    ├── [01;35mtrain_batch0.jpg[00m
    ├── [01;35mtrain_batch1.jpg[00m
    ├── [01;35mtrain_batch2.jpg[00m
    ├── [01;35mtrain_batch5920.jpg[00m
    ├── [01;35mtrain_batch5921.jpg[00m
    ├── [01;35mtrain_batch5922.jpg[00m
    ├── [01;35mval_batch0_labels.jpg[00m
    ├── [01;35mval_batch0_pred.jpg[00m
    ├── [01;35mval_batch1_labels.jpg[00m
    ├── [01;35mval_batch1_pred.jpg[00m
    ├── [01;35mval_batch2_labels.jpg[00m
    ├── [01;35mval_batch2_pred.jpg[00m
    └── [01;34mweights[00m
        ├── best.pt
        └── last.pt
    
    1 directory, 25 files


**Task 3.4.16:** Display and examine the precision-recall curves for the model.  They are plotted in `PR_curve.png`.  Remember that the more area under the curve, the better the model is performing.  Which classes does the model do well at detecting?


```python
Image.open(save_dir / "PR_curve.png")
```




    
![png](output_80_0.png)
    



**Task 3.4.17:** Display the confusion matrix, which is already plotted for us in `confusion_matrix_normalized.png`.  Note which classes the model is most effective on.  Are they the same as those suggested by the precision-recall curves?


```python
Image.open(save_dir / "confusion_matrix_normalized.png")
```




    
![png](output_82_0.png)
    



Data about the training process are available in the `results.csv` file.  We can load it in with pandas, although we need a few options to make the result look nice.


```python
df = pd.read_csv(save_dir / "results.csv", skipinitialspace=True).set_index(
    "epoch"
)
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>time</th>
      <th>train/box_loss</th>
      <th>train/cls_loss</th>
      <th>train/dfl_loss</th>
      <th>metrics/precision(B)</th>
      <th>metrics/recall(B)</th>
      <th>metrics/mAP50(B)</th>
      <th>metrics/mAP50-95(B)</th>
      <th>val/box_loss</th>
      <th>val/cls_loss</th>
      <th>val/dfl_loss</th>
      <th>lr/pg0</th>
      <th>lr/pg1</th>
      <th>lr/pg2</th>
    </tr>
    <tr>
      <th>epoch</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>35.1072</td>
      <td>1.45223</td>
      <td>3.32406</td>
      <td>1.16199</td>
      <td>0.59554</td>
      <td>0.11979</td>
      <td>0.10365</td>
      <td>0.06298</td>
      <td>1.36083</td>
      <td>2.15096</td>
      <td>1.11174</td>
      <td>0.000133</td>
      <td>0.000133</td>
      <td>0.000133</td>
    </tr>
    <tr>
      <th>2</th>
      <td>67.5593</td>
      <td>1.42356</td>
      <td>2.24330</td>
      <td>1.17132</td>
      <td>0.67236</td>
      <td>0.15496</td>
      <td>0.13611</td>
      <td>0.08650</td>
      <td>1.33130</td>
      <td>1.89784</td>
      <td>1.10711</td>
      <td>0.000257</td>
      <td>0.000257</td>
      <td>0.000257</td>
    </tr>
    <tr>
      <th>3</th>
      <td>99.9395</td>
      <td>1.38437</td>
      <td>2.04035</td>
      <td>1.16109</td>
      <td>0.56155</td>
      <td>0.18055</td>
      <td>0.16316</td>
      <td>0.10168</td>
      <td>1.30970</td>
      <td>1.70307</td>
      <td>1.10191</td>
      <td>0.000373</td>
      <td>0.000373</td>
      <td>0.000373</td>
    </tr>
    <tr>
      <th>4</th>
      <td>131.5240</td>
      <td>1.36695</td>
      <td>1.91453</td>
      <td>1.15004</td>
      <td>0.60072</td>
      <td>0.19290</td>
      <td>0.17265</td>
      <td>0.10846</td>
      <td>1.28410</td>
      <td>1.66555</td>
      <td>1.08647</td>
      <td>0.000360</td>
      <td>0.000360</td>
      <td>0.000360</td>
    </tr>
    <tr>
      <th>5</th>
      <td>163.3040</td>
      <td>1.32782</td>
      <td>1.81436</td>
      <td>1.12958</td>
      <td>0.55458</td>
      <td>0.21755</td>
      <td>0.19436</td>
      <td>0.12272</td>
      <td>1.25971</td>
      <td>1.53956</td>
      <td>1.07159</td>
      <td>0.000347</td>
      <td>0.000347</td>
      <td>0.000347</td>
    </tr>
  </tbody>
</table>
</div>



**Task 3.4.18:** Plot the classification loss over time for both the training and validation data.  Comparing these curves helps us understand if we are overfitting.

<div class="alert alert-info" role="alert">
<p>Don't see the plot?  Run the code below, and then try the plotting cell again.</p>
<p><pre><code>plt.close('all')
plt.switch_backend('module://matplotlib_inline.backend_inline')</code>
</pre>
<p>This happens with YOLO experiences an error while generating its plots.  It doesn't reset the matplotlib settings correctly.  The code above sets things right again.
</div>


```python
%matplotlib
```

    Using matplotlib backend: module://matplotlib_inline.backend_inline



```python
# The `.plot` method on DataFrames may be useful.
df[["train/cls_loss", "val/cls_loss"]].plot(marker='.');
```


    
![png](output_88_0.png)
    


Snapshots of the model are saved in the `weights` subdirectory.  The weights for the last training epoch as well as the best performing epoch, as measured by the validation data, are saved.  This allows us to load the trained model to make predictions.

**Task 3.4.19:** Load the weights that gave the best performance.


```python
saved_model = YOLO(save_dir / "weights" / "best.pt")

print(saved_model)
```

    YOLO(
      (model): DetectionModel(
        (model): Sequential(
          (0): Conv(
            (conv): Conv2d(3, 16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
            (bn): BatchNorm2d(16, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
            (act): SiLU(inplace=True)
          )
          (1): Conv(
            (conv): Conv2d(16, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
            (bn): BatchNorm2d(32, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
            (act): SiLU(inplace=True)
          )
          (2): C2f(
            (cv1): Conv(
              (conv): Conv2d(32, 32, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(32, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
              (act): SiLU(inplace=True)
            )
            (cv2): Conv(
              (conv): Conv2d(48, 32, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(32, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
              (act): SiLU(inplace=True)
            )
            (m): ModuleList(
              (0): Bottleneck(
                (cv1): Conv(
                  (conv): Conv2d(16, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                  (bn): BatchNorm2d(16, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
                  (act): SiLU(inplace=True)
                )
                (cv2): Conv(
                  (conv): Conv2d(16, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                  (bn): BatchNorm2d(16, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
                  (act): SiLU(inplace=True)
                )
              )
            )
          )
          (3): Conv(
            (conv): Conv2d(32, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
            (bn): BatchNorm2d(64, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
            (act): SiLU(inplace=True)
          )
          (4): C2f(
            (cv1): Conv(
              (conv): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(64, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
              (act): SiLU(inplace=True)
            )
            (cv2): Conv(
              (conv): Conv2d(128, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(64, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
              (act): SiLU(inplace=True)
            )
            (m): ModuleList(
              (0-1): 2 x Bottleneck(
                (cv1): Conv(
                  (conv): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                  (bn): BatchNorm2d(32, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
                  (act): SiLU(inplace=True)
                )
                (cv2): Conv(
                  (conv): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                  (bn): BatchNorm2d(32, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
                  (act): SiLU(inplace=True)
                )
              )
            )
          )
          (5): Conv(
            (conv): Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
            (bn): BatchNorm2d(128, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
            (act): SiLU(inplace=True)
          )
          (6): C2f(
            (cv1): Conv(
              (conv): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(128, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
              (act): SiLU(inplace=True)
            )
            (cv2): Conv(
              (conv): Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(128, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
              (act): SiLU(inplace=True)
            )
            (m): ModuleList(
              (0-1): 2 x Bottleneck(
                (cv1): Conv(
                  (conv): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                  (bn): BatchNorm2d(64, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
                  (act): SiLU(inplace=True)
                )
                (cv2): Conv(
                  (conv): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                  (bn): BatchNorm2d(64, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
                  (act): SiLU(inplace=True)
                )
              )
            )
          )
          (7): Conv(
            (conv): Conv2d(128, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
            (bn): BatchNorm2d(256, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
            (act): SiLU(inplace=True)
          )
          (8): C2f(
            (cv1): Conv(
              (conv): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(256, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
              (act): SiLU(inplace=True)
            )
            (cv2): Conv(
              (conv): Conv2d(384, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(256, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
              (act): SiLU(inplace=True)
            )
            (m): ModuleList(
              (0): Bottleneck(
                (cv1): Conv(
                  (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                  (bn): BatchNorm2d(128, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
                  (act): SiLU(inplace=True)
                )
                (cv2): Conv(
                  (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                  (bn): BatchNorm2d(128, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
                  (act): SiLU(inplace=True)
                )
              )
            )
          )
          (9): SPPF(
            (cv1): Conv(
              (conv): Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(128, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
              (act): SiLU(inplace=True)
            )
            (cv2): Conv(
              (conv): Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(256, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
              (act): SiLU(inplace=True)
            )
            (m): MaxPool2d(kernel_size=5, stride=1, padding=2, dilation=1, ceil_mode=False)
          )
          (10): Upsample(scale_factor=2.0, mode='nearest')
          (11): Concat()
          (12): C2f(
            (cv1): Conv(
              (conv): Conv2d(384, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(128, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
              (act): SiLU(inplace=True)
            )
            (cv2): Conv(
              (conv): Conv2d(192, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(128, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
              (act): SiLU(inplace=True)
            )
            (m): ModuleList(
              (0): Bottleneck(
                (cv1): Conv(
                  (conv): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                  (bn): BatchNorm2d(64, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
                  (act): SiLU(inplace=True)
                )
                (cv2): Conv(
                  (conv): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                  (bn): BatchNorm2d(64, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
                  (act): SiLU(inplace=True)
                )
              )
            )
          )
          (13): Upsample(scale_factor=2.0, mode='nearest')
          (14): Concat()
          (15): C2f(
            (cv1): Conv(
              (conv): Conv2d(192, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(64, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
              (act): SiLU(inplace=True)
            )
            (cv2): Conv(
              (conv): Conv2d(96, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(64, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
              (act): SiLU(inplace=True)
            )
            (m): ModuleList(
              (0): Bottleneck(
                (cv1): Conv(
                  (conv): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                  (bn): BatchNorm2d(32, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
                  (act): SiLU(inplace=True)
                )
                (cv2): Conv(
                  (conv): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                  (bn): BatchNorm2d(32, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
                  (act): SiLU(inplace=True)
                )
              )
            )
          )
          (16): Conv(
            (conv): Conv2d(64, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
            (bn): BatchNorm2d(64, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
            (act): SiLU(inplace=True)
          )
          (17): Concat()
          (18): C2f(
            (cv1): Conv(
              (conv): Conv2d(192, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(128, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
              (act): SiLU(inplace=True)
            )
            (cv2): Conv(
              (conv): Conv2d(192, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(128, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
              (act): SiLU(inplace=True)
            )
            (m): ModuleList(
              (0): Bottleneck(
                (cv1): Conv(
                  (conv): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                  (bn): BatchNorm2d(64, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
                  (act): SiLU(inplace=True)
                )
                (cv2): Conv(
                  (conv): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                  (bn): BatchNorm2d(64, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
                  (act): SiLU(inplace=True)
                )
              )
            )
          )
          (19): Conv(
            (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
            (bn): BatchNorm2d(128, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
            (act): SiLU(inplace=True)
          )
          (20): Concat()
          (21): C2f(
            (cv1): Conv(
              (conv): Conv2d(384, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(256, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
              (act): SiLU(inplace=True)
            )
            (cv2): Conv(
              (conv): Conv2d(384, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(256, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
              (act): SiLU(inplace=True)
            )
            (m): ModuleList(
              (0): Bottleneck(
                (cv1): Conv(
                  (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                  (bn): BatchNorm2d(128, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
                  (act): SiLU(inplace=True)
                )
                (cv2): Conv(
                  (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                  (bn): BatchNorm2d(128, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
                  (act): SiLU(inplace=True)
                )
              )
            )
          )
          (22): Detect(
            (cv2): ModuleList(
              (0): Sequential(
                (0): Conv(
                  (conv): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                  (bn): BatchNorm2d(64, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
                  (act): SiLU(inplace=True)
                )
                (1): Conv(
                  (conv): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                  (bn): BatchNorm2d(64, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
                  (act): SiLU(inplace=True)
                )
                (2): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1))
              )
              (1): Sequential(
                (0): Conv(
                  (conv): Conv2d(128, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                  (bn): BatchNorm2d(64, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
                  (act): SiLU(inplace=True)
                )
                (1): Conv(
                  (conv): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                  (bn): BatchNorm2d(64, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
                  (act): SiLU(inplace=True)
                )
                (2): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1))
              )
              (2): Sequential(
                (0): Conv(
                  (conv): Conv2d(256, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                  (bn): BatchNorm2d(64, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
                  (act): SiLU(inplace=True)
                )
                (1): Conv(
                  (conv): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                  (bn): BatchNorm2d(64, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
                  (act): SiLU(inplace=True)
                )
                (2): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1))
              )
            )
            (cv3): ModuleList(
              (0): Sequential(
                (0): Conv(
                  (conv): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                  (bn): BatchNorm2d(64, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
                  (act): SiLU(inplace=True)
                )
                (1): Conv(
                  (conv): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                  (bn): BatchNorm2d(64, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
                  (act): SiLU(inplace=True)
                )
                (2): Conv2d(64, 21, kernel_size=(1, 1), stride=(1, 1))
              )
              (1): Sequential(
                (0): Conv(
                  (conv): Conv2d(128, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                  (bn): BatchNorm2d(64, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
                  (act): SiLU(inplace=True)
                )
                (1): Conv(
                  (conv): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                  (bn): BatchNorm2d(64, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
                  (act): SiLU(inplace=True)
                )
                (2): Conv2d(64, 21, kernel_size=(1, 1), stride=(1, 1))
              )
              (2): Sequential(
                (0): Conv(
                  (conv): Conv2d(256, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                  (bn): BatchNorm2d(64, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
                  (act): SiLU(inplace=True)
                )
                (1): Conv(
                  (conv): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                  (bn): BatchNorm2d(64, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
                  (act): SiLU(inplace=True)
                )
                (2): Conv2d(64, 21, kernel_size=(1, 1), stride=(1, 1))
              )
            )
            (dfl): DFL(
              (conv): Conv2d(16, 1, kernel_size=(1, 1), stride=(1, 1), bias=False)
            )
          )
        )
      )
    )


<div class="alert alert-info" role="alert">
Each time you run the predict function loop, YOLO writes out information to <code>runs/detect/predict##</code>.  This can be useful if you want to assemble many results.  If you want to clean up all the old data, run: <code>!rm -r /runs/detect/predict*</code>
</div>

**Task 3.4.20:** Conduct object detection on frame 600 that was extracted from the video.


```python
predict_results = saved_model.predict(
    "data_video/extracted_frames/frame_600.jpg", # Path to an image

    # Only return objects detected at this confidence level or higher
    conf=0.5,
    # Save output to disk
    save=True,
)

f"Results type: {type(predict_results)}, length {len(predict_results)}"
```

    
    image 1/1 /app/data_video/extracted_frames/frame_600.jpg: 384x640 7 cars, 1 suv, 3 three wheelers (CNG)s, 108.1ms
    Speed: 2.6ms preprocess, 108.1ms inference, 218.9ms postprocess per image at shape (1, 3, 384, 640)
    Results saved to [1mruns/detect/predict[0m





    "Results type: <class 'list'>, length 1"



This gives us a list with a single element.  The bounding boxes detected are listed in the `boxes` attribute.


```python
predict_results[0].boxes
```




    ultralytics.engine.results.Boxes object with attributes:
    
    cls: tensor([ 5.,  5., 17.,  5., 17.,  5., 15., 17.,  5.,  5.,  5.], device='cuda:0')
    conf: tensor([0.9604, 0.9439, 0.9421, 0.8939, 0.7899, 0.7891, 0.7002, 0.6514, 0.6015, 0.5221, 0.5190], device='cuda:0')
    data: tensor([[1.7001e-01, 2.2476e+02, 1.6071e+02, 3.5945e+02, 9.6036e-01, 5.0000e+00],
            [2.6941e+02, 1.5886e+02, 4.6822e+02, 2.9230e+02, 9.4386e-01, 5.0000e+00],
            [5.6848e+02, 1.4075e+02, 5.9965e+02, 1.7218e+02, 9.4212e-01, 1.7000e+01],
            [5.0122e+02, 1.4614e+02, 5.6323e+02, 1.8404e+02, 8.9386e-01, 5.0000e+00],
            [3.6133e+02, 1.2966e+02, 3.8017e+02, 1.4575e+02, 7.8993e-01, 1.7000e+01],
            [4.3392e+02, 1.4806e+02, 4.9631e+02, 2.0146e+02, 7.8912e-01, 5.0000e+00],
            [8.5944e-01, 1.2990e+02, 2.3154e+02, 2.5031e+02, 7.0019e-01, 1.5000e+01],
            [3.0862e+02, 1.3703e+02, 3.6140e+02, 1.7356e+02, 6.5137e-01, 1.7000e+01],
            [2.5644e+02, 1.3687e+02, 3.1232e+02, 1.7165e+02, 6.0147e-01, 5.0000e+00],
            [3.8901e+02, 1.4571e+02, 4.2773e+02, 1.6276e+02, 5.2207e-01, 5.0000e+00],
            [4.8636e+02, 1.4414e+02, 5.2142e+02, 1.7205e+02, 5.1898e-01, 5.0000e+00]], device='cuda:0')
    id: None
    is_track: False
    orig_shape: (360, 640)
    shape: torch.Size([11, 6])
    xywh: tensor([[ 80.4396, 292.1061, 160.5392, 134.6830],
            [368.8176, 225.5822, 198.8063, 133.4417],
            [584.0615, 156.4654,  31.1709,  31.4354],
            [532.2274, 165.0890,  62.0083,  37.9059],
            [370.7474, 137.7061,  18.8354,  16.0948],
            [465.1164, 174.7568,  62.3911,  53.3998],
            [116.2020, 190.1029, 230.6852, 120.4108],
            [335.0104, 155.2986,  52.7853,  36.5304],
            [284.3798, 154.2598,  55.8854,  34.7714],
            [408.3734, 154.2349,  38.7213,  17.0570],
            [503.8933, 158.0958,  35.0625,  27.9070]], device='cuda:0')
    xywhn: tensor([[0.1257, 0.8114, 0.2508, 0.3741],
            [0.5763, 0.6266, 0.3106, 0.3707],
            [0.9126, 0.4346, 0.0487, 0.0873],
            [0.8316, 0.4586, 0.0969, 0.1053],
            [0.5793, 0.3825, 0.0294, 0.0447],
            [0.7267, 0.4854, 0.0975, 0.1483],
            [0.1816, 0.5281, 0.3604, 0.3345],
            [0.5235, 0.4314, 0.0825, 0.1015],
            [0.4443, 0.4285, 0.0873, 0.0966],
            [0.6381, 0.4284, 0.0605, 0.0474],
            [0.7873, 0.4392, 0.0548, 0.0775]], device='cuda:0')
    xyxy: tensor([[1.7001e-01, 2.2476e+02, 1.6071e+02, 3.5945e+02],
            [2.6941e+02, 1.5886e+02, 4.6822e+02, 2.9230e+02],
            [5.6848e+02, 1.4075e+02, 5.9965e+02, 1.7218e+02],
            [5.0122e+02, 1.4614e+02, 5.6323e+02, 1.8404e+02],
            [3.6133e+02, 1.2966e+02, 3.8017e+02, 1.4575e+02],
            [4.3392e+02, 1.4806e+02, 4.9631e+02, 2.0146e+02],
            [8.5944e-01, 1.2990e+02, 2.3154e+02, 2.5031e+02],
            [3.0862e+02, 1.3703e+02, 3.6140e+02, 1.7356e+02],
            [2.5644e+02, 1.3687e+02, 3.1232e+02, 1.7165e+02],
            [3.8901e+02, 1.4571e+02, 4.2773e+02, 1.6276e+02],
            [4.8636e+02, 1.4414e+02, 5.2142e+02, 1.7205e+02]], device='cuda:0')
    xyxyn: tensor([[2.6565e-04, 6.2435e-01, 2.5111e-01, 9.9847e-01],
            [4.2096e-01, 4.4128e-01, 7.3159e-01, 8.1195e-01],
            [8.8824e-01, 3.9097e-01, 9.3695e-01, 4.7829e-01],
            [7.8316e-01, 4.0593e-01, 8.8005e-01, 5.1123e-01],
            [5.6458e-01, 3.6016e-01, 5.9401e-01, 4.0487e-01],
            [6.7800e-01, 4.1127e-01, 7.7549e-01, 5.5960e-01],
            [1.3429e-03, 3.6083e-01, 3.6179e-01, 6.9530e-01],
            [4.8222e-01, 3.8065e-01, 5.6469e-01, 4.8212e-01],
            [4.0068e-01, 3.8021e-01, 4.8800e-01, 4.7679e-01],
            [6.0783e-01, 4.0474e-01, 6.6833e-01, 4.5212e-01],
            [7.5994e-01, 4.0040e-01, 8.1473e-01, 4.7791e-01]], device='cuda:0')



Some properties worth noting:
- `cls` lists the class indices
- `conf` gives the confidence of there being an object
- `xywh` lists the centers and sizes of the bounding boxes in pixels
- `xywhn` gives the same in normalized coordinates.
- `xyxy` and `xyxyn` give the corners of bounding boxes in pixels and normalized coordinate.

**Task 3.4.21:** Get the names of the classes of objects detected in this image.


```python
# Note that the tensor is on the GPU
[classes[int(ind)] for ind in predict_results[0].boxes.cls.cpu()]
```




    ['car',
     'car',
     'three wheelers (CNG)',
     'car',
     'three wheelers (CNG)',
     'car',
     'suv',
     'three wheelers (CNG)',
     'car',
     'car',
     'car']



We could use the bounding box information to plot these objects on the image.  But with the `save=True` option we gave the predict call, this image has already been created for us:


```python
Image.open(pathlib.Path(predict_results[0].save_dir) / "frame_600.jpg")
```




    
![png](output_101_0.png)
    



---
&#169; 2024 by [WorldQuant University](https://www.wqu.edu/)
