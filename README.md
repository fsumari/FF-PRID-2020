# Full-Frame person Re-identification

* This the code for security-level of person Re-identification on real scenarios.

## part 1. Quick start
1. Clone this file
```bashrc
$ git clone https://github.com/fsumari/FF-PRID-2020.git
```
2.  The code is compatible with Python 3.7. The following dependencies are needed to run the FF-PRID:

```bashrc
Tensorflow 1.4.0
NumPy
OpenCV
pandas
matplotlib
```
3. Exporting loaded COCO weights as TF checkpoint(`yolov3_coco.ckpt`)【[BaiduCloud](https://pan.baidu.com/s/11mwiUy8KotjUVQXqkGGPFQ&shfl=sharepset)】
```bashrc
$ cd RW-PRID-2020/pYOLO
$ cd checkpoint
$ wget https://github.com/YunYang1994/tensorflow-yolov3/releases/download/v1.0/yolov3_coco.tar.gz
$ tar -xvf yolov3_coco.tar.gz
$ cd ..
$ python convert_weight.py
$ python freeze_graph.py
```
4. Exporting loaded Re-id weights as TF checkpoint(`model.ckpt`)
```bashrc
$ cd RW-PRID-2020/logs
$ wget https://drive.google.com/file/d/1pFAIkjLrNB0KeKLWuUlS0hoJIwXR00oV/view?usp=sharing
$ unzip weightsReid.zip
$ cd ..
```
5. Run the simple test for FF-PRID
* Execute test for classic ReID:
```bashrc
$ python run.py --mode=cl_test --query_path=querys/person_0015.png --cropps_path=data/seq1/cropps --top=10
```
* Execute test for FF-PRID:
```
python run.py --mode=ff_test --query_path=querys/person_0015.png --video_path=data/seq1/video_in.avi --top=10
```
## part 2. Evaluation

Two steps are required as follows:

### 2.1 Organization of Dataset

1. You have that get the raw videos of authors PRID2011 and download official PRID2011

```bashrc
$ cd data
$ wget link_of_prid2011_videos
$ wget link_of_prid2011
```
2. Extract into one directory and rename them, after that with script `generateFF-PRID-videos.py` many videos. You have put on the script, the path of `anno_a.mat` and `anno_b.mat`.
```bashrc
$ cd prid2011_videos
$ python generateFF-PRID-videos.py
```
3. You could generate the splitting the raw videos on sub-videos and ground_truth with `generateGT.py` on `--t_skip=frames` frames for sub-video.
```bashrc
$ python generateGT.py --dataset=path_FF-PRID --anno=path_annotations --t_skip=frames
```
4. For our evaluation we had the following basic structure.

```bashrc

FF-PRID    # path:  /home/oliver/dataset/FF-PRID
├── A-B
|    └──000001 #video directory
|    |   └──person_0017 #query .png
|    |   └──person_0018
|    |   └──person_0019
|    |   └──person_0020
|    └──000002
|    |   └──person_0011
|    |   └──person_0014
|    |   └──person_0015
|    |   └──person_0026
|    └──000003
|    |   └──person_0022
|    |   └──person_0023
|    |   └──person_0027
|    |   └──person_0039
|    └──000004
|    |   └──person_0006
|    |    └──person_0007
|    |    └──person_0008
|    |    └──person_0009
|    └──000005
|    |   └──person_0003
|    |   └──person_0030
|    |   └──person_0036
|    └──000006
|    |   └──person_0004
|    |   └──person_0005
|    |   └──person_0034
|    |   └──person_0054
|    └──000007
|    |   └──person_0001
|    |   └──person_0002
|    |   └──person_0033
|    └──000008
|    |   └──person_0043
|    |   └──person_0044
|    |   └──person_0046
|    |    └──person_0048
|    └──000010
|    |   └──person_0042
|    |   └──person_0049
|    |   └──person_0050
|    └──000011
|        └──person_0052
|        └──person_0053
|        └──person_0063
|        └──person_0064
└── B-A
     └──000001
     |   └──person_0015
     |   └──person_0017
     |   └──person_0018
     |   └──person_0020
     └──000002
     |   └──person_0021
     |   └──person_0022
     |   └──person_0023
     └──000003
     |   └──person_0011
     |   └──person_0013
     |   └──person_0039
     |   └──person_0040
     └──000004
     |   └──person_0008
     |   └──person_0009
     |   └──person_0010
     |   └──person_0028
     └──000005
     |   └──person_0006
     |   └──person_0007
     └──000006
     |   └──person_0003
     |   └──person_0005
     |   └──person_0036
     |   └──person_0041
     └──000007
     |   └──person_0001
     |   └──person_0032
     |   └──person_0033
     |   └──person_0035
     └──000008
     |   └──person_0043
     |   └──person_0044
     |   └──person_0045
     └──000009
     |   └──person_0042
     |   └──person_0046
     |   └──person_0048
     |   └──person_0049
     └──000011
         └──person_0052
         └──person_0053
         └──person_0054
         └──person_0058
```  
### 2.2 Description of ground_truth and prediction
1. ground truth
 
 `(id, frame_0, size_list, ulx, uly, brx, bry, seg_appears, sub_seq_appears)`

 * `id` = id of person
 * `frame_0` = first frame where appears id
 * `size_list` = size of frames where appears id
 * `seg_appears` = time where appears id(timestamp)
 * `sub_seq_appears` = sub sequence where appears id
 
2. prediction manual
 `(id, sub_seq_appears, rank, true_score, false_score)`
- puede repetirse un id, si aparece en varias sub sequencias

* `id` = id of person
* `sub_seq_appear` = example 000002
* `rank` = position on the TOP10
* `true_score` = example 0.86
* `false_score` = example 0.2

### 2.3 Execute mode evaluation

* Execute for Dataset:

`python run.py --mode=data --data_dir=data/FF-PRID01 --t_skip= frames`

* Validation:

`python run.py --mode=val --data_dir=data/FF-PRID01 --p_name=RW-01-predict`

* Graphs of metrics

`python run.py --mode=graph --data_dir=data/FF-PRID01 --p_name=RW-01-predict`


