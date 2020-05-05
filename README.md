 
 #ground truth
 
 (id, frame_0, size_list, ulx, uly, brx, bry, seg_appears, sub_seq_appears)

 * id= id of person
 * frame_0 = first frame where appears id
 * size_list = size of frames where appears id
 * seg_appears = time where appears id(timestamp)
 * sub_seq_appears = sub sequence where appears id


 
 #prediction manual

(id, sub_seq_appears, rank, true_score, false_score)
- puede repetirse un id, si aparece en varias sub sequencias

* id = id of person
* sub_seq_appear = example 000002
* rank = position on the TOP10
* true_score = example 0.86
* false_score = example 0.2

*Exute test for standard ReID:

´python run.py --mode=classic_test --query_path=querys/person_0015.png --cropps_path=data/seq1/cropps --top=10´

*Exute test for RW ReID:

´python run.py --mode=rw_test --query_path=querys/person_0015.png --video_path=data/seq1/video_in.avi --top=10´

*Execute for Dataset, 750 frames are 30 seconds:

´python run.py --mode=data --data_dir=data/RW-PRID01 --t_skip=750´

* Validation

´python run.py --mode=val --data_dir=data/RW-PRID01 --p_name=RW-01-predict´

* Graphs of metrics

´python run.py --mode=graph --data_dir=data/RW-PRID01 --p_name=RW-01-predict´


