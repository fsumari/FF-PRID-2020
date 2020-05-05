#!python
#!/usr/bin/env python
############################
# code for read .mat
# annotations of PRID2011
# Felix Oliver Sumari Huayta
############################

from scipy.io import loadmat
import numpy as np
import cv2
import glob
import os
import csv
import pandas as pd
import tensorflow as tf
import math

FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_string('dataset', '', 'path of dataset')
tf.flags.DEFINE_string('anno', '', 'path of annotations')
tf.flags.DEFINE_string('t_skip', '', 'number of subsequence')

def read_anno(anno):
    x = loadmat(anno)

    list_data = []
    for e in x:
        if '__' not in e and 'readme' not in e:
            ann = {}
            for i in x[e]:
                ann = i
            print("num of elements: ", len(ann))
            id = 1
            for e in ann:            
                list_data.append((id, e['frame'].ravel(), \
                    e['ulx'].ravel(), e['uly'].ravel(), \
                    e['brx'].ravel() ,e['bry'].ravel()))
                id = id+1
    return list_data

######################################
#list_data = (id_person, list_frame, \
#               list_ulx, list_uly, \
#               list_brx, list_bry)
######################################
#ordeno lista por frame
#read A
list_data_A = read_anno('anno_a.mat')
print("shapes", np.shape(list_data_A))
list_data_A.sort(key = lambda tup: tup[1][0])
# read B
list_data_B = read_anno('anno_b.mat')
print("shapes", np.shape(list_data_B))
list_data_B.sort(key = lambda tup: tup[1][0])
'''
count = 0

### single shot
def getFrameNumber(count):
    count = count + 1
    return count
'''
def write_gt(path_files, list_data):
    fps = 25
    size_seq = 3000 # 25 * 120 , fps * segundos por sequencia
    #size_sub = size_seq / int(FLAGS.t_skip)
    count = 0
    
    for seq in path_files:
        path, name = os.path.split(seq)
        seq_t = int(name)
        list_csv = []
        print('limite' ,seq_t * size_seq)
        while(True):
            frame_a = list_data[count][1][0] 
            l_max = seq_t * size_seq
            l_min = l_max - size_seq
            if(frame_a > l_min):
                if(frame_a <= l_max):
                    sub = frame_a - l_min
                    sub = math.ceil(sub/int(FLAGS.t_skip))
                    print('seq: ',seq_t,'id: ', list_data[count][0],'frame: ',frame_a,' sub_seq: ', sub)
                    list_csv.append([list_data[count][0],\
                            list_data[count][1][0],\
                            len(list_data[count][1]),\
                            list_data[count][2][0],\
                            list_data[count][3][0],\
                            list_data[count][4][0],\
                            list_data[count][5][0],\
                            list_data[count][1][0]/ float(fps),\
                            sub ])
                            
                    count=count+1
                else:
                    break
            elif(frame_a <=l_min):
                count=count+1
        df = pd.DataFrame(np.array(list_csv))
        df.to_csv(seq+'/ground_truth.csv', header = False)

#************************************************************
testAB = sorted(glob.glob(FLAGS.dataset + '/A-B/*'))
testBA = sorted(glob.glob(FLAGS.dataset + '/B-A/*'))
write_gt(testAB, list_data_B)
write_gt(testBA, list_data_A)        
'''
def single_shot(tam_per_sequence):
    #tam_seq = sec_per_seq * num_seq

    #fps = cap.get(cv2.CAP_PROP_FPS)

    cont_sub_seq = 0
    count_sub_seq = 1
    #tam_per_subsequence = fps * sec_per_seq
    
    #print ("Frames per second using video.get(cv2.CAP_PROP_FPS) : {0}".format(fps))
    global count
    #25 FRAMES POR SEGUNDO TIENEN
    count_frame = 0
    #30 segundos , para cada sequencia
    tam_per_sequence = fps * tam_seq # 25 f en una sequencia
    print("tam sequence: ", tam_per_sequence)
    cont_seq = 0
    #carpet 
    carpet = 0
    cont_ids = 0# para avanzar en los ids
    ncarpet = ''
    list_csv = []
    
    global frame_array 
    frame_array = []
    while(True):
        #Capture frame-by-frame
        ret, frame = cap.read()
        
        # if frame exists
        if ret == True:
            
            height, width, layers = frame.shape
            size = (width,height)

            if(cont_seq <= tam_per_sequence):
                ##contador para SUB SEQUENCIAS
                if(cont_sub_seq < tam_per_subsequence):
                    cont_sub_seq = cont_sub_seq +1
                if(cont_sub_seq == tam_per_subsequence):
                    cont_sub_seq = 0
                    count_sub_seq = count_sub_seq +1
                    
                ##
                #print('cont_seq: ',cont_seq)
                if(cont_seq == tam_per_sequence):
                    df = pd.DataFrame(np.array(list_csv))
                    df.to_csv("CamB_FINAL/"+ncarpet+"/ground_truth.csv", header = False)
                    cont_seq = 0
                    count_sub_seq = 1
                    carpet = carpet + 1                    
                    #print('len : ',len(frame_array))
                    #for i in range(len(frame_array)):
                        # writing to a image array
                    #    out.write(frame_array[i])
                    #frame_array = []
                    #out.release()                   
                elif(cont_ids == np.shape(list_data)[0]):
                    df = pd.DataFrame(np.array(list_csv))
                    df.to_csv("CamB_FINAL/"+ncarpet+"/ground_truth.csv", header = False)
                    out.release()
                    break
                
                ##########
                if(cont_seq == 0):
                    print('CONT_SEQ: ',cont_seq)
                    list_csv = []
                    ncarpet = '{0:06}'.format(getFrameNumber(carpet))                    
                    os.system('mkdir CamB_FINAL/'+ncarpet)
                    print('ncarpet: ',ncarpet, carpet)

                    pathOut = "CamB_FINAL/"+ncarpet+"/video_in.avi"
                    out = cv2.VideoWriter(pathOut, cv2.VideoWriter_fourcc(*'DIVX'), fps, size)
                ####
                out.write(frame)
                ####                
                
                #frame_array.append(frame)
                #print(frame_array)
                #print('frame ',frame)
                ####
                nfile = '{0:06}'.format(getFrameNumber(count))
                #cv2.imwrite("CamB_FINAL/"+ncarpet+"/"+nfile+".jpg", frame)

                ##if el frame actual, es igual al de list, agregamos a un archivo
                #print(nfile, cont_ids, list_data[cont_ids][1][0])
                if(int(nfile) == int(list_data[cont_ids][1][0])):
                    #with open("CamA_new/"+ncarpet+"/ground_truth.csv","w") as f:
                    #wr = csv.writer(f, delimiter=",")#solo el primer frame
                    ##(id, frame_0,size_list,ulx,uly,brx,bry,seg_appears, sub_seq_appears)
                    ## size_list, es para frame e coords...
                    list_csv.append([list_data[cont_ids][0],\
                        list_data[cont_ids][1][0],\
                        len(list_data[cont_ids][1]),\
                        list_data[cont_ids][2][0],\
                        list_data[cont_ids][3][0],\
                        list_data[cont_ids][4][0],\
                        list_data[cont_ids][5][0],\
                        count/ float(fps),\
                        count_sub_seq ])
                    #por si existen 2 ids q aparecen en el mismo frame, no aumento el id
                    if(int(list_data[cont_ids][1][0]) == int(list_data[cont_ids+1][1][0])):
                        count = count-1#para no avanzar el frame actual
                    cont_ids = cont_ids +1           
                    #creo lista de primeros elementos de cada lista
                    #en que segundo aparece
                cont_seq = cont_seq + 1
            
                            
            #print ("esribio")
            #print (nfile)
            # increment counter
            count = count + 1
        else:
            break

        # ESC to stop algorithm
        key = cv2.waitKey(7) % 0x100
        if key == 27:
            break
            cap.release()
    cap.release()


#######

single_shot( sec_per_seq = 30  , num_seq = 4 )#10 sequencias de 30 segundos
### multi shot
'''