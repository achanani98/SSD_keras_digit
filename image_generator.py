import os
import numpy as np
import pandas as pd
import tensorflow as tf
import cv2
import tensorflow.keras.datasets.mnist as mnist
import matplotlib.pyplot as plt
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train[x_train>10] = 255
x_train[x_train<=10] = 0

generation_samples = 10
tot_images = x_train.shape[0]

#Blank edge length
edge_len = 300
SAVE_PATH = "."
ANNOTATIONS_PATH = "."
if(not os.path.isdir(SAVE_PATH+'/train/')):
    os.mkdir(SAVE_PATH + '/train/')
    os.mkdir("./train/Images/")
    os.mkdir("./train/labels/")
if(not os.path.isdir(SAVE_PATH+'/val/')):
    os.mkdir(SAVE_PATH + '/val/')
    os.mkdir("./val/Images/")
    os.mkdir("./val/labels/")
for mode in ['train','val']:
    csv_list = []
    if(mode=='train'):
        generation_samples = 400
    else:
        generation_samples = 80
    
    for generation in range(generation_samples):
        if(mode=='val'):
            generation += 400
        MAX_IMAGES_PER_FRAME = 4
        num_images = np.random.randint(1,MAX_IMAGES_PER_FRAME+1)
        indices = np.random.randint(0,tot_images,num_images)
        Blank = np.zeros((edge_len,edge_len))
        label_info = []
        for indice in indices:
            img = x_train[indice]
            # +1 addition to label as background is considered as zero class and not literal 0 digit
            label = y_train[indice]+1

            #zoom range(0-3)
            z = 3*np.random.rand()
            img = cv2.resize(img,(int(28*(1+z)),int(28*(1+z))))
            (h,w) = img.shape
            #Horizontal shift and vertical shifts
            hs = np.random.randint(1,edge_len-h)
            vs = np.random.randint(1,edge_len-w)

            label_info.append([label,hs,vs,h,w])

            Blank[hs:hs+h,vs:vs+w] = np.logical_or(img,Blank[hs:hs+h,vs:vs+w])
            #csv list info: filename height width class xmin ymin xmax ymax 
            csv_list.append([mode + str(generation) + '.jpg',hs,vs,hs+h,vs+w,str(label),h,w])

        
        #saving generated image and labels
        generated_img = Blank
        Blank = cv2.convertScaleAbs(Blank, alpha=(255.0))
        cv2.imwrite(SAVE_PATH + '/train/' + '/Images/' + mode + str(generation) + '.jpg',Blank)
        
        with open(SAVE_PATH + '/train/' + mode + '.txt','a+') as f:
            f.write(str(mode) + str(generation) + '.jpg ' + str(generation) + '.txt\n')
        
        with open(ANNOTATIONS_PATH + '/train' + '/labels/' + str(generation) +  '.txt','a+') as f:
            for label in label_info:
                str_label = [str(_) for _ in label]
                f.write(" ".join(str_label)+"\n")

        with open(ANNOTATIONS_PATH + '/' + mode + '/annotations.txt','a+') as f:
            for label in label_info:
                str_label = [str(_) for _ in label]
                f.write(SAVE_PATH + "/" + mode + "/" +mode +str(generation) + '.jpg '+" ".join(str_label)+"\n")
    
    csv_columns = ['filename','xmin','xmax','ymin','ymax','class_id','width','height']
    CSV_dataframe = pd.DataFrame(csv_list,columns=csv_columns)
    CSV_dataframe.to_csv(SAVE_PATH + "/" + mode +'/labels_' + mode + '.csv',index = None)
# cv2.destroyAllWindows()