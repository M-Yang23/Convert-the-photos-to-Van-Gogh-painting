import os
import matplotlib.pyplot as plt

from models.cycleGAN import CycleGAN
from GDL_code.utils import loaders

import keras
from keras import layers
import numpy as np
import pydot
import PIL


SECTION = 'data'
RUN_ID = '001' 
DATA_NAME = 'vangogh2photo'  
FOLDER = 'C:/Users/59119/Desktop/program/'
RUN_FOLDER = FOLDER.format(SECTION)


RUN_FOLDER += '_'.join([RUN_ID, DATA_NAME])

if not os.path.exists(RUN_FOLDER):
    os.mkdir(RUN_FOLDER)
    os.mkdir(os.path.join(RUN_FOLDER, 'viz'))
    os.mkdir(os.path.join(RUN_FOLDER, 'images'))
    os.mkdir(os.path.join(RUN_FOLDER, 'weights'))

mode =  'build' # 'build' # 

IMAGE_SIZE = 128 

data_loader = loaders.DataLoader(dataset_name=DATA_NAME, img_res=(IMAGE_SIZE, IMAGE_SIZE))

gan = CycleGAN(
    input_dim = (IMAGE_SIZE,IMAGE_SIZE,3)
    ,learning_rate = 0.0002
    , buffer_max_length = 1
    , lambda_validation = 10
    , lambda_reconstr = 10
    , lambda_id = 5
    , generator_type = 'unet'
    , gen_n_filters = 32
    , disc_n_filters = 64
    )

if mode == 'build':
    gan.save(RUN_FOLDER)
else:
    gan.load_weights(os.path.join(RUN_FOLDER, 'weights/weights.h5'))
    
gan.g_BA.summary()
gan.g_AB.summary()

gan.d_A.summary()
gan.d_B.summary()


BATCH_SIZE = 1
EPOCHS = 15
PRINT_EVERY_N_BATCHES = 100
TEST_A_FILE = 'M0.jpg'
TEST_B_FILE = 'M0.jpg'

gan.train(data_loader
        , run_folder = RUN_FOLDER
        , epochs=EPOCHS
        , test_A_file = TEST_A_FILE
        , test_B_file = TEST_B_FILE
        , batch_size=BATCH_SIZE
        , sample_interval=PRINT_EVERY_N_BATCHES)



fig = plt.figure(figsize=(20,10))

plt.plot([x[1] for x in gan.g_losses], color='green', linewidth=0.1) #DISCRIM LOSS
# plt.plot([x[2] for x in gan.g_losses], color='orange', linewidth=0.1)
plt.plot([x[3] for x in gan.g_losses], color='blue', linewidth=0.1) #CYCLE LOSS
# plt.plot([x[4] for x in gan.g_losses], color='orange', linewidth=0.25)
plt.plot([x[5] for x in gan.g_losses], color='red', linewidth=0.25) #ID LOSS
# plt.plot([x[6] for x in gan.g_losses], color='orange', linewidth=0.25)

plt.plot([x[0] for x in gan.g_losses], color='black', linewidth=0.25)

# plt.plot([x[0] for x in gan.d_losses], color='black', linewidth=0.25)

plt.xlabel('batch', fontsize=18)
plt.ylabel('loss', fontsize=16)

plt.ylim(0, 5)

plt.show()


for p in range(2):
    if p == 0:
        set1 = "testA"
        load = 'data/%s/testA/%s'
    else:
        set1 = "testB"
        load = 'data/%s/testB/%s'
    path2 = FOLDER + SECTION + "/" + DATA_NAME + "/" + set1
    print(path2)

    files= os.listdir(path2) 
    listA = []
    for i in range(len(files)):
        listA = listA+ [files[i]]
        
    for i in listA:
        if p == 0:
            imgs_A = data_loader.load_img('data/%s/testA/%s' % (data_loader.dataset_name, i))
            fake_B = gan.g_AB.predict(imgs_A)[0]
        else:
            imgs_A = data_loader.load_img('data/%s/testB/%s' % (data_loader.dataset_name, i))
            fake_B = gan.g_BA.predict(imgs_A)[0]
        fake_B = ((fake_B+1)*128).astype("uint8")
        img = Image.fromarray(fake_B)
        img.save(path2 +"/"+ i + "_trans.jpg")


