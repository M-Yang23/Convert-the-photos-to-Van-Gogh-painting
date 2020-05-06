import glob
import os
import numpy as np
from PIL import Image

path = "dir to files contents pictures"

def data_pre_process(path,size = 128,dim = 3,file_name="Modified Data"):
    #path is the dir of the pic file
    #size is the pix length
    #dim = 1,3,4... -> BW， RGB， RGBA...
    #get data from the path
    path2 = path + "/"
    files= os.listdir(path) 
    listA = []
    for i in range(len(files)):
        listA = listA+ [path2 + files[i]]

    dataA = []

    count = 0
    for i in listA:
        img = Image.open(i)
        
        arr = np.asarray(img, dtype="float32")
        if len(arr.shape) != dim:
            continue
        if img.size[0]!=img.size[1]:
            l = img.size[0]
            w = img.size[1]
            x = max(0,int((l-w)/2))
            y = max(0,int((w-l)/2))
            box = (x,y,x+min(l,w),y+min(l,w))
            img = img.crop(box)
        
        img = img.resize((size, size)) 
        
        dataA.append(img)
    
    data_num = len(dataA)
    
    #create a file to store modified picture
    # if os.path.exists(file_name):
    #     os.mkdir(file_name + " new")
    #     file_n = file_name + " new"
    #     print("The given file name already exists. The data is stored in a new file.")

    # else:
    #     os.mkdir(file_name)
    #     file_n = file_name
    
    count = 0
    
    for i in dataA:
        name = "M"+str(count)
        i.save(path+"/"+ name + ".jpg")
        count += 1
    
    return("Finish processing "+str(data_num)+" pictures")


data_pre_process(path)








