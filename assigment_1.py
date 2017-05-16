from __future__ import print_function
from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import tarfile
from IPython.display import display, Image
from scipy import ndimage
from sklearn.linear_model import LogisticRegression
import sklearn
from six.moves.urllib.request import urlretrieve
from six.moves import cPickle as pickle
import random 

#KONSTANTE

url = 'http://commondatastorage.googleapis.com/books1000/'
data_root='/home/josip/Documents/Udacity_machine_learning/'
num_classes = 10
np.random.seed(133)
image_size = 28 
pixel_depth = 255.0  
train_size = 20000
valid_size = 1000
test_size = 1000

p_data='model_dataset'
#GOTOVE KONSTANTE


#POMOCNE METODE
def download_process_hook(count,blockSize,totalSize):
    return 1

def maybe_download(filename,exp_bytes,force=False):
    dest_file_path=os.path.join(data_root,filename)
    if force or not os.path.exists(dest_file_path):
        print("Downloading data....\n")
        data=urlretrieve(url+filename,dest_file_path)
        print("Downloading completed")
    stats=os.stat(dest_file_path)
    if stats.st_size == exp_bytes:
        print("Data verified")
    else:
        raise Exception("Data is comporomised")
    return dest_file_path


def maybe_extract(filename,force=False):
    root=os.path.splitext(os.path.splitext(filename)[0])[0]
    if os.path.isdir(root) and not force:
        print("Data already extracted")
    else:
        print("Extracting data....")
        tar=tarfile.open(filename)
        sys.stdout.flush()
        tar.extractall(root)
        tar.close()
    root=os.path.join(root,os.listdir(root)[0])
    data_folders=[os.path.join(root,d) for d in sorted(os.listdir(root))]
    for d in data_folders:
        if os.path.isfile(d):
            data_folders.remove(d)
    return data_folders


def load_data(folder,min_num_images=0):
    image_files=os.listdir(folder)
    dataset=np.ndarray(shape=(len(image_files),image_size,image_size),dtype=np.float32)
    num_images=0
    for image in image_files:
        try:
            image_d=os.path.join(folder,image)
            image_data=(ndimage.imread(image_d).astype(float)- pixel_depth/2) / pixel_depth
            dataset[num_images,:,:]=image_data
            num_images = num_images + 1
        except Exception as e:
            print("Cannot read image "+image+" error "+e.message)
    
    if num_images < min_num_images:
        print("Num of images is too low ("+str(num_images)+")")
    
    print("Full dataset:",dataset.shape)
    print("Dataset mean",np.mean(dataset))
    print("std:",np.std(dataset))
    return dataset
    
def maybe_pickle(data_folder,min_images=0,force=False):
    dataset_names=[]
    for folder in data_folder:
        set_folder=folder+".pickle"
        dataset_names.append(set_folder)
        if os.path.exists(set_folder) and not force:
            print("File exists ",set_folder)
        else:
            dataset=load_data(set_folder)
            try:
                with open(set_folder,'wb') as f:
                    pickle.dump(dataset,f,pickle.HIGHEST_PROTOCOL)
            except Exception as e:
                print('Unable to save data to', set_folder, ':', e)
    return dataset_names

def load_random_pickle(folders):
    with open(folders[random.randint(0,len(folders)-1)]) as f:
        l=pickle.load(f)
    return l
 
def make_array(num_rows,img_size):
    data=np.ndarray(shape=(num_rows,image_size,img_size),dtype=np.float32)
    labels=np.ndarray(num_rows,dtype=np.int)
    return data,labels

def merge_datasets(pickle_files,train_size,valid_size=0):
    
    num_classes=len(pickle_files)
    valid_dataset,valid_labels=make_array(valid_size,image_size)
    train_dataset,train_labels=make_array(train_size,image_size)
    vsize_per_class = valid_size // num_classes
    tsize_per_class = train_size // num_classes
    start_v, start_t = 0, 0
    end_v, end_t = vsize_per_class, tsize_per_class
    end_l = vsize_per_class+tsize_per_class
    
    for label,pickle_file in enumerate(pickle_files):
            try:
                with open(pickle_file,'rb') as f:
                    letter_file=pickle.load(f)
                    np.random.shuffle(letter_file)
                    if valid_dataset is not None:
                        valid_letter = letter_file[:vsize_per_class, :, :]
                        valid_dataset[start_v:end_v, :, :] = valid_letter
                        valid_labels[start_v:end_v] = label
                        start_v += vsize_per_class
                        end_v += vsize_per_class
                    
                    train_letter = letter_file[vsize_per_class:end_l, :, :]
                    train_dataset[start_t:end_t, :, :] = train_letter
                    train_labels[start_t:end_t] = label
                    start_t += tsize_per_class
                    end_t += tsize_per_class
            except Exception as e:
                print('Unable to process data from', pickle_file, ':', e)
                raise
    
    return valid_dataset, valid_labels, train_dataset, train_labels

       
def randomize(dataset,labels):
    permutation=np.random.permutation(labels.shape[0])
    sh_dataset=dataset[permutation,:,:]
    sh_labels=labels[permutation]
    return sh_dataset,sh_labels

def save_datasets(file_name,train_dataset,train_labels,valid_dataset,valid_labels,test_dataset,test_labels):
    _path=os.path.join(data_root,file_name)
    with open(_path,'wb') as f:
        save={
                'train_dataset':train_dataset,
                'train_labels':train_labels,
                'valid_dataset':valid_dataset,
                'valid_labels':valid_labels,
                'test_dataset':test_dataset,
                'test_labels':test_labels
                }
        try:
            pickle.dump(save,f,pickle.HIGHEST_PROTOCOL)
        except Exception as e:
            print("Unable to save dataset ",e)

def get_new_dataset():
    train_filename = maybe_download('notMNIST_large.tar.gz', 247336696)
    test_filename = maybe_download('notMNIST_small.tar.gz', 8458043)

    train_folder=maybe_extract(train_filename)
    test_folder=maybe_extract(test_filename)

    train_datasets = maybe_pickle(train_folder, 45000)
    test_datasets = maybe_pickle(test_folder, 1800)

    valid_dataset, valid_labels, train_dataset, train_labels = merge_datasets(
    train_datasets, train_size, valid_size)
    _, _, test_dataset, test_labels = merge_datasets(test_datasets, test_size)

    train_dataset, train_labels = randomize(train_dataset, train_labels)
    test_dataset, test_labels = randomize(test_dataset, test_labels)
    valid_dataset, valid_labels = randomize(valid_dataset, valid_labels)

    save_datasets(p_data,train_dataset,train_labels,valid_dataset,valid_labels,test_dataset,test_labels)
            
def load_dataset(filename):
    _path=os.path.join(data_root,filename)
    train_d=np.ndarray(shape=(train_size,image_size,image_size))
    train_l=np.ndarray(shape=(train_size))
    valid_d=np.ndarray(shape=(valid_size,image_size,image_size))
    valid_l=np.ndarray(shape=(valid_size))
    test_d=np.ndarray(shape=(test_size,image_size,image_size))
    test_l=np.ndarray(test_size)
    with open(_path,'rb') as f:
        try:
            data=pickle.load(f)
            train_d=data['train_dataset']
            train_l=data['train_labels']
            
            valid_d=data['valid_dataset']
            valid_l=data['valid_labels']
            
            test_d=data['test_dataset']
            test_l=data['test_labels']
            
        except Exception as e:
            print('Unable to load data',e)
        
    return train_d,train_l,valid_d,valid_l,test_d,test_l

def problem_1(folder):
    display(Image(filename=os.path.join(folder,os.listdir(folder)[100])))
    

def problem_2(folders):
    l=load_random_pickle(folders)
    sample=l[random.randint(0,len(l)-1),:,:]
    plt.imshow(sample)

def problem_3(folders):
    num_of_img=[]
    for i,folder in enumerate(folders):
        with open(folder,'rb') as f:
            l=pickle.load(f)
            num_of_img.append(l.shape[0])
    print(num_of_img)



def problem_4(dataset):
    plt.imshow(dataset[random.randint(0,len(dataset)-1)])

def problem_6(train_dataset,train_labels,test_dataset,test_labels):
    model=sklearn.linear_model.LogisticRegression(max_iter=1000)
    X=train_dataset.reshape(train_dataset.shape[0],train_dataset.shape[1]*train_dataset.shape[2])
    X_test=test_dataset.reshape(test_dataset.shape[0],test_dataset.shape[1]*train_dataset.shape[2])
    model.fit(X,train_labels)
    print("model: ",model.score(X_test,test_labels))
#GOTOVE POMOCNE METODE

def main():
    train_dataset,train_labels,valid_dataset,valid_labels,test_dataset,test_labels=load_dataset(p_data)
    problem_6(train_dataset,train_labels,test_dataset,test_labels)

    
if __name__ == '__main__':
    main()