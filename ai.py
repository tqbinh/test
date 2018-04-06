from random import shuffle
from array import array
from sys import byteorder as system_endian
from os import stat
import tensorflow as tf
import prepareData as pr
import numpy as np
import os
import glob
import collections
import clr
import time
from time import gmtime, strftime
import datetime
import gc
import sys
import ctypes
import System
from System import Array, Int32
from System.Runtime.InteropServices import GCHandle, GCHandleType
import gc
from multiprocessing import Pool
from pathlib2 import Path

scriptFolder = os.path.dirname(os.path.realpath(__file__))
dllPath = scriptFolder + "//" + "PrePostProcessingImage.dll"
clr.AddReference(dllPath)

from PrePostProcessingImage import PreProcessingImage

fileProcessing = PreProcessingImage()
Datasets = collections.namedtuple('Datasets', ['data'])

_MAP_NET_NP = {
    'Single' : np.dtype('float32'),
    'Double' : np.dtype('float64'),
    'SByte'  : np.dtype('int8'),
    'Int16'  : np.dtype('int16'), 
    'Int32'  : np.dtype('int32'),
    'Int64'  : np.dtype('int64'),
    'Byte'   : np.dtype('uint8'),
    'UInt16' : np.dtype('uint16'),
    'UInt32' : np.dtype('uint32'),
    'UInt64' : np.dtype('uint64'),
    'Boolean': np.dtype('bool'),
}

_MAP_NP_NET = {
    np.dtype('float32'): System.Single,
    np.dtype('float64'): System.Double,
    np.dtype('int8')   : System.SByte,
    np.dtype('int16')  : System.Int16,
    np.dtype('int32')  : System.Int32,
    np.dtype('int64')  : System.Int64,
    np.dtype('uint8')  : System.Byte,
    np.dtype('uint16') : System.UInt16,
    np.dtype('uint32') : System.UInt32,
    np.dtype('uint64') : System.UInt64,
    np.dtype('bool')   : System.Boolean,
}

def asNumpyArrayFromInt16Pointer(netArrayInt16Pointer, imgSize):
    '''
    Given a CLR `System.Array` returns a `numpy.ndarray`.  See _MAP_NET_NP for 
    the mapping of CLR types to Numpy dtypes.
    '''
    try:
        npArray = np.empty(imgSize, order='C', dtype=_MAP_NET_NP['Int16'])
    except KeyError:
        raise NotImplementedError("asNumpyArray does not yet support System type {}".format(netType) )

    #try: # Memmove 
    destPtr = npArray.__array_interface__['data'][0]
    sourcePtr = netArrayInt16Pointer.ToInt64()
    ctypes.memmove(destPtr, sourcePtr, npArray.nbytes)
    #finally:
    #    sourceHandle = GCHandle.FromIntPtr(netArrayInt16Pointer)
    #    if sourceHandle.IsAllocated: 
    #        print("Free", netArrayInt16Pointer)
    #        sourceHandle.Free()
    return npArray

def asNumpyArrayEx(netArray):
    '''
    Given a CLR `System.Array` returns a `numpy.ndarray`.  See _MAP_NET_NP for 
    the mapping of CLR types to Numpy dtypes.
    '''
    netType = 'Int16'

    try:
        npArray = np.empty(95*95, order='C', dtype=_MAP_NET_NP[netType])
    except KeyError:
        raise NotImplementedError("asNumpyArray does not yet support System type {}".format(netType) )

    # try: # Memmove 
    #     destPtr = npArray.__array_interface__['data'][0]
    #     sourcePtr = netArray.ToInt64()
    # # print sourcePtr
    # # print type(sourcePtr)
    #     ctypes.memmove(destPtr, sourcePtr, npArray.nbytes)
    # finally:
    #     sourceHandle = GCHandle.FromIntPtr(netArray)
    #     if sourceHandle.IsAllocated: sourceHandle.Free()

    destPtr = npArray.__array_interface__['data'][0]
    sourcePtr = netArray.ToInt64()
    # # print sourcePtr
    # # print type(sourcePtr)
    ctypes.memmove(destPtr, sourcePtr, npArray.nbytes)
    # finally:
    #     sourceHandle = GCHandle.FromIntPtr(netArray)
    #     if sourceHandle.IsAllocated: sourceHandle.Free()

    return npArray


def asNumpyArray(netArray):
    '''
    Given a CLR `System.Array` returns a `numpy.ndarray`.  See _MAP_NET_NP for 
    the mapping of CLR types to Numpy dtypes.
    '''
    dims = np.empty(netArray.Rank, dtype=int)
    for I in range(netArray.Rank):
        dims[I] = netArray.GetLength(I)
    netType = netArray.GetType().GetElementType().Name

    try:
        npArray = np.empty(dims, order='C', dtype=_MAP_NET_NP[netType])
    except KeyError:
        raise NotImplementedError("asNumpyArray does not yet support System type {}".format(netType) )

    try: # Memmove 
        sourceHandle = GCHandle.Alloc(netArray, GCHandleType.Pinned)
        sourcePtr = sourceHandle.AddrOfPinnedObject().ToInt64()
        destPtr = npArray.__array_interface__['data'][0]
        ctypes.memmove(destPtr, sourcePtr, npArray.nbytes)
    finally:
        if sourceHandle.IsAllocated: sourceHandle.Free()
    return npArray

###################################
#Convert output to numpy array
###################################
def convertToNumpyArray(outputData):
    # noElem = 0
    image = [] #list
    for i in outputData:
        image.append(i)
        # noElem = noElem+1
    # print("type of image:",type(image))
    # print("type of label:",type(label))
    npImage = np.asarray(image,np.float32)
    npImage = np.reshape(npImage,(95,95,1))
    # print("type of image:",type(npImage))
    # print(npImage)
    # gc.collect()
    return npImage

#How to use it:
# lsImage, lsLabel = convertToNumpyArray(outputData,mask)
# print("lend lsImage:",len(lsImage))
# print("lend lsLabel:",lsLabel)

class color:
   PURPLE = '\033[95m'
   CYAN = '\033[96m'
   DARKCYAN = '\033[36m'
   BLUE = '\033[94m'
   GREEN = '\033[92m'
   YELLOW = '\033[93m'
   RED = '\033[91m'
   BOLD = '\033[1m'
   UNDERLINE = '\033[4m'
   END = '\033[0m'

# print color.BOLD + 'Hello World !' + color.END
########################
#CHECK FILE EXIST Or NOT
#Return:
#   + True if Exist
#   + False if NotExist
########################
def isExist(filename):
    # print(filename)
    filename = Path(filename)
    #Check the exist of filename
    if filename.is_dir():
        return True
    else:
        # print("{} does not exist.\nPlease, Check again.!".format(my_file))
        print color.RED + '{} does not exist.\nPlease, Check again.\nSTOP APPLICATION!!'.format(filename) + color.END
        return False

def showMessageError_ListFileNotExistence (listFileNotExist):
    print color.BOLD + color.RED + 'The below mask files not exist! \n Please, check again.' + color.END
    for f in listFileNotExist:
        print color.YELLOW + f
        # print("\n")
    print color.END


################################
#Read image at (row,col)
################################
# If train = "train" --> shuffle training data
# else --> don't shuffle data
def extractDicomImageAndLabel(imgFilePath, mskFilePath, slicedWidth, slicedHeight,train):
    fileProcessing.Processing(imgFilePath, mskFilePath, slicedWidth, slicedHeight)
    # create a row and a col list with value in range 0 and 511
    row = np.arange(0,512)
    col = np.arange(0,512)
    # row = np.arange(0,5) #This is only use for testing
    # col = np.arange(0,5) #This is only use for testing
    # print(type(row))
    # print(type(col))

    #shuffle row and col. 
    # Note: we only shuffle for training data.
    if train=="train":
        c = list(zip(row, col))
        shuffle(c)
        row, col = zip(*c)

    countTotal=0
    noElem_notFat_notMuscle = 0
    noElem_Fat = 0
    noElem_Muscle=0


    lsImage = []
    lsLabel = []
    #Create a loop to extract a 95x95 image and their mask
    print("Cutting Image")
    start_time = time.time()
    for r in row:
        for c in col:
            # print("({0},{1})".format(r,c))
            # outputData = fileProcessing.GetSlicedDataMul(r, c) #Return a 2D array class int64[]
            # img = fileProcessing.GetSlicedData(r, c) #Return a 1D array class int64[]
            imgPtr = fileProcessing.GetSlicedDataAddress(c, r) #return a pointer
            # print("**********img*******")
            # print("img:",len(img))
            # for im in img:
            #     print(im)
            mask = fileProcessing.GetMaskForSlicedData(c, r) #Return a int scalar mask for outputData
            if mask==0:
                noElem_notFat_notMuscle = noElem_notFat_notMuscle + 1;
            elif mask==1:
                noElem_Fat = noElem_Fat +1
                # print("mask value:",mask)
            else:
                noElem_Muscle = noElem_Muscle +1
            # convert image to numpy array and reshape to 95*95
            # npImage = convertToNumpyArray(img)
            # print("array img",len(img))
            # img = asNumpyArray(img)
            img = asNumpyArrayEx(imgPtr)
            # fileProcessing.FreeIntPtr(imgPtr)
            # print("numpy array img",len(img))
            # print(type(img))
            # print(img.dtype)
            npImage =np.asarray(img ,np.float32)
            # print(id(npImage))
            # print("****npI*******")
            # for npI in npImage:
            #     print(npI)
            # print("img:",len(img))
            # print("len(npImage):",len(npImage))
            npImage =np.reshape(npImage,(95,95,1))
            # print(id(npImage))
            # print(npImage.dtype)
            #append image to a list
            lsImage.append(npImage)
            # print("*******")
            #append mask to a list
            lsLabel.append(mask)
    fileProcessing.FreeAllSlicedDataMem()
    countTotal = noElem_Fat + noElem_Muscle + noElem_notFat_notMuscle
    #convert lsImage to numpy array
    npImages = np.asarray(lsImage)
    # print(id(npImages))
    # del lsImage[:]
    #convert lsLabel to numpy array
    npLabels = np.asarray(lsLabel)
    end_time = time.time()
    total_time=end_time-start_time
    processTime = str(datetime.timedelta(seconds=total_time))
    print("Total time: ",processTime)
    file = open("TimeCutImages.txt","a") 
    currentTime = str(datetime.datetime.now())
    # print(currentTime)
    file.write(currentTime)
    file.write(",")
    file.write(imgFilePath)
    file.write(",")
    file.write(processTime) 
    file.write(",")
    file.write(str(end_time))
    file.write(",")
    file.write("\n")
    file.close() 
    # del lsImage[:]
    # # print(npImages)
    # print(npLabels)
    # print("shape npImages:",npImages.shape)
    # print("shape npLabel:",npLabels.shape)
    # print("len(npImages):",len(npImages))
    # print("len(npLabel):",len(npLabels))
    # print("noElem_Fat:",noElem_Fat)
    # print("noElem_Muscle:",noElem_Muscle)
    # print("noElem_notFat_notMuscle:",noElem_notFat_notMuscle)
    # print(countTotal)

    # fileProcessing.SaveImage(r"img.jpg");
    # fileProcessing.SaveMaskImage(r"mask.jpg");

    # print(outputData)
    # print(len(outputData))
    collect = gc.collect()
    print("Have collect ", collect)
    return npImages, npLabels, noElem_Fat,noElem_Muscle,noElem_notFat_notMuscle, countTotal

def extractDicomImageAndLabel2(imgFilePath, mskFilePath, slicedWidth, slicedHeight,train):
    fileProcessing.Processing(imgFilePath, mskFilePath, slicedWidth, slicedHeight)
    # create a row and a col list with value in range 0 and 511
    row = np.arange(0,512)
    col = np.arange(0,512)
    # row = np.arange(0,5) #This is only use for testing
    # col = np.arange(0,5) #This is only use for testing
    # print(type(row))
    # print(type(col))

    #shuffle row and col. 
    # Note: we only shuffle for training data.
    if train=="train":
        c = list(zip(row, col))
        shuffle(c)
        row, col = zip(*c)

    countTotal=0
    noElem_notFat_notMuscle = 0
    noElem_Fat = 0
    noElem_Muscle=0


    lsImage = []
    lsLabel = []
    #Create a loop to extract a 95x95 image and their mask
    print("Cutting Image")
    start_time = time.time()
    for r in row:
        for c in col:

            mask = fileProcessing.GetMaskForSlicedData(c, r) #Return a int scalar mask for outputData
            if mask==0:
                noElem_notFat_notMuscle = noElem_notFat_notMuscle + 1;
            elif mask==1:
                noElem_Fat = noElem_Fat +1
                # print("mask value:",mask)
            else:
                noElem_Muscle = noElem_Muscle +1

            #npImage = slicing_image3(c, r, slicedWidth, slicedHeight)
            npImage = slicing_image(c, r, slicedWidth, slicedHeight)
        
            # print(id(npImage))
            # print(npImage.dtype)
            #append image to a list
            lsImage.append(npImage)
            # print("*******")
            #append mask to a list
            lsLabel.append(mask)
    fileProcessing.FreeAllSlicedDataMem()
    countTotal = noElem_Fat + noElem_Muscle + noElem_notFat_notMuscle
    #convert lsImage to numpy array
    npImages = np.asarray(lsImage)
    # print(id(npImages))
    # del lsImage[:]
    #convert lsLabel to numpy array
    npLabels = np.asarray(lsLabel)
    end_time = time.time()
    total_time=end_time-start_time
    processTime = str(datetime.timedelta(seconds=total_time))
    print("Total time: ",processTime)
    file = open("TimeCutImages.txt","a") 
    currentTime = str(datetime.datetime.now())
    # print(currentTime)
    file.write(currentTime)
    file.write(",")
    file.write(imgFilePath)
    file.write(",")
    file.write(processTime) 
    file.write(",")
    file.write(str(end_time))
    file.write(",")
    file.write("\n")
    file.close() 
    # del lsImage[:]
    # # print(npImages)
    # print(npLabels)
    # print("shape npImages:",npImages.shape)
    # print("shape npLabel:",npLabels.shape)
    # print("len(npImages):",len(npImages))
    # print("len(npLabel):",len(npLabels))
    # print("noElem_Fat:",noElem_Fat)
    # print("noElem_Muscle:",noElem_Muscle)
    # print("noElem_notFat_notMuscle:",noElem_notFat_notMuscle)
    # print(countTotal)

    # fileProcessing.SaveImage(r"img.jpg");
    # fileProcessing.SaveMaskImage(r"mask.jpg");

    # print(outputData)
    # print(len(outputData))
    collect = gc.collect()
    print("Have collect ", collect)
    return npImages, npLabels, noElem_Fat,noElem_Muscle,noElem_notFat_notMuscle, countTotal


def slicing_image(c, r, slicedWidth, slicedHeight):
    # not use because of poor performance
    #img = fileProcessing.GetSlicedData(c, r) 
    #img = asNumpyArray(img)
    imgPointer = fileProcessing.GetSlicedDataAddress(c, r)
    imgSize    = slicedWidth * slicedHeight
    img = asNumpyArrayFromInt16Pointer(imgPointer, imgSize)
    npImage = np.asarray(img, np.float32)
    npImage = np.reshape(npImage, (95, 95, 1))
    return npImage

def extractDicomImage(imgFilePath, slicedWidth, slicedHeight, startIndex = 0, endIndex = 0):
    fileProcessing.Processing(imgFilePath, slicedWidth, slicedHeight)
    # create a row and a col list with value in range 0 and 511
    endSlicedIndex   = 512 * 512;
    lsImage = []
    #Create a loop to extract a 95x95 image and their mask
    print("[Cutting Image]")
    start_time = time.time()

    if endIndex != 0:
        endSlicedIndex = endIndex

    # SERIAL RUNNING
    for index in range(startIndex, endSlicedIndex):
        npImage = slicing_image(index, slicedWidth, slicedHeight)
        #npImage = slicing_image2(index, slicedWidth, slicedHeight)
        #append image to a list
        lsImage.append(npImage)

    # PARALEL RUNNING
    #indexInputs = range(currentIndex, endIndex)
    #pool = Pool(1)
    #results = pool.map(slicing_image, indexInputs)
    #for result in results:
    #    lsImage.append(result)

    fileProcessing.FreeAllSlicedDataMem()
    #convert lsImage to numpy array
    npImages = np.asarray(lsImage)
    # print(id(npImages))
    # del lsImage[:]
    end_time = time.time()
    total_time=end_time-start_time
    processTime = str(datetime.timedelta(seconds=total_time))
    print("Total time: ", processTime)
    print("[End cutting Image]")
    file = open("TimeCutImages.txt","a") 
    currentTime = str(datetime.datetime.now())
    # print(currentTime)
    file.write(currentTime)
    file.write(",")
    file.write(imgFilePath)
    file.write(",")
    file.write(processTime) 
    file.write(",")
    file.write(str(end_time))
    file.write(",")
    file.write("\n")
    file.close() 

    collect = gc.collect()
    print("Have collect ", collect)
    return npImages


#How to use it:
# wSize=95
# imgFilePath = "./DicomImages/000000.dcm"
# mskFilePath = "./DicomImagesMask/000000-mask.msk"
# npImages, npLabels, noElem_Fat,noElem_Muscle,noElem_notFat_notMuscle, countTotal = extractDicomImageAndLabel(imgFilePath, mskFilePath, wSize, wSize,"train")
# print(npImages.shape)
# print(id(npImages))

def sqrt(x):
    return np.sqrt(x)

#################################################
#Parallel image cutting
#################################################
def extractDicomImageAndLabel_para(imgFilePath, mskFilePath, slicedWidth, slicedHeight,train):
    fileProcessing.Processing(imgFilePath, mskFilePath, slicedWidth, slicedHeight)
    # create a row and a col list with value in range 0 and 511
    row = np.arange(0,512)
    # col = np.arange(0,512)
    # row = np.arange(0,5) #This is only use for testing
    # col = np.arange(0,5) #This is only use for testing
    # print(type(row))
    # print(type(col))

    #shuffle row and col. 
    # Note: we only shuffle for training data.
    start_time = time.time()
    pool = Pool()
    rootspara = pool.map(sqrt, range(512*512*512))
    print("rootspara:",len(rootspara))
    #Create a loop to extract a 95x95 image and their mask
    end_time = time.time()
    total_time=end_time-start_time
    processTime1 = str(datetime.timedelta(seconds=total_time))



    start_time = time.time()
    roots=[]
    for x in range(512*512*512):
        roots.append(sqrt(x))
    print("rootsnormal:",len(roots))
    #Create a loop to extract a 95x95 image and their mask
    end_time = time.time()
    total_time=end_time-start_time
    processTime = str(datetime.timedelta(seconds=total_time))
    print("Total time para: ",processTime1)
    print("Total time seri: ",processTime)

#How to use it:
# wSize=95
# imgFilePath = "./DicomImages/000000.dcm"
# mskFilePath = "./DicomImagesMask/000000-mask.msk"
# extractDicomImageAndLabel_para(imgFilePath, mskFilePath, wSize, wSize,"test")



def showNumpyArray(npArray):
    for i in range(len(npArray)):
        print (npArray[i])

def read_binary_file(filename, endian):
    count = stat(filename).st_size / 2
    with file(filename, 'rb') as f:
        result = array('h')
        result.fromfile(f, count)
        if endian != system_endian: result.byteswap()
        return result

def read_create_labels_shuffle_binary_files(pathPatternFile):
    #get all binary filenames from pathPatternFile and store in a list addrs
    addrs = glob.glob(pathPatternFile)
    #create labels for specify filenames in addrs and store in a list labels
    labels = []
    for i in addrs:
        if 'F' in i:
            labels.append(0)
        elif 'M' in i:
            labels.append(1)
        else:
            labels.append(2)

    #pairing (image,label) and shuffle them. Then return addrs and labels
    shuffle_data = True

    if shuffle_data:
        c = list(zip(addrs, labels))
        # print(c)
        shuffle(c)
        addrs, labels = zip(*c)
    return addrs, labels

def start_to_create_4D_numpy_array_data(wSize,addrs,endian):
    if len(addrs) <=0:
        print ("There is no data.")
        return -1
    img = read_binary_file(addrs[0],endian)
    npTemp1 = np.asarray(img,np.float32)
    img = read_binary_file(addrs[1],endian)
    npTemp2 =np.asarray(img,np.float32)
    img = read_binary_file(addrs[2],endian)
    npTemp3 =np.asarray(img,np.float32)
    img = read_binary_file(addrs[3],endian)
    npTemp4 =np.asarray(img,np.float32)
    npTemp1 = npTemp1.reshape(1,wSize,wSize,1)
    npTemp2 = npTemp2.reshape(1,wSize,wSize,1)
    npTemp3 = npTemp3.reshape(1,wSize,wSize,1)
    npTemp4 = npTemp4.reshape(1,wSize,wSize,1)
    npTemp1 = np.append(npTemp1,npTemp2,axis=0)
    npTemp1 = np.append(npTemp1,npTemp3,axis=0)
    npTemp1 = np.append(npTemp1,npTemp4,axis=0)
    # print("********************npTemp1**********************")
    # print(img)
    return npTemp1


def load_data_ai(wSize,pathPatternFile,endian):
    # 1. Read binary filename:
    addrs, label = read_create_labels_shuffle_binary_files(pathPatternFile)
    label = np.asarray(label,np.uint8)
    # print (addrs)
    # print (labels)
    data=start_to_create_4D_numpy_array_data(wSize,addrs,endian)
    # print(data.shape)
    for i in range(len(addrs)):
        # print i
        if i>3:
            img = read_binary_file(addrs[i],endian)
            npArray =np.asarray(img,np.float32)
            npArray = npArray.reshape(1,wSize,wSize,1)
            data = np.append(data,npArray,axis=0)
    # print("************************LOAD DATA********************")
    # print(data.shape)
    # print(data)
    return data, label


#Class DataSet
class DataSet(object):

  def __init__(self,
               images,
               labels,
               one_hot=False,
               dtype=np.float32,
               reshape=True,
               seed=None):
    assert images.shape[0] == labels.shape[0], (
        'images.shape: %s labels.shape: %s' % (images.shape, labels.shape))
    self._num_examples = images.shape[0]

    # Convert shape from [num examples, rows, columns, depth]
    # to [num examples, rows*columns] (assuming depth == 1)
    if reshape:
      assert images.shape[3] == 1
      images = images.reshape(images.shape[0],
                              images.shape[1] * images.shape[2])
    # if dtype == np.float32:
    #   # Convert from [0, 255] -> [0.0, 1.0].
    #   images = images.astype(numpy.float32)
    #   # images = numpy.multiply(images, 1.0 / 255.0) #Normalization of input data
    self._images = images
    self._labels = labels
    self._epochs_completed = 0
    self._index_in_epoch = 0

  @property
  def images(self):
    return self._images

  @property
  def labels(self):
    return self._labels

  @property
  def num_examples(self):
    return self._num_examples

  @property
  def epochs_completed(self):
    return self._epochs_completed

  def next_batch(self, batch_size, fake_data=False, shuffle=True):
    """Return the next `batch_size` examples from this data set."""
    if fake_data:
      fake_image = [1] * 784
      if self.one_hot:
        fake_label = [1] + [0] * 9
      else:
        fake_label = 0
      return [fake_image for _ in xrange(batch_size)], [
          fake_label for _ in xrange(batch_size)
      ]
    start = self._index_in_epoch
    # Shuffle for the first epoch
    if self._epochs_completed == 0 and start == 0 and shuffle:
      perm0 = numpy.arange(self._num_examples)
      numpy.random.shuffle(perm0)
      self._images = self.images[perm0]
      self._labels = self.labels[perm0]
    # Go to the next epoch
    if start + batch_size > self._num_examples:
      # Finished epoch
      self._epochs_completed += 1
      # Get the rest examples in this epoch
      rest_num_examples = self._num_examples - start
      images_rest_part = self._images[start:self._num_examples]
      labels_rest_part = self._labels[start:self._num_examples]
      # Shuffle the data
      if shuffle:
        perm = numpy.arange(self._num_examples)
        numpy.random.shuffle(perm)
        self._images = self.images[perm]
        self._labels = self.labels[perm]
      # Start next epoch
      start = 0
      self._index_in_epoch = batch_size - rest_num_examples
      end = self._index_in_epoch
      images_new_part = self._images[start:end]
      labels_new_part = self._labels[start:end]
      return numpy.concatenate(
          (images_rest_part, images_new_part), axis=0), numpy.concatenate(
              (labels_rest_part, labels_new_part), axis=0)
    else:
      self._index_in_epoch += batch_size
      end = self._index_in_epoch
      return self._images[start:end], self._labels[start:end]


# imgFilePath, mskFilePath,train,wSize,validation_size
def read_data_sets(imgFilePath,
                    mskFilePath,
                    train,
                    wSize,
                    one_hot=False,
                    dtype=np.float32,
                    reshape=True,
                    seed=None):
    fileProcessing.Processing(imgFilePath, mskFilePath, wSize, wSize)


    images, labels, noElem_Fat,noElem_Muscle,noElem_notFat_notMuscle, countTotal = extractDicomImageAndLabel2(imgFilePath, mskFilePath, wSize, wSize,train)
    # train_images,train_labels = load_data_ai(wSize,pathPatternFile,endian)
    print("train_image shape:",images.shape)
    print("train_labels shape", labels.shape)
    print("noElem_Fat:",noElem_Fat)
    print("noElem_Muscle:",noElem_Muscle)
    print("noElem_notFat_notMuscle:",noElem_notFat_notMuscle)
    print("Total: ",countTotal)

    options = dict(dtype=dtype, reshape=reshape, seed=seed)
    data = DataSet(images, labels, **options)

    return Datasets(data=data)


######################
#load_ai_dataset
######################

def load_ai_dataset(imgFilePath, mskFilePath, train,wSize):
  datasets=read_data_sets(imgFilePath, mskFilePath,train,wSize)
  return datasets




#How to use load_ai_dataset
# imgFilePath = r"./DicomImages/img1.dcm"
# mskFilePath = r"./DicomImagesMask/img1_mask.msk"
# wSize = 95
# datasets = load_ai_dataset(imgFilePath, mskFilePath, train="train",wSize=95,validation_size=8)
# train_data = datasets.train.images  # Returns np.array
# train_labels = np.asarray(datasets.train.labels, dtype=np.int32)
# eval_data = datasets.validation.images  # Returns np.array
# eval_labels = np.asarray(datasets.validation.labels, dtype=np.int32)
# print("****************")
# print("Rank of train_data",train_data.ndim) #xem rank cua du lieu
# print("Shape of train_data",train_data.shape) #xem chieu cua du lieu
# print("Type of train_data",type(train_data[0])) # Kieu cua train_data: numpy.ndarray
# print("Cotent of 1st train_data:",train_data[0]) # xem noi dung du lieu thu nhat

# pathPatternFileDicomImage = './DicomImages/*.dcm'
# pathPatternFileDicomImageMask = './DicomImagesMask/*.msk'
def read_path_file(folder):
	dcmPaths = []
	maskPaths = []
	errorPaths = []
	# Read all file dcm in folder
	for root, dirs, files in os.walk(folder):
	    for file in files:
	        if file.endswith('.dcm'):
	            dcmPaths.append(os.path.join(root, file))
	for dcmFilePath in dcmPaths:
		tmpPath = os.path.split(dcmFilePath)
		maskFileName = tmpPath[1].replace(".dcm","-mask.msk")
		head,tail = os.path.split(tmpPath[0])
		maskFilePath = os.path.join(head,'MaskBinary',maskFileName)
		if os.path.isfile(maskFilePath):
			maskPaths.append(maskFilePath)
		else:
			errorPaths.append(dcmFilePath)

	if len(errorPaths) > 0:
		for path in errorPaths:
			dcmPaths.remove(path)
		print("List dcm file doesn't exist mask file: ")
		print(errorPaths)
	print("total dicom file: {0}".format(len(dcmPaths)))
	print("total mask file: {0}".format(len(maskPaths)))
	return dcmPaths, maskPaths

def read_dicom_image_file_and_mask(pathPatternFileDicomImage):
    #get all binary filenames from pathPatternFileDicomImage and store in a list addrs
    addrs_dicom_image, addrs_dicom_image_mask = read_path_file(pathPatternFileDicomImage)
    train=[]
    val=[]
    test=[]
    for t in addrs_dicom_image:
        c = list(zip(addrs_dicom_image, addrs_dicom_image_mask))
        shuffle(c)
        addrs, labels = zip(*c)
        # Divide the hata into 60% train, 20% validation, and 20% test
        train_addrs = addrs[0:int(0.6*len(addrs))]
        train_labels = labels[0:int(0.6*len(labels))]
        train = list(zip(train_addrs, train_labels))
        
        val_addrs = addrs[int(0.6*len(addrs)):int(0.8*len(addrs))]
        val_labels = labels[int(0.6*len(addrs)):int(0.8*len(addrs))]
        val = list(zip(val_addrs, val_labels))

        test_addrs = addrs[int(0.8*len(addrs)):]
        test_labels = labels[int(0.8*len(labels)):]
        test = list(zip(test_addrs, test_labels))

    return train, val, test

#How to use it:
# train, val, test = read_dicom_image_file_and_mask(pathPatternFileDicomImage)
# # for i in train:
# #     print(i[0])
# #     print(i[1])
# #     print("\n")
# print(len(train))
# print(len(val))
# print(len(test))

# train_images, val_images, test_images = read_dicom_image_file_and_mask(pathPatternFileDicomImage)
# print(len(train_images))
# print(len(val_images))
# print(len(test_images))


# for train, val, test in zip(train_images, val_images, test_images):
#     # if count==0:
#     imgFilePath_train = train[0] #Dicom image filename path
#     mskFilePath_train = train[1] #Mask filename path

#     imgFilePath_val = val[0]
#     mskFilePath_val = val[1]

#     imgFilePath_test = test[0]
#     mskFilePath_test = test[1]
#     print("*************Train data************")
#     print(imgFilePath_train)
#     print(mskFilePath_train)
#     print("*************Evaluation data************")
#     print(imgFilePath_val)
#     print(mskFilePath_val)
#     print("*************Test data************")
#     print(imgFilePath_test)
#     print(mskFilePath_test)
