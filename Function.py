import csv
import cv2
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import copy
import random
import os 
import math
from scipy.ndimage import sobel, label
import numpy as np
from Sigmoid import *

def readFile(File):
    dataset =[]
    with open(File, 'r', encoding='utf-8-sig') as file:
        csv_reader = csv.reader(file)
        for row in csv_reader:
            dataset.append(row)
    
    return dataset

def get_images_and_Labels(dataset):
    data= []
    for row in dataset:
        label = int(row[0])
        image = np.array(row[1:], dtype=np.uint8).reshape(28, 28)
        data.append([label,image])
    return data

def Density(img):
    totalSum = np.sum(img)
    avg = totalSum / img.size
    return avg

def measureSymmetry(img):
    imgReversed = np.flipud(img)  
    xorIMG = np.bitwise_xor(img, imgReversed)  
    denOFxor = Density(xorIMG)  
    return denOFxor

def grayToBin(img):
    binImage = np.where(img > 128, 0, 1)
    return binImage


def VertivalIntersections(img):

    binImg = grayToBin(img)
    AllIntersections = []

    for col in range(img.shape[1]):
        col_intersections =0
        for row in range(1,img.shape[0]):

            
            if binImg [row][col] != binImg [row-1][col]:
                col_intersections+=1
        
        AllIntersections.append(col_intersections)
    
    MaxIntersection= max(AllIntersections)
    avg = sum(AllIntersections)/28

    return MaxIntersection, avg

def HorizontalIntersection(img):

    binImg =grayToBin(img)
    AllIntersections= []

   
    for row in range(img.shape[0]):
        row_intersections=0
        for col in range(1,img.shape[1]):
            
            if binImg[row][col] != binImg[row][col-1]:
                row_intersections +=1
        
        AllIntersections.append(row_intersections)

    maxIntersection = max(AllIntersections)
    avg = sum(AllIntersections)/28

    
    return maxIntersection , avg

def central_pixel_density(img):
    
    center_region = img [9:19, 9:19]  
    total_sum = np.sum(center_region)
    density = total_sum / (10 * 10)  
    return density

def number_of_loops(img):
    bin_img = img > 0
    labeled_array, num_features = label(bin_img)
    return num_features

def aspect_ratio(img):
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    aspect_ratio = (cmax - cmin + 1) / (rmax - rmin + 1)
    return aspect_ratio



def number_of_corners(img):
    img_float = np.float32(img)
    dst = cv2.cornerHarris(img_float, blockSize=2, ksize=3, k=0.04)
    dst = cv2.dilate(dst, None)
    corners = dst > 0.01 * dst.max()
    num_corners = np.sum(corners)
    return num_corners

def density_per_quadrant(img):
    h, w = img.shape
    half_h, half_w = h // 2, w // 2
    
    # Divide the image into four quadrants
    top_left = img[:half_h, :half_w]
    top_right = img[:half_h, half_w:]
    bottom_left = img[half_h:, :half_w]
    bottom_right = img[half_h:, half_w:]
    
    # Calculate the density of pixels in each quadrant
    density_tl = np.sum(top_left) / top_left.size
    density_tr = np.sum(top_right) / top_right.size
    density_bl = np.sum(bottom_left) / bottom_left.size
    density_br = np.sum(bottom_right) / bottom_right.size
    
    return density_tl, density_tr, density_bl, density_br



def edge_density(img):
    edges = sobel(img)
    edge_density = np.sum(edges > 0) / img.size
    return edge_density

def extract_features(img):
    density = Density(img)
    symmetry = measureSymmetry(img)
    horizonatalMax, horizontalAvg = HorizontalIntersection(img)
    verticalMax, verticalAvg = VertivalIntersections(img)
    numLoops = number_of_loops(img)
    cornerpts = number_of_corners(img)
    dtL,dtR,dbL,dbR = density_per_quadrant(img)
    ed = edge_density(img)
    
    

    features=[
        density,
        symmetry,
        horizonatalMax,
        horizontalAvg,
        verticalMax,
        verticalAvg,
        numLoops,
        cornerpts,
        dtL,
        dtR,
        dbL,
        dbR,
        ed,
        
        
    ]
    return np.array(features)


def perceptron_train(Train,Val,epochs=1000, lr=0.01):


    X_train = []
    y_train = []
    X_valid =[]
    y_valid =[]
    for imgidx in range(len(Train)):
        label, img = Train[imgidx]
        features = extract_features(img)
        X_train.append(features)
        y_train.append(1 if label == 9 else -1)
    
    for imgidx in range(len(Val)):
        label, img = Val[imgidx]
        features = extract_features(img)
        X_valid.append(features)
        y_valid.append(1 if label == 9 else -1)

    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_valid = np.array(X_valid)
    y_valid = np.array(y_valid)

    X_train = (X_train - X_train.mean(axis=0)) / X_train.std(axis=0)
    X_valid = (X_valid - X_valid.mean(axis=0)) / X_valid.std(axis=0)
    
    
    weights = np.random.uniform(-0.1, 0.1, X_train.shape[1])
    best_weights = weights
    min_error = float('inf')
    min_error_fraction = 0.0

    for epoch in range(epochs):
        for x, y in zip(X_train, y_train):
            prediction = np.sign(np.dot(weights, x))
            weights += lr * (y - prediction) * x
        # Validate
        valid_predictions = np.sign(np.dot(X_valid, weights))
        error = np.mean(valid_predictions != y_valid)
        if error < min_error:
            min_error = error
            min_error_fraction = error
            best_weights = weights.copy()
    return best_weights,min_error_fraction
    


def extract_folder(Dir):
    filenames = sorted(os.listdir(Dir))
    Files=[]
    for file in filenames:
        file_path = os.path.join(Dir, file)
        if os.path.isfile(file_path):
            Files.append(file_path)

    DataSetFiles = []
    for file in Files:
        DataSetFiles.append(readFile(file))
    
    Dataset = []

    for file in DataSetFiles:
        Dataset += get_images_and_Labels(file)
        

    return Dataset


def perceptron_train_mulitClass(Train,Val,epochs = 1000, lr= 0.01):

    X_train = []
    y_train = []
    X_valid =[]
    y_valid =[]

    for imgidx in range(len(Train)):
        label, img = Train[imgidx]
        features = extract_features(img)
        X_train.append(features)
        y_train.append(label)
    
    for imgidx in range(len(Val)):
        label, img = Val[imgidx]
        features = extract_features(img)
        X_valid.append(features)
        y_valid.append(label)

    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_valid = np.array(X_valid)
    y_valid = np.array(y_valid)

    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    X_train = (X_train - mean) / std
    X_valid = (X_valid - mean) / std


    num_classes = 10
    weights = np.random.uniform(-0.1, 0.1, (num_classes, X_train.shape[1]))
    best_weights = weights.copy()
    min_error = float('inf')
    min_error_fraction = 0.0

    for epoch in range(epochs):
        for x, y in zip(X_train, y_train):
            scores = np.dot(weights, x)
            prediction = np.argmax(scores)
            if prediction != y:
                weights[y] += lr * x
                weights[prediction] -= lr * x
        
        #validate
        valid_predictions = np.argmax(np.dot(X_valid, weights.T), axis=1)
        error = np.mean(valid_predictions != y_valid)
        if error < min_error:
            min_error = error
            min_error_fraction = error
            best_weights = weights.copy() 

    print(f"Minimum Error Fraction: {min_error_fraction}")
    return best_weights, min_error_fraction








def img_to_binary(img):
   
    binary_img = (img >= 128).astype(np.uint8)
    
    return binary_img


def softmax(v):
    exp_v = np.exp(v - np.max(v))  
    return exp_v / np.sum(exp_v)


def BackProp(W1, W2, X, D):
    alpha = 0.1
    N = X.shape[0]
    for k in range(N):
        x = X[k].reshape(-1, 1) 
        d = D[k, :].reshape(-1, 1)  
        
        v1 = np.dot(W1, x)
        y1 = Sigmoid(v1)
        v = np.dot(W2, y1)
        y = softmax(v)
        
        e = d - y
        delta = e
        e1 = np.dot(W2.T, delta)
        delta1 = y1 * (1 - y1) * e1  
        
        dW1 = alpha * np.dot(delta1, x.T)
        W1 = W1 + dW1
        
        dW2 = alpha * np.dot(delta, y1.T)
        W2 = W2 + dW2
        
    return W1, W2




def one_hot_encode(labels, num_classes):
    return np.eye(num_classes)[labels]

def TestBackpropMmt(Train, Val):
    print("Starting BackProp Training")
    X_train = []
    y_train = []
    X_valid = []
    y_valid = []

    for imgidx in range(len(Train)):
        label, img = Train[imgidx]
        binImg = img_to_binary(img)
        X_train.append(binImg.flatten()) 
        y_train.append(label)
    
    for imgidx in range(len(Val)):
        label, img = Val[imgidx]
        binImg = img_to_binary(img)
        X_valid.append(binImg.flatten()) 
        y_valid.append(label)


    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_valid = np.array(X_valid)
    y_valid = np.array(y_valid)
    y_train = one_hot_encode(y_train, 10)
    y_valid = one_hot_encode(y_valid, 10)

    print("X_train shape:", X_train.shape) 
    print("X_valid shape:", X_valid.shape) 

    
    input_size = 784
    hidden_size = 100
    output_size = 10
    
    W1 = 2 * np.random.random((hidden_size, input_size)) - 1
    W2 = 2 * np.random.random((output_size, hidden_size)) - 1
    
    best_accuracy = 0
    best_W1 = W1.copy()
    best_W2 = W2.copy()

    for _epoch in range(1000):
        
        W1, W2 = BackProp(W1, W2, X_train, y_train)
        N = X_valid.shape[0]
        correct_predictions = 0

        for k in range(N):
            x = X_valid[k]
            v1 = np.dot(W1, x)
            y1 = Sigmoid(v1)
            v = np.dot(W2, y1)
            y = softmax(v)
            
            predicted_label = np.argmax(y)
            true_label = np.argmax(y_valid[k])
            
            if predicted_label == true_label:
                correct_predictions += 1
        
        accuracy = correct_predictions / N
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            print(f"Best accuracy: {best_accuracy}  Epoch:{_epoch}")
            best_W1 = W1.copy()
            best_W2 = W2.copy()
        if best_accuracy>0.9:
            break
    
    print(f"Best Accuracy after training: {best_accuracy}")
    return best_W1, best_W2, best_accuracy




