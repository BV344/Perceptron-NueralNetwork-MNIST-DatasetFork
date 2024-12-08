# Group Members
# Ian Boskin, Luis Carmona, and Brayan Vizcaino

from Function import *
import numpy as np


data_7_t = './Train/train7.csv'
data_7_v = './Valid/valid7.csv'
data_9_t = './Train/train9.csv'
data_9_v = './Valid/valid9.csv'



data7_t = readFile(data_7_t)
data9_t = readFile(data_9_t)

dataset_7_t = get_images_and_Labels(data7_t)
dataset_9_t = get_images_and_Labels(data9_t)


train_data7, test_data7 = train_test_split(dataset_7_t, test_size=0.2)
train_data9, test_data9 = train_test_split(dataset_9_t, test_size=0.2)

val7 = readFile(data_7_v)
val7= get_images_and_Labels(val7)

val9 = readFile(data_9_v)
val9= get_images_and_Labels(val9)

combined_train = train_data7 + train_data9
combined_test = test_data7 + test_data9

random.shuffle(combined_train)
VAL = val7 +val9
best_weights,min_error_frac = perceptron_train(combined_train,VAL,epochs=10000)


X_test = []
y_test = []

for label, img in combined_test:
    features = extract_features(img)
    X_test.append(features)
    y_test.append(1 if label == 9 else -1)


X_test = np.array(X_test)
y_test = np.array(y_test)
X_test = (X_test - X_test.mean(axis=0)) / X_test.std(axis=0)

test_predictions = np.sign(np.dot(X_test, best_weights))
test_accuracy = np.mean(test_predictions == y_test)

# for prediction in test_predictions:
#     if prediction == 1:
#         print(9)
#     else: 
#         print(7)

print("PART 1:")
print("_________________________________")
print(f"Optimal Weights: {best_weights}")
print(f"Min Error Fraction: {min_error_frac}")
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")


Train_dir = './Train'
Val_dir = './Valid'


Train_dataset = extract_folder(Train_dir)
Valid_dataset = extract_folder(Val_dir)


best_weights,min_error_frac = perceptron_train_mulitClass(Train_dataset, Valid_dataset)

testFile ='test1.csv'
testFile = readFile(testFile)
test_data = get_images_and_Labels(testFile)

X_test = []
y_test = []

for label, img in test_data:
    features = extract_features(img)
    X_test.append(features)
    y_test.append(label)  

X_test = np.array(X_test)
y_test = np.array(y_test)


X_test = (X_test - X_test.mean(axis=0)) / X_test.std(axis=0)

test_predictions = np.argmax(np.dot(X_test, best_weights.T), axis=1)
test_accuracy = np.mean(test_predictions == y_test)


print("PART 2:")
print("_________________________________")
print(f"Optimal Weights: {best_weights}")
print(f"Min Error Fraction: {min_error_frac}")
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

W1,W2,best_success = TestBackpropMmt(Train_dataset,Valid_dataset)


test_data = get_images_and_Labels(testFile)

X_test = []
y_test = []

for Label, img in test_data:
    binImg = img_to_binary(img)
    X_test.append(binImg.flatten()) 
    y_test.append(Label)  

X_test = np.array(X_test)
y_test = np.array(y_test)
y_test = one_hot_encode(y_test , 10)



correct_predictions = 0
N= X_test.shape[0]
for k in range(N):
    x = X_test[k]
    v1 = np.dot(W1, x)
    y1 = Sigmoid(v1)
    v = np.dot(W2, y1)
    y = softmax(v)
    
    predicted_label = np.argmax(y)
    true_label = np.argmax(y_test[k])
    
    if predicted_label == true_label:
        correct_predictions += 1

accuracy = correct_predictions / N

test_accuracy = correct_predictions / X_test.shape[0]


print("PART 3:")
print("_________________________________")
# print(f"Best Weights: {best_weights}")
print(f"Best Success Ratio: {best_success}")
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")




    
