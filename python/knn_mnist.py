# check images and labels

import numpy as np
import matplotlib.pyplot as plt

def fetch_data():
    #from tensorflow.examples.tutorials.mnist import input_data    

    mnist = input_data.read_data_sets("./data/MNIST_data/", one_hot=True)

    print ("mnist type:" , type(mnist) )
    print ("train size:" , mnist.train.num_examples )
    
    print ("validation size: ", mnist.validation.num_examples )
    print ("test size:",mnist.test.num_examples )
    return mnist

mnist = fetch_data()

def data_type_check(mnist):
    

    print("讓我們看一下 MNIST 訓練還有測試的資料集長得如何")
    train_img = mnist.train.images
    train_label = mnist.train.labels
    test_img = mnist.test.images
    test_label = mnist.test.labels
    print
    print(" train_img 的 type : %s" % (type(train_img)))
    print(" train_img 的 dimension : %s" % (train_img.shape,))
    print(" train_label 的 type : %s" % (type(train_label)))
    print(" train_label 的 dimension : %s" % (train_label.shape,))
    print(" test_img 的 type : %s" % (type(test_img)))
    print(" test_img 的 dimension : %s" % (test_img.shape,))
    print(" test_label 的 type : %s" % (type(test_label)))
    print(" test_label 的 dimension : %s" % (test_label.shape,))
#data_type_check(mnist)
def data_content_check(data):
    
    trainimg = mnist.train.images
    trainlabel = mnist.train.labels
    print(trainimg.shape)
    print (trainlabel.shape)

    #idx = 1
    for idx in range(0,3):
        curr_img   = np.reshape(trainimg[idx, :], (28, 28)) # 28 by 28 matrix 
        curr_label = np.argmax(trainlabel[idx, :] ) # Label
        print("train label:", curr_label)
        plt.matshow(curr_img, cmap=plt.get_cmap('gray'))
        plt.title("" + str(idx+1) + "th Training Data " 
                  + "Label is " + str(curr_label))

#data_content_check(mnist)

trainimg = mnist.train.images
trainlabel = mnist.train.labels

# here the image for both train and test are 
def calc_dis(train_image,test_image):
    dist=np.linalg.norm(train_image-test_image)
    return dist


def one_hot2number(one_hot):
    xx = np.nonzero(one_hot)
    return xx[0][0]

def knn_find_labels(train_images, train_labels, test_image,k):
    labels = []
    all_dis = []
    for i in range(len(train_images)):
        t_img = train_images[i,:]
        dis = calc_dis(t_img, test_image)
        all_dis.append(dis)
    # sort according to list item
    # image with smallest distance is nearest image compared to test_image here.
    sorted_dis = np.argsort(all_dis)
    for i in range(0,k):
        #print("top ",i)
        idx = sorted_dis[i]
        #print(" idx of top is: ", idx)
        labels.append(train_labels[idx,:])
    return labels

test_img = mnist.test.images
test_label = mnist.test.labels

def top1_label(labels):
    label = []
    for l in labels:
        label.append(one_hot2number(l))
    counts = np.bincount(label)
    return np.argmax(counts)


K = 3

def test_knn(test_img, test_label, trainimg, testlabel):
    #for i in range(len(test_img)):
    total = len(test_img)
    hit = 0
    for i in range(total):
        test_1 = test_img[i,:]
        test_1_gt = test_label[i,:]
        
        test_1_img   = np.reshape(test_1, (28, 28))
        #plt.matshow(test_1_img, cmap=plt.get_cmap('gray'))
        labels = knn_find_labels(trainimg, trainlabel,test_1,K)
        #print("labels: ", labels)
        #print("labels: ", top1_label(labels))
        p_n = top1_label(labels)
        gt_n = one_hot2number(test_1_gt)
        
        curr_img   = np.reshape(test_1, (28, 28)) # 28 by 28 matrix 
        #if debug:
            #plt.matshow(curr_img, cmap=plt.get_cmap('gray'))
            #plt.title("" + str(i+1) + "th  " 
            #          + " gt Label is " + str(gt_n) + " predicted label is: " + str(p_n))

        #print("")
        if p_n == gt_n:
           hit +=1 
    return float(hit)/total

        #print(" predicted test 1 img is: ", p_n)
        #print(" gt of test 1 img is: ", gt_n)



x = test_knn(test_img, test_label, trainimg, trainlabel)
print("precision is : ", str(x))