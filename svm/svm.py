import numpy as np
import matplotlib.pyplot as plt
import sys

def generate_training_data_binary(num):
    if num == 1:
        data = np.zeros((10,3))
        for i in range(5):
            data[i] = [i-5, 0, 1]
            data[i+5] = [i+1, 0, -1]

    elif num == 2:
        data = np.zeros((10,3))
        for i in range(5):
            data[i] = [0, i-5, 1]
            data[i+5] = [0, i+1, -1]

    elif num == 3:
        data = np.zeros((10,3))
        data[0] = [3, 2, 1]
        data[1] = [6, 2, 1]
        data[2] = [3, 6, 1]
        data[3] = [4, 4, 1]
        data[4] = [5, 4, 1]
        data[5] = [-1, -2, -1]
        data[6] = [-2, -4, -1]
        data[7] = [-3, -3, -1]
        data[8] = [-4, -2, -1]
        data[9] = [-4, -4, -1]
    elif num == 4:
        data = np.zeros((10,3))
        data[0] = [-1, 1, 1]
        data[1] = [-2, 2, 1]
        data[2] = [-3, 5, 1]
        data[3] = [-3, -1, 1]
        data[4] = [-2, 1, 1]
        data[5] = [3, -6, -1]
        data[6] = [0, -2, -1]
        data[7] = [-1, -7, -1]
        data[8] = [1, -10, -1]
        data[9] = [0, -8, -1]
    elif num == 5:
        data = np.zeros((4,3))
        data[0] = [-1,-1,-1]
        data[1] = [-1,1,1]
        data[2] = [1,1,-1]
        data[3] = [1,-1,1]

    else:
        print("Incorrect num", num, "provided to generate_training_data_binary.")
        sys.exit()
    return data

def generate_training_data_multi(num):
    if num == 1:
        data = np.zeros((20,3))
        for i in range(5):
          data[i] = [i-5, 0, 1]
          data[i+5] = [i+1, 0, 2]
          data[i+10] = [0, i-5, 3]
          data[i+15] = [0, i+1, 4]
        C = 4

    elif num == 2:
        data = np.zeros((15,3))
        data[0] = [-5, -5, 1]
        data[1] = [-3, -2, 1]
        data[2] = [-5, -3, 1]
        data[3] = [-5, -4, 1]
        data[4] = [-2, -9, 1]
        data[5] = [0, 6, 2]
        data[6] = [-1, 3, 2]
        data[7] = [-2, 1, 2]
        data[8] = [1, 7, 2]
        data[9] = [1, 5, 2]
        data[10] = [6, 3, 3]
        data[11] = [9, 2, 3]
        data[12] = [10, 4, 3]
        data[13] = [8, 1, 3]
        data[14] = [9, 0, 3]
        C = 3

    else:
        print("Incorrect num", num, "provided to generate_training_data_binary.")
        sys.exit()

    return [data, C]

def plot_training_data_binary(data):
  for item in data:
    if item[-1] == 1:
      plt.plot(item[0], item[1], 'b+')
    else:
      plt.plot(item[0], item[1], 'ro')
  m = max(data.max(), abs(data.min()))+1
  plt.axis([-m, m, -m, m])
  plt.show()


#PART 1
def plot_training_data_multi(data):
  colors = ['b', 'r', 'g', 'm']
  shapes = ['+', 'o', '*', '.']

  for item in data:
    plt.plot(item[0], item[1], colors[int(item[2])-1] + shapes[int(item[2])-1])
  m = max(data.max(), abs(data.min()))+1
  plt.axis([-m, m, -m, m])
  plt.show()

#Finds the distance of every point to the hyperplane
def distance_point_to_hyperplane(pt, w, b):
    pt = np.array(pt)
    w = np.array(w)
    return ((pt[0]*w[0])+ (pt[1]*w[1]) + b) / (np.sqrt( (w[0]*w[0]) + (w[1]*w[1]) ))

#returns 1 if x is above the hyperplane or -1 if it is below  or on the hyperplane
def svm_test_brute(w,b,x):
    d = ( w[0]*x[0]) + (w[1]*x[1]) + b
    if d > 0:
        return 1
    else:
        return -1

#Computes distance between every point and hyperplane.
#margin is the smallest distance between point and hyperplane
def compute_margin(data,w,b):
    data = np.array(data)
    #dist_min stores the minimum distance between a point and a hyperplane
    dist_min = np.abs(distance_point_to_hyperplane(data[0], w, b))
    for pt in data:
        dist = np.abs(distance_point_to_hyperplane(pt, w, b))
        if dist < dist_min:
            dist_min = dist

    return dist_min


#gets all the pairs of support vectors that correctly classify the data
def get_support_vectors2(class1, class2):
    support_vectors2 = []
    #Finding all the sets of 2 support vectors
    for c1 in class1:
        for c2 in class2:
            w1 = c1[0:2] - c2[0:2]
            #magnitude_w = np.sqrt(w.dot(w))
            #midpoint in the midpoint of p and n
            midpoint = (c1[0:2] + c2[0:2])/2

            w = np.array([w1[0],w1[1]])
            b = -(w[0]*midpoint[0] +  w[1]*midpoint[1])

            pred_class1 = [ svm_test_brute(w,b,pt) for pt in class1[:,0:2] ]
            pred_class2 = [ svm_test_brute(w,b,pt) for pt in class2[:,0:2] ]

            if (np.unique(pred_class2)[0] != np.unique(pred_class1)[0] and len(np.unique(pred_class1)) == 1) and (len(np.unique(pred_class2)) == 1 ):
                support_vectors2.append([w,b,c1,c2])

    return support_vectors2

#gets all possible triplet support vectors that correctly classify the data
def get_support_vectors3(class1,class2):
    support_vectors3 = []
    #Finding all the set of 3 support vectors that correctly classify the class:
    for i in range(0,len(class1)-1):
        for j in range(i+1, len(class1)):
            for k in range(0, len(class2)):
                #w is the gradient of the line between pos[i] and pos[j]
                w1 = class1[i,0:2]-class1[j,0:2]
                w1_unit_v = w1/np.sqrt(w1.dot(w1))
                #w2 is the gradient of the line perpendicular to line
                w2 = class2[k,0:2]-class1[j,0:2]

                proj = w2.dot(w1_unit_v)

                mid = ((proj * w1_unit_v) + class1[j,0:2] +  class2[k,0:2])/2
                b = (-mid[1]*w1_unit_v[0]) +(mid[0]*w1_unit_v[1])
                w = [-w1_unit_v[1],w1_unit_v[0]]

                pred_class1 = [ svm_test_brute(w,b,pt) for pt in class1[:,0:2] ]
                pred_class2 = [ svm_test_brute(w,b,pt) for pt in class2[:,0:2] ]

                if (np.unique(pred_class2)[0] != np.unique(pred_class1)[0] and len(np.unique(pred_class1)) == 1) and (len(np.unique(pred_class2)) == 1 ):
                    support_vectors3.append([w,b,class1[i],class1[j],class2[k]])

    return support_vectors3

#returns the decision boundary and a list of support vectors for the svm
def svm_train_brute(training_data):
    data = np.array(training_data)

    #divide the data into the the two classes representing label 1 and -1
    pos = []
    neg = []
    for d in data:
        if d[2] == 1:
            pos.append(d)
        else:
            neg.append(d)
    pos = np.array(pos)
    neg = np.array(neg)

    #support_vectors2 stores pairs of support vectors
    support_vectors2 = np.array(get_support_vectors2(pos, neg))
    #Find the best support vectors among all pairs in support_vectors2
    largest_margin2 = 0
    smallest_distance2  = 0
    s_v2 = [[0,0],0,[0,0],[0,0]]
    if (len(support_vectors2) > 0):
        largest_margin2 = compute_margin(data,support_vectors2[0,0], support_vectors2[0,1])
        smallest_distance2 = np.abs(distance_point_to_hyperplane(support_vectors2[0,2], support_vectors2[0,0], support_vectors2[0,1]))
        s_v2 = support_vectors2[0]   #best support vector so far among the support_vectors2
        for pt in support_vectors2:
            margin = compute_margin(data, pt[0],pt[1])
            distance = np.abs(distance_point_to_hyperplane(pt[2],pt[0], pt[1]))
            if margin >= largest_margin2 and distance < smallest_distance2:
                largest_margin2 = margin
                smallest_distance2 = distance
                s_v2 = pt

    #support_vectors3 stores all triplets of possible support vectors
    support_vectors3 = get_support_vectors3(pos,neg)
    support_vectors3.extend(get_support_vectors3(neg,pos))
    support_vectors3 = np.array(support_vectors3)
    #find the best support_vectors among all triplets in support_vectors3
    largest_margin3 = 0
    smallest_distance3  = 0
    s_v3 = [[0,0],0,[0,0],[0,0],[0,0]]
    if (len(support_vectors3) > 0):
        largest_margin3 = compute_margin(data,support_vectors3[0,0], support_vectors3[0,1])
        smallest_distance3 = np.abs(distance_point_to_hyperplane(support_vectors3[0,2], support_vectors3[0,0], support_vectors3[0,1]))
        s_v3 = support_vectors3[0] #best support vector so far among the support_vectors3
        for pt in support_vectors3:
            margin = compute_margin(data, pt[0],pt[1])
            distance = np.abs(distance_point_to_hyperplane(pt[2],pt[0], pt[1]))
            #and distance < smallest_distance3
            if margin >= largest_margin3 :
                largest_margin3 = margin
                smallest_distance3 = distance
                s_v3 = pt

    #Find out between s_v2 and s_v3, which has the best margin
    #and smallest_distance2 < smallest_distance
    if largest_margin2 >= largest_margin3 :
        return (s_v2[0],s_v2[1], s_v2[2:4])
    else:
        return (s_v3[0],s_v3[1], s_v3[2:5])

############################################################
#PART 2

#returns decision boundaries of one vs rest classifiers
def svm_train_multiclass(training_data):
    W = []
    B = []
    train_data = np.array(training_data)
    num_classes = len(np.unique(train_data[:,2]))

    #loop through the data and get one vs rest decision boundaries for each class
    for c in range(1,num_classes+1):
        data_m = np.copy(train_data)
        data = []
        #transform the data so that it can be fed into svm_train_brute method
        for d in data_m:
            if d[2] == c:
                d[2] = 1
                data.append(d)
            else:
                d[2] = -1
                data.append(d)

        data_np = np.array(data)
        w,b,s = svm_train_brute(data_np)
        W.append(w)
        B.append(b)
    return W,B

#function takes in a bunch of one vs rest decision boundaries and
#returns the class of point x
def svm_test_multiclass(W,B,x):
    W = np.array(W)
    B = np.array(B)

    prediction = -1
    distance_from_hyperplane = 0
    for i in range(0,len(W)):
        pred = svm_test_brute(W[i],B[i],x)
        dist = np.abs(distance_point_to_hyperplane(x,W[i],B[i]))
        if prediction == -1 and pred == 1:
            prediction = i
        elif pred == 1 and dist > distance_from_hyperplane:
            prediction = i
            distance_from_hyperplane = dist

    return prediction


def kernel_svm_train(training_data):
    data = np.array(training_data)

    #kernel transformation
    for d in data:
        d[1] = d[0]*d[1]

    w,b,s = svm_train_brute(data)
    return (w,b)
