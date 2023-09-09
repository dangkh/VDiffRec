import csv
import numpy as np
user, item = [0]*6100, [0]*4000
maxu, maxi = 0, 0
uiMat = {}
with open('../datasets/ml-1m/dml-1m.csv', newline='') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
    counter = 0
    for row in spamreader:
        if counter >= 1:
            u, i = [int(x) for x in row[0].split(',')]
            if str(u) not in uiMat:
                uiMat[str(u)] = []
            
            uiMat[str(u)].append(i)

            user[u] = 1
            item[i] = 1
            maxu = max(maxu, u)
            maxi = max(maxi, i)
        counter += 1
print("num of interact", counter)
# print(maxu, maxi)
# print(sum(user), sum(item))
train_path = '../datasets/ml-1m_clean/train_list.npy'
train_list = np.load(train_path, allow_pickle=True)

uiMat2 = {}
counter = 0
maxu, maxi = 0, 0
for u, i in train_list:
    if str(u) not in uiMat2:
        uiMat2[str(u)] = []
    
    uiMat2[str(u)].append(i)
    counter += 1
    maxu = max(maxu, u)
    maxi = max(maxi, i)

# train_path = '../datasets/ml-1m_clean/valid_list.npy'
# train_list = np.load(train_path, allow_pickle=True)

# for u, i in train_list:
#     if str(u) not in uiMat2:
#         uiMat2[str(u)] = []
    
#     uiMat2[str(u)].append(i)
#     counter += 1
#     maxu = max(maxu, u)
#     maxi = max(maxi, i)
    


# train_path = '../datasets/ml-1m_clean/test_list.npy'
# train_list = np.load(train_path, allow_pickle=True)

# for u, i in train_list:
#     if str(u) not in uiMat2:
#         uiMat2[str(u)] = []
    
#     uiMat2[str(u)].append(i)
#     counter += 1
#     maxu = max(maxu, u)
#     maxi = max(maxi, i)

# print("num of interact", counter)
# print(maxu, maxi)
keys1 = [x for x in uiMat.keys()]
keys2 = [x for x in uiMat2.keys()]

idxItem = [-1]*5000
counter = 0
for ii in range(len(uiMat)):
    idx1 = uiMat[keys1[ii]]
    idx2 = uiMat2[keys2[ii]]
    for idx in range(len(idx2)):
        if (idxItem[idx2[idx]] == -1):
            idxItem[idx2[idx]] = idx1[idx]
        else:
            if (idxItem[idx2[idx]] != idx1[idx]):
                print(idx, idx2[idx], idxItem[idx2[idx]], idx1[idx], len(idx2))
                print('fault: ', ii)
                stop
np.save('mapItem.npy', np.asarray(idxItem))
# print(idxItem)

