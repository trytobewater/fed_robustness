import pickle

train_accuracy = []
train_loss = []
test_accuracy = []
test_loss = []

# test_react_light_5
file = open('/data/yc/leaf_te_data/celeba/result/non_FL/test_react_mask_5/history.pckl', 'rb')
data = pickle.load(file)

train_accuracy = (data['accuracy'])
train_loss = (data['loss'])
test_accuracy = (data['val_accuracy'])
test_loss = (data['val_loss'])

with open('/data/yc/leaf_te_data/celeba/result/non_FL/test_react_mask_5/init_train_Accuracy.txt', "w",
          encoding='utf-8') as fw:
    j = 0
    for word in train_accuracy:
        if j != len(train_accuracy) - 1:
            fw.write(str(word) + ',')
        else:
            fw.write(str(word))
        j += 1
with open('/data/yc/leaf_te_data/celeba/result/non_FL/test_react_mask_5/init_train_Loss.txt', "w", encoding='utf-8') as fw:
    j = 0
    for word in train_loss:
        if j != len(train_loss) - 1:
            fw.write(str(word) + ',')
        else:
            fw.write(str(word))
        j += 1
with open('/data/yc/leaf_te_data/celeba/result/non_FL/test_react_mask_5/init_test_Accuracy.txt', "w",
          encoding='utf-8') as fw:
    j = 0
    for word in test_accuracy:
        if j != len(test_accuracy) - 1:
            fw.write(str(word) + ',')
        else:
            fw.write(str(word))
        j += 1
with open('/data/yc/leaf_te_data/celeba/result/non_FL/test_react_mask_5/init_test_Loss.txt', "w", encoding='utf-8') as fw:
    j = 0
    for word in test_loss:
        if j != len(test_loss) - 1:
            fw.write(str(word) + ',')
        else:
            fw.write(str(word))
        j += 1

file.close()