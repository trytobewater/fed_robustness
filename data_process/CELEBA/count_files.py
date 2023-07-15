import os

def findAllFile(base,img):
    for root, ds, fs in os.walk(base):
        for f in fs:
            if f.endswith('.jpg'):
                fullname = os.path.join(root, f)
                img.append(fullname)

train_img_list=[]
test_img_list=[]
findAllFile("/data/yc/leaf_new_data/celeba/data/train_need_data/",train_img_list)
findAllFile("/data/yc/leaf_new_data/celeba/data/test_need_data/",test_img_list)
print(len(train_img_list))
print(len(test_img_list))
print(len(train_img_list)+len(test_img_list))