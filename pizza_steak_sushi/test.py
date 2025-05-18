import os

path = "data/train"
class_names = ['pizza','steak', 'sushi']
# for i in os.listdir(path):
#     print (i)

# class_files = {}

# for classes in class_names:
#     class_name = os.path.join(path, classes)
#     class_files[classes] = [os.path.join(class_name, i) for i in os.listdir(class_name)]

# print(class_files)

class_label = {class_name: class_name for  i,class_name in enumerate(class_names)}
results = []

for classes in class_names:
    class_dir = os.path.join(path, classes)
    for paths in os.listdir(class_dir):
        img_path = os.path.join(class_dir, paths)
        results.append((img_path, classes))

print(results[:5])
