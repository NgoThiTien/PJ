import numpy as np
import cv2
import os
import glob
from os.path import join
import json
from data_utility import image_normalization


# load data directly from the npz file (small dataset, 48k and 5k for train and test)
def load_data_from_npz(file):

    # print("Loading dataset from npz file...", end='')
    # npzfile = np.load(file)
    # train_eye_left = npzfile["train_eye_left"]
    # train_eye_right = npzfile["train_eye_right"]
    # train_face = npzfile["train_face"]
    # train_face_mask = npzfile["train_face_mask"]
    # train_y = npzfile["train_y"]
    # val_eye_left = npzfile["val_eye_left"]
    # val_eye_right = npzfile["val_eye_right"]
    # val_face = npzfile["val_face"]
    # val_face_mask = npzfile["val_face_mask"]
    # val_y = npzfile["val_y"]
    # print("Done.")

    train_eye_left = np.load(file + 'train_eye_left.npy')
    train_eye_right = np.load(file + 'train_eye_right')
    train_face = np.load(file + 'train_face.npy')
    train_face_mask = np.load(file + 'train_face_mask.npy')
    train_y = np.load(file + 'train_y.npy')
    val_eye_left = np.load(file + 'val_eye_left.npy')
    val_eye_right = np.load(file + 'val_eye_right.npy')
    val_face = np.load(file + 'val_face.npy')
    val_face_mask = np.load(file + 'val_face_mask.npy')
    val_y = np.load(file + 'val_y.npy')
    train_eye_right = np.load(file + 'train_eye_right.npy')
    return [train_eye_left, train_eye_right, train_face, train_face_mask, train_y], [val_eye_left, val_eye_right, val_face, val_face_mask, val_y]


# load a batch with data loaded from the npz file
def load_batch(data, img_ch, img_cols, img_rows):

    # useful for debug
    save_images = True

    # if save images, create the related directory
    img_dir = "images"
    if save_images:
        print("if save images, create the related directory")
        if not os.path.exists(img_dir):
            os.makedir(img_dir)

    # create batch structures
    left_eye_batch = np.zeros(shape=(data[0].shape[0], img_ch, img_cols, img_rows), dtype=np.float32)
    right_eye_batch = np.zeros(shape=(data[0].shape[0], img_ch, img_cols, img_rows), dtype=np.float32)
    face_batch = np.zeros(shape=(data[0].shape[0], img_ch, img_cols, img_rows), dtype=np.float32)
    face_grid_batch = np.zeros(shape=(data[0].shape[0], 1, 25, 25), dtype=np.float32)
    y_batch = np.zeros((data[0].shape[0], 2), dtype=np.float32)

    # load left eye
    for i, img in enumerate(data[0]):
        img = cv2.resize(img, (img_cols, img_rows))
        if save_images:
            cv2.imwrite(join(img_dir, "left" + str(i) + ".png"), img)
        img = image_normalization(img)
        left_eye_batch[i] = img.transpose(2, 0, 1)

    # load right eye
    for i, img in enumerate(data[1]):
        img = cv2.resize(img, (img_cols, img_rows))
        if save_images:
            cv2.imwrite("images/right" + str(i) + ".png", img)
        img = image_normalization(img)
        right_eye_batch[i] = img.transpose(2, 0, 1)

    # load faces
    for i, img in enumerate(data[2]):
        img = cv2.resize(img, (img_cols, img_rows))
        if save_images:
            cv2.imwrite("images/face" + str(i) + ".png", img)
        img = image_normalization(img)
        face_batch[i] = img.transpose(2, 0, 1)

    # load grid faces
    for i, img in enumerate(data[3]):
        if save_images:
            cv2.imwrite("images/grid" + str(i) + ".png", img)
        face_grid_batch[i] = img.reshape((1, img.shape[0], img.shape[1]))

    # load labels
    for i, labels in enumerate(data[4]):
        y_batch[i] = labels

    return [right_eye_batch, left_eye_batch, face_batch, face_grid_batch], y_batch

    # create a list of all names of images in the dataset
# def load_data_names(path):
#     print("path ",path)
#     seq_list = []
#     seqs = sorted(glob.glob(join("0", "0***.jpg")))    

#     for seq in seqs:
#         print("load_data_names ",seq)
#         seq = str(path)+str(seq).rjust(9,"0")
#         file = open(seq, "rb")
#         content = file.read()#.splitlines()
#         for line in content:
#             seq_list.append(str(line).rjust(9,"0"))#+".jpg")

    # return seq_list
def load_data_names(path):

    seq_list = []
    seqs = sorted(glob.glob(join(path, "0*.jpg")))

    for seq in seqs:
    	#print("images link:" , seq)
    	file = open(seq, "rb")
    	content = file.read()#.splitlines()
    	for line in content:
    		seq_list.append(line)
    return seq_list
# def load_data_names(path):

#     seq_list = []
#     seqs = sorted(glob.glob(join(path, "0*")))

#     for seq in seqs:
#         file = open(seq, "rb")
#         contentName = file.read()#.splitlines()
#         contentEncode = file.read().splitlines()
#         #print( 
#         content = sorted(glob.glob(join(contentName,contentEncode))
#         #)
#     return seq_list
#         #print(content)
#         #for line in content:
#             #seq_list.append(line)

    #return seq_list


# # create a list of all names of images in the dataset
# def load_data_names(path):
# 	seq_list = []
# 	for x in xrange(1,500):
#          pass seq = sorted(glob.glob(join(path, str(x))))
# 	print(path)
# 	for seq in seqs:
# 		print("seq ", seq)
# 		file = open(seq, "rb")
# 		content_name = file.read()
# 		content = file.read().splitlines()
# 		for line in content:
# 			seq_list.append(line)#str(line).rjust(5,"0") + str(line).rjust(5,"0").splitlines())#+".jpg")
# 			# print("seq_list1  ",line )
# 			# print("seq_list2  ",line )
# 		return seq_list




# load a batch given a list of names (all images are loaded)
def load_batch_from_names(names, path, img_ch, img_cols, img_rows):

    save_img = True

    # data structures for batches
    left_eye_batch = np.zeros(shape=(len(names), img_ch, img_cols, img_rows), dtype=np.float32)
    right_eye_batch = np.zeros(shape=(len(names), img_ch, img_cols, img_rows), dtype=np.float32)
    face_batch = np.zeros(shape=(len(names), img_ch, img_cols, img_rows), dtype=np.float32)
    face_grid_batch = np.zeros(shape=(len(names), 1, 25, 25), dtype=np.float32)
    y_batch = np.zeros((len(names), 2), dtype=np.float32)
    print("dir dir dir" , )

    for i, img_name in enumerate(names):

        # directory
#        dir = img_name[:5]
        dir = img_name[:9]
        print("dir dir dir" , )

        # frame name
        frame = img_name[6:]

        # index of the frame inside the sequence
#        idx = int(frame[:-4])
        idx = int(frame[-9:-4])

        # open json files
        print("open json files",str(join(path, dir, "appleFace.json")) )
        face_file = open(join(path, dir, "appleFace.json"))
        left_file = open(join(path, dir, "appleLeftEye.json"))
        right_file = open(join(path, dir, "appleRightEye.json"))
        dot_file = open(join(path, dir, "dotInfo.json"))
        grid_file = open(join(path, dir, "faceGrid.json"))

        # load json content
        left_json = json.load(left_file)
        right_json = json.load(right_file)
        dot_json = json.load(dot_file)
        grid_json = json.load(grid_file)

        # open image
        img = cv2.imread(join(path, dir, frame))

        # debug stuff
        # if img is None:
        #     print("Error opening image: {}".format(join(path, dir, "frames", frame)))
        #     continue
        #
        # if int(face_json["X"][idx]) < 0 or int(face_json["Y"][idx]) < 0 or \
        #     int(left_json["X"][idx]) < 0 or int(left_json["Y"][idx]) < 0 or \
        #     int(right_json["X"][idx]) < 0 or int(right_json["Y"][idx]) < 0:
        #     print("Error with coordinates: {}".format(join(path, dir, "frames", frame)))
        #     continue

        # get face
        tl_x_face = int(face_json["X"][idx])
        tl_y_face = int(face_json["Y"][idx])
        w = int(face_json["W"][idx])
        h = int(face_json["H"][idx])
        br_x = tl_x_face + w
        br_y = tl_y_face + h
        face = img[tl_y_face:br_y, tl_x_face:br_x]

        # get left eye
        tl_x = tl_x_face + int(left_json["X"][idx])
        tl_y = tl_y_face + int(left_json["Y"][idx])
        w = int(left_json["W"][idx])
        h = int(left_json["H"][idx])
        br_x = tl_x + w
        br_y = tl_y + h
        left_eye = img[tl_y:br_y, tl_x:br_x]

        # get right eye
        tl_x = tl_x_face + int(right_json["X"][idx])
        tl_y = tl_y_face + int(right_json["Y"][idx])
        w = int(right_json["W"][idx])
        h = int(right_json["H"][idx])
        br_x = tl_x + w
        br_y = tl_y + h
        right_eye = img[tl_y:br_y, tl_x:br_x]

        # get face grid (in ch, cols, rows convention)
        face_grid = np.zeros(shape=(1, 25, 25), dtype=np.float32)
        tl_x = int(grid_json["X"][idx])
        tl_y = int(grid_json["Y"][idx])
        w = int(grid_json["W"][idx])
        h = int(grid_json["H"][idx])
        br_x = tl_x + w
        br_y = tl_y + h
        face_grid[0, tl_y:br_y, tl_x:br_x] = 1

        # get labels
        y_x = dot_json["XCam"][idx]
        y_y = dot_json["YCam"][idx]

        # resize images
        face = cv2.resize(face, (img_cols, img_rows))
        left_eye = cv2.resize(left_eye, (img_cols, img_rows))
        right_eye = cv2.resize(right_eye, (img_cols, img_rows))

        # save images (for debug)
        #if save_img:
        cv2.imwrite("images/face.png", face)
        cv2.imwrite("images/right.png", right_eye)
        cv2.imwrite("images/left.png", left_eye)
        cv2.imwrite("images/image.png", img)

        # normalization
        face = image_normalization(face)
        left_eye = image_normalization(left_eye)
        right_eye = image_normalization(right_eye)

        ######################################################

        # transpose images
        face = face.transpose(2, 0, 1)
        left_eye = left_eye.transpose(2, 0, 1)
        right_eye = right_eye.transpose(2, 0, 1)

        # check data types
        face = face.astype('float32')
        left_eye = left_eye.astype('float32')
        right_eye = right_eye.astype('float32')

        # add to the related batch
        left_eye_batch[i] = left_eye
        right_eye_batch[i] = right_eye
        face_batch[i] = face
        face_grid_batch[i] = face_grid
        y_batch[i][0] = y_x
        y_batch[i][1] = y_y

    return [right_eye_batch, left_eye_batch, face_batch, face_grid_batch], y_batch


# load a batch of random data given the full list of the dataset
def load_batch_from_names_random(names, path, batch_size, img_ch, img_cols, img_rows):

    save_img = True

    # data structures for batches
    left_eye_batch = np.zeros(shape=(batch_size, img_ch, img_cols, img_rows), dtype=np.float32)
    right_eye_batch = np.zeros(shape=(batch_size, img_ch, img_cols, img_rows), dtype=np.float32)
    face_batch = np.zeros(shape=(batch_size, img_ch, img_cols, img_rows), dtype=np.float32)
    face_grid_batch = np.zeros(shape=(batch_size, 1, 25, 25), dtype=np.float32)
    y_batch = np.zeros((batch_size, 2), dtype=np.float32)

    # counter for check the size of loading batch
    b = 0
    #Note len(names) = 5
    print("batch_size ", batch_size)
    while b < batch_size:

        # lottery
        i = np.random.randint(0, len(names))
        #print("b ", b)
        #i=b        

        # get the lucky one
        #print("names[i] :",names[i])
        print("names[",i,"]) ",names[i])
        img_name = (str(names[i]).rjust(5,"0")+".jpg")
        print("img_name ",img_name)
        # directory
#        dir = img_name[:5]
        dir = img_name[:5]
        #print("dir ",dir)
        #print("img_name load_batch_from_names_random",img_name)

        # frame name
        frame = img_name[5:]
        #print("frame load_batch_from_names_random",frame)

        # index of the frame into a sequence

        idx = int(img_name[:5])
        print("index of the frame into a sequence", idx)
#        idx = (frame[-9:-4])

        # open json files
        # face_file = open(join(path, dir, "appleFace.json"))
        # print("face_file ", face_file)
        # left_file = open(join(path, dir, "appleLeftEye.json"))
        # right_file = open(join(path, dir, "appleRightEye.json"))
        # dot_file = open(join(path, dir, "dotInfo.json"))
        # grid_file = open(join(path, dir, "faceGrid.json"))
        face_file = open(join(path, "appleFace.json"))
        #print("face_file ", face_file)
        left_file = open(join(path, "appleLeftEye.json"))
        right_file = open(join(path, "appleRightEye.json"))
        dot_file = open(join(path, "dotInfo.json"))
        grid_file = open(join(path, "faceGrid.json"))

        # load json content
        face_json = json.load(face_file)
        left_json = json.load(left_file)
        right_json = json.load(right_file)
        dot_json = json.load(dot_file)
        grid_json = json.load(grid_file)

        # open image
        img = cv2.imread(join(path, "frames", img_name))

        # if image is null, skip
        if img is None:
            print("Error opening image: {}".format(join(path, "frames", img_name)))
            continue

        # if coordinates are negatives, skip (a lot of negative coords!)
        if int(face_json["X"][idx]) < 0 or int(face_json["Y"][idx]) < 0 or \
            int(left_json["X"][idx]) < 0 or int(left_json["Y"][idx]) < 0 or \
            int(right_json["X"][idx]) < 0 or int(right_json["Y"][idx]) < 0:
#            print("Error with coordinates: {}".format(join(path, dir, "frames", frame)))
            print("Error with coordinates: {}".format(join(path, "frames", img_name)))
            continue

        # get face
        tl_x_face = int(face_json["X"][idx])
        tl_y_face = int(face_json["Y"][idx])
        w = int(face_json["W"][idx])
        h = int(face_json["H"][idx])
        br_x = tl_x_face + w
        br_y = tl_y_face + h
        face = img[tl_y_face:br_y, tl_x_face:br_x]

        # get left eye
        tl_x = tl_x_face + int(left_json["X"][idx])
        tl_y = tl_y_face + int(left_json["Y"][idx])
        w = int(left_json["W"][idx])
        h = int(left_json["H"][idx])
        br_x = tl_x + w
        br_y = tl_y + h
        left_eye = img[tl_y:br_y, tl_x:br_x]

        # get right eye
        tl_x = tl_x_face + int(right_json["X"][idx])
        tl_y = tl_y_face + int(right_json["Y"][idx])
        w = int(right_json["W"][idx])
        h = int(right_json["H"][idx])
        br_x = tl_x + w
        br_y = tl_y + h
        right_eye = img[tl_y:br_y, tl_x:br_x]

        # get face grid (in ch, cols, rows convention)
        face_grid = np.zeros(shape=(1, 25, 25), dtype=np.float32)
        tl_x = int(grid_json["X"][idx])
        tl_y = int(grid_json["Y"][idx])
        w = int(grid_json["W"][idx])
        h = int(grid_json["H"][idx])
        br_x = tl_x + w
        br_y = tl_y + h
        face_grid[0, tl_y:br_y, tl_x:br_x] = 1

        # get labels
        y_x = dot_json["XCam"][idx]
        y_y = dot_json["YCam"][idx]

        # resize images
        #print("test os.path.exists:")
        #if face is not None:
	        #if os.path.exists(face):
        print("resize have run method:")
        # if face is not None:
        #     img = cv2.resize(face, (img_cols, img_rows))
        #     training_data.append([np.array(img), np.array(label)])
        # else:
        #     print("image not loaded")
        #if face is None:
        face = cv2.resize(face, (img_cols, img_rows))
        print(face)
        #if(left_eye.any):
	    #    if os.path.exists(left_eye):
        #print("if not os.path.exists(left_eye):if not os.path.exists(left_eye):")
        left_eye = cv2.resize(left_eye, (img_cols, img_rows))
        #if(right_eye.any):
	    #    if os.path.exists(right_eye):
        #print("if not os.path.exists(right_eye):if not os.path.exists(right_eye):")
        right_eye = cv2.resize(right_eye, (img_cols, img_rows))
        path_folder_save = path[-6:-2]
   	    # save images (for debug)
        if (save_img):
            print("test save_img before: ")
            str_idx = str(idx)
            path_folder=  join("images",path_folder_save,str_idx)
            img_dir = "images"
            if save_img:
                print("if save images, create the related directory")
                if not os.path.exists(path_folder):
                    print("create the related directory")
                    os.makedirs(path_folder)
                    print("path_folder ",path_folder)
            cv2.imwrite(join(path_folder,"face.png"), face)
            print("path", path)
            #print("test save_img Eye-Tracking-for-Everyone-master ", join(path_folder,"face.png"))

            cv2.imwrite(join(path_folder,"right.png"), right_eye)            
            cv2.imwrite(join(path_folder,"left.png"), left_eye)
            cv2.imwrite(join(path_folder,"image.png"), img)

        # normalization
        face = image_normalization(face)
        left_eye = image_normalization(left_eye)
        right_eye = image_normalization(right_eye)

        ######################################################

        # transpose images
        face = face.transpose(2, 0, 1)
        left_eye = left_eye.transpose(2, 0, 1)
        right_eye = right_eye.transpose(2, 0, 1)

        # check data types
        face = face.astype('float32')
        left_eye = left_eye.astype('float32')
        right_eye = right_eye.astype('float32')

        # add to the related batch
        left_eye_batch[b] = left_eye
        right_eye_batch[b] = right_eye
        face_batch[b] = face
        face_grid_batch[b] = face_grid
        y_batch[b][0] = y_x
        y_batch[b][1] = y_y

        # increase the size of the current batch
        b += 1

    return [right_eye_batch, left_eye_batch, face_batch, face_grid_batch], y_batch


if __name__ == "__main__":

	# debug
#   seq_list = load_data_names("/cvgl/group/GazeCapture/test")
    #seq_list = load_data_names("/media/ngotien/D/lvtn/dataset/MPIIFaceGaze/note Eye-Tracking-for-Everyone-master/eye_tracker_train_and_val/test")
	
	seq_list = load_data_names("/media/ngotien/D/dataset/00028/frames")
	batch_size = len(seq_list)
    #print("len(seq_list) ",len(seq_list))
#    dataset_path = "/cvgl/group/GazeCapture/gazecapture"
    #dataset_path = "/media/ngotien/D/lvtn/dataset/MPIIFaceGaze/note Eye-Tracking-for-Everyone-master/eye_tracker_train_and_val/"
    
	dataset_path = "/media/ngotien/D/dataset/00010/frames"
	img_ch = 3
	img_cols = 64
	img_rows = 64
	test_batch = load_batch_from_names_random(seq_list, dataset_path, batch_size, 3, 64, 64)

	print("Loaded: {} data".format(len(test_batch[0][0])))
