import os
from keras.optimizers import SGD, Adam
from keras.callbacks import  EarlyStopping, ModelCheckpoint
from load_data import load_data_from_npz, load_batch, load_data_names, load_batch_from_names_random,load_batch_from_names
from models import get_eye_tracker_model


# generator for data loaded from the npz file
def generator_npz(data, batch_size, img_ch, img_cols, img_rows):
    while True:
        for it in list(range(0, data[0].shape[0], batch_size)):
            x, y = load_batch([l[it:it + batch_size] for l in data], img_ch, img_cols, img_rows)
            yield x, y


# generator with random batch load (train)
def generator_train_data(names, path, batch_size, img_ch, img_cols, img_rows):
    while True:
        print("load_batch_from_names")
        x, y = load_batch_from_names_random(names, path, batch_size, img_ch, img_cols, img_rows)
        yield x, y


# generator with random batch load (validation)
def generator_val_data(names, path, batch_size, img_ch, img_cols, img_rows):

    while True:
        x, y = load_batch_from_names_random(names, path, batch_size, img_ch, img_cols, img_rows)
        yield x, y


def train(args):

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.dev

    #todo: manage parameters in main
    if args.data == "big":
        #dataset_path = "/cvgl/group/GazeCapture/gazecapture"
        dataset_path = "/media/ngotien/D/dataset/00010/"
    if args.data == "small":
        dataset_path = "/cvgl/group/GazeCapture/eye_tracker_train_and_val.npz"

    if args.data == "big":
    #    train_path = "/cvgl/group/GazeCapture/train"
    #    val_path = "/cvgl/group/GazeCapture/validation"
    #    test_path = "/cvgl/group/GazeCapture/test"
        train_path = "/media/ngotien/D/dataset/00010/frames/"
        val_path = "/media/ngotien/D/dataset/00034/frames/"
        test_path = "/media/ngotien/D/dataset/00033/frames/"

    #print("{} dataset: {}".format(args.data, dataset_path))

    # train parameters
    n_epoch = args.max_epoch
    batch_size = args.batch_size
    print(batch_size)
    patience = args.patience

    # # image parameter
    img_cols = 64
    img_rows = 64
    img_ch = 3
    # image parameter
    # img_cols = 480
    # img_rows = 640
    # img_ch = 3

    # model
    print("get_eye_tracker_model(img_ch, img_cols, img_rows)")
    model = get_eye_tracker_model(img_ch, img_cols, img_rows)

    # model summary
    model.summary()

#co comment
    #weights
    # print("Loading weights...",  end='')
    # weights_path = "weights/weights.003-4.05525.hdf5"
    # model.load_weights(weights_path)
    # print("Done.")
#co comment

    # optimizer
    sgd = SGD(lr=1e-1, decay=5e-4, momentum=9e-1, nesterov=True)
    adam = Adam(lr=1e-3)

    # compile model
    model.compile(optimizer=adam, loss='mse')

    # data
    # todo: parameters not hardocoded
    if args.data == "big":
        # train data
        train_names = load_data_names(train_path)
        # validation data
        val_names = load_data_names(val_path)
        #print("val_names", val_names)
        # test data
        test_names = load_data_names(test_path)

    if args.data == "small":
        train_data, val_data = load_data_from_npz(dataset_path)

    # debug
    # x, y = load_batch([l[0:batch_size] for l in train_data], img_ch, img_cols, img_rows)
    # x, y = load_batch_from_names(train_names[0:batch_size], dataset_path, img_ch, img_cols, img_rows)

    # last dataset checks
    if args.data == "small":
        print("train data sources of size: {} {} {} {} {}".format(
            train_data[0].shape[0], train_data[1].shape[0], train_data[2].shape[0],
            train_data[3].shape[0], train_data[4].shape[0]))
        print("validation data sources of size: {} {} {} {} {}".format(
            val_data[0].shape[0], val_data[1].shape[0], val_data[2].shape[0],
            val_data[3].shape[0], val_data[4].shape[0]))


    if args.data == "big":
    	print ("batch_size  ", batch_size)#print ("batch_size  ", len(val_names))

    	model.fit_generator(        	
            generator=generator_train_data(train_names, dataset_path, batch_size, img_ch, img_cols, img_rows),
            steps_per_epoch=(len(train_names)) / batch_size,
            epochs=n_epoch,
            verbose=0,
            validation_data=generator_val_data(val_names, dataset_path, batch_size, img_ch, img_cols, img_rows),
            validation_steps=(len(val_names)) / batch_size,
            #validation_steps=20,
            callbacks=[EarlyStopping(patience=patience),
                       ModelCheckpoint("weights_big/weights.{epoch:03d}-{val_loss:.5f}.hdf5", save_best_only=True)
                       ]
        )
    if args.data == "small":
        model.fit_generator(
            generator=generator_npz(train_data, batch_size, img_ch, img_cols, img_rows),
            steps_per_epoch=(train_data[0].shape[0])/batch_size,
            epochs=n_epoch,
            verbose=1,
            validation_data=generator_npz(val_data, batch_size, img_ch, img_cols, img_rows),
            validation_steps=(val_data[0].shape[0])/batch_size,
            callbacks=[EarlyStopping(patience=patience),
                       ModelCheckpoint("weights/weights.{epoch:03d}-{val_loss:.5f}.hdf5", save_best_only=True)
                       ]
        )
    if args.data == "":
        model.fit(train_tensors, train_targets, 
          validation_data=(valid_tensors, valid_targets),
          epochs=epochs, batch_size=20, callbacks=[checkpointer], verbose=1)
        model.load_weights('saved_models/weights.best.from_scratch.hdf5')      

        # get index of predicted dog breed for each image in test set
        dog_breed_predictions = [np.argmax(model.predict(np.expand_dims(tensor, axis=0))) for tensor in test_tensors]

        # report test accuracy
        test_accuracy = 100*np.sum(np.array(dog_breed_predictions)==np.argmax(test_targets, axis=1))/len(dog_breed_predictions)
        print('Test accuracy: %.4f%%' % test_accuracy)




