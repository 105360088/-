# classification-105360088  
classification-105360088 created by GitHub Classroom
# 作業要求
1.用train檔裡的.jpg來訓練模型\
2.用test檔裡的.jpg來輸入訓練好的模型並產生sample-submission.cvs檔\
3.上傳sample-submission.cvs檔\
4.看kaggle出來的分數夠不夠高\
5.是否要更改模型

# 工作環境
1.python3.7\
2.Tensorflow\
3.spyder

# 引入所需資料庫
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import cv2
    import tensorflow as tf
    import os
    
# 引入檔案
    simpson_img = pd.read_csv('../simpsons_dataset/simpson_img_list.csv')
    print(simpson_img.head())
    
# 訓練模型
    x_train, y_train = simpson_img, simpson_y

    img = cv2.resize(img, (50, 50))
    img = img.flatten()
    print('input_data shape: training {training_shape}'.format(
            training_shape=(len(x_train), img.shape[0])))
    print('y_true shape: training {training_shape}'.format(
            training_shape=y_train.shape))
          from sklearn.utils import shuffle

    def simpson_train_batch_generator(x, y, bs, shape):
        x_train = np.array([]).reshape((0, shape))
        y_train = np.array([]).reshape((0, y.shape[1]))
        while True:
            new_ind = shuffle(range(len(x)))
            x = x.take(new_ind)
            y = np.take(y, new_ind, axis = 0)
            for i in range(len(x)):
                dir_img = '../simpsons_dataset/' + x.img.iloc[i]
                img = cv2.imread(dir_img, 0)
                img = cv2.resize(img, (50,50))
                x_train = np.row_stack([x_train, img.flatten()])
                y_train = np.row_stack([y_train, y[i]])
                if x_train.shape[0] == bs:
                    x_batch = x_train.copy()
                    x_batch /= 255.
                    y_batch = y_train.copy()
                    x_train = np.array([]).reshape((0 ,shape))
                    y_train = np.array([]).reshape((0 ,y.shape[1]))        
                    yield x_batch, y_batch 
                    tf.reset_default_graph()
                    
# 建立dense層神經網路 
    x1 = tf.layers.dense(input_data, 512, activation = tf.nn.sigmoid, name='hidden1')
    x2 = tf.layers.dense(x1, 256, activation = tf.nn.sigmoid, name = 'hidden2')
    x3 = tf.layers.dense(x2, 128, activation = tf.nn.sigmoid, name = 'hidden3')
    x4 = tf.layers.dense(x3, 64, activation = tf.nn.sigmoid, name = 'hidden4')
    x5 = tf.layers.dense(x4, 32, activation = tf.nn.sigmoid, name = 'hidden5')
    out = tf.layers.dense(x5, y_train.shape[1], name = 'output')
    y_pred = out
    
# 定義 loss            
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = y_true, logits = y_pred))
    
# 定義輸出

    from tqdm import tqdm 
    from sklearn.metrics import accuracy_score

    epoch = 30 #epoch
    bs = 32 #batch size
    update_per_epoch = 100 

    tr_loss = list() 
    tr_acc = list() 
    train_gen = simpson_train_batch_generator(x_train, y_train, bs, img.shape[0])

    for i in range(epoch):
        training_loss = 0
        training_acc = 0
        bar = tqdm(range(update_per_epoch))
    
    for j in tqdm(range(update_per_epoch)):
        x_batch, y_batch = next(train_gen)
        tr_pred, training_loss_batch, _ = sess.run([y_pred, loss, update], feed_dict = {
            input_data : x_batch,
            y_true : y_batch
        })
        training_loss += training_loss_batch
        training_acc_batch = accuracy_score(np.argmax(y_batch, axis=1), np.argmax(tr_pred, axis=1))
        training_acc += training_acc_batch
        
        if j % 5 == 0:
            bar.set_description('loss: %.4g' % training_loss_batch) 
            
    training_loss /= update_per_epoch
    training_acc /= update_per_epoch
    
    tr_loss.append(training_loss)
    tr_acc.append(training_acc)
    
    print('epoch {epochs}: training loss {training_loss}'.format(
            epochs = (i+1), 
            training_loss = training_loss))
            
