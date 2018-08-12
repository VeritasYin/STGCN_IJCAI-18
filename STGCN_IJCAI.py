#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
tf.Session(config=config)

import numpy as np
import pandas as pd
import random
from scipy.sparse.linalg import eigs


def Scaled_Laplacian(W):
    n = np.shape(W)[0]
    d = []
    L = -W
    for i in range(n):
        d.append(np.sum(W[i, :]))
        L[i, i] = d[i]
    for i in range(n):
        for j in range(n):
            if (d[i] > 0) and (d[j] > 0):
                L[i, j] = L[i, j] / np.sqrt(d[i] * d[j])
    lambda_max = eigs(L, k=1, which='LR')[0][0].real
    # lambda_max \approx 2.0
    return np.matrix(2 * L / lambda_max - np.identity(n))


def Cheb_Poly(L, Ks, n):
    L0 = np.matrix(np.identity(n))
    L1 = np.matrix(np.copy(L))
    L_list = [np.copy(L0), np.copy(L1)]
    for i in range(2, Ks):
        Ln = np.matrix(2 * L * L1 - L0)
        L_list.append(np.copy(Ln))
        L0 = np.matrix(np.copy(L1))
        L1 = np.matrix(np.copy(Ln))
    return np.concatenate(L_list, axis=-1)
# L_lsit (Ks, n*n), Lk (n, Ks*n)


def First_Approx(W, n):
    A = W + np.identity(n)
    d = []
    for i in range(n):
        d.append(np.sum(A[i, :]))
    sinvD = np.sqrt(np.matrix(np.diag(d)).I)
    return np.identity(n) + sinvD * A * sinvD


def Z_Score(x, mean, std):
    return (x - mean) / std


def Z_Inverse(x, mean, std):
    return x * std + mean


def MAPE(v, v_):
    return np.mean(np.abs(v_ - v) / (v + 1e-5))


def MSE(v, v_):
    return np.sqrt(np.mean((v_ - v)**2))


def MAE(v, v_):
    return np.mean(np.abs(v_ - v))


def Data_Gen(file_path, n_train, n_val, n_test, n_route, n_frame=21, day_slot=288):
    # generate training and testing data
    data = pd.read_csv(file_path, header=None).as_matrix()
    # sample_size, frame_num, signal_size, fin
    n_slot = day_slot - n_frame + 1
    C0 = 1
    xtr = np.zeros((n_train * n_slot, n_frame, n_route, C0))
    xva = np.zeros((n_val * n_slot, n_frame, n_route, C0))
    xte = np.zeros((n_test * n_slot, n_frame, n_route, C0))
    for i in range(n_train):
        for j in range(n_slot):
            sta, end = i * day_slot + j, i * day_slot + j + n_frame
            xtr[i * n_slot + j, 0:n_frame, :,
                :] = np.reshape(data[sta:end, :], [n_frame, n_route, C0])
    for i in range(n_val):
        for j in range(n_slot):
            sta, end = (i + n_train) * day_slot + \
                j, (i + n_train) * day_slot + j + n_frame
            xva[i * n_slot + j, 0:n_frame, :,
                :] = np.reshape(data[sta:end, :], [n_frame, n_route, C0])
    for i in range(n_test):
        for j in range(n_slot):
            sta, end = (i + n_train + n_val) * day_slot + \
                j, (i + n_train + n_val) * day_slot + j + n_frame
            xte[i * n_slot + j, 0:n_frame, :,
                :] = np.reshape(data[sta:end, :], [n_frame, n_route, C0])
    frame = list(range(n_frame))
    xmean = np.mean(xtr[:, frame, :, :])
    xstd = np.std(xtr[:, frame, :, :])
    xtr[:, frame, :, :] = Z_Score(xtr[:, frame, :, :], xmean, xstd)
    xva[:, frame, :, :] = Z_Score(xva[:, frame, :, :], xmean, xstd)
    xte[:, frame, :, :] = Z_Score(xte[:, frame, :, :], xmean, xstd)
    return xtr, xva, xte, xmean, xstd


def Conv_Graph(x, theta, Ks, C_in, C_out):
    ker = tf.get_collection('graph_kernel')[0]
    n = ker.get_shape().as_list()[0]
    x_new = tf.reshape(tf.transpose(x, [0, 2, 1]), [-1, n])
    x_ker = tf.reshape(tf.transpose(tf.reshape(
        tf.matmul(x_new, ker), [-1, C_in, Ks, n]), [0, 3, 1, 2]), [-1, C_in * Ks])
    return tf.reshape(tf.matmul(x_ker, theta), [-1, n, C_out])


def Conv_T(x, n, Kt, C_in, C_ot, act='relu'):
    T = x.get_shape().as_list()[1]
    if (C_in > C_ot):
        w_input = tf.get_variable(
            'wt_input', shape=[1, 1, C_in, C_ot], dtype=tf.float32)
        tf.add_to_collection(name='weight_decay', value=tf.nn.l2_loss(w_input))
        x_input = tf.nn.conv2d(x, w_input, strides=[
                               1, 1, 1, 1], padding='SAME')
    elif (C_in < C_ot):
        x_input = tf.concat(
            [x, tf.zeros([tf.shape(x)[0], T, n, C_ot - C_in])], axis=3)
    else:
        x_input = x

    x_input = x_input[:, Kt - 1:T, :, :]

    if (act == 'linear') or (act == 'relu'):
        wt = tf.get_variable(
            name='wt', shape=[Kt, 1, C_in, C_ot], dtype=tf.float32)
        tf.add_to_collection(name='weight_decay', value=tf.nn.l2_loss(wt))
        bt = tf.get_variable(
            name='bt', initializer=tf.zeros([C_ot]), dtype=tf.float32)
        x = tf.nn.conv2d(x, wt, strides=[1, 1, 1, 1], padding='VALID') + bt
        if (act == 'relu'):
            x = tf.nn.relu(x + x_input)
    if (act == 'GLU'):
        wt = tf.get_variable(
            name='wt', shape=[Kt, 1, C_in, 2 * C_ot], dtype=tf.float32)
        tf.add_to_collection(name='weight_decay', value=tf.nn.l2_loss(wt))
        bt = tf.get_variable(name='bt', initializer=tf.zeros(
            [2 * C_ot]), dtype=tf.float32)
        x = tf.nn.conv2d(x, wt, strides=[1, 1, 1, 1], padding='VALID') + bt
        x = (x[:, :, :, 0:C_ot] + x_input) * \
            tf.nn.sigmoid(x[:, :, :, C_ot:2 * C_ot])
    if (act == 'sigmoid'):
        wt = tf.get_variable(
            name='wt', shape=[Kt, 1, C_in, C_ot], dtype=tf.float32)
        tf.add_to_collection(name='weight_decay', value=tf.nn.l2_loss(wt))
        bt = tf.get_variable(
            name='bt', initializer=tf.zeros([C_ot]), dtype=tf.float32)
        x = tf.nn.sigmoid(tf.nn.conv2d(
            x, wt, strides=[1, 1, 1, 1], padding='VALID') + bt)
    return x


def Conv_S(x, n, Ks, C_ot, C_out):
    T = x.get_shape().as_list()[1]
    if (C_ot > C_out):
        w_input = tf.get_variable(
            'ws_input', shape=[1, 1, C_ot, C_out], dtype=tf.float32)
        tf.add_to_collection(name='weight_decay', value=tf.nn.l2_loss(w_input))
        x_input = tf.nn.conv2d(x, w_input, strides=[
                               1, 1, 1, 1], padding='SAME')
    elif (C_ot < C_out):
        x_input = tf.concat(
            [x, tf.zeros([tf.shape(x)[0], T, n, C_out - C_ot])], axis=3)
    else:
        x_input = x

    ws = tf.get_variable(name='ws', shape=[Ks * C_ot, C_out], dtype=tf.float32)
    tf.add_to_collection(name='weight_decay', value=tf.nn.l2_loss(ws))
    bs = tf.get_variable(
        name='bs', initializer=tf.zeros([C_out]), dtype=tf.float32)
    x = tf.reshape(Conv_Graph(tf.reshape(
        x, [-1, n, C_ot]), ws, Ks, C_ot, C_out) + bs, [-1, T, n, C_out])
    x = tf.nn.relu(x[:, :, :, 0:C_out] + x_input)
    return x


def LN(y0, scope):
    size_list = y0.get_shape().as_list()
    T, N, C = size_list[1], size_list[2], size_list[3]
    mu, sigma = tf.nn.moments(y0, axes=[1, 2, 3], keep_dims=True)
    with tf.variable_scope(scope):
        gamma = tf.get_variable('gamma', initializer=tf.ones([1, T, N, C]))
        beta = tf.get_variable('beta', initializer=tf.zeros([1, T, N, C]))
        y0 = (y0 - mu) / tf.sqrt(sigma + 1e-6) * gamma + beta
    return y0


def STGCN(x, n, n_his, Ks, Kt, keep_prob, is_training):
    y = x[:, 0:n_his, :, :]
    # ST-Block 0
    with tf.variable_scope('1'):
        y = Conv_T(y, n, Kt, 1, 32, act='GLU')
        y = Conv_S(y, n, Ks, 32, 32)
    with tf.variable_scope('2'):
        y = Conv_T(y, n, Kt, 32, 64, act='GLU')
    y = LN(y, 'ln1')
    y = tf.nn.dropout(y, keep_prob)
    # ST-Block 1
    with tf.variable_scope('3'):
        y = Conv_T(y, n, Kt, 64, 32, act='GLU')
        y = Conv_S(y, n, Ks, 32, 32)
    with tf.variable_scope('4'):
        y = Conv_T(y, n, Kt, 32, 128, act='GLU')
    y = LN(y, 'ln2')
    y = tf.nn.dropout(y, keep_prob)
    # Output Layer
    with tf.variable_scope('5'):
        y = Conv_T(y, n, 4, 128, 128, act='GLU')
    y = LN(y, 'ln3')
    with tf.variable_scope('6'):
        y = Conv_T(y, n, 1, 128, 128, act='sigmoid')

    w = tf.get_variable(name='w_output', shape=[
                        1, 1, 128, 1], dtype=tf.float32)
    tf.add_to_collection(name='weight_decay', value=tf.nn.l2_loss(w))
    b = tf.get_variable(
        name='b_output', initializer=tf.zeros([n, 1]), dtype=tf.float32)
    y = tf.nn.conv2d(y, w, strides=[1, 1, 1, 1], padding='SAME') + b
    print(y.get_shape().as_list())

    tf.add_to_collection(name='train_loss', value=tf.nn.l2_loss(
        y - x[:, n_his:n_his + 1, :, :]))
    tf.add_to_collection(name='copy_loss', value=tf.nn.l2_loss(
        x[:, n_his - 1:n_his, :, :] - x[:, n_his:n_his + 1, :, :]))
    tf.add_to_collection(name='single_step_predict', value=y[:, 0, :, :])
    return tf.get_collection('train_loss')[0]


def STGCN_Test(time_step, x, n_his, keep_prob, test_set, batch_size):
    n_tsample = np.shape(test_set)[0]
    n_tbatch = int(n_tsample / float(batch_size))
    ans_list = []
    for j in range(n_tbatch):
        sta, end = j * batch_size, (j + 1) * batch_size
        Xte = np.copy(test_set[sta:end, 0:n_his + 1, :, :])
        for k in range(time_step):
            Y = sess.run(tf.get_collection('single_step_predict')[0], feed_dict={
                         x: Xte, keep_prob: 1.0, is_training: False})
            Xte[:, 0:n_his - 1, :, :] = Xte[:, 1:n_his, :, :]
            Xte[:, n_his - 1, :, :] = Y
        ans_list.append(Y)
    return np.concatenate(ans_list, axis=0), n_tbatch * batch_size

# Load wighted adjacency matrix W
W = pd.read_csv('./PeMS_Sta_Dis.csv', header=None).as_matrix()
n = np.shape(W)[0]
sigma2, epsilon = 0.1, 0.5
W = W / 10000.
W = np.exp(-W * W / sigma2) * (np.exp(-W * W / sigma2) >
                               epsilon) * (np.ones([n, n]) - np.identity(n))
# Kernel Parameters
Ks, Kt = 3, 3
L = Scaled_Laplacian(W)
Lk = Cheb_Poly(L, Ks, n)
tf.add_to_collection(name='graph_kernel',
                     value=tf.cast(tf.constant(Lk), tf.float32))

# Data Preprocessing
n_train, n_val, n_test, n_his = 34, 5, 5, 12
xtr, xva, xte, xmean, xstd = Data_Gen(
    './PeMS_Sta_V.csv', n_train, n_val, n_test, n)
print(xmean, xstd)

EPOCH = 50
batch_size = 50
batch_label = np.ones([batch_size, n, 1])

x = tf.placeholder(tf.float32, [None, n_his + 1, n, 1])
keep_prob = tf.placeholder(tf.float32)
is_training = tf.placeholder(tf.bool)

train_loss = STGCN(x, n, n_his, Ks, Kt, keep_prob, is_training)
copy_loss = tf.add_n(tf.get_collection('copy_loss'))

global_step = tf.Variable(0, trainable=False)
n_sample = np.shape(xtr)[0]
n_batch = int(n_sample / float(batch_size))
lr = tf.train.exponential_decay(
    1e-2, global_step, decay_steps=5 * n_batch, decay_rate=0.7, staircase=True)
step_op = tf.assign_add(global_step, 1)

with tf.control_dependencies([step_op]):
    train_op = tf.train.RMSPropOptimizer(lr).minimize(train_loss)

saver = tf.train.Saver()
sess = tf.Session()
sess.run(tf.global_variables_initializer())

min_va_mape9 = min_va_mape6 = min_va_mape3 = 0.4
min_va_mse9 = min_va_mse6 = min_va_mse3 = 1e5
min_va_mae9 = min_va_mae6 = min_va_mae3 = 1e5
min_mape9 = min_mape6 = min_mape3 = 0.4
min_mse9 = min_mse6 = min_mse3 = 1e5
min_mae9 = min_mae6 = min_mae3 = 1e5
flag3 = flag6 = flag9 = False

for i in range(EPOCH):
    epoch_id = np.arange(n_sample)
    random.shuffle(epoch_id)
    for j in range(n_batch):
        sta, end = j * batch_size, (j + 1) * batch_size
        sess.run(train_op, feed_dict={
                 x: xtr[epoch_id[sta:end], 0:n_his + 1, :, :], keep_prob: 1.0, is_training: True})
        if(j % 50 == 0):
            loss_value = \
                sess.run([train_loss, copy_loss], feed_dict={
                         x: xtr[epoch_id[sta:end], 0:n_his + 1, :, :], keep_prob: 1.0, is_training: False})
            print('Epoch %d, Step %d:' % (i, j), loss_value)

        time_step = 3
        yva, n_y = STGCN_Test(time_step, x, n_his, keep_prob,
                              np.copy(xva), batch_size)
        v, v_ = Z_Inverse(xva[0:n_y, n_his + time_step - 1, :, :],
                          xmean, xstd), Z_Inverse(yva, xmean, xstd)
        mape3, mse3, mae3 = MAPE(v, v_), MSE(v, v_), MAE(v, v_)
        if(mape3 < min_va_mape3):
            min_va_mape3 = mape3
            yte, n_y = STGCN_Test(time_step, x, n_his, keep_prob,
                                  np.copy(xte), batch_size)
            vt, vt_ = Z_Inverse(xte[0:n_y, n_his + time_step - 1, :, :],
                                xmean, xstd), Z_Inverse(yte, xmean, xstd)
            min_mape3 = MAPE(vt, vt_)
            flag3 = True
        if(mse3 < min_va_mse3):
            min_va_mse3 = mse3
            if(flag3):
                min_mse3 = MSE(vt, vt_)
            else:
                yte, n_y = STGCN_Test(time_step, x, n_his, keep_prob,
                                      np.copy(xte), batch_size)
                vt, vt_ = Z_Inverse(xte[0:n_y, n_his + time_step - 1, :, :],
                                    xmean, xstd), Z_Inverse(yte, xmean, xstd)
                min_mse3 = MSE(vt, vt_)
                flag3 = True
        if(mae3 < min_va_mae3):
            min_va_mae3 = mae3
            if(flag3):
                min_mae3 = MAE(vt, vt_)
            else:
                yte, n_y = STGCN_Test(time_step, x, n_his, keep_prob,
                                      np.copy(xte), batch_size)
                vt, vt_ = Z_Inverse(xte[0:n_y, n_his + time_step - 1, :, :],
                                    xmean, xstd), Z_Inverse(yte, xmean, xstd)
                min_mae3 = MAE(vt, vt_)
        print('T3 Epoch %d, Step %d: MAPE %g, %g; MAE %g, %g; RMSE %g, %g' % (
            i, j, min_va_mape3, min_mape3, min_va_mae3, min_mae3, min_va_mse3, min_mse3))
        flag3 = False

        time_step = 6
        yva, n_y = STGCN_Test(time_step, x, n_his, keep_prob,
                              np.copy(xva), batch_size)
        v, v_ = Z_Inverse(xva[0:n_y, n_his + time_step - 1, :, :],
                          xmean, xstd), Z_Inverse(yva, xmean, xstd)
        mape6, mse6, mae6 = MAPE(v, v_), MSE(v, v_), MAE(v, v_)
        if(mape6 < min_va_mape6):
            min_va_mape6 = mape6
            yte, n_y = STGCN_Test(time_step, x, n_his, keep_prob,
                                  np.copy(xte), batch_size)
            vt, vt_ = Z_Inverse(xte[0:n_y, n_his + time_step - 1, :, :],
                                xmean, xstd), Z_Inverse(yte, xmean, xstd)
            min_mape6 = MAPE(vt, vt_)
            flag6 = True
        if(mse6 < min_va_mse6):
            min_va_mse6 = mse6
            if (flag6):
                min_mse6 = MSE(vt, vt_)
            else:
                yte, n_y = STGCN_Test(time_step, x, n_his, keep_prob,
                                      np.copy(xte), batch_size)
                vt, vt_ = Z_Inverse(xte[0:n_y, n_his + time_step - 1, :, :],
                                    xmean, xstd), Z_Inverse(yte, xmean, xstd)
                min_mse6 = MSE(vt, vt_)
                flag6 = True
        if(mae6 < min_va_mae6):
            min_va_mae6 = mae6
            if(flag6):
                min_mae6 = MAE(vt, vt_)
            else:
                yte, n_y = STGCN_Test(time_step, x, n_his, keep_prob,
                                      np.copy(xte), batch_size)
                vt, vt_ = Z_Inverse(xte[0:n_y, n_his + time_step - 1, :, :],
                                    xmean, xstd), Z_Inverse(yte, xmean, xstd)
                min_mae6 = MAE(vt, vt_)
        print('T6 Epoch %d, Step %d: MAPE %g, %g; MAE %g, %g; RMSE %g, %g' % (
            i, j, min_va_mape6, min_mape6, min_va_mae6, min_mae6, min_va_mse6, min_mse6))
        flag6 = False

        time_step = 9
        yva, n_y = STGCN_Test(time_step, x, n_his, keep_prob,
                              np.copy(xva), batch_size)
        v, v_ = Z_Inverse(xva[0:n_y, n_his + time_step - 1, :, :],
                          xmean, xstd), Z_Inverse(yva, xmean, xstd)
        mape9, mse9, mae9 = MAPE(v, v_), MSE(v, v_), MAE(v, v_)
        if(mape9 < min_va_mape9):
            min_va_mape9 = mape9
            yte, n_y = STGCN_Test(time_step, x, n_his, keep_prob,
                                  np.copy(xte), batch_size)
            vt, vt_ = Z_Inverse(xte[0:n_y, n_his + time_step - 1, :, :],
                                xmean, xstd), Z_Inverse(yte, xmean, xstd)
            min_mape9 = MAPE(vt, vt_)
            flag9 = True
        if(mse9 < min_va_mse9):
            min_va_mse9 = mse9
            if (flag9):
                min_mse9 = MSE(vt, vt_)
            else:
                yte, n_y = STGCN_Test(time_step, x, n_his, keep_prob,
                                      np.copy(xte), batch_size)
                vt, vt_ = Z_Inverse(xte[0:n_y, n_his + time_step - 1, :, :],
                                    xmean, xstd), Z_Inverse(yte, xmean, xstd)
                min_mse9 = MSE(vt, vt_)
                flag9 = True
        if(mae9 < min_va_mae9):
            min_va_mae9 = mae9
            if (flag9):
                min_mae9 = MAE(vt, vt_)
            else:
                yte, n_y = STGCN_Test(time_step, x, n_his, keep_prob,
                                      np.copy(xte), batch_size)
                vt, vt_ = Z_Inverse(xte[0:n_y, n_his + time_step - 1, :, :],
                                    xmean, xstd), Z_Inverse(yte, xmean, xstd)
                min_mae9 = MAE(vt, vt_)
        print('T9 Epoch %d, Step %d: MAPE %g, %g; MAE %g, %g; RMSE %g, %g' % (
            i, j, min_va_mape9, min_mape9, min_va_mae9, min_mae9, min_va_mse9, min_mse9))
        flag9 = False
sess.close()
