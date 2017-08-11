

from __future__ import print_function

import tensorflow as tf
#from tensorflow.python.ops import rnn ,rnn_cell
import numpy as np
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt

# Parameters
learning_rate = 0.001
training_iters = 500000
batch_size = 128
display_step = 20
r_sigma = .1

alpha = .1

# Network Parameters
n_in = 16 #ins
n_steps = 200 # timesteps
n_hidden = 400 # hidden layer num of features !!Must be a perfect square
n_out = 16 # outs -- same as n_in

n_grid = np.sqrt(n_hidden).astype(int)

# tf Graph input
x = tf.placeholder("float", [batch_size, n_steps, n_in])
#y = tf.placeholder("float", [None, n_out])
y = tf.placeholder("float", [batch_size,n_steps, n_out])
init_state = tf.zeros([batch_size, n_hidden])

# 2D
location = np.array(np.meshgrid(range(n_grid),range(n_grid))).reshape([2,n_grid**2]).T
dist_mat = 1+squareform(pdist(location))

# 1D
#dist_mat = 1+sp.distance.squareform(sp.distance.pdist(np.arange(n_hidden).reshape([n_hidden,1])))

# data
def gen_data(coh,sigma=1.):
    xdata = sigma*np.random.randn(batch_size,n_steps,n_in)
    ydata = np.zeros([batch_size,n_steps,n_out])
    dirs =  np.random.randint(0,n_in,size=batch_size)
    stim_time = range(10,60)
    out_time = range(20,200)
    for ii in range(batch_size):
        ydata[ii,out_time, dirs[ii]] = 1

        ydata[ii,out_time, np.mod(dirs[ii]-1,16) ] = .8
        ydata[ii,out_time, np.mod(dirs[ii]+1,16) ] = .8
        ydata[ii,out_time, np.mod(dirs[ii]-2,16) ] = .4
        ydata[ii,out_time, np.mod(dirs[ii]+2,16) ] = .4
        ydata[ii,out_time, np.mod(dirs[ii]-3,16) ] = .1
        ydata[ii,out_time, np.mod(dirs[ii]+3,16) ] = .1
        ydata[ii,out_time, np.mod(dirs[ii]-4,16) ] = .05
        ydata[ii,out_time, np.mod(dirs[ii]+4,16) ] = .05

        xdata[ii,stim_time, dirs[ii]] += coh

        xdata[ii,stim_time, np.mod(dirs[ii]-1,16) ] += coh*.8
        xdata[ii,stim_time, np.mod(dirs[ii]+1,16) ] += coh*.8
        xdata[ii,stim_time, np.mod(dirs[ii]-2,16) ] += coh*.4
        xdata[ii,stim_time, np.mod(dirs[ii]+2,16) ] += coh*.4
        xdata[ii,stim_time, np.mod(dirs[ii]-3,16) ] += coh*.1
        xdata[ii,stim_time, np.mod(dirs[ii]+3,16) ] += coh*.1

    return xdata,ydata
    
def gen_test_data(coh,sigma=1.):
    xdata = sigma*np.random.randn(16,n_steps,n_in)
    ydata = np.zeros([16,n_steps,n_out])
    dirs =  range(16)
    stim_time = range(10,60)
    out_time = range(20,200)
    for ii in range(16):
        ydata[ii,out_time, dirs[ii]] = 1

        ydata[ii,out_time, np.mod(dirs[ii]-1,16) ] = .8
        ydata[ii,out_time, np.mod(dirs[ii]+1,16) ] = .8
        ydata[ii,out_time, np.mod(dirs[ii]-2,16) ] = .4
        ydata[ii,out_time, np.mod(dirs[ii]+2,16) ] = .4
        ydata[ii,out_time, np.mod(dirs[ii]-3,16) ] = .1
        ydata[ii,out_time, np.mod(dirs[ii]+3,16) ] = .1
        ydata[ii,out_time, np.mod(dirs[ii]-4,16) ] = .05
        ydata[ii,out_time, np.mod(dirs[ii]+4,16) ] = .05

        xdata[ii,stim_time, dirs[ii]] += coh

        xdata[ii,stim_time, np.mod(dirs[ii]-1,16) ] += coh*.8
        xdata[ii,stim_time, np.mod(dirs[ii]+1,16) ] += coh*.8
        xdata[ii,stim_time, np.mod(dirs[ii]-2,16) ] += coh*.4
        xdata[ii,stim_time, np.mod(dirs[ii]+2,16) ] += coh*.4
        xdata[ii,stim_time, np.mod(dirs[ii]-3,16) ] += coh*.1
        xdata[ii,stim_time, np.mod(dirs[ii]+3,16) ] += coh*.1

    return xdata,ydata
    
    
def get_tuning(win,wrec,wout,b,bout):
    
    xtest,ytest = gen_test_data(1,sigma=.1)
    tun = np.zeros([n_hidden,16])
    for ii in range(16):
        r,s,out = run_net(xtest[ii,:,:],win,wrec.T,wout,b,bout)
        tun[:,ii] = np.mean(r[:,100:],1)
        
    return tun
    
def activation(x):
    return np.tanh(x)
      
    
def run_net(inp,win,wrec,wout,b,bout):
    
    r = np.zeros([wrec.shape[0],inp.shape[0]])
    y = np.zeros([wrec.shape[0],inp.shape[0]])
    z = np.zeros([wout.shape[1],inp.shape[0]])
    
    for ii in range(1,inp.shape[0]):
        r[:,ii] = (1.-alpha)*r[:,ii-1] + alpha*(wrec.dot(y[:,ii-1]) + win.T.dot(inp[ii,:]) + b + r_sigma*np.random.randn(r.shape[0],1).T)
        y[:,ii] = activation(r[:,ii])
        z[:,ii] = y[:,ii].dot(wout) + bout
        
    return r,y,z

# Define weights
weights = {
    'out': tf.Variable(tf.random_normal([n_hidden, n_out]))
}
biases = {
    'out': tf.Variable(tf.random_normal([n_out]))
}

with tf.variable_scope('rnn_cell'):
    W = tf.get_variable('W', [n_hidden, n_hidden])
    U = tf.get_variable('U',[n_in,n_hidden])
    b = tf.get_variable('b', [n_hidden], initializer=tf.constant_initializer(0.0))

def rnn_cell(rnn_in, state):
    with tf.variable_scope('rnn_cell', reuse=True):
        W = tf.get_variable('W', [n_hidden, n_hidden])
        U = tf.get_variable('U',[n_in,n_hidden])
        b = tf.get_variable('b', [n_hidden], initializer=tf.constant_initializer(0.0))
    #return tf.nn.relu(tf.matmul(state, W) + tf.matmul(rnn_in,U) + b)
    return (1.-alpha)*state + alpha*tf.tanh(tf.matmul(state, W) + tf.matmul(rnn_in,U) + b\
            + r_sigma*tf.random_normal(state.get_shape(), mean=0.0, stddev=1.0))


def RNN(x, weights, biases):
    
    rnn_inputs = tf.unstack(x,axis=1)
    
    state = init_state
    rnn_outputs = []
    y_out = []
    for rnn_input in rnn_inputs:
        state = rnn_cell(rnn_input, state)
        rnn_outputs.append(state)
        y_out.append(tf.matmul(state,weights['out']) + biases['out'])
    final_state = rnn_outputs[-1]

    #out = tf.matmul(tf.transpose(rnn_outputs,[1,0,2]),weights['out'] + biases['out'])

    # Linear activation, using rnn inner loop last output
    return tf.matmul(final_state, weights['out']) + biases['out'] , y_out, rnn_outputs

pred,y_out,r_out = RNN(x, weights, biases)


def reg_loss(pred,y):
    with tf.variable_scope('rnn_cell',reuse=True):
        W = tf.get_variable('W', [n_hidden, n_hidden])
    return tf.reduce_mean(tf.square(pred-y)) + 1.*tf.reduce_mean(tf.square(dist_mat*W))
        
# Define loss and optimizer
cost = reg_loss(tf.transpose(y_out,[1,0,2]),y)
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Evaluate model
correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y[:,-1,:],1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initializing the variables
init = tf.initialize_all_variables()

if __name__ == "__main__":

    # Launch the graph
    with tf.Session() as sess:
        sess.run(init)
        step = 1
        acc = 0.
        # Keep training until reach max iterations
        while step * batch_size < training_iters:
            c = 1. + 0**np.random.rand()
            batch_x, batch_y = gen_data(coh=c,sigma=.1)
            sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})
            if step % display_step == 0:
                # Calculate batch accuracy
                acc = sess.run(accuracy, feed_dict={x: batch_x, y: batch_y})
                # Calculate batch loss
                loss = sess.run(cost, feed_dict={x: batch_x, y: batch_y})
                #pred = sess.run(pred, feed_dict={x: batch_x, y: batch_y})
                #state = sess.run(final_state, feed_dict={x: batch_x, y: batch_y})
                print("Iter " + str(step*batch_size) + ", Minibatch Loss= " + \
                      "{:.6f}".format(loss)  + ", Training Accuracy= " + \
                      "{:.5f}".format(acc))
            step += 1
            
            if acc > 1.75:
                break
            
        print("Optimization Finished!")
        y = sess.run(y_out, feed_dict={x: batch_x, y: batch_y})
        #z = sess.run(r_out, feed_dict={x: batch_x, y: batch_y})
        wrec = sess.run(W)
        win = sess.run(U)
        wout = sess.run(weights)
        bout = sess.run(biases)
        b = sess.run(b)
        init_state = sess.run(init_state)
    
    
    bout = bout['out']
    wout = wout['out']    
    
    plt.figure()
    plt.pcolor(wrec,cmap='BuPu') 
    plt.title('Recurrent Weights')
    plt.colorbar()
    
    plt.figure()
    plt.title('Distance Matrix')
    plt.pcolor(dist_mat)
    plt.colorbar()
    
    plt.figure()
    plt.scatter(dist_mat.flatten(),wrec.flatten())
    plt.xlabel('distance')
    plt.ylabel('weigth')
    
    
    #Get Tuning for units during memory epoch
    t = get_tuning(win,wrec,wout,b,bout)
    t = t.T
    t_norm = (t-np.min(t,axis=0))/(np.max(t,axis=0)-np.min(t,axis=0)) #Normalize between 0 and 1
    
    #Get tuning distance between cells (Euclidean)
    t_dist = squareform(pdist(t.T))
    
    #Identify preferred direction argmax (Should this be abs value?)
    pref_direction = np.argmax(t_norm,0)
    color = pref_direction/15.
    
    
    #Plot topological map -- preferred direction of each neuron on the grid
    plt.figure()
    for ii in range(len(location)): 
        plt.plot(location[ii,0],location[ii,1],'o',c=[0,color[ii],(1-color[ii])],markersize=10);
    plt.title('topological map')
    
    plt.figure()
    plt.scatter(dist_mat.flatten(),t_dist.flatten())
    plt.xlabel('distance')
    plt.ylabel('tuning similarity (euclidean)')
    
    
    #Coherence dependent variance of units
    x,y = gen_test_data(1,sigma=.1)
    r = np.zeros([n_hidden,n_steps,n_in])
    for ii in range(16): 
        r[:,:,ii],s,out = run_net(x[ii,:,:],win,wrec.T,wout,b,bout)
     
    pref_mat = np.zeros([np.sqrt(n_hidden).astype(int),np.sqrt(n_hidden).astype(int)])
    for ii in range(len(location)): 
        pref_mat[location[ii,0],location[ii,1]] = pref_direction[ii]
    
    plt.figure()    
    plt.plot(np.var(r,axis=2).T);
    plt.xlabel('time')
    plt.ylabel('variance across conditions')
    





