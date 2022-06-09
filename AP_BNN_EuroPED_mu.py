#!/usr/bin/env python
# coding: utf-8



import logging
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from sklearn.model_selection import train_test_split

from tensorflow.keras.layers import Input, Concatenate, Dense, Lambda, LeakyReLU
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import L2
from tensorflow_probability import distributions as tfd
from tensorflow_probability import layers as tfpl
from scipy.stats import norm
import tensorflow_probability
import plotly.graph_objects as go
from scipy.linalg import lstsq





logging.getLogger('tensorflow').setLevel(logging.ERROR)

rng = np.random.RandomState(120)  # For reproducibility of results

# ============================
# Specifying Hyper-parameters:
# ============================

num_epochs = 8000  # Setting the number of training epochs
batch_size = 64 # Setting the size of training batches
num_select = 100  # Amount of data points available

sigma_x1 = 0.01  # Tune how far from original data points is sampled (x1 dimension)
sigma_x2 = 0.001  # Tune how far from original data points is sampled (x2 dimension)
sigma_x3 = 0.01 
sigma_x4 = 0.02
sigma_x5 = 0.03
sigma_x6 = 0.01 
sigma_x7 = 0.5 
sigma_x8 = 0.2 
sigma_x9=0.05

mean_prior_weight = 0.01
noise_prior_weight = 40
nll_weight=2
nll_weight2=3
nll_weight3=5

p_error=0.2


initial_lr = 0.001  # Setting the learning rate for the optimizer
decay_rate = 0.5  # Decay of the learning rate scheduler


#---------------DATASET-------------------

plasma0=pd.read_csv('../data/europed_predictions_more_input.csv')
def myscaler(data):
    std=data.std()
    mean=data.mean()
    return (data-mean)/std,mean,std
plasma0=plasma0[(plasma0['Ip[MA]']!=0)&(plasma0['Te_ped_SC']!=0)&(plasma0['beta_n_pred']!=0)]
plasma0['epsilon']=plasma0['rminor[m]']/plasma0['Rmag[M]']
mu0=4*np.pi*10**(-7)
plasma0['mu'] = (mu0 /(2*np.pi)) * 1e6 * plasma0['Ip[MA]'] / (plasma0['Bt[T]'] * plasma0['rminor[m]'])

plasma0['Delta_SC']=plasma0.Delta_EPED*np.sqrt(plasma0.Te_ped_SC/plasma0.Te_ped_EPED) 
plasma=myscaler(plasma0)[0]

sigmay2=(plasma0.Delta_exp-plasma0.Delta_SC).abs()/myscaler(plasma0.Delta_SC)[2]
sigmay2[sigmay2==0]=0.00001
plasma['Sigma_Te_ped_SC']=(plasma0.Te_ped_SC-plasma0.Te_ped_exp).abs()/myscaler(plasma0.Te_ped_SC)[2]

plasma['Sigma_Delta_SC']=sigmay2
plasma['Sigma_neped']= (plasma0.ne_ped_exp-plasma0.neped_pre_2).abs()/myscaler(plasma0.neped_pre_2)[2]


selected_dataset=plasma0.loc[:,['mu', 'Bt[T]', 'epsilon', 'kappa', 'Triang.',
     'P_tot[MW]',  'Gas[1e22/s]', 'Zeff_h','Delta_SC',  'Te_ped_SC', 'neped_pre_2']]






train,test=train_test_split(plasma.loc[:,['mu', 'Bt[T]', 'epsilon', 'kappa', 'Triang.',
     'P_tot[MW]',  'Gas[1e22/s]', 'Zeff_h','Delta_SC',  'Te_ped_SC', 'neped_pre_2','Sigma_Te_ped_SC','Sigma_Delta_SC','Sigma_neped']],test_size=0.098,shuffle=True,random_state=2)
sigma_x1 = 0.02/myscaler(selected_dataset)[2][0]#mu
sigma_x2 = 0.001/myscaler(selected_dataset)[2][1]#Bt
sigma_x3 = 0.01/myscaler(selected_dataset)[2][2] #eps
sigma_x4 = 0.01/myscaler(selected_dataset)[2][3]#kap
sigma_x5 = 0.01/myscaler(selected_dataset)[2][4]#trian
sigma_x6 = 0.5/myscaler(selected_dataset)[2][5]#ptot
sigma_x7 = 0.2/myscaler(selected_dataset)[2][6]#gas
sigma_x8 = 0.05/myscaler(selected_dataset)[2][7]#zeff



x1_train=tf.constant(train.loc[:,'mu'].values, dtype=tf.float32)
x2_train=tf.constant(train.loc[:,'Bt[T]'].values, dtype=tf.float32)
x3_train=tf.constant(train.loc[:,'epsilon'].values, dtype=tf.float32)
x4_train=tf.constant(train.loc[:,'kappa'].values, dtype=tf.float32)
x5_train=tf.constant(train.loc[:,'Triang.'].values, dtype=tf.float32)
x6_train=tf.constant(train.loc[:,'P_tot[MW]'].values, dtype=tf.float32)
x7_train=tf.constant(train.loc[:,'Gas[1e22/s]'].values, dtype=tf.float32)
x8_train=tf.constant(train.loc[:,'Zeff_h'].values, dtype=tf.float32)
y_train=tf.constant(train.loc[:,'Delta_SC'].values, dtype=tf.float32)
y2_train=tf.constant(train.loc[:,'Te_ped_SC'].values, dtype=tf.float32)
y3_train=tf.constant(train.loc[:,'neped_pre_2'].values, dtype=tf.float32)
sigma_y_train=tf.constant(train.loc[:,'Sigma_Te_ped_SC'].values, dtype=tf.float32)
sigma_y2_train=tf.constant(train.loc[:,'Sigma_Delta_SC'].values, dtype=tf.float32)
sigma_y3_train=tf.constant(train.loc[:,'Sigma_neped'].values, dtype=tf.float32)
Train=train



# =========================
# Creating model structure:
# =========================

def mean_dist_fn(variational_layer):
    def mean_dist(inputs):
        bias_mean = variational_layer.bias_posterior.mean()

        kernel_mean = variational_layer.kernel_posterior.mean()
        kernel_std = variational_layer.kernel_posterior.stddev()
        

        mu_mean = tf.matmul(inputs, kernel_mean) + bias_mean
        mu_var = tf.matmul(inputs ** 2, kernel_std ** 2)
        mu_std = tf.sqrt(mu_var)
        return tfd.Normal(mu_mean, mu_std)
        

    return mean_dist

def create_model(n_hidden=200):
    leaky_relu = LeakyReLU(alpha=0.2)
    variational_layer = tfpl.DenseReparameterization(1, name='mu')
    variational_layer2 = tfpl.DenseReparameterization(1, name='mu2')
    variational_layer3 = tfpl.DenseReparameterization(1, name='mu3')
    
    input_x1 = Input(shape=(1,))  
    input_x2 = Input(shape=(1,))  
    input_x3 = Input(shape=(1,))
    input_x4 = Input(shape=(1,))
    input_x5 = Input(shape=(1,))
    input_x6 = Input(shape=(1,))
    input_x7 = Input(shape=(1,))
    input_x8 = Input(shape=(1,))
    input_combined = Concatenate(axis=1)([input_x1,input_x2,
        input_x3,input_x4,input_x5,input_x6,input_x7,input_x8])
    d_combined = Dense(n_hidden, input_dim=8, activation=leaky_relu)(input_combined)
    d_combined = Dense(n_hidden, activation=leaky_relu)(d_combined)
    s1 = Dense(1, activation='softplus', name='sigma1')(d_combined)
    s2 = Dense(1, activation='softplus', name='sigma2')(d_combined)
    s3 = Dense(1, activation='softplus', name='sigma3')(d_combined)
    m = variational_layer(d_combined)
    m2=variational_layer2(d_combined)
    m3=variational_layer3(d_combined)
    

    mean_dist = Lambda(mean_dist_fn(variational_layer))(d_combined)
    mean_dist2 = Lambda(mean_dist_fn(variational_layer2))(d_combined)
    mean_dist3 = Lambda(mean_dist_fn(variational_layer3))(d_combined)
    ndim_out = Lambda(lambda p: tfd.Normal(p[0],p[1]))((m,s1))
    ndim_out2=Lambda(lambda p: tfd.Normal(p[0],p[1]))((m2,s2))
    ndim_out3=Lambda(lambda p: tfd.Normal(p[0],p[1]))((m3,s3))
    return Model([input_x1,input_x2,
        input_x3,input_x4,input_x5,input_x6,input_x7,input_x8], [ndim_out, mean_dist,ndim_out2,mean_dist2, ndim_out3, mean_dist3])


model = create_model()

# =======================================
# Creating model training infrastructure:
# =======================================
def lr_scheduler(decay_steps):
    return tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=initial_lr,
        decay_steps=decay_steps,
        decay_rate=decay_rate)


def optimizer_function(lr):
    return tf.keras.optimizers.Adam(learning_rate=lr)


def data_loader(x1, x2,x3,x4,x5,x6,x7,x8, y,y2,y3 ,sigma_y,sigma_y2,sigma_y3, batch_size):
    ds = tf.data.Dataset.from_tensor_slices((x1, x2,x3,x4,x5,x6,x7,x8, y,y2,y3,sigma_y,sigma_y2, sigma_y3))
    ds = ds.shuffle(x1.shape[0])
    return ds.batch(batch_size)


def backprop(model, loss, tape):
    trainable_vars = model.trainable_variables
    gradients = tape.gradient(loss, trainable_vars)
    return zip(gradients, trainable_vars)



# The function below uses a lot of manually shaped tensors, so careful generalization is needed
@tf.function
def NCP_train_step(model, optimizer, x1, x2,x3,x4,x5,x6,x7,x8, y,y2,y3,sigma_y,sigma_y2,sigma_y3):
    # Format input data
    y_reshape = tf.reshape(y, [batch_size, 1])  # MANUAL RESHAPING
    y_nll = [y_reshape]  # MANUAL RESHAPING
    y2_reshape = tf.reshape(y2, [batch_size, 1])  # MANUAL RESHAPING
    y2_nll = [y2_reshape] 
    y3_reshape = tf.reshape(y3, [batch_size, 1])  # MANUAL RESHAPING
    y3_nll = [y3_reshape] 
    sigma_y_reshape = tf.reshape(sigma_y, [batch_size, 1])
    sigma_y2_reshape = tf.reshape(sigma_y2, [batch_size, 1])
    sigma_y3_reshape = tf.reshape(sigma_y3, [batch_size, 1])
    
    # Generate random OOD data from training data
    ood_x1 = x1 + tf.random.normal(tf.shape(x1), stddev=sigma_x1)
    ood_x2 = x2 + tf.random.normal(tf.shape(x2), stddev=sigma_x2)
    ood_x3 = x3 + tf.random.normal(tf.shape(x3), stddev=sigma_x3)
    ood_x4 = x4 + tf.random.normal(tf.shape(x4), stddev=sigma_x4)
    ood_x5 = x5 + tf.random.normal(tf.shape(x5), stddev=sigma_x5)
    ood_x6 = x6 + tf.random.normal(tf.shape(x6), stddev=sigma_x6)
    ood_x7 = x7 + tf.random.normal(tf.shape(x7), stddev=sigma_x7)
    ood_x8 = x8 + tf.random.normal(tf.shape(x8), stddev=sigma_x8)
    

    # NCP output prior
    ood_mean_prior = tfd.Normal(y_reshape, sigma_y_reshape)
    ood_mean_prior2 = tfd.Normal(y2_reshape, sigma_y2_reshape)
    ood_mean_prior3 = tfd.Normal(y3_reshape, sigma_y3_reshape)
    with tf.GradientTape() as tape:
        # For train data inputs
        ndim_dist, mean_dist,ndim_dist2, mean_dist2, ndim_dist3, mean_dist3 = model([x1, x2,x3,x4,x5,x6,x7,x8], training=True)

        # For OOD data inputs
        ood_ndim_dist, ood_mean_dist,ood_ndim_dist2, ood_mean_dist2, ood_ndim_dist3, ood_mean_dist3 = model([ood_x1, ood_x2,ood_x3,ood_x4,ood_x5,ood_x6,ood_x7,ood_x8], training=True)
        
        # A single combined Negative Log-Likelihood for all dimensions
        nll = -ndim_dist.log_prob(y_nll)
        nll_reshape = tf.reshape(nll, [1, batch_size, 1])  # MANUAL RESHAPING
        nll2 = -ndim_dist2.log_prob(y2_nll)
        nll2_reshape = tf.reshape(nll2, [1, batch_size, 1])  # MANUAL RESHAPING
        nll3 = -ndim_dist3.log_prob(y3_nll)
        nll3_reshape = tf.reshape(nll3, [1, batch_size, 1])

        
        # KL divergence between output prior and OOD mean distribution
        kl_ood_mean = tfd.kl_divergence(ood_mean_prior, ood_mean_dist)#epistemic
        kl_ood_mean_reshape = tf.reshape(kl_ood_mean, [1, batch_size, 1])  # MANUAL RESHAPING
        kl_ood_mean2 = tfd.kl_divergence(ood_mean_prior2, ood_mean_dist2)#epistemic
        kl_ood_mean_reshape2 = tf.reshape(kl_ood_mean2, [1, batch_size, 1])
        kl_ood_mean3 = tfd.kl_divergence(ood_mean_prior3, ood_mean_dist3)#epistemic
        kl_ood_mean_reshape3 = tf.reshape(kl_ood_mean3, [1, batch_size, 1])
        
        # Encouraging aleatoric uncertainty to be a set amount for OOD data
        y_reshape_0=y_reshape*myscaler(plasma0.Delta_SC)[2]+myscaler(plasma0.Delta_SC)[1]
        y2_reshape_0=y2_reshape*myscaler(plasma0.Te_ped_SC)[2]+myscaler(plasma0.Te_ped_SC)[1]
        y3_reshape_0=y3_reshape*myscaler(plasma0.beta_n_pred)[2]+myscaler(plasma0.beta_n_pred)[1]
        exp_noise_vec1 = 0.3*0.5*y_reshape_0/myscaler(plasma0.Delta_SC)[2]
        exp_noise_vec2 = p_error*y2_reshape_0/myscaler(plasma0.Te_ped_SC)[2]
        exp_noise_vec3 = 0.3*y3_reshape_0/myscaler(plasma0.neped_pre_2)[2]
    
        exp_noise_dist = tfd.Normal(0, exp_noise_vec1)#aleatoric
        mean_noise_dist = tfd.Normal(0, ood_ndim_dist.stddev())#what increases output error or input error
        exp_noise_dist2 = tfd.Normal(0, exp_noise_vec2)
        mean_noise_dist2 = tfd.Normal(0, ood_ndim_dist2.stddev())
        exp_noise_dist3 = tfd.Normal(0, exp_noise_vec3)
        mean_noise_dist3 = tfd.Normal(0, ood_ndim_dist3.stddev())
        
        # KL-Divergence between the noise distributions to fit towards the prior:
        kl_ood_noise = tfd.kl_divergence(exp_noise_dist, mean_noise_dist)#aleatoric
        kl_ood_noise_reshape = tf.reshape(kl_ood_noise, [1, batch_size, 1])  # MANUAL RESHAPING
        kl_ood_noise2 = tfd.kl_divergence(exp_noise_dist2, mean_noise_dist2)#aleatoric
        kl_ood_noise_reshape2 = tf.reshape(kl_ood_noise2, [1, batch_size, 1])
        kl_ood_noise3 = tfd.kl_divergence(exp_noise_dist3, mean_noise_dist3)#aleatoric
        kl_ood_noise_reshape3 = tf.reshape(kl_ood_noise3, [1, batch_size, 1])
        # Calculate the combined loss term
        loss = tf.reduce_sum(
            nll_weight*(nll_reshape) +nll_weight2*nll2_reshape+nll_weight3*nll3_reshape+ mean_prior_weight * (kl_ood_mean_reshape+kl_ood_mean_reshape2+kl_ood_mean_reshape3) + noise_prior_weight * (kl_ood_noise_reshape+kl_ood_noise_reshape2+kl_ood_noise_reshape3))
        nll=tf.reduce_sum(nll_reshape)
        nll2=tf.reduce_sum(nll2_reshape)
        nll3=tf.reduce_sum(nll3_reshape)
        epistemic_loss=tf.reduce_sum(kl_ood_mean_reshape+kl_ood_mean_reshape2+kl_ood_mean_reshape3)
        aleatoric_loss=tf.reduce_sum(kl_ood_noise_reshape+kl_ood_noise_reshape2+kl_ood_noise_reshape3)

    optimizer.apply_gradients(backprop(model, loss, tape))
    return loss, mean_dist.mean(), mean_dist2.mean(),mean_dist3.mean(),epistemic_loss, aleatoric_loss, nll,nll2,nll3

def train(model, x1, x2,x3,x4,x5,x6,x7,x8, y,y2,y3,sigma_y,sigma_y2,sigma_y3, batch_size, epochs, step_fn):
    steps_per_epoch = int(np.ceil(y.shape[0] / batch_size))
    steps = epochs * steps_per_epoch

    scheduler = lr_scheduler(steps)
    optimizer = optimizer_function(scheduler)
    loader = data_loader(x1, x2,x3,x4,x5,x6,x7,x8, y,y2,y3,sigma_y,sigma_y2,sigma_y3, batch_size=batch_size)

    loss_tracker = tf.keras.metrics.Mean(name='loss')
    mse_tracker = tf.keras.metrics.MeanSquaredError(name='mse')
    mse_tracker2 = tf.keras.metrics.MeanSquaredError(name='mse2')
    mse_tracker3 = tf.keras.metrics.MeanSquaredError(name='mse3')

    loss_list = []
    mse_list = []
    mse2_list = []
    mse3_list=[]
    epi=[]
    alea=[]
    nll_list=[]
    nll2_list=[]
    nll3_list=[]

    for epoch in range(1, epochs + 1):
        for x1_batch, x2_batch,x3_batch,x4_batch,x5_batch,x6_batch,x7_batch,x8_batch, y_batch, y2_batch,y3_batch,sigma_y_batch,sigma_y2_batch,sigma_y3_batch in loader:
            loss, y_pred,y2_pred,y3_pred, epi_loss, alea_loss, nll,nll2,nll3 = step_fn(model, optimizer, x1_batch,x2_batch,x3_batch,x4_batch,x5_batch,x6_batch,x7_batch,x8_batch, y_batch,y2_batch,y3_batch,sigma_y_batch,sigma_y2_batch,sigma_y3_batch)

            loss_tracker.update_state(loss)
            mse_tracker.update_state(y_batch, y_pred)
            mse_tracker2.update_state(y2_batch, y2_pred)
            mse_tracker3.update_state(y3_batch, y3_pred)

        loss_list.append(loss_tracker.result().numpy())
        mse_list.append(mse_tracker.result().numpy())
        mse2_list.append(mse_tracker2.result().numpy())
        mse3_list.append(mse_tracker3.result().numpy())
        epi.append(epi_loss)
        alea.append(alea_loss)
        nll_list.append(nll)
        nll2_list.append(nll2)
        nll3_list.append(nll3)

        if 1 and epoch % 100 == 0:
            print(f'epoch {epoch}: loss = {loss_tracker.result():.3f}, mse = {mse_tracker.result():.3f}, mse2={mse_tracker2.result():.3f}, mse3={mse_tracker3.result():.3f}, epi={epi_loss:.3f}, alea={alea_loss:.3f}, nll={nll:.3f},nll2={nll2:.3f},nll3={nll3:.3f}')
        loss_tracker.reset_states()
        mse_tracker.reset_states()
        mse_tracker2.reset_states()
        mse_tracker3.reset_states()
    metric_tensor = [loss_list, mse_list,mse2_list,mse3_list,epi,alea,nll_list,nll2_list,nll3_list]
    return metric_tensor


#---------------TRAIN-----------------------

history = train(model, x1_train, x2_train, x3_train,x4_train,x5_train,x6_train,x7_train,x8_train, y_train,y2_train,y3_train,sigma_y_train,sigma_y2_train,sigma_y3_train, batch_size=batch_size, epochs=1000, step_fn=NCP_train_step)


# model.save_weights('../weights/BNN_EUROPED_mu_17-04_1k')
# model.load_weights('../weights/BNN_EUROPED_mu_17-04_1k')

#--------------PLOTS---------------
# Plotting training performance
fig = plt.figure(figsize=(16, 15))
ax = fig.add_subplot(331)
ax2 = fig.add_subplot(332)
ax3 = fig.add_subplot(333)
ax4 = fig.add_subplot(334)
ax5 = fig.add_subplot(335)
ax6=fig.add_subplot(336)
ax7=fig.add_subplot(337)
ax8=fig.add_subplot(338)
ax9=fig.add_subplot(339)
ax.set_title("Train Mean Loss")
ax.plot(history[0], color='red', label='BNN-NCP',scaley='log')
ax.set_yscale('log')
ax.legend()

ax2.set_title(" MSE 1")
ax2.plot(history[1], color='red', label='BNN-NCP')
ax2.set_yscale('log')
ax2.legend()

ax4.set_title("MSE2 loss")
ax4.plot(history[2], color='red', label='BNN-NCP')
ax4.set_yscale('log')
ax4.legend()

ax3.set_title("MSE3 loss")
ax3.plot(history[3], color='red', label='BNN-NCP')
ax3.set_yscale('log')
ax3.legend()

ax5.set_title("EPI loss")
ax5.plot(mean_prior_weight*np.array(history[4]), color='red', label='BNN-NCP')
ax5.set_yscale('log')
ax5.legend()

ax6.set_title("Alea loss")
ax6.plot(noise_prior_weight*np.array(history[5]), color='red', label='BNN-NCP')
ax6.set_yscale('log')
ax6.legend()

ax7.set_title("NLL1")
ax7.plot(nll_weight*(np.array(history[6])), color='red', label='BNN-NCP')
ax7.set_yscale('log')
ax7.legend()

ax8.set_title("NLL2")
ax8.plot(nll_weight2*(np.array(history[7])), color='red', label='BNN-NCP')
ax8.set_yscale('log')
ax8.legend()

ax9.set_title("NLL3")
ax9.plot(nll_weight3*(np.array(history[8])), color='red', label='BNN-NCP')
ax9.set_yscale('log')
ax9.legend()

# plt.savefig('../figures/delete_later.png')

plt.show(block=False)

#-------------CALL THE MODEL----------------------
ndim_out_dist, mean_dist,ndim_out_dist2,mean_dist2,ndim_out_dist3,mean_dist3=model([tf.constant(plasma['mu[A]'].values,dtype=tf.float32),tf.constant(plasma['Bt[T]'].values,dtype=tf.float32),
    tf.constant(plasma['epsilon'].values,dtype=tf.float32),
    tf.constant(plasma['kappa'].values,dtype=tf.float32),tf.constant(plasma['Triang.'].values,dtype=tf.float32),tf.constant(plasma['P_tot[MW]'].values,dtype=tf.float32),
    tf.constant(plasma['Gas[1e22/s]'].values,dtype=tf.float32),tf.constant(plasma['Zeff_h'].values,dtype=tf.float32)])

#POLOIDAL BETA FUNCTIONS

def beta_fun(Te_ped=plasma0.Te_ped_SC.values,ne_ped=plasma0.neped_pre_2.values,triang=plasma0['Triang.'].values,rminor=plasma0['rminor[m]'].values,kappa=plasma0.kappa.values,ip=plasma0['Ip[MA]'].values,zeff=plasma0.Zeff_h.values):
    mu0=4*np.pi*10**(-7)
    theta = np.linspace(0, 2*np.pi, 361).reshape(1,-1)
    dx = np.sin(theta + np.arcsin(triang.reshape(-1,1)) * np.sin(theta)) * (1 + np.arcsin(triang.reshape(-1,1)) * np.cos(theta))
    dy = kappa.reshape(-1,1) * np.cos(theta)
    Lp = rminor.reshape(-1,1) * np.trapz(np.sqrt(dx**2 + dy**2), theta).reshape(-1,1)
    Bp2 = (ip.reshape(-1,1)*1e6 * mu0 / Lp)**2
    pe_ped = ne_ped.reshape(-1,1)*1e19 * Te_ped.reshape(-1,1)*1.6e-16
    Zimp = 4
    pi_ped = pe_ped * (Zimp - zeff.reshape(-1,1)) / (Zimp - 1)
    beta = (pe_ped + pi_ped) * (2 * mu0) / Bp2
    return beta
def beta_error(Te_ped_error,Te_ped,ne_ped_error,ne_ped,triang=plasma0['Triang.'].values,rminor=plasma0['rminor[m]'].values,kappa=plasma0.kappa.values,ip=plasma0['Ip[MA]'].values,zeff=plasma0.Zeff_h.values):
    mu0=4*np.pi*10**(-7)
    theta = np.linspace(0, 2*np.pi, 361).reshape(1,-1)
    dx = np.sin(theta + np.arcsin(triang.reshape(-1,1)) * np.sin(theta)) * (1 + np.arcsin(triang.reshape(-1,1)) * np.cos(theta))
    dy = kappa.reshape(-1,1) * np.cos(theta)
    Lp = rminor.reshape(-1,1) * np.trapz(np.sqrt(dx**2 + dy**2), theta).reshape(-1,1)
    Bp2 = (ip.reshape(-1,1)*1e6 * mu0 / Lp)**2
    pe = ne_ped*1e19 * Te_ped*1.6e-16
    dpe=pe*(ne_ped_error/ne_ped+Te_ped_error/Te_ped)
    
    Zimp=4
    dpi=dpe*(Zimp-zeff.reshape(-1))/(Zimp-1)
    
    dbeta=(dpe+dpi)*2*mu0/Bp2.reshape(-1)
    return dbeta.reshape(-1)

from scipy.optimize import curve_fit
from sklearn.metrics import r2_score
def f(x,a):
    return a*np.sqrt(x)
#----------------PLOTS-------------------

#EXP
beta=plasma0.Te_ped_exp.values*plasma0.ne_ped_exp.values/plasma0['Ip[MA]'].values**2
delta=plasma0.Delta_exp.values
dfbeta=pd.DataFrame({'delta_nn':delta.reshape(-1),'beta_nn':beta.reshape(-1)})
# dfbeta[dfbeta['beta_nn']>100]=np.nan
# dfbeta=dfbeta.dropna(axis=0)
popt_exp,pcov=curve_fit(f,dfbeta.beta_nn,dfbeta.delta_nn,maxfev=50000)
x=np.linspace(0.5,1.4,100)
plt.plot(x,f(x,popt_exp),'r--',label='square root fit')
plt.scatter(beta,delta,s=5,label='Experimental')
plt.ylabel('Delta')
plt.xlabel('Beta')
r2_exp=r2_score(dfbeta.delta_nn, f(dfbeta.beta_nn,popt_exp))
plt.annotate('R^2='+str(np.round(r2_exp,3))+'\n'+'c='+str(np.round(popt_exp[0],3)),xy=[1.8,0.07])
plt.legend()
plt.show()

#EUROPED
beta_europed=beta_fun()#plasma.Te_ped_europed.values*plasma.ne_ped_exp.values/plasma['Ip[MA]'].values**2
delta_europed=plasma0.Delta_SC.values
beta_n=plasma0.beta_n_pred
dfbeta=pd.DataFrame({'delta_nn':delta_europed.reshape(-1),'beta_nn':beta_europed.reshape(-1),'z':beta_n.values.reshape(-1)})
dfbeta[dfbeta['beta_nn']>100]=np.nan
dfbeta=dfbeta.dropna(axis=0)
popt_eped,pcov=curve_fit(f,dfbeta.beta_nn,dfbeta.delta_nn,maxfev=50000)
x=np.linspace(0.1,0.5,100)
plt.plot(x,f(x,popt_eped),'r--',label='square root fit')

plt.scatter(beta_europed,delta_europed,s=5,label='EuroPED')


plt.ylabel(r'$\Delta$')
plt.xlabel(r'$\beta_p$')

r2_eped=r2_score(dfbeta.delta_nn, f(dfbeta.beta_nn,popt_eped))
plt.annotate('R^2='+str(np.round(r2_eped,3))+'\n'+'c='+str(np.round(popt_eped[0],8)),xy=[0.42,0.03])
plt.legend()
plt.show()

dfbeta_europed=dfbeta
def un_norm(a,var):
    return a*myscaler(plasma0[var])[2]+myscaler(plasma0[var])[1]

#NN Epistemic
beta_nn=beta_fun((mean_dist2.mean().numpy()*myscaler(plasma0.Te_ped_SC)[2])+myscaler(plasma0.Te_ped_SC)[1],(mean_dist3.mean().numpy()*myscaler(plasma0.neped_pre_2)[2])+myscaler(plasma0.neped_pre_2)[1])#(mean_dist.mean().numpy().reshape(-1)*plasma.ne_ped_exp.values)/(plasma['Ip[MA]'].values)**2
delta_nn=(mean_dist.mean().numpy()*myscaler(plasma0.Delta_SC)[2])+myscaler(plasma0.Delta_SC)[1]
dfbeta=pd.DataFrame({'delta_nn':delta_nn.reshape(-1),'beta_nn':beta_nn.reshape(-1),
    'epi_beta':beta_error(Te_ped_error=mean_dist2.stddev().numpy().reshape(-1)*myscaler(plasma0.Te_ped_SC)[2],ne_ped_error=mean_dist3.stddev().numpy().reshape(-1)*myscaler(plasma0.neped_pre_2)[2],Te_ped=un_norm(mean_dist2.mean().numpy().reshape(-1),'Te_ped_SC'),ne_ped=un_norm(mean_dist3.mean().numpy().reshape(-1),'neped_pre_2')),
    'epi_delta':mean_dist.stddev().numpy().reshape(-1)*myscaler(plasma0.Delta_SC)[2],'z':plasma0.beta_n_pred.values.reshape(-1)})
dfbeta[dfbeta['beta_nn']>100]=np.nan
dfbeta=dfbeta.dropna(axis=0)
plt.figure(dpi=200)
plt.errorbar(dfbeta.beta_nn,dfbeta.delta_nn,yerr=dfbeta.epi_delta,xerr=dfbeta.epi_beta,label='NN_pred epistemic error',fmt='o',markersize=2,elinewidth=0.09)

popt_nn,pcov=curve_fit(f,dfbeta.beta_nn,dfbeta.delta_nn,sigma=dfbeta.epi_delta,maxfev=50000)
# plt.xlim(0.1,0.5)
x=np.linspace(0.1,0.5,100)
plt.plot(x,f(x,popt_nn),'r--',label='square root fit')
r2_nn=r2_score(dfbeta.delta_nn, f(dfbeta.beta_nn,popt_nn))
plt.annotate('R^2='+str(np.round(r2_nn,3))+'\n'+'c='+str(np.round(popt_nn[0],5)),xy=[0.4,0.025])
plt.ylabel('Delta')
plt.xlabel('Beta')
plt.legend()
plt.show()
dfbeta_epi=dfbeta

#NN Aleatoric
beta_nn=beta_fun((mean_dist2.mean().numpy()*myscaler(plasma0.Te_ped_SC)[2])+myscaler(plasma0.Te_ped_SC)[1],(mean_dist3.mean().numpy()*myscaler(plasma0.neped_pre_2)[2])+myscaler(plasma0.neped_pre_2)[1])#(mean_dist.mean().numpy().reshape(-1)*plasma.ne_ped_exp.values)/(plasma['Ip[MA]'].values)**2
delta_nn=(mean_dist.mean().numpy()*myscaler(plasma0.Delta_SC)[2])+myscaler(plasma0.Delta_SC)[1]
dfbeta=pd.DataFrame({'delta_nn':delta_nn.reshape(-1),'beta_nn':beta_nn.reshape(-1),
    'epi_beta':beta_error(Te_ped_error=ndim_out_dist2.stddev().numpy().reshape(-1)*myscaler(plasma0.Te_ped_SC)[2],ne_ped_error=ndim_out_dist3.stddev().numpy().reshape(-1)*myscaler(plasma0.neped_pre_2)[2],Te_ped=un_norm(mean_dist2.mean().numpy().reshape(-1),'Te_ped_SC'),ne_ped=un_norm(mean_dist3.mean().numpy().reshape(-1),'neped_pre_2')),
    'epi_delta':ndim_out_dist.stddev().numpy().reshape(-1)*myscaler(plasma0.Delta_SC)[2],'z':plasma0.beta_n_pred.values.reshape(-1)})
dfbeta[dfbeta['beta_nn']>100]=np.nan
dfbeta=dfbeta.dropna(axis=0)
plt.figure(dpi=200)
plt.errorbar(dfbeta.beta_nn,dfbeta.delta_nn,yerr=dfbeta.epi_delta,xerr=dfbeta_epi.epi_beta,label='NN_pred aleatoric error',fmt='o',markersize=2,elinewidth=0.09)

popt_nn_alea,pcov=curve_fit(f,dfbeta.beta_nn,dfbeta.delta_nn,sigma=dfbeta_epi.epi_delta,maxfev=50000)

x=np.linspace(0.1,0.5,100)
plt.plot(x,f(x,popt_nn),'r--',label='square root fit')
r2_nn=r2_score(dfbeta.delta_nn, f(dfbeta.beta_nn,popt_nn))
plt.annotate('R^2='+str(np.round(r2_nn,3))+'\n'+'c='+str(np.round(popt_nn_alea[0],5)),xy=[0.5,0.025])
plt.ylabel('Delta')
plt.xlabel('Beta')
plt.legend()
plt.show()

#----------------3D PLOT--------------

def specific_3d_plot(data,titulo):
    # regular grid covering the domain of the data
    X,Y = np.meshgrid(np.arange(data['beta_nn'].min(),data['beta_nn'].max(), 0.01), np.arange(data['delta_nn'].min(), data['delta_nn'].max(), 0.005))
    XX = X.flatten()
    YY = Y.flatten()

    order = 1    # 1: linear, 2: quadratic
    if order == 1:
        # best-fit linear plane
        A = np.c_[data['beta_nn'].values, data['delta_nn'].values, np.ones(data.shape[0])]#
        C,_,_,_ = lstsq(A, data['z'].values)    # coefficients
        
        # evaluate it on grid
        Z = C[0]*X + C[1]*Y + C[2]
        
        # or expressed using matrix/vector product
        #Z = np.dot(np.c_[XX, YY, np.ones(XX.shape)], C).reshape(X.shape)

    elif order == 2:
        # best-fit quadratic curve
        A = np.c_[np.ones(data.shape[0]), data[:,:2], np.prod(data[:,:2], axis=1), data[:,:2]**2]
        C,_,_,_ = lstsq(A, data[:,2])
        
        # evaluate it on a grid
        Z = np.dot(np.c_[np.ones(XX.shape), XX, YY, XX*YY, XX**2, YY**2], C).reshape(X.shape)
    
    fig = go.Figure(data = [
        go.Surface(x = X,y = Y,z = Z,opacity=0.3,text='SJDSJHFBSDFH'), 
        go.Scatter3d(x=data['beta_nn'],y=data['delta_nn'], z=data['z'],mode='markers',marker=dict(
        size=1,
        colorscale='Viridis'),
        line=dict(
        color='darkblue',
        width=2
    ))])
    
    fig.update_layout(title=titulo,
        scene=dict(xaxis_title='Beta Poloidal',
            yaxis_title='Delta (psi)',
            zaxis_title='Beta Total',annotations=[
            dict(
                showarrow=False,
                x=0.4,
                y=0.05,
                z=3.5,
                text="Beta_n="+str(C[0].round(2))+'Beta_p'+str(C[1].round(2))+'Delta+'+str(C[2].round(2)),
                xanchor="left",
                xshift=10,
                opacity=0.7)]
        ),
    )
    fig.show()
    # fig.write_html('../figures/3Dplot/europed_plane_beta_n_'+titulo+'.html')
specific_3d_plot(dfbeta_europed,'EuroPED')
# fig.write_html('../figures/europed_plane_beta_n_pred0,0,0.html')
# specific_3d_plot(dfbeta)

#-----------------RANDOM DATA PLOTS------------------

size=500
def unif_sample(var,size=size):
    return np.random.uniform(low=1*plasma[var].values.min(), high=1*plasma[var].values.max(),size=size).reshape(-1)
selected_dataset['rminor[m]']=plasma0['rminor[m]']
selected_dataset['Zeff_h']=plasma0['Zeff_h']
# selected_dataset=selected_dataset.loc[:,['Ip[MA]', 'Bt[T]', 'Triang.', 'epsilon', 'kappa', 'beta_n_exp',
#        'ne_ped_exp', 'Te_ped_EPED', 'Delta_EPED', 'rminor[m]', 'Zeff_h']]
cov_plasma=np.cov(selected_dataset.values.T)+0*np.cov(selected_dataset.values.T)
mean_plasma=selected_dataset.mean().values
# mean_plasma[0]=mean_plasma[0]*0.7
# mean_plasma[-5]=mean_plasma[-5]*1.5
Bt_rn=unif_sample('Bt[T]')
Ip_rn=unif_sample('Ip[MA]')
mu_rn=unif_sample('mu[A]')
Trian_rn=unif_sample('Triang.')
eps_rn=unif_sample('epsilon')
kap_rn=unif_sample('kappa')
beta_rn=unif_sample('beta_n_exp')
ne_ped_rn=unif_sample('ne_ped_exp')
ptot_rn=unif_sample('P_tot[MW]')
gas_rn=unif_sample('Gas[1e22/s]')
zeff_rn=unif_sample('Zeff_h')
rmin_rn=unif_sample('rminor[m]')
def un_norm(a,var):
    return a*myscaler(plasma0[var])[2]+myscaler(plasma0[var])[1]
Bt_r=un_norm(Bt_rn,'Bt[T]')
mu_r=un_norm(mu_rn,'mu[A]')
Ip_r=un_norm(Ip_rn,'Ip[MA]')
Trian_r=un_norm(Trian_rn,'Triang.')
eps_r=un_norm(eps_rn,'epsilon')
kap_r=un_norm(kap_rn,'kappa')
beta_r=un_norm(beta_rn,'beta_n_exp')
ne_ped_r=un_norm(ne_ped_rn,'ne_ped_exp')
ptot_r=un_norm(ptot_rn,'P_tot[MW]')
gas_r=un_norm(gas_rn,'Gas[1e22/s]')
zeff_r=un_norm(zeff_rn,'Zeff_h')
rmin_r=un_norm(rmin_rn,'rminor[m]')
sample=np.random.multivariate_normal(mean_plasma,cov_plasma,size)
sample_norm=tf.constant(myscaler(sample)[0],dtype=tf.float32)
# sample_norm
alea_dist_rand,mean_dist_rand,alea_dist_rand2,mean_dist_rand2,alea_dist_rand3,mean_dist_rand3,=model([tf.constant(mu_rn,dtype=tf.float32),
    tf.constant(Bt_rn,dtype=tf.float32),tf.constant(eps_rn,dtype=tf.float32),
    tf.constant(kap_rn,dtype=tf.float32),
    tf.constant(Trian_rn,dtype=tf.float32),tf.constant(ptot_rn,dtype=tf.float32),
    tf.constant(gas_rn,dtype=tf.float32),tf.constant(zeff_rn,dtype=tf.float32)])


beta_nn=beta_fun(mean_dist_rand2.mean().numpy()*myscaler(plasma0.Te_ped_SC)[2]+myscaler(plasma0.Te_ped_SC)[1],ip=Ip_r,
    ne_ped=(mean_dist_rand3.mean().numpy()*myscaler(plasma0.neped_pre_2)[2])+myscaler(plasma0.neped_pre_2)[1],
    triang=Trian_r,rminor=rmin_r,kappa=kap_r,zeff=zeff_r)
delta_nn=mean_dist_rand.mean().numpy()*myscaler(plasma0.Delta_SC)[2]+myscaler(plasma0.Delta_SC)[1]
dfbeta_random=pd.DataFrame({'delta_nn':delta_nn.reshape(-1),'beta_nn':beta_nn.reshape(-1),
    'epi_beta':beta_error(Te_ped_error=mean_dist_rand2.stddev().numpy().reshape(-1)*myscaler(plasma0.Te_ped_SC)[2],ne_ped_error=mean_dist_rand3.stddev().numpy().reshape(-1)*myscaler(plasma0.neped_pre_2)[2],Te_ped=un_norm(mean_dist_rand2.mean().numpy().reshape(-1),'Te_ped_SC'),ne_ped=un_norm(mean_dist_rand3.mean().numpy().reshape(-1),'neped_pre_2'),ip=Ip_r,triang=Trian_r,rminor=rmin_r,kappa=kap_r,zeff=zeff_r),
    'epi_delta':mean_dist_rand.stddev().numpy().reshape(-1)*myscaler(plasma0.Delta_SC)[2],
    'alea_beta':beta_error(Te_ped_error=alea_dist_rand2.stddev().numpy().reshape(-1)*myscaler(plasma0.Te_ped_SC)[2],ne_ped_error=alea_dist_rand3.stddev().numpy().reshape(-1)*myscaler(plasma0.neped_pre_2)[2],Te_ped=un_norm(mean_dist_rand2.mean().numpy().reshape(-1),'Te_ped_SC'),ne_ped=un_norm(mean_dist_rand3.mean().numpy().reshape(-1),'neped_pre_2'),ip=Ip_r,triang=Trian_r,rminor=rmin_r,kappa=kap_r,zeff=zeff_r),
    'alea_delta':alea_dist_rand.stddev().numpy().reshape(-1)*myscaler(plasma0.Delta_SC)[2]})
dfbeta_random[dfbeta_random['beta_nn']>100]=np.nan
dfbeta_random=dfbeta_random.dropna(axis=0)
dfbeta_random=dfbeta_random.loc[(dfbeta_random['beta_nn']>0.),:]
plt.figure(dpi=200)
plt.errorbar(dfbeta_random.beta_nn,dfbeta_random.delta_nn,yerr=dfbeta_random.epi_delta,xerr=dfbeta_random.epi_beta,label='Random Values epistemic error',fmt='o',markersize=2,elinewidth=0.08)
x=np.linspace(0,1,100)
plt.plot(x,f(x,popt_nn),'r--',label='square root fit')
plt.annotate('c='+str(np.round(popt_nn[0],4)),xy=[1.5,0.01])
plt.ylabel('Delta')
plt.xlabel('Beta')
plt.legend()
plt.show()
plt.figure(dpi=200)
plt.errorbar(dfbeta_random.beta_nn,dfbeta_random.delta_nn,yerr=dfbeta_random.alea_delta,xerr=dfbeta_random.alea_beta,label='Random Values aleatoric error',fmt='o',markersize=2,elinewidth=0.08)
# plt.xlim(-0.25,2)
# plt.ylim(-0.075,0.1)
# popt_nn,pcov=curve_fit(f,dfbeta_random.beta_nn,dfbeta_random.delta_nn,sigma=dfbeta_random.epi_delta,maxfev=50000)

x=np.linspace(0,1,100)
plt.plot(x,f(x,popt_nn),'r--',label='square root fit')
# r2_nn=r2_score(dfbeta_random.delta_nn, f(dfbeta_random.beta_nn,popt_nn))
plt.annotate('c='+str(np.round(popt_nn[0],4)),xy=[1.5,0.01])
plt.ylabel('Delta')
plt.xlabel('Beta')
plt.legend()
plt.show()
def min_dist(beta,delta):
    x=np.linspace(0,2,100)
    y=f(x,popt_nn)
    d=np.min(np.sqrt((x-beta)**2+100*(y-delta)**2))
    return d
dfbeta.beta_nn.mean()
def min_dist_data(beta,delta):
    x=dfbeta_europed.beta_nn.mean()
    y=dfbeta_europed.delta_nn.mean()
    # x=dfbeta.beta_nn.values.reshape(-1)
    # y=dfbeta.delta_nn.values.reshape(-1)
    d=np.sqrt((x-beta)**2+100*(y-delta)**2)
    return d
dfbeta_random['dist_curve']=np.vectorize(min_dist)(dfbeta_random.loc[:,'beta_nn'],dfbeta_random.loc[:,'delta_nn'])
dfbeta_random['dist_data']=min_dist_data(dfbeta_random.loc[:,'beta_nn'],dfbeta_random.loc[:,'delta_nn'])
dfbeta_random['dist']=dfbeta_random['dist_data']+dfbeta_random['dist_curve']

fig,ax=plt.subplots(2,2,dpi=200)
plt.suptitle('Relative error vs Distance to data')

sns.scatterplot(data=dfbeta_random,x=dfbeta_random['dist_data'],y=dfbeta_random['epi_beta']/dfbeta_random['beta_nn'],ax=ax[0,0])
ax[0,0].set_ylim(0,1)
ax[0,0].set_ylabel('epi_beta')
# sns.scatterplot(data=data_weird,x='dist_data',y='epi_beta',color='r',label='weird points',ax=ax[0,0])
sns.scatterplot(data=dfbeta_random,x=dfbeta_random['dist_data'],y=dfbeta_random['epi_delta']/dfbeta_random['delta_nn'],ax=ax[1,0])
# ax[1,0].ylabel('epi_delta')
ax[1,0].set_ylim(0,1)
ax[1,0].set_ylabel('epi_delta')
# sns.scatterplot(data=data_weird,x='dist_data',y='epi_delta',color='r',label='weird points',ax=ax[1,0])
sns.scatterplot(data=dfbeta_random,x=dfbeta_random['dist_data'],y=dfbeta_random['alea_beta']/dfbeta_random['beta_nn'],ax=ax[0,1])
ax[0,1].set_ylim(0,1)
ax[0,1].set_ylabel('alea_beta')
# sns.scatterplot(data=data_weird,x='dist_data',y='alea_beta',color='r',label='weird points',ax=ax[0,1])
sns.scatterplot(data=dfbeta_random,x=dfbeta_random['dist_data'],y=dfbeta_random['alea_delta']/dfbeta_random['delta_nn'],ax=ax[1,1])
# sns.scatterplot(data=data_weird,x='dist_data',y='alea_delta',color='r',label='weird points',ax=ax[1,1])
ax[1,1].set_ylim(0,1)
ax[1,1].set_ylabel('alea_delta')
plt.show()

fig,ax=plt.subplots(2,2,dpi=200)
plt.suptitle('Distance to data')
sns.scatterplot(data=dfbeta_random,x='dist_data',y='epi_beta',ax=ax[0,0])

# sns.scatterplot(data=data_weird,x='dist_data',y='epi_beta',color='r',label='weird points',ax=ax[0,0])
sns.scatterplot(data=dfbeta_random,x='dist_data',y='epi_delta',ax=ax[1,0])
ax[1,0].set_ylim(0,0.01)

# sns.scatterplot(data=data_weird,x='dist_data',y='epi_delta',color='r',label='weird points',ax=ax[1,0])
sns.scatterplot(data=dfbeta_random,x='dist_data',y='alea_beta',ax=ax[0,1])

# sns.scatterplot(data=data_weird,x='dist_data',y='alea_beta',color='r',label='weird points',ax=ax[0,1])
sns.scatterplot(data=dfbeta_random,x='dist_data',y='alea_delta',ax=ax[1,1])
ax[1,1].set_ylim(0,0.01)
# sns.scatterplot(data=data_weird,x='dist_data',y='alea_delta',color='r',label='weird points',ax=ax[1,1])

plt.show()