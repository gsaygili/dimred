#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  2 15:32:51 2021

@author: busraozgode
"""
import numpy as np
import matplotlib.pyplot as plt
import sys
from sklearn.metrics import mean_squared_error
import pickle
from sklearn.linear_model import LinearRegression
from scipy.stats import variation
from scipy.optimize import curve_fit

def evaluate_regression (model, X_test, y_te_score, y_te_ind, y_npr, save_folder):
  y_pred = model.predict(X_test)
  save_folder = "/Users/busraozgode/Desktop/t-SNE/Datasets/AMB_integrated/results/"
  ind_corr = [ n for n,i in enumerate(y_te_score) if i>=0.5 ] 
  ind_err = [ n for n,i in enumerate(y_te_score) if i<0.5 ] # same variable with y_te_ind 
  
  #for writing log screen in a file
  stdoutOrigin=sys.stdout 
  sys.stdout = open(save_folder+"log.txt", "w")

  data = [y_npr[ind_corr],y_npr[ind_err],y_pred[ind_corr],y_pred[ind_err]]
  fig = plt.figure(figsize =(10, 7))
  plt.rc('font', size=32)
  # Creating axes instance
  ax = fig.add_axes([0, 0, 1, 1])  
  # x-axis labels
  ax.set_xticklabels(['NPR (1)', 'NPR (0)',
                    'RF (1)', 'RF(0)'], fontsize=40)
  plt.ylabel("RF/NPR Score", fontsize=40)
  # Creating plot
  bp = ax.boxplot(data)
  plt.savefig(save_folder+"box_plot.pdf", bbox_inches="tight")
  # show plot
  plt.show()

  # Calculate the absolute errors
  errors = np.abs(y_pred - y_te_score)

  print('y_pred_mean:',np.mean(y_pred))
  print('y_pred_std:',np.std(y_pred))

  print('y_test_mean:',np.mean(y_te_score))
  print('y_test_std:',np.std(y_te_score))

  # print('y_train_mean:',np.mean(y_tr_score))
  # print('y_train_std:',np.std(y_tr_score))
  
# Calculate the absolute errors for y_npr
  errors2 = np.abs(y_npr-y_te_score)
  print('y_npr_mean:',np.mean(y_npr))
  print('y_npr_std:',np.std(y_npr))

  
  # Print out the mean absolute error (mae)
  print('Mean Absolute Error:', round(np.mean(errors), 2), 'degrees.')
  print('Mean Absolute Error for npr:', round(np.mean(errors2), 2), 'degrees.')

  # Calculate mean squared error (MSE)
  mse=mean_squared_error(y_te_score, y_pred)
  mse2=mean_squared_error(y_te_score, y_npr)
  print('Mean Squared Error:', mse)
  print('Mean Squared Error for npr:', mse2)
  
  rmse = np.sqrt(mse)
  rmse2= np.sqrt(mse2)
  print('Root Mean Squared Error:', rmse)
  print('Root Mean Squared Error for npr:', rmse2)
  
  print('coefficient of variation (CV):', variation(y_pred))
  print('coefficient of variation (CV) for npr:', variation(y_npr))
  
  #plotting regression line
  
  sort_index=np.argsort(y_pred,axis=0)
  y_pred_sorted=y_pred[(sort_index)]
  sorted_test=y_te_score[(sort_index)]
  
  sample_size=y_te_score.shape[0]
  fig = plt.figure(figsize =(10, 7))
  plt.rc('font', size=20)
  # define the true objective function
  def objective(x, a, b):
	  return a * x + b
  # curve fit
  popt, _ = curve_fit(objective,y_te_score, y_pred)
  # summarize the parameter values
  a, b = popt
  print('y = %.5f * x + %.5f' % (a, b))
  # plot input vs output
  plt.scatter(sorted_test[0:sample_size], y_pred_sorted[0:sample_size])
  plt.xlabel("Target Score", fontsize=30)
  plt.ylabel("RF Score", fontsize=30)
  plt.plot(sorted_test[0:sample_size], objective(sorted_test[0:sample_size], *popt), 'r--',
         label='fit: a=%5.3f, b=%5.3f' % tuple(popt)) 
  plt.savefig(save_folder+"RF_Regression_pred.pdf", bbox_inches="tight")
  plt.show()
  
   #plotting regression line for npr
   
  sort_index2=np.argsort(y_npr,axis=0)
  y_npr_sorted=y_npr[(sort_index2)]
  sorted_test2=y_te_score[(sort_index2)]
  
  sample_size2=y_te_score.shape[0]
  fig = plt.figure(figsize =(10, 7))
  # define the true objective function
  def objective(x, a, b):
	  return a * x + b
  # curve fit
  popt, _ = curve_fit(objective,y_te_score, y_npr)
  # summarize the parameter values
  a, b = popt
  print('y = %.5f * x + %.5f' % (a, b))
  # plot input vs output
  plt.scatter(sorted_test2[0:sample_size2], y_npr_sorted[0:sample_size2])
  plt.xlabel("Target Score", fontsize=30)
  plt.ylabel("NPR Score", fontsize=30)
  plt.plot(sorted_test2[0:sample_size2], objective(sorted_test2[0:sample_size2], *popt), 'r--',
         label='fit: a=%5.3f, b=%5.3f' % tuple(popt))
  plt.savefig(save_folder+"RF_Regression_npr.pdf", bbox_inches="tight")
  plt.show()
  
  #Box_plot2
  si= [ n for n,i in enumerate(y_te_score) if i<=0.1 ]
  si1= [ n for n,i in enumerate(y_te_score) if (i<=0.2 and i>0.1) ]
  si2= [ n for n,i in enumerate(y_te_score) if (i<=0.3 and i>0.2) ]
  si3= [ n for n,i in enumerate(y_te_score) if (i<=0.4 and i>0.3) ]
  si4= [ n for n,i in enumerate(y_te_score) if (i<=0.5 and i>0.4) ]
  si5= [ n for n,i in enumerate(y_te_score) if (i<=0.6 and i>0.5) ]
  si6= [ n for n,i in enumerate(y_te_score) if (i<=0.7 and i>0.6) ]
  si7= [ n for n,i in enumerate(y_te_score) if (i<=0.8 and i>0.7) ]
  si8= [ n for n,i in enumerate(y_te_score) if (i<=0.9 and i>0.8) ]
  si9= [ n for n,i in enumerate(y_te_score) if (i>0.9) ]

  data2 = [y_pred[si],y_pred[si1],y_pred[si2],y_pred[si3],y_pred[si4],y_pred[si5],y_pred[si6],y_pred[si7],y_pred[si8],y_pred[si9]]
  fig = plt.figure()
  # Creating axes instance
  ax2 = fig.add_axes([0, 0, 1, 1])
  # x-axis labels
  ax2.set_xticklabels(['0.0-0.1', '0.1-0.2','0.2-0.3','0.3-0.4', '0.4-0.5', '0.5-0.6',
                    '0.6-0.7', '0.7-0.8','0.8-0.9', '0.9-1.0'], fontsize=10)
  # Creating plot
  bp = ax2.boxplot(data2)
  plt.rc('font', size=10)
  plt.xlabel("Target Score", fontsize=20)
  plt.ylabel("RF Score", fontsize=20)
  plt.savefig(save_folder+"box_plot_pred.pdf", bbox_inches="tight")
  # show plot
  plt.show()
  
  #Box_plot2 for npr
  si= [ n for n,i in enumerate(y_te_score) if i<=0.1 ]
  si1= [ n for n,i in enumerate(y_te_score) if (i<=0.2 and i>0.1) ]
  si2= [ n for n,i in enumerate(y_te_score) if (i<=0.3 and i>0.2) ]
  si3= [ n for n,i in enumerate(y_te_score) if (i<=0.4 and i>0.3) ]
  si4= [ n for n,i in enumerate(y_te_score) if (i<=0.5 and i>0.4) ]
  si5= [ n for n,i in enumerate(y_te_score) if (i<=0.6 and i>0.5) ]
  si6= [ n for n,i in enumerate(y_te_score) if (i<=0.7 and i>0.6) ]
  si7= [ n for n,i in enumerate(y_te_score) if (i<=0.8 and i>0.7) ]
  si8= [ n for n,i in enumerate(y_te_score) if (i<=0.9 and i>0.8) ]
  si9= [ n for n,i in enumerate(y_te_score) if (i>0.9) ]

  data3 = [y_npr[si],y_npr[si1],y_npr[si2],y_npr[si3],y_npr[si4],y_npr[si5],y_npr[si6],y_npr[si7],y_npr[si8],y_npr[si9]]
  fig = plt.figure()
  # Creating axes instance
  ax2 = fig.add_axes([0, 0, 1, 1])
  ax2.set_xticklabels(['0.0-0.1', '0.1-0.2','0.2-0.3','0.3-0.4', '0.4-0.5', '0.5-0.6',
                    '0.6-0.7', '0.7-0.8','0.8-0.9', '0.9-1.0'], fontsize=10)
  # Creating plot
  bp = ax2.boxplot(data3) 
  plt.rc('font', size=10)
  plt.xlabel("Target Score", fontsize=20)
  plt.ylabel("NPR Score", fontsize=20)
  plt.savefig(save_folder+"box_plot_npr.pdf", bbox_inches="tight")
  # show plot
  plt.show()
  #calculate slop and r2
  model = LinearRegression()
  model.fit(sorted_test.reshape((-1, 1)),y_pred_sorted.reshape((-1, 1)))
  r_sq = model.score(sorted_test.reshape((-1, 1)),y_pred_sorted.reshape((-1, 1)))
  print('coefficient of determination (R2) for pred:', r_sq)
  print('slope for pred:', model.coef_)
  
  #calculate slop and r2 for npr
  model2 = LinearRegression()
  model2.fit(sorted_test2.reshape((-1, 1)),y_npr_sorted.reshape((-1, 1)))
  r_sq2 = model2.score(sorted_test2.reshape((-1, 1)),y_npr_sorted.reshape((-1, 1)))
  print('coefficient of determination (R2)for npr:', r_sq2)
  print('slope for npr:', model2.coef_)
  
  #Correct prediction of last N error
  print("-----for prediction-------")
  
  ind = np.argsort(y_pred)

  print('corrects in last'+ str(len(y_te_ind)) +' sample:', len(np.intersect1d(ind[:len(y_te_ind)], y_te_ind)))
  print('corrects in last 100 sample:', len(np.intersect1d(ind[:100], y_te_ind)))
  print('corrects in last 50 sample:', len(np.intersect1d(ind[:50], y_te_ind)))
  print('corrects in last 10 sample:', len(np.intersect1d(ind[:10], y_te_ind)))
  
  #for npr
  print("-----for npr-------")
  
  ind2 = np.argsort(y_npr)

  print('corrects in last'+ str(len(y_te_ind)) +' sample:', len(np.intersect1d(ind2[:len(y_te_ind)], y_te_ind)))
  print('corrects in last 100 sample:', len(np.intersect1d(ind2[:100], y_te_ind)))
  print('corrects in last 50 sample:', len(np.intersect1d(ind2[:50], y_te_ind)))
  print('corrects in last 10 sample:', len(np.intersect1d(ind2[:10], y_te_ind)))
  
  np.save(save_folder,"y_npr.npy", y_npr)
  sys.stdout.flush()
  sys.stderr.flush() 
  sys.stdout.close()
  sys.stdout=stdoutOrigin
get_model = ".../model_file/"
model = pickle.load(open(get_model, 'rb'))
# X_test: fetures of dataset, y_te_score:target values, 
# y_te_ind: indexes of errors obtained from find_error_score function, y_npr:NPR values obtained from calc_npr
evaluate_regression(model, X_test, y_te_score, y_te_ind, y_npr) 