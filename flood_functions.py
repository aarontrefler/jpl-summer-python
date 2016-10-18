####################
# plot_grace_feature_histogram: Plots grace feature at time of each flood as a histogram
# Parameters:
# feature_description: to be used in x-axis title
# feature_values: Pandas Series of feature values
# num_bins: number of bins for histogram
# ylabel: y-axis label

def plot_grace_feature_histogram(feature_description, feature_values, \
    title='Ground Water Level Prior to Flood\n04/02 to 05/16', ylabel='Number of Floods',\
    num_bins=50, middle_value=0):

    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    # GRACE levels during flood events
    y,_,_ = plt.hist(feature_values, bins=num_bins,color='b')
    plt.title(title)
    plt.xlabel(feature_description)
    plt.ylabel(ylabel)
    
    # add median and zero lines
    
    y_max = int(y.max())
    # median line
    feature_median = np.median(feature_values)
    plt.plot( [feature_median]*y_max , range(y_max), 'r', linewidth=1.5)  
    # zero line
    plt.plot( [middle_value]*y_max, range(y_max), 'g', linewidth=1.5)
    
    #plt.savefig(dir_figures + 'floods_graceFeaure_' + feature_description)
    #plt.show()

####################
# brick_convert_highres: Creates higher resolution data brick 
#
# Parameters:
# data_brick: 3D matrix of values
# scale: increase resolution of data by scale factor
#
# Returns: 3D matrix of values at higherresolution

def brick_convert_highres(data_brick, scale):

    import numpy as np
    import warnings
    
    # function variables
    dim = data_brick.shape
    d1 = dim[0]  
    d2 = dim[1]
    tp = dim[2]
    
    d1_highres = d1 * scale
    d2_highres = d2 * scale
    
    data_brick_highres = np.empty([d1_highres,d2_highres,tp])
    data_brick_highres.fill(np.nan)
    
    for i in range(tp):       
        for j in range(d1_highres):          
            for k in range(d2_highres):

                data_brick_highres[j,k,i] = data_brick[j/scale, k/scale, i]
    
    return data_brick_highres


####################
# grace_brick_convert_lowres: Converts GRACE brick from 0.5x0.5 to 3x3 degree resolution
#
# Parameters:
# grace_brick: 3D matrix of values at 0.5x0.5 degree resolution
# scale: downsize high resolution data by scale factor
#
# Returns: 3D matrix of values at 3x3 degree resolution

def grace_brick_convert_lowres(data_brick_highres, scale):

    import numpy as np
    import warnings
    
    # function variables
    dim = data_brick_highres.shape
    d1 = dim[0]  
    d2 = dim[1]
    tp = dim[2]
    
    d1_lowres = d1/scale
    d2_lowres = d2/scale
    
    data_brick_lowres = np.empty([d1_lowres,d2_lowres,tp])
    data_brick_lowres.fill(np.nan)
    for i in range(tp):
        
        for j in range(0,d1,scale):
            idx_d1 = (j+scale-1)/scale
            r_d1 = range(j,j+scale-1)
            r_d1_beg = min(r_d1)
            r_d1_end = max(r_d1)+1

            for k in range(0,d2,scale):
                idx_d2 = (k+scale-1)/scale
                r_d2 = range(k,k+scale-1)
                r_d2_beg = min(r_d2)
                r_d2_end = max(r_d2)+1

                # expecting warning when calculating mean of all NaN values
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=RuntimeWarning)    
                    data_brick_lowres[idx_d1,idx_d2,i] = \
                        np.nanmean(data_brick_highres\
                        [r_d1_beg:r_d1_end,r_d2_beg:r_d2_end,i])
    
    return data_brick_lowres 


####################
# grace_2d_convert_lowres: Converts GRACE map from 0.5x0.5 to 3x3 degree resolution
#
# Parameters:
# grace_map: 3D matrix of values at 0.5x0.5 degree resolution
# scale: downsize high resolution data by scale factor
#
# Returns: 2D matrix of values at 3x3 degree resolution

def grace_2d_convert_lowres(data_highres, scale):

    import numpy as np
    import warnings

    dim = data_highres.shape
    d1 = dim[0]  
    d2 = dim[1]

    d1_lowres = d1/scale
    d2_lowres = d2/scale

    data_lowres = np.empty([d1_lowres,d2_lowres])
    data_lowres.fill(np.nan)

    for j in range(0,d1,scale):
        idx_d1 = (j+scale-1)/scale
        r_d1 = range(j,j+scale-1)
        r_d1_beg = min(r_d1)
        r_d1_end = max(r_d1)+1

        for k in range(0,d2,scale):
            idx_d2 = (k+scale-1/scale)
            r_d2 = range(k,k+scale-1)
            r_d2_beg = min(r_d2)
            r_d2_end = max(r_d2)+1

            # expecting warning when calculating mean of all NaN values
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)    
                
                data_lowres[idx_d1,idx_d2] = \
                    np.nanmean(data_highres\
                    [r_d1_beg:r_d1_end,r_d2_beg:r_d2_end])
    
    return data_lowres


####################
# calculate_confusion_matrix_bricks: Calculates confusion-matrix metrics for across all geogrpahical locations and time-points
#
# Parameters:
# label_brick: 3D grace brick of labels
# pred_brick: 3D grace brick of predictions
#
# Returns: 3D bricks for true positive, true negative, false postivie, and false negative metrics

def calculate_confusion_matrix_bricks(label_brick, pred_brick):

    import numpy as np

    # function variables
    dim = pred_brick.shape
    d1 = dim[0]  
    d2 = dim[1]
    tp = dim[2]

    # create confusion matrix bricks
    metric_tp_brick = np.zeros([d1,d2,tp]) #hit
    metric_fp_brick = np.zeros([d1,d2,tp]) #type I error
    metric_fn_brick = np.zeros([d1,d2,tp]) #type II error
    metric_tn_brick = np.zeros([d1,d2,tp]) #correct regection

    for i in range(d1):
        for j in range(d2):
            for k in range(tp):
                
                if(pred_brick[i,j,k] == 1 and label_brick[i,j,k] == 1):
                    metric_tp_brick[i,j,k] = 1
                elif(pred_brick[i,j,k] == 1 and label_brick[i,j,k] == 0):
                    metric_fp_brick[i,j,k] = 1
                elif(pred_brick[i,j,k] == 0 and label_brick[i,j,k] == 0):
                    metric_tn_brick[i,j,k] = 1
                elif(pred_brick[i,j,k] == 0 and label_brick[i,j,k] == 1):
                    metric_fn_brick[i,j,k] = 1

    return {'true_positive':metric_tp_brick, \
            'true_negative':metric_tn_brick, \
            'false_positive':metric_fp_brick, \
            'false_negative':metric_fn_brick}

####################
# calculate_ml_metric_maps: Calculates machine-learning metric maps ffor each geogrpahical location
#
# Parameters:
# label_brick: 3D grace brick of labels
# metric_tp_brick: true positive brick
# metric_fp_brick: false positive brick
# metric_tn_brick: true negative brick
# metric_fn_brick: false negative brick
#
# Returns: 2D maps for accuracy, miss-classification rate, true positive rate, false positive rate, specificity, precision, and prevalence

def calculate_ml_metric_maps(label_brick, pred_brick, metric_tp_brick, metric_fp_brick, metric_tn_brick, metric_fn_brick):

    import numpy as np

    # function variables
    dim = label_brick.shape
    d1 = dim[0]  
    d2 = dim[1]
    tp = dim[2]

    # create ML metric maps

    #accuracy
    metric_accuracy_map = np.empty([d1,d2]) 
    metric_accuracy_map.fill(np.nan)
    #miss-classifcation rate
    metric_missClassRate_map = np.empty([d1,d2]) 
    metric_missClassRate_map.fill(np.nan)
    #true positive rate
    metric_tpRate_map = np.empty([d1,d2]) 
    metric_tpRate_map.fill(np.nan)
    #false positive rate
    metric_fpRate_map = np.empty([d1,d2]) 
    metric_fpRate_map.fill(np.nan)
    #specificity
    metric_specificity_map = np.empty([d1,d2]) 
    metric_specificity_map.fill(np.nan)
    #precision
    metric_precision_map = np.empty([d1,d2]) 
    metric_precision_map.fill(np.nan)
    #prevalence
    metric_prevalence_map = np.empty([d1,d2]) 
    metric_prevalence_map.fill(np.nan)

    for i in range(d1):
        for j in range(d2):
        
            if (np.nansum(label_brick[i,j,:]) > 0): #flood events occured
                
                metric_accuracy_map[i,j] = \
                    (np.sum(metric_tp_brick[i,j,:]) + np.sum(metric_tn_brick[i,j,:])) / tp
                metric_missClassRate_map[i,j] = \
                    (np.sum(metric_fp_brick[i,j,:]) + np.sum(metric_fn_brick[i,j,:])) / tp
                metric_tpRate_map[i,j] = \
                    np.sum(metric_tp_brick[i,j,:]) /  np.sum(label_brick[i,j,:])
                metric_fpRate_map[i,j] = \
                    np.sum(metric_fp_brick[i,j,:]) /  np.sum(label_brick[i,j,:] == 0)
                metric_specificity_map[i,j] = \
                    np.sum(metric_tn_brick[i,j,:]) /  np.sum(label_brick[i,j,:] == 0)
                metric_precision_map[i,j] = \
                    np.sum(metric_tp_brick[i,j,:]) /  np.sum(pred_brick[i,j,:] == 1)
                metric_prevalence_map[i,j] = \
                    np.sum(label_brick[i,j,:] == 1)  / tp

    return {'accuracy':metric_accuracy_map, \
            'missClassRate':metric_missClassRate_map, \
            'tpRate':metric_tpRate_map, \
            'fpRate':metric_fpRate_map, \
            'specificity':metric_specificity_map, \
            'precision':metric_precision_map, \
            'prevalence':metric_prevalence_map}

