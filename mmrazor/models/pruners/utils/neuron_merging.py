
import numpy as np
from sklearn.metrics import pairwise_distances

def create_scaling_mat_ip_thres_bias(weight, ind, threshold, model_type):
    '''
    weight - 2D matrix (n_{i+1}, n_i), np.ndarray
    ind - chosen indices to remain, np.ndarray
    threshold - cosine similarity threshold
    '''
    assert(type(weight) == np.ndarray)
    assert(type(ind) == np.ndarray)

    cosine_sim = 1-pairwise_distances(weight, metric="cosine")
    weight_chosen = weight[ind, :]
    scaling_mat = np.zeros([weight.shape[0], weight_chosen.shape[0]])

    for i in range(weight.shape[0]):
        if i in ind: # chosen
            ind_i, = np.where(ind == i)
            assert(len(ind_i) == 1) # check if only one index is found
            scaling_mat[i, ind_i] = 1
        else: # not chosen
            if model_type == 'prune':
                continue
            max_cos_value = np.max(cosine_sim[i][ind])
            max_cos_value_index = np.argpartition(cosine_sim[i][ind], -1)[-1]

            if threshold and max_cos_value < threshold:
                continue

            baseline_weight = weight_chosen[max_cos_value_index]
            current_weight = weight[i]
            baseline_norm = np.linalg.norm(baseline_weight)
            current_norm = np.linalg.norm(current_weight)
            scaling_factor = current_norm / baseline_norm
            scaling_mat[i, max_cos_value_index] = scaling_factor

    return scaling_mat

def create_scaling_mat_conv_thres_bn(weight, ind, threshold,
                                     bn_weight, bn_bias,
                                     bn_mean, bn_var, lam, model_type):
    '''
    weight - 4D tensor(n, c, h, w), np.ndarray
    ind - chosen indices to remain
    threshold - cosine similarity threshold
    bn_weight, bn_bias - parameters of batch norm layer right after the conv layer
    bn_mean, bn_var - running_mean, running_var of BN (for inference)
    lam - how much to consider cosine sim over bias, float value between 0 and 1
    '''
    assert(type(weight) == np.ndarray)
    assert(type(ind) == np.ndarray)
    assert(type(bn_weight) == np.ndarray)
    assert(type(bn_bias) == np.ndarray)
    assert(type(bn_mean) == np.ndarray)
    assert(type(bn_var) == np.ndarray)
    assert(bn_weight.shape[0] == weight.shape[0])
    assert(bn_bias.shape[0] == weight.shape[0])
    assert(bn_mean.shape[0] == weight.shape[0])
    assert(bn_var.shape[0] == weight.shape[0])
    
    
    weight = weight.reshape(weight.shape[0], -1)

    cosine_dist = pairwise_distances(weight, metric="cosine")

    weight_chosen = weight[ind, :]
    scaling_mat = np.zeros([weight.shape[0], weight_chosen.shape[0]])

    for i in range(weight.shape[0]):
        if i in ind: # chosen
            ind_i, = np.where(ind == i)
            assert(len(ind_i) == 1) # check if only one index is found
            scaling_mat[i, ind_i] = 1
        else: # not chosen

            if model_type == 'prune':
                continue

            current_weight = weight[i]
            current_norm = np.linalg.norm(current_weight)
            current_cos = cosine_dist[i]
            gamma_1 = bn_weight[i]
            beta_1 = bn_bias[i]
            mu_1 = bn_mean[i]
            sigma_1 = bn_var[i]
            
            # choose one
            cos_list = []
            scale_list = []
            bias_list = []
            
            for chosen_i in ind:
                chosen_weight = weight[chosen_i]
                chosen_norm = np.linalg.norm(chosen_weight, ord = 2)
                chosen_cos = current_cos[chosen_i]
                gamma_2 = bn_weight[chosen_i]
                beta_2 = bn_bias[chosen_i]
                mu_2 = bn_mean[chosen_i]
                sigma_2 = bn_var[chosen_i]
                
                # compute cosine sim
                cos_list.append(chosen_cos)
                
                # compute s
                s = current_norm/chosen_norm
                
                # compute scale term
                scale_term_inference = s * (gamma_2 / gamma_1) * (sigma_1 / sigma_2)
                scale_list.append(scale_term_inference)
                
                # compute bias term
                bias_term_inference = abs((gamma_2/sigma_2) * (s * (-(sigma_1*beta_1/gamma_1) + mu_1) - mu_2) + beta_2)

                bias_term_inference = bias_term_inference/scale_term_inference

                bias_list.append(bias_term_inference)

            assert(len(cos_list) == len(ind))
            assert(len(scale_list) == len(ind))
            assert(len(bias_list) == len(ind))
            

            # merge cosine distance and bias distance
            bias_list = (bias_list - np.min(bias_list)) / (np.max(bias_list)-np.min(bias_list))

            score_list = lam * np.array(cos_list) + (1-lam) * np.array(bias_list)


            # find index and scale with minimum distance
            min_ind = np.argmin(score_list)

            min_scale = scale_list[min_ind]
            min_cosine_sim = 1-cos_list[min_ind]

            # check threshold - second
            if threshold and min_cosine_sim < threshold:
                continue
            
            scaling_mat[i, min_ind] = min_scale

    return scaling_mat

