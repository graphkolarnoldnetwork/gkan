from __future__ import division
from __future__ import print_function

import time
import argparse
import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim

from pygcn_all.pygcn.utils import load_data, accuracy, normalize
from pygcn_all.pygcn.models import GCN
from pykan.kan import *
from utils import *
import matplotlib.pyplot as plt
import os
import pickle
import sys
import pickle 


# settigns for the input arguments
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False,
                    help='Validate during training pass.')
parser.add_argument('--input_dataset', type=str, choices=['cora', 'citeseer','pubmed'], default='citeseer',
                    help='The dataset to use. Options are "cora", "citeseer", and "pubmed".')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=600,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.001,#0.0005
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=.0,#5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=5,#50(pubmed),#100,#50,  #104
                    help='Number of hidden units in GCN.')


parser.add_argument('--optimizer_gkan', type=str, choices=['Adam', 'LBFGS'], default='Adam',
                    help='The optimizer of GKAN. Options are "Adams" and "LBFGS".')
parser.add_argument('--hidden_GKAN', type=int, default=16,#10(pubmed),#20,#10,     ## <=15
                    help='Number of hidden units in GKAN architectures.')
parser.add_argument('--k_GKAN', type=int, default=1,
                    help='degree of polynomial in KAN.')
parser.add_argument('--g_GKAN', type=int, default=3, #<=======2
                    help='Number of intervals in KAN.')
parser.add_argument('--grid_eps', type=float, default=.01, # 0.01(so far) default 1.0=======
                    help='grid eps (1.0 means all uniform interval and 0.0 means all from data).')
parser.add_argument('--grid_update_num', type=int, default=1*10,#10*10,#450//3, # default 1.0=======
                    help='# of times for updating the grid.')
parser.add_argument('--stop_grid_update_step', type=int, default=100,#450, # default 1.0=======
                    help='stopping point of updating grid.')
parser.add_argument('--batch_kan', type=int, default=-1,#450, # default 1.0=======
                    help='batch size of GKAN, in case of selecting -1, it will consider the whole data.')
parser.add_argument('--weight_decay_kan', type=float, default=0,#1e-2, #5e-3
                    help='Weight decay KAN Adam Optimizer(L2 loss on parameters).')

parser.add_argument('--lamb', type=float, default=.0,#5e-4,#1.,#0.01,#0.01(Archit 1),#0.00005
                    help='lamb value in regularization of KAN.')
parser.add_argument('--lamb_l1', type=float, default=1.0,
                    help='lamb l1 value in regularization of KAN.')
parser.add_argument('--lamb_entropy', type=float, default=1.0,#0.0
                    help='lamb entropy value in regularization of KAN.')
parser.add_argument('--lamb_coef', type=float, default=1.,#.0
                    help='lamb_coef value in regularization of KAN.')
parser.add_argument('--lamb_coefdiff', type=float, default=.0,#1.,#.0
                    help='lamb_coefdiff value in regularization of KAN.')
parser.add_argument('--dropout', type=float, default=0,
                    help='Dropout rate (1 - keep probability).')

args = parser.parse_args()

optimizaer_GKAN_type = args.optimizer_gkan  # LBFGS or Adam
name_dataset = args.input_dataset # 'cora' or 'citeseer' or 'pubmed'
args.cuda = not args.no_cuda and torch.cuda.is_available()


np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# checking the avaliability of CUDA
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    
    # Get the default CUDA device 
    device = torch.device('cuda')
    total_memory = torch.cuda.get_device_properties(device).total_memory
    print(f'Total GPU Memory: {total_memory / 1024**3:.2f} GB')
    # Print allocated memory
    allocated_memory = torch.cuda.memory_allocated(device)
    print(f'Allocated GPU Memory: {allocated_memory / 1024**3:.2f} GB')
    # Print free memory (total - allocated)
    free_memory = torch.cuda.memory_reserved(device) - allocated_memory
    print(f'Free GPU Memory: {free_memory / 1024**3:.2f} GB')
else:
    device = torch.device('cpu')
    print("CUDA is not available. Check your installation and GPU.")
print(f"Device is {device}")
# load data
with open('datasets/'+name_dataset+'_dataset.pkl', 'rb') as f:  # 'rb' denotes read binary mode
    adj, features, labels, idx_train, idx_val, idx_test = pickle.load(f)
print(f"idx_train, idx_val, idx_test {idx_train.shape, idx_val.shape, idx_test.shape}")
print(f"training {idx_train}")

# cora
# idx_train = torch.arange(0, 1000)
# idx_val = torch.arange(1000, 1100)
# idx_test = torch.arange(1100, 1560)

# cora
# idx_train = torch.arange(0, 500)
# idx_val = torch.arange(500, 501)
# idx_test = torch.arange(640, 1640)
# citeseer
# idx_train = torch.arange(0, 500)
# idx_val = torch.arange(500, 620)
# idx_test = torch.arange(620, 1620)

#pubmed
# idx_train = torch.arange(0, 500)
# idx_val = torch.arange(500, 560)
# idx_test = torch.arange(560, 1560)


number_of_input_features = 100#features.shape[1] # number of input features to consider
input_feat_KAN = number_of_input_features # the number of input features should match the input dimension of GKAN architectures 
# set "number_of_input_features" to a lower value than the number of features in case you want to truncate the 
features = features[:,0:number_of_input_features]
 
# adjust training and test data as well as labels
dataset = {}
dataset['all_input'] = features
dataset['all_label'] = labels
dataset['idx_train'] = idx_train
dataset['idx_test'] = idx_test
dataset['idx_val'] = idx_val
dataset['train_input'] = features[idx_train]
dataset['train_label'] = labels[idx_train]
dataset['test_input'] = features[idx_test]
dataset['test_label'] = labels[idx_test]
adj = adj.to_dense()
dataset['adj'] = adj


# create the GKAN models as well as GCN
model_kan_archit_2 = KAN(width=[input_feat_KAN, args.hidden_GKAN, labels.max().item() + 1], grid=args.g_GKAN, k=args.k_GKAN, noise_scale=0.1, seed=0,base_fun=torch.nn.ReLU(),symbolic_enabled=False,bias_trainable=True,device=device, architecture='2',original_input_dim = features.shape[1],dimension_transformation_layer = None, grid_eps=args.grid_eps)
model_kan_archit_2_k2 = KAN(width=[input_feat_KAN, args.hidden_GKAN, labels.max().item() + 1], grid=args.g_GKAN, k=args.k_GKAN+2, noise_scale=0.1, seed=0,base_fun=torch.nn.ReLU(),symbolic_enabled=False,bias_trainable=True,device=device, architecture='2',original_input_dim = features.shape[1],dimension_transformation_layer = None, grid_eps=args.grid_eps)
model_kan_archit_2_k3 = KAN(width=[input_feat_KAN, args.hidden_GKAN, labels.max().item() + 1], grid=args.g_GKAN, k=args.k_GKAN+4, noise_scale=0.1, seed=0,base_fun=torch.nn.ReLU(),symbolic_enabled=False,bias_trainable=True,device=device, architecture='2',original_input_dim = features.shape[1],dimension_transformation_layer = None, grid_eps=args.grid_eps)

#model_kan_archit_1 = KAN(width=[input_feat_KAN, args.hidden_GKAN, labels.max().item() + 1], grid=args.g_GKAN, k=args.k_GKAN, noise_scale=0.1, seed=0,base_fun=torch.nn.ReLU(),symbolic_enabled=False,bias_trainable=True,device=device, architecture='1',original_input_dim = features.shape[1],dimension_transformation_layer = None, grid_eps=args.grid_eps)

model = GCN(nfeat=dataset['train_input'].shape[1],
             nhid=args.hidden,
             nclass=labels.max().item() + 1,
             dropout=args.dropout)
optimizer = optim.Adam(model.parameters(),
                        lr=args.lr, weight_decay=args.weight_decay)

# load models and dataset on CUDA (in case of being available)
if args.cuda:
    torch.cuda.empty_cache()
    model_kan_archit_2.cuda()
    model_kan_archit_2_k2.cuda()
    model_kan_archit_2_k3.cuda()
    
    #model_kan_archit_1.cuda()
    model.cuda()
    features = features.cuda()
    adj = adj.cuda()
    labels = labels.cuda()
    idx_train = idx_train.cuda()
    idx_val = idx_val.cuda()
    idx_test = idx_test.cuda()

    dataset['all_input'] = dataset['all_input'].cuda()
    dataset['all_label'] = dataset['all_label'].cuda()
    dataset['idx_train'] = dataset['idx_train'].cuda()
    dataset['idx_test'] = dataset['idx_test'].cuda()
    dataset['idx_val'] = dataset['idx_val'].cuda()
    dataset['train_input'] = dataset['train_input'].cuda()
    dataset['train_label'] = dataset['train_label'].cuda()
    dataset['test_input'] = dataset['test_input'].cuda()
    dataset['test_label'] = dataset['test_label'].cuda()
    dataset['adj'] = dataset['adj'].cuda()
    print(f" Model and Data are on CUDA {dataset['test_label'].device} ")


# train GCN model (and record the results)
result_loss_train_GCN, result_acc_train_GCN, result_loss_val_GCN, result_acc_val_GCN = [], [], [], []
result_loss_test_GCN, result_acc_test_GCN = [], []
t_total = time.time()
for epoch in range(args.epochs):
    loss_train_GCN, acc_train_GCN, loss_val_GCN, acc_val_GCN = train_GCN(model,features,adj,idx_train,idx_val,labels,optimizer,epoch)
    result_loss_train_GCN.append(loss_train_GCN.cpu().detach().numpy())
    result_acc_train_GCN.append(acc_train_GCN.cpu().detach().numpy())
    result_loss_val_GCN.append(loss_val_GCN.cpu().detach().numpy())
    result_acc_val_GCN.append(acc_val_GCN.cpu().detach().numpy())

    loss_test_GCN, acc_test_GCN = test_GCN(model,features,adj,idx_test,labels)
    result_loss_test_GCN.append(loss_test_GCN.cpu().detach().numpy())
    result_acc_test_GCN.append(acc_test_GCN.cpu().detach().numpy())

resutls_gcn = {}
resutls_gcn['train_loss'] = result_loss_train_GCN
resutls_gcn['test_loss'] = result_loss_test_GCN
resutls_gcn['train_acc'] = result_acc_train_GCN
resutls_gcn['test_acc'] = result_acc_test_GCN
print(np.mean(result_acc_test_GCN[-50:]))


# train GKAN architectures
resutls_archit_2 = model_kan_archit_2.train_kan(dataset,opt=optimizaer_GKAN_type, steps=args.epochs, lamb=args.lamb, lamb_l1=args.lamb_l1,lamb_entropy=args.lamb_entropy,lamb_coef=args.lamb_coef, lamb_coefdiff=args.lamb_coefdiff, lr=args.lr,device=device, weight_decay = args.weight_decay_kan, grid_update_num = args.grid_update_num, stop_grid_update_step = args.stop_grid_update_step, batch=args.batch_kan)
resutls_archit_2_k2 = model_kan_archit_2.train_kan(dataset,opt=optimizaer_GKAN_type, steps=args.epochs, lamb=args.lamb, lamb_l1=args.lamb_l1,lamb_entropy=args.lamb_entropy,lamb_coef=args.lamb_coef, lamb_coefdiff=args.lamb_coefdiff, lr=args.lr,device=device, weight_decay = args.weight_decay_kan, grid_update_num = args.grid_update_num, stop_grid_update_step = args.stop_grid_update_step, batch=args.batch_kan)
resutls_archit_2_k3 = model_kan_archit_2.train_kan(dataset,opt=optimizaer_GKAN_type, steps=args.epochs, lamb=args.lamb, lamb_l1=args.lamb_l1,lamb_entropy=args.lamb_entropy,lamb_coef=args.lamb_coef, lamb_coefdiff=args.lamb_coefdiff, lr=args.lr,device=device, weight_decay = args.weight_decay_kan, grid_update_num = args.grid_update_num, stop_grid_update_step = args.stop_grid_update_step, batch=args.batch_kan)

#resutls_archit_1 = model_kan_archit_1.train_kan(dataset,opt=optimizaer_GKAN_type, steps=args.epochs, lamb=args.lamb, lamb_l1=args.lamb_l1,lamb_entropy=args.lamb_entropy,lamb_coef=args.lamb_coef, lamb_coefdiff=args.lamb_coefdiff, lr=args.lr,device=device, weight_decay = args.weight_decay_kan, grid_update_num = args.grid_update_num, stop_grid_update_step = args.stop_grid_update_step, batch=args.batch_kan)




# ====== save results ======
name_of_setting = 'Feat_'+str(number_of_input_features)+'h_gcn_'+str(args.hidden)+'h_'+str(args.hidden_GKAN)+'k_'+str(args.k_GKAN)+'g_'+str(args.g_GKAN)

with open('results/GCN'+name_of_setting+'_'+name_dataset+'.pickle', 'wb') as handle:
    pickle.dump(resutls_gcn, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open('results/GKAN_2'+name_of_setting+'_'+name_dataset+'.pickle', 'wb') as handle:
    pickle.dump(resutls_archit_2, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open('results/GKAN_2_k2'+name_of_setting+'_'+name_dataset+'.pickle', 'wb') as handle:
    pickle.dump(resutls_archit_2, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open('results/GKAN_2_k3'+name_of_setting+'_'+name_dataset+'.pickle', 'wb') as handle:
    pickle.dump(resutls_archit_2, handle, protocol=pickle.HIGHEST_PROTOCOL)
# with open('results/GKAN_1'+name_of_setting+'_'+name_dataset+'.pickle', 'wb') as handle:
#     pickle.dump(resutls_archit_1, handle, protocol=pickle.HIGHEST_PROTOCOL)

# ======  plot results ======
path_current = os.getcwd()

# plot test accuracy
plt.figure()  # Create a new figure
#plt.plot([x+1 for x in range(args.epochs)], result_acc_train_GCN, 'ro-',label='GCN',markersize=4, linewidth=1)  
plt.plot([x+1 for x in range(args.epochs)], resutls_archit_2['train_acc'], 'bo-',label='GKAN Archit. 2 w. g='+str(args.g_GKAN),markersize=4, linewidth=1)  
plt.plot([x+1 for x in range(args.epochs)], resutls_archit_2_k2['train_acc'], 'mo-',label='GKAN Archit. 2 w. g='+str(args.g_GKAN+4),markersize=4, linewidth=1)  
plt.plot([x+1 for x in range(args.epochs)], resutls_archit_2_k3['train_acc'], 'ko-',label='GKAN Archit. 2 w. g='+str(args.g_GKAN+8),markersize=4, linewidth=1)  

#plt.plot([x+1 for x in range(args.epochs)], resutls_archit_1['train_acc'], 'go-',label='GKAN Archit. 1',markersize=4, linewidth=1)  
plt.title('Training Accuracy Over Epochs - '+name_dataset+' dataset')  
plt.xlabel('Epoch')  
plt.ylabel('Accuracy')
plt.grid(True, color='gray', linestyle='--', linewidth=0.5)
plt.legend(loc='lower right')
plt.savefig('plots/plot_acc_train'+name_of_setting+'_'+name_dataset+'.png')

# plot train accuracy
plt.figure()  
#plt.plot([x+1 for x in range(args.epochs)], result_acc_test_GCN, 'ro-',label='GCN',markersize=4, linewidth=1)  
plt.plot([x+1 for x in range(args.epochs)], resutls_archit_2['test_acc'], 'bo-',label='GKAN Archit. 2 w. g='+str(args.g_GKAN),markersize=4, linewidth=1)  
plt.plot([x+1 for x in range(args.epochs)], resutls_archit_2_k2['test_acc'], 'mo-',label='GKAN Archit. 2 w. g='+str(args.g_GKAN+4),markersize=4, linewidth=1)  
plt.plot([x+1 for x in range(args.epochs)], resutls_archit_2_k3['test_acc'], 'ko-',label='GKAN Archit. 2 w. g='+str(args.g_GKAN+8),markersize=4, linewidth=1)  

#plt.plot([x+1 for x in range(args.epochs)], resutls_archit_1['test_acc'], 'ko-',label='GKAN Archit. 1',markersize=4, linewidth=1)  
plt.title('Test Accuracy Over Epochs - '+name_dataset+' dataset')  
plt.xlabel('Epoch')  
plt.ylabel('Accuracy')
plt.grid(True, color='gray', linestyle='--', linewidth=0.5)
plt.legend(loc='lower right')
plt.savefig('plots/plot_acc_test'+name_of_setting+'_'+name_dataset+'.png')

# # plot train loss
# plt.figure() 
# #plt.plot([x+1 for x in range(args.epochs)], result_loss_train_GCN, 'ro-',label='GCN',markersize=4, linewidth=1)  
# plt.plot([x+1 for x in range(args.epochs)], resutls_archit_2['train_loss'], 'bo-',label='GKAN Archit. 2 w. g='+str(args.k_GKAN),markersize=4, linewidth=1)
# plt.plot([x+1 for x in range(args.epochs)], resutls_archit_2_k2['train_loss'], 'mo-',label='GKAN Archit. 2 w. g='+str(args.k_GKAN+4),markersize=4, linewidth=1)
# plt.plot([x+1 for x in range(args.epochs)], resutls_archit_2_k3['train_loss'], 'ko-',label='GKAN Archit. 2 w. g='+str(args.k_GKAN+8),markersize=4, linewidth=1)

# #plt.plot([x+1 for x in range(args.epochs)], resutls_archit_1['train_loss'], 'go-',label='GKAN Archit. 1',markersize=4, linewidth=1)
# plt.title('Training Loss per Epoch - '+name_dataset+' dataset')  
# plt.xlabel('Epoch')  
# plt.ylabel('Train Loss')
# plt.grid(True, color='gray', linestyle='--', linewidth=0.5)
# plt.legend()
# plt.savefig('plots/plot_loss_train'+name_of_setting+'_'+name_dataset+'.png')
# # plot test loss
# plt.figure() 
# plt.plot([x+1 for x in range(args.epochs)], result_loss_test_GCN, 'ro-',label='GCN',markersize=4, linewidth=1)  
# plt.plot([x+1 for x in range(args.epochs)], resutls_archit_2['test_loss'], 'bo-',label='GKAN Archit. 2',markersize=4, linewidth=1)
# plt.plot([x+1 for x in range(args.epochs)], resutls_archit_1['test_loss'], 'go-',label='GKAN Archit. 1',markersize=4, linewidth=1)
# plt.title('Test Loss per Epoch - '+name_dataset+' dataset')  
# plt.xlabel('Epoch')  
# plt.ylabel('Loss')
# plt.grid(True, color='gray', linestyle='--', linewidth=0.5)
# plt.legend()
# plt.savefig('plots/plot_loss_test'+name_of_setting+'_'+name_dataset+'.png')


# trainable_params_gcn = sum(p.numel() for p in model.parameters() if p.requires_grad)
# trainable_params_gkan_achit_2 = sum(p.numel() for p in model_kan_archit_2.parameters() if p.requires_grad)
# trainable_params_gkan_achit_1 = sum(p.numel() for p in model_kan_archit_1.parameters() if p.requires_grad)

# print(f"Number of parameters GCN: {trainable_params_gcn} gkan_achit_2 {trainable_params_gkan_achit_2} gkan_achit_1 {trainable_params_gkan_achit_1}")

# load results for printing the resutls
# with open('results/GCN'+name_of_setting+'_'+name_dataset+'.pickle', 'rb') as handle_gcn:
#     loaded_dict_gcn = pickle.load(handle_gcn)
#     print(f"GCN: Training:{loaded_dict_gcn['train_acc'][-1]} Test: {np.mean(loaded_dict_gcn['test_acc'][-50:])}")
# with open('results/GKAN_2'+name_of_setting+'_'+name_dataset+'.pickle', 'rb') as handle_gkan_2:
#     loaded_dict_gkan_2 = pickle.load(handle_gkan_2)
#     print(f"GKAN 2: Training:{loaded_dict_gkan_2['train_acc'][-1]} Test: {np.mean(loaded_dict_gkan_2['test_acc'][-50:])} ")
# with open('results/GKAN_1'+name_of_setting+'_'+name_dataset+'.pickle', 'rb') as handle_gkan_1:
#     loaded_dict_gkan_1 = pickle.load(handle_gkan_1)
#     print(f"GKAN 1: Training:{loaded_dict_gkan_1['train_acc'][-1]} Test: {np.mean(loaded_dict_gkan_1['test_acc'][-50:])}")
