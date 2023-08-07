import sys
import os
import numpy as np
import torch
import torch.utils.data as du
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import util
from util import *
from DrugGene_Network import *
import argparse
from sklearn.preprocessing import MinMaxScaler


def predict_drugGene(predict_data, gene_dim, drug_dim, model_file, hidden_folder, batch_size, result_file, cell_features, drug_features):

	feature_dim = gene_dim + drug_dim
	model = torch.load(model_file, map_location='cuda:%d' % CUDA_ID)
	predict_feature, predict_label = predict_data
	predict_label_gpu = predict_label.cuda(CUDA_ID)

	model.cuda(CUDA_ID)

	test_loader = du.DataLoader(du.TensorDataset(predict_feature,predict_label), batch_size=batch_size, shuffle=False)
	test_predict = torch.zeros(0, 0).cuda(CUDA_ID)
	term_hidden_map = {}	

	batch_num = 0
	for i, (inputdata, labels) in enumerate(test_loader):
		features = build_input_vector(inputdata, cell_features, drug_features)

		cuda_features = Variable(features.cuda(CUDA_ID), requires_grad=False)
		# aux_out_map, term_hidden_map, final_input, gene_out, drug_out = model(cuda_features)
		aux_out_map, term_hidden_map = model(cuda_features)
		# print("final_input: ", final_input)

		# print("drug_out: ", drug_out)

		# gene_out_np = gene_out.cpu().detach().numpy()
		# drug_out_np = drug_out.cpu().detach().numpy()

		# np.savetxt("./Result_sample/Embedding" + '/gene_embedding.txt', gene_out_np, '%.4e', delimiter=",")
		# np.savetxt("./Result_sample/Embedding" + '/drug_embedding.txt', drug_out_np, '%.4e', delimiter=",")

		if test_predict.size()[0] == 0:
			test_predict = aux_out_map['final'].data
		else:
			test_predict = torch.cat([test_predict, aux_out_map['final'].data], dim=0)

		# for term, hidden_map in term_hidden_map.items():
		# 	hidden_file = hidden_folder+'/'+term+'.hidden'
		# 	with open(hidden_file, 'ab') as f:
		# 		np.savetxt(f, hidden_map.data.cpu().numpy(), '%.4e')

		batch_num += 1
		print(batch_num)

	test_corr = pearson_corr(test_predict, predict_label_gpu)
	print("Test pearson corr\t%s\t%.6f" % (model.subsystem_root, test_corr))
	# print(test_predict)
	np.savetxt(result_file+'/drugGene.txt', test_predict.cpu().numpy(), '%.4e')


parser = argparse.ArgumentParser(description='Train dcell')
parser.add_argument('-predict', help='Dataset to be predicted', type=str)
parser.add_argument('-batchsize', help='Batchsize', type=int, default=2000)
parser.add_argument('-gene2id', help='Gene to ID mapping file', type=str, default=1000)
parser.add_argument('-drug2id', help='Drug to ID mapping file', type=str, default=1000)
parser.add_argument('-cell2id', help='Cell to ID mapping file', type=str, default=1000)
parser.add_argument('-load', help='Model file', type=str, default='MODEL/model_200')
parser.add_argument('-hidden', help='Hidden output folder', type=str, default='Hidden/')
parser.add_argument('-result', help='Result file name', type=str, default='Result/')
parser.add_argument('-cuda', help='Specify GPU', type=int, default=0)
parser.add_argument('-genotype', help='Mutation information for cell lines', type=str)
parser.add_argument('-fingerprint', help='Morgan fingerprint representation for drugs', type=str)

parser.add_argument('-exp', help='exp information for cell lines', type=str)
parser.add_argument('-cn', help='cn information for cell lines', type=str)

opt = parser.parse_args()
torch.set_printoptions(precision=5)

predict_data, cell2id_mapping, drug2id_mapping = prepare_predict_data(opt.predict, opt.cell2id, opt.drug2id)
gene2id_mapping = load_mapping(opt.gene2id)

cell_mutation = np.genfromtxt(opt.genotype, delimiter=',')
cell_exp = np.genfromtxt(opt.exp, delimiter=',')
cell_cn = np.genfromtxt(opt.cn, delimiter=',')
drug_features = np.genfromtxt(opt.fingerprint, delimiter=',')

# encoding
# max = cell_mutation.shape[1] - 1
# for line in cell_features:
# 	for i in range(max, -1, -1):
# 		if i == 0:
# 			continue
# 		if line[i] == line[i - 1]:
# 			line[i] = 0
# 		else:
# 			line[i] = 1

min_max_scaler = MinMaxScaler()
exp_result = min_max_scaler.fit_transform(cell_exp)
cell_exp = np.around(exp_result, 5)
cn_result = min_max_scaler.fit_transform(cell_cn)
cell_cn = np.around(cn_result, 5)
cell_features = cell_mutation + cell_exp + cell_cn

num_cells = len(cell2id_mapping)
num_drugs = len(drug2id_mapping)
num_genes = len(gene2id_mapping)
drug_dim = len(drug_features[0, :])

CUDA_ID = opt.cuda

print("batch: ")
predict_drugGene(predict_data, num_genes, drug_dim, opt.load, opt.hidden, opt.batchsize, opt.result, cell_features, drug_features)
