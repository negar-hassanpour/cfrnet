import os
import csv
import numpy as np

from causalbenchmark import evaluate

which_data = 'censoring_practice'

# file_names = [
# 	'd1546da12d8e4daf8fe6771e2187954d.csv'
# 	]
data_dir = '/home/negar/cfrnet/data/'+which_data+'/'
file_names = os.listdir(data_dir)
file_names.sort()

def load_config(cfg_file):
	cfg = {}

	with open(cfg_file,'r') as f:
		for l in f:
			l = l.strip()
			if len(l)>0 and not l[0] == '#':
				vs = l.split(': ')
				if len(vs)>0:
					k,v = (vs[0], (vs[1]))
					if not isinstance(v,list):
						v = [v]
					cfg[k] = v
	for key in ['outdir', 'data_test', 'datadir', 'dataform', 'output_csv', 'save_rep']:
		cfg.pop(key, None)
	return cfg

def load_meta_data(dir):
	mode_lines = {}
	config = {}
	with open(dir+'/configs_sorted.txt', 'r') as f:
		for l in f:
			l = l.strip()
			l = l.split('\t')
			line_num = l[0]
			l = l[1].split(', ')
			for c in l:
				key, val = c.split('=')
				config[key] = val
			if 'mode' in config.keys():
				if 'm_'+config['mode'] not in mode_lines.keys():
					mode_lines['m_'+config['mode']] = line_num
			else:
				mode_lines['m_2.0'] = line_num
				break

	with open(dir+'/folders_sorted.txt', 'r') as f:
		for l in f:
			l = l.strip()
			l = l.split('\t')
			line_num = l[0]
			l = l[1].split('/')
			for mode in mode_lines.keys():
				if line_num == mode_lines[mode]:
					mode_lines[mode] = l[-1]
	
	return mode_lines, mode_lines.keys()


os.chdir('results/'+which_data)

import shutil
if os.path.isfile('m_0.0'):
	shutil.rmtree('m_0.0')
if os.path.isfile('m_2.0'):
	shutil.rmtree('m_2.0')

for file_name in file_names:
	file_name = file_name[:-4]
	if file_name[-3:] != '_cf':
		if file_name in os.listdir(os.getcwd()):

			flag = True
			os.chdir(file_name)
			mode_lines, modes_avail = load_meta_data(os.getcwd())
			print mode_lines

			for mode in modes_avail:
				results_dir = '../'+mode+'/'
				if not os.path.exists(results_dir):
					os.makedirs(results_dir)
				
				if which_data[-4:] != 'test':
					score = evaluate.evaluate(predictions_location=os.getcwd()+'/'+mode_lines[mode],
												  cf_dir_location='../../../data/'+which_data,
												  is_individual_prediction=True)
					with open(os.getcwd()+"/results_IBM.txt", "a") as txtfile:
						if flag == True: # write the header only once
							txtfile.write('\t\t' + '\t'.join(score.keys()) + '\n')
							flag = False
						temp = mode + '\t'
						for key in score.keys():
							temp += '%.4f ' % score[key] + '\t'
						txtfile.write(temp + '\n')

				os.chdir(mode_lines[mode])
				files = os.listdir(os.getcwd())
				for file in files:
					if file.endswith( '.csv' ):
						os.system('cp '+file+' '+'../'+results_dir)
				
				os.chdir('..')
			os.chdir('..')


# main_path = os.getcwd()
# data_dir = main_path + '/data_cf'
# if not os.path.exists(data_dir):
# 	os.makedirs(data_dir)

# os.chdir('../../data/'+which_data)
# for file_name in file_names:
# 	if file_name[-4-3:-4] == '_cf':
# 		os.system('cp' +' '+ file_name +' '+ data_dir)
# 		# os.system('mv' +' '+ data_dir+'/'+file_name +' '+ data_dir+'/'+file_name[:-4-3]+'.csv')


if which_data[-4:] != 'test':
	os.chdir('../../results/'+which_data)
	flag = True
	modes_avail.append('BART')
	for mode in modes_avail:
		score = evaluate.evaluate(predictions_location=mode,
									  cf_dir_location='../../data/'+which_data,
									  is_individual_prediction=True)
		with open('../../results/'+which_data+"/results_IBM.txt", "a") as txtfile:
			if flag == True: # write the header only once
				txtfile.write('\t\t' + '\t'.join(score.keys()) + '\n')
				flag = False
			temp = mode + '\t'
			for key in score.keys():
				temp += '%.4f ' % score[key] + '\t'
			txtfile.write(temp + '\n')
