import os
import subprocess
from multiprocessing import Pool

from cfr.util import *

curr_dir = "C:\\Users\\Samad\\Desktop\\cfrnet\\"

def dump_csv(file_path, data):
	with open(file_path, "w") as csvfile:
		writer = csv.writer(csvfile, delimiter=',')
		writer.writerows(data)

def del_CSVs(files_path, file_name):
	for suffix in ['x','t','y','id']:
		os.remove(files_path+file_name+'_'+suffix+'.csv')


temp_dir = curr_dir+'temp/'
if not os.path.isdir(temp_dir):
	os.makedirs(temp_dir)

data_dir = curr_dir+'data/censoring_practice/'
file_names = os.listdir(data_dir)
file_names.sort()				# bonanza
# file_names.sort(reverse=True)	# pipestone

# file_names = []
for file_name in file_names:
	file_name = file_name[:-4]
	if file_name[:-3] != "_cf":
# def parallel_bart(input):
# 	curr_dir  = input[0]
# 	data_dir  = input[1]
# 	temp_dir  = input[2]
# 	file_name = input[3]
	
		if not os.path.isfile(curr_dir+"/results/BART/"+file_name+".csv"):
			data = load_data(data_dir+file_name)

			''' handle censored data '''
			I_uncensored = [i for i,yf in enumerate(data['yf']) if not np.isnan(yf)]
			x = data['x'][I_uncensored,:,0]
			t = data['t'][I_uncensored]
			yf = data['yf'][I_uncensored]
			
			dump_csv(temp_dir+file_name+"_x.csv",  x)
			dump_csv(temp_dir+file_name+"_x_tst.csv",  data['x'])
			dump_csv(temp_dir+file_name+"_t.csv",  t)
			dump_csv(temp_dir+file_name+"_y.csv",  yf)
			dump_csv(temp_dir+file_name+"_id.csv", data['id'])

			retcode = subprocess.call(['E:\\Program Files\\R\\R-3.3.2\\bin\\Rscript','BART/BART_cmd.R', curr_dir, file_name])

			del_CSVs(temp_dir, file_name)

# pool = Pool(processes = 8)
# work_results = pool.map(parallel_bart, (curr_dir, data_dir, temp_dir, file_names))
# pool.close()
# pool.join()
