import os
import subprocess
from multiprocessing import Pool

from cfr.util import *

curr_dir = '/home/negar/cfrnet/'
which_data = 'scaling_practice'

file_names = [
	'ea8ec4f5364049a19cb6cf92df0e2593.csv',
	'd09f96200455407db569ae33fe06b0d3.csv'
	]
data_dir = curr_dir+'/data/'+which_data+'/'
# file_names = os.listdir(data_dir)
# file_names.sort()

temp_dir = curr_dir+'temp/'
if not os.path.isdir(temp_dir):
	os.makedirs(temp_dir)

def dump_csv(file_path, data):
	with open(file_path, "wb") as csvfile:
		writer = csv.writer(csvfile, delimiter=',')
		writer.writerows(data)

def del_CSVs(files_path, file_name):
	for suffix in ['x','t','y','id']:
		os.remove(files_path+file_name+'_'+suffix+'.csv')


for file_name in file_names:
# def parallel_bart(input):
# 	curr_dir  = input[0]
# 	data_dir  = input[1]
# 	temp_dir  = input[2]
# 	file_name = input[3]
	
	if not os.path.isfile(curr_dir+"/results/BART/"+file_name[:-4]+".csv"):
		data = load_data(data_dir+file_name[:-4])
		
		dump_csv(temp_dir+file_name[:-4]+"_x.csv",  data['x'][:,:,0])
		dump_csv(temp_dir+file_name[:-4]+"_t.csv",  data['t'])
		dump_csv(temp_dir+file_name[:-4]+"_y.csv",  data['yf'])
		dump_csv(temp_dir+file_name[:-4]+"_id.csv", data['id'])

		retcode = subprocess.call(['/usr/bin/Rscript','BART/BART_cmd.R', curr_dir, file_name[:-4]])

		del_CSVs(temp_dir, file_name[:-4])

# pool = Pool(processes = 8)
# work_results = pool.map(parallel_bart, (curr_dir, data_dir, temp_dir, file_names))
# pool.close()
# pool.join()
