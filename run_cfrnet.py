import os

which_data = 'censoring_test'

if not os.path.isdir('results/'+which_data):
	os.makedirs('results/'+which_data)

data_dir = '/home/negar/cfrnet/data/'+which_data+'/'
file_names = os.listdir(data_dir)
file_names.sort()#reverse=True

def create_config_file(file_name, which_data):
	with open("config_ref.txt", "r") as txtfile:
		config_ref = txtfile.read()

	with open("configs/"+file_name+".txt", "w") as txtfile:
		config_ref += "outdir=\'results/"+which_data+"/"+file_name+"\'\n"
		config_ref += "datadir=\'data/"+which_data+"/\'\n"
		config_ref += "dataform=\'"+file_name+"\'\n"
		txtfile.write(config_ref)

for file_name in file_names:
	file_name = file_name[:-4]
	if file_name[-3:] != '_cf':
		create_config_file(file_name, which_data)
		
		results_path = 'results/'+which_data+'/'+file_name+'/'
		if not os.path.isdir(results_path):
			os.makedirs(results_path)
		
		# if os.path.isfile(results_path+'used_configs.txt'):
		# 	os.remove(results_path+'used_configs.txt') # for test purposes

		# os.system('python cfr_param_search.py configs/'+file_name+'.txt _')
		os.system('python cfr_param_search.py configs/'+file_name+'.txt _weighted_')

		# exit()
