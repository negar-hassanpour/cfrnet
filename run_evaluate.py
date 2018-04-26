import os

which_data = 'censoring_practice'

# file_names = [
# 	'd1546da12d8e4daf8fe6771e2187954d.csv'
# 	]
data_dir = '/home/negar/cfrnet/data/'+which_data+'/'
file_names = os.listdir(data_dir)
file_names.sort()

for file_name in file_names:
	file_name = file_name[:-4]
	if file_name[-3:] != "_cf":

		with open('configs/'+file_name+'.txt', "r") as txtfile:
			configs = txtfile.read()
		configs = configs.split("\n")
		configs[36] = "outdir=\'results/"+which_data+"/"+file_name+"\'"
		configs[37] = "datadir=\'data/"+which_data+"/"+"\'"
		configs = "\n".join(configs)
		with open('configs/'+file_name+'.txt', "w") as txtfile:
			txtfile.write(configs)

		os.system('python evaluate.py configs/'+file_name+'.txt 1')
	
		# os.chdir('IBM-Causal-Inference-Benchmarking-Framework/code/')
		# os.system('python score.py '+file_name)
		
		# exit()