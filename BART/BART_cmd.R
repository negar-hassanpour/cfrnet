args <- commandArgs(TRUE)
curr_dir <- args[1]
data_name <- args[2]
source( paste( c(curr_dir,"/BART/BART.R"), collapse="" ) )
BART(curr_dir, data_name)