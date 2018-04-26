BART <- function(curr_dir, data_name){

	library(BayesTree)
	set.seed(1)

	ntree = 100
		
	# read data
	x 		= as.matrix( read.csv( paste( c(curr_dir,"temp/",data_name,"_x.csv"),  collapse="" ), header = FALSE ) )
	y 		= as.matrix( read.csv( paste( c(curr_dir,"temp/",data_name,"_y.csv"),  collapse="" ), header = FALSE ) )
	t 		= as.matrix( read.csv( paste( c(curr_dir,"temp/",data_name,"_t.csv"),  collapse="" ), header = FALSE ) )
	IDs		= as.matrix( read.csv( paste( c(curr_dir,"temp/",data_name,"_id.csv"), collapse="" ), header = FALSE ) )

	idx0 = t == 0
	idx1 = t == 1

	yhat = matrix(, nrow = nrow(x), ncol = 3)
	yhat[,1] = IDs

	fit = bart(x[idx0,],y[idx0], x, ntree)
	yhat[,2] = fit$yhat.test.mean

	fit = bart(x[idx1,],y[idx1], x, ntree)
	yhat[,3] = fit$yhat.test.mean

	filename = paste( c(curr_dir,"/results/BART/",data_name,".csv"), collapse="" )
	cat("sample_id,y0,y1\n", file=filename)
	write.table(yhat, file = filename, sep=",", row.names = FALSE, col.names = FALSE , append = TRUE)
}