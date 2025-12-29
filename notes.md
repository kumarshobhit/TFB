once experiment is run, to get actual and predicted values and to visualize them : 
1. go to the test report results dir and unzip the tar file 
tar -xzvf TST.1766013506.uc2n566.localdomain.115181.csv.tar.gz

2. place python extract_results.py file in the current results directory
change the csv file name in the script. Then execute it. 

3. run the visualize_sample.py with location of sample_0 
python visualize_sample.py result/ETTh1/TST_dmodel512_sinespe_period24_27thdec/ETTh1_TST_batch_size32d_model512dim_feedforward2048dropout0.1horizon96lr0.0001n_heads8normtruenum_epochs100num_layers1patience10period24pos_encodingsinespeseq_len96/1766860899445/sample_0/