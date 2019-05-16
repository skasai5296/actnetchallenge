nohup scp -r -P 10022 /ssd2/dsets/valtestdata.tar.gz aab10821no@localhost:~/datasets/ > valtestscp.out 2>&1 &
nohup scp -r -P 10022 /ssd1/dsets/traindata.tar.gz aab10821no@localhost:~/datasets/ > trainscp.out 2>&1 &
