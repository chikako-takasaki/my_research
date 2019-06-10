#nohup sh -c 'python wb_gridsearch.py > ./log/wb_gridsearch2.txt ; python rw_gridsearch.py > ./log/rw_gridsearch2.txt ; python all_gridsearch.py > ./log/all_gridsearch.txt' &
nohup sh -c 'python all_gridsearch.py > ./log/all_gridsearch.txt ; python openpose_wb.py > ./log/wb4_2000.txt ; python openpose_rw.py > ./log/rw5_2000.txt' &
