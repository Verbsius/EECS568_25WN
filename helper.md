only apply for cpu:
salloc --account eecs568s001w25_class --partition standard \
--nodes 1 --ntasks 1 --cpus-per-task 4 --mem 16G --time 01:00:00

apply for GPU for training:
salloc --account eecs568s001w25_class --partition gpu_mig40 \
--nodes 1 --ntasks 1 --cpus-per-task 1 --gpus 1 --mem 16G --time 00:30:00

eecs556w25_class_root

watch -n 1 nvidia-smi

du -sh .

v
0 ok 
1 ok 
2 ok
3 ok
4 ok
5 ok
6 ok
7 ok
8 ok
9 ok 
10 ok

 退出当前屏幕： Ctrl + A + D 
 重新链接：screen -r <window name>