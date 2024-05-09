# 运行第一个 Python 程序，并将其放入后台
nohup python Real-time_detect/real_time_pub.py > output1.log &
sleep 0.5
# 运行第二个 Python 程序，并将其放入后台
nohup python Real-time_detect/real_time_sub.py > output2.log &