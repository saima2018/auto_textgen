#!/bin/sh
NAME=auto_run_all
echo $NAME
ID=`ps -ef | grep "$NAME" | grep -v "$0" | grep -v "grep" | awk '{print $2}'`
echo $ID
for id in $ID
do
kill -9 $id
echo "killed $id"
done

NAME1=service_monitor_
echo $NAME1
ID=`ps -ef | grep "$NAME1" | grep -v "$0" | grep -v "grep" | awk '{print $2}'`
echo $ID
for id in $ID
do
kill -9 $id
echo "killed $id"
done


sh ./start/service_monitor_ts.sh & sh ./start/service_monitor_autogen_18092.sh & sh ./start/service_monitor_autogen_18093.sh & sh ./start/service_monitor_1092.sh
