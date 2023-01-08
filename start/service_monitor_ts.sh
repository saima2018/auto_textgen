#!/bin/sh
while :
do
NAME=TaskSchedule.py
echo $NAME
ID=`ps -ef | grep "$NAME" | grep -v "$0" | grep -v "grep" | awk '{print $2}'`
echo $ID
if [ "$ID" ]; then
for id in $ID
do
kill -9 $id
echo "killed $id"
done
else
echo "starting taskschedule service..."
python schedule/TaskSchedule.py
fi
sleep 5
done
