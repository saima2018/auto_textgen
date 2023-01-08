#!/bin/sh
while :
do
NAME=start_service_1092.py
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
echo "starting 1092 service..."
python start_service_1092.py
fi
sleep 5
done
