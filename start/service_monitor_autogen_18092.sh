#!/bin/sh
while :
do
NAME1="18092"
NAME2="autogen"

echo $NAME1
ID=`ps -ef | grep "$NAME1" |grep "$NAME2" | grep -v "$0" | grep -v "grep" | awk '{print $2}'`
echo $ID
if [ "$ID" ]; then
for id in $ID
do
kill -9 $id
echo "killed $id"
done
else
echo "starting 18092 service..."
python ./autogen.py --host '0.0.0.0' --port '18092' --gpu '0'
fi
sleep 5
done
