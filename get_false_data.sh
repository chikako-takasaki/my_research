#!/bin/bash
# usage
# bash get_false_data.sh log_file row_num output_filename
# eg.
# bash get_false_data.sh keras/lstm/log/s10_v2/d2_50_2.txt 127 lstm_d2_50_falsedata.csv
FILENAME=$1
ROWNUM=$2
OUTFILE=$3
n=0
echo data_index, predict_class, correct_class >> ${OUTFILE}
for row in $(tail -n ${ROWNUM} ${FILENAME} | cut -f2,5,8 -d' ') 
do
 # echo ${row:0:4}
 if [ $(( $n % 3 )) -eq 0 ] ; then
  index=$row
  n=$((n+1))
 elif [ $(( $n % 3 )) -eq 1 ] ; then
  predict=$row
  n=$((n+1))
 else
  correct=$row
  n=0
  echo ${index%,}, ${predict%,}, ${correct} >> ${OUTFILE}
 fi
done
