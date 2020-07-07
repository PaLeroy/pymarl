#!/usr/bin/env bash
for (( i=3; i<8; i++ ))
do
  for (( j=3; j<= 8; j++))
  do
  ./run_test_duo.sh $i $j
  done
done