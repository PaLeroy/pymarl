#!/usr/bin/env bash
for (( i=1; i<=10; i++ ))
do
  for (( j=1; j<=10; j++))
  do
  ./run_test_duo.sh $i $j $1
  done
done