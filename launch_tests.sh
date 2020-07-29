#!/usr/bin/env bash
for (( i=1; i<=10; i++ ))
do
  ./run_test.sh $i $1
done