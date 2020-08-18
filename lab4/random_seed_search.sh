#!/bin/bash

range="${range:="1000"}"
file="${file:="0.86_0.02.pt"}"
size="${size:="512"}"

for i in $(seq 0 $range); do python test.py --wts_path ckpt/"$file" --hidden_size "$size" --seed $i | grep "Gaussian" | awk -v var="$i" '{if ($NF>0) print var, $NF}'; done
