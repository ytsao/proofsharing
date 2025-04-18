#!/bin/bash

echo ""
echo "Computing results for Table 6"
echo ""

mkdir -p results
for dataset in mnist cifar
do
	for rep in "1" "2" "3"
	do
		for net in "7x200_best.pth" "9x500_best.pth"
		do
			method=base
			layer=base
			fn=results/patches_${dataset}_${net//_best.pth/}_${layer// /+}_${method//_/}_${rep}.txt
			if test -f "$fn"; then
				echo "$fn exists; skipping."
			else
				python . -p --netname ${net} --dataset ${dataset} --num_tests 100 --relu_transformer zonotope --patch_size 2 |& tee "$fn"
			fi

			layer="1 2"
			method="l_infinity"
			fn=results/patches_${dataset}_${net//_best.pth/}_${layer// /+}_${method//_/}_${rep}.txt
			if test -f "$fn"; then
				echo "$fn exists; skipping."
			else
				python . -p --netname ${net} --dataset ${dataset} --num_tests 100 --relu_transformer zonotope --patch_size 2 --template_method ${method} --template_domain box --template_layers ${layer} |& tee "$fn"
			fi
		done
	done
done

python scripts/summarize_results.py --table 6 | tee results/table6.txt


