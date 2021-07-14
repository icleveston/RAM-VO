#!/bin/zsh

glimpses=(0 1 2 3 4 5 6 7 8)
train_seq=(0 2 4 5 6 8 9) 
val_seq=(10)
test_seq=(1 3 7) 

mkdir out/$1/results
mkdir out/$1/results/train
mkdir out/$1/results/val
mkdir out/$1/results/test

python tools/plot_loss.py --data_dir $1 --minibatch=false --save=true
python tools/plot_loss.py --data_dir $1 --minibatch=true --save=true

cd out/$1/
mv loss_epoch.pdf results/loss_epoch.pdf
mv loss_minibatch.pdf results/loss_minibatch.pdf
cd ../..
    
for x in $train_seq
do 
	python main.py --test $1 --dataset 'kitti' --test_seq $x
	python tools/plot_glimpse.py --dir $1/results/$x --epoch test
	
	for g in $glimpses
	do 
		python tools/plot_heatmap.py --dir $1/results/$x --glimpse $g --train false
	done
	
	python tools/gen_metrics.py --data_dir $1/results/$x
	
	mv out/$1/results/$x out/$1/results/train/$x

done

for x in $val_seq
do 
	python main.py --test $1 --dataset 'kitti' --test_seq $x
	python tools/plot_glimpse.py --dir $1/results/$x --epoch test
	
	for g in $glimpses
	do 
		python tools/plot_heatmap.py --dir $1/results/$x --glimpse $g --train false
	done
	
	python tools/gen_metrics.py --data_dir $1/results/$x
	
	mv out/$1/results/$x out/$1/results/val/$x

done
  
for x in $test_seq
do 
	python main.py --test $1 --dataset 'kitti' --test_seq $x
	python tools/plot_glimpse.py --dir $1/results/$x --epoch test
	
	for g in $glimpses
	do 
		python tools/plot_heatmap.py --dir $1/results/$x --glimpse $g --train false
	done
	
	python tools/gen_metrics.py --data_dir $1/results/$x
	
	mv out/$1/results/$x out/$1/results/test/$x

done
