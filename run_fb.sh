for i in 0.1
do
for j in 0.7 # 0.0 = original rgcn
do
for n in 2
do
    python3 main_fb.py --n-epochs 10000 --gpu 0 --regularization ${i} --a ${j} --n-layer ${n}
    echo "FB_${n}_layer" >> log/fb_layer_${n}_link_${i}
    echo ${i} ${j} ${k} ${w} >> log/fb_layer_${n}_link_${i}
    echo ${i} ${j} ${k} ${w}
done
done
done
