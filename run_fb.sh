# Inference Mode on: INFER=1
TRAIN=1
EVAL=1
INFER=0
for i in 0.05
do
for j in 0.7
do
for n in 2
do
    python3 main_fb.py --n-epochs 15000 --gpu 0 --regularization ${i} --a ${j} --nlayer ${n} --infer $INFER --tran $TRAIN --eval $EVAL
    echo "FB_${n}_layer" >> log/fb_layer_${n}_mu_${j}_reg_${i}
    echo ${i} ${j} ${k} ${w} >> log/fb_layer_${n}_mu_${j}_reg_${i}
    echo ${i} ${j} ${k} ${w}
done
done
done
