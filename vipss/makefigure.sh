
# replace the out_dir with your own output dir
# create a output dir, default is in data/output

# Figure 11 
./nnvipss -i ../../data/points/torus_n50.xyz -o ../../data/output/torus_n50.ply
./nnvipss -i ../../data/points/torus_halfsample.xyz -o ../../data/output/torus_halfsample.ply 
./nnvipss -i ../../data/points/bathtub_1k.xyz -o ../../data/output/bathtub_1k.ply 
./nnvipss -i ../../data/points/chair36.xyz -o ../../data/output/chair36.ply 

# Figure 12 
./nnvipss -i ../../data/points/torus_wires.xyz -o ../../data/output/torus_wires.ply
./nnvipss -i ../../data/points/doghead.xyz -o ../../data/output/doghead.ply
./nnvipss -i ../../data/points/hand_ok.xyz -o ../../data/output/hand_ok.ply
./nnvipss -i ../../data/points/walrus.xyz -o ../../data/output/walrus.ply -l 0.0005 --max_iter 1000

# Figure 13 
./nnvipss -i ../../data/points/Helmet.xyz -o ../../data/output/Helmet.ply
./nnvipss -i ../../data/points/brain_Ih.xyz -o ../../data/output/brain_Ih.ply
./nnvipss -i ../../data/points/Mobius.xyz -o ../../data/output/Mobius.ply

# Figure 14
./nnvipss -i ../../data/points/lord_quas.xyz -o ../../data/output/lord_quas.ply -l 0.0002 
./nnvipss -i ../../data/points/anchor.xyz -o ../../data/output/anchor.ply -l 0.001 

# Figure 1
./nnvipss -i ../../data/points/helmetMoustache.xyz -o ../../data/output/helmetMoustache.ply