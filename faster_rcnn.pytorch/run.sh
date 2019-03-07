./prepare_demo_frames.sh
CUDA_VISIBLE_DEVICES=$1 python3 test_net.py --dataset car_type_0 --net vgg16 --checksession 1 --checkepoch 50 --checkpoint 150 --cuda
