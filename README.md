# Environments
1. Python 3.6
2. Pytorch 0.4.0
3. $HOME=TrajectoryDrawing


# Preparation
1. install python3.6
2. install pytorch 0.4.0
	1. download whl file "cu90/torch-0.4.0-cp36-cp36m-linux_x86_64.whl" from https://pytorch.org/get-started/previous-versions/
	2. pip3 install torch_XXX.whl
3. cd ${HOME}/faster_rcnn.pytorch
4. pip3 install requirements.txt

# Directory hierarchy
TrajectoryDrawing/  
    faster_rcnn.pytorch/  
        demo_video/  
            Moon_morning_2_1.mp4  
            Moon_morning_2_3.mp4  
        data/  
            pretrained_model/  
                vgg16_caffe.pth (pretrained vgg16 for training and testing faster-RCNN backend)  
        models/  
            vgg16/  
                car_type_0/  
                    faster_rcnn_1_50_150.pth (trained faster-RCNN model, used for predicting)  
        prepare_demo_frames.sh (將影片分割成frame，然後放進適當的資料夾)  
        test_net.py (predict每個frame，預測bbox)  
        output_video.sh (產生有bbox的影片到./demo_video中)  
        visualize_movement.py (產生trajectory的影片到./demo_video中)  
    dataset/  
        car_type_0/  
            results/  
                demo_type_0/  
            data/  
                Images/  
                    (prepare_demo_frames.sh output的frame們都會放在這)  
                ImageSets/  
                    (prepare_demo_frames.sh 產生的demo_type_0.txt會放在這，文件包含所有要被test的frame)  
				

# Dataset type
1. type 0: 俯視
2. type 1: 測俯視




# Execution
1. type 0  
	# model name: models/vgg16/car_type_0/faster_rcnn_1_50_150.pth  
	./prepare_demo_frames.sh  
	CUDA_VISIBLE_DEVICES=2 python3 test_net.py --dataset car_type_0 --net vgg16 --checksession 1 --checkepoch 50 --checkpoint 150 --cuda  
	./output_video.sh  
	python3 visualize_movement.py --video 1 --start 600 --end 1050 --object pedestrian --object_idx 26 --output_dir ./demo_video   
	python3 visualize_movement.py --video 2 --start 960 --end 1080 --object bicycle --object_idx 0 --output_dir ./demo_video   


2. type 1  
	# model name: models/vgg16/car_type_1/faster_rcnn_1_50_128.pth   

	CUDA_VISIBLE_DEVICES=2 python3 test_net.py --dataset car_type_1 --net vgg16 --checksession 1 --checkepoch 50 --checkpoint 128 --cuda  

