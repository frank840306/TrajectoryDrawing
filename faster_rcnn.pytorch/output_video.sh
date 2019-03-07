PYTHON=python3.6


# type 0

$PYTHON nms.py ../dataset/car_type_0/results/demo_type_0/comp4_det_demo_type_0_bicycle.txt
$PYTHON nms.py ../dataset/car_type_0/results/demo_type_0/comp4_det_demo_type_0_pedestrian.txt
$PYTHON visualize_result.py -i ../dataset/car_type_0/data/Images/ -a ../dataset/car_type_0/results/demo_type_0/ -o ../dataset/car_type_0/results/demo_type_0/result_img/
$PYTHON frames2video.py ../dataset/car_type_0/results/demo_type_0/result_img/ ./demo_video
# type 1

# $PYTHON nms.py ../dataset/car_type_1/results/demo_type_1/comp4_det_demo_type_1_bicycle.txt
# $PYTHON nms.py ../dataset/car_type_1/results/demo_type_1/comp4_det_demo_type_1_pedestrian.txt
# $PYTHON visualize_result.py -i ../dataset/car_type_1/data/Images/ -a ../dataset/car_type_1/results/demo_type_1/ -o ../dataset/car_type_1/results/demo_type_1/result_img/
# $PYTHON frames2video.py ../dataset/car_type_1/results/demo_type_1/result_img/ /tmp2/frank840306/demo_video
