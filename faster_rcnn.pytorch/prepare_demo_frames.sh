PYTHON=python3

# type 0

echo "convert video to frame"
$PYTHON video2frames.py 0


echo "move file"
mv demo_video/Moon_morning_2_1/* ../dataset/car_type_0/data/Images/
# mv demo_video/Moon_morning_2_2/* ../dataset/car_type_0/data/Images/
mv demo_video/Moon_morning_2_3/* ../dataset/car_type_0/data/Images/


echo "ImageSet file"
$PYTHON demo_imageset.py

mv demo_type_0.txt ../dataset/car_type_0/data/ImageSets/




# type 1
