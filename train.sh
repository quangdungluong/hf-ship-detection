#!bin/bash
echo "Start Training Model"
eval "$(conda shell.bash hook)"
conda activate ship
cd yolov5
cat > training.yaml <<EOF
train: ../train_data/
val: ../val_data/

nc: 1
names: ['ship']
EOF
python train.py --img 2048 --batch 4 --epochs 15 --data training.yaml --weights yolov5s.pt --name exp1
echo "Training model done!!!"