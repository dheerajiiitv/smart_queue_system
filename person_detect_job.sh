model=$1
device=$2 
input_file=$3 
extensions=$4 

python3 person_detect.py --model ${model} --extensions ${extensions} --visualise --video ${input_file}