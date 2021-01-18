


############
#
# Update gcloud
#
#
#gcloud components update
#


#
# Requirement: Install SDK
#
# Requirement: Create bucket in zone eu-west-4
#
# TPU instructions
# ----------------
#
# Local: Launch terminal
#
#
#
# COMPUTE INSTANCE VM: CREATE
#
# gcloud compute instances create tpu-driver-eur --machine-type=n1-standard-2 --image-project=ml-images --image-family=tf-1-9 --scopes=cloud-platform
# gcloud compute instances create anders-tf1-10 --machine-type=n1-standard-2 --image-project=ml-images --image-family=tf-1-10 --scopes=cloud-platform
#
#
# CREATE VM DISK
#
# gcloud compute disk-types list
# gcloud compute disks create tmpt2t --size 200 --type pd-ssd
# disks create tmpt2t --size 1500 --type pd-standard
#
# gcloud compute instances attach-disk atcompute --disk tmpt2t
#
# sudo lsblk
#
# sudo mkfs.ext4 -m 0 -F -E lazy_itable_init=0,lazy_journal_init=0,discard /dev/sdbXXX
#
# sudo mkdir -p /mnt/disks/tmp_mnt
#
# sudo mount -o discard,defaults /dev/sdb /mnt/disks/tmp_mnt
#
# sudo chmod 777 -R /mnt/disks/tmp_mnt
#
# DELETE VM DISK
#
# gcloud compute instances detach-disk tpu-driver-eur --disk=tmpt2t
# gcloud compute disks delete tmpt2t
#
#
#
#
# COMPUTE INSTANCE VM: SETUP
#
#
# https://www.datacamp.com/community/tutorials/google-cloud-data-science
#
#
#sudo apt-get update
#sudo apt-get install bzip2 git libxml2-dev
#
#wget https://repo.continuum.io/archive/Anaconda3-5.0.1-Linux-x86_64.sh
#
#bash Anaconda3-5.2.0-Linux-x86_64.sh
#rm Anaconda3-5.2.0-Linux-x86_64.sh
#
#
# pip install --upgrade pip
# pip install tensorflow==1.9
#
# pip install --upgrade google-api-python-client
# pip install --upgrade oauth2client
#
#pip install --upgrade cloud-tpu-profiler
#
# COMPUTE INSTANCE VM: LOGIN
#
# (LOCAL) gcloud compute ssh USERNAME@tpu-driver-eur
#
# TODO - check out integrated ssh on windows: https://www.thomasmaurer.ch/2017/11/install-ssh-on-windows-10-as-optional-feature/
#
# source .bashrc
# source myscript.sh
#
#
# COMPUTE INSTANCE VM: CONFIGURATION
#
# Create a script or run these commands on the command line.
#
#
# (VM):
#
# #!/bin/bash
# gcloud config set compute/region europe-west4
# gcloud config set compute/zone europe-west4-a
# export STORAGE_BUCKET=gs://anders_eu
# export TPU_NAME='preempt-1-9'
#
#
#
#
#
# CHECK ENVIRONMENT VARIABLE
#
# (VM) echo "$TPU_NAME"
# => tpu-anders-eur
#
#
# CREATE TPU
#
#
# gcloud compute tpus create preempt-1-10 --network=default --range=10.240.1.0/29 --version=1.10 --preemptible
# gcloud compute tpus create preempt-1-9 --network=default --range=10.240.1.8/29 --version=1.9 --preemptible
#
# note '8', ,'0' to coexist with neighbouring CIDR addresses.
#
#
#
#
# CHECK TPU STATUS
#
#(VM OR LOCAL) gcloud compute tpus list
#
#
# UPLOAD CODE (*this* very file, mnist.py, and convert_to_records.py)
#
#
# USERNAME often firstname_lastname
#
#(DESKTOP) gcloud compute scp .\mnist.py USERNAME@tpu-driver-eur:.
#(DESKTOP) gcloud compute scp .\convert_to_records.py USERNAME@tpu-driver-eur:.
#
#
# DOWNLOAD DATASET TO INSTANCEVM) python convert_to_records.py --directory=./data
# (VM) gunzip -d on all .gz files.
#
#
# MOVE DATASET FROM INSTANCE TO GS
#
# (VM) gsutil cp -r ./data ${STORAGE_BUCKET}
# (VM) rm -rf ./data/
#
#
# EXECUTE:
#
# (VM) python ./mnist.py --tpu=$TPU_NAME --data_dir=${STORAGE_BUCKET}/data --model_dir=${STORAGE_BUCKET}/output --use_tpu=True --iterations=500 --train_steps=9000
#
#
#
#
# CODE RE-TRANSFER DESKTOP -> VM:
#
# gcloud compute scp .\mnist.py USERNAME@tpu-driver-eur:.
#
# gcloud compute scp .\*.py anders_topper@atcompute:/mnt/disks/tmp_mnt/data

# ... and run again
#
#
# MOVE OUTPUT FROM GS TO VM
#
# mkdir output
# gsutil cp -r ${STORAGE_BUCKET}/output .
#


#DELETE OUTPUT IN BUCKET
# gsutil rm ${STORAGE_BUCKET}/output/*

#
# MOVE OUTPUT FROM VM TO LOCAL:
#
# (LOCAL):


# Create folder output
# gcloud compute scp --recurse anders_topper@tpu-driver-eur:./output/* ".\output\"
#
#
# STOP SYSTEM
#
# Log out from VM
#
# (LOCAL OR VM) gcloud compute tpus list
# (LOCAL OR VM) gcloud compute tpus stop tpu-anders-eur
#
#
#
# (LOCAL) gcloud compute instances list
# (LOCAL) gcloud compute instances stop tpu-driver-eur
# (LOCAL) gcloud compute instances start tpu-driver-eur
#
#
# RUN LOCAL
#
# Get data - note direct access local to GS possible:
#
# (LOCAL)gsutil cp -r gs://anders_eu/data .
#
# (LOCAL) python ./mnist.py --data_dir=./data --model_dir=./output --use_tpu=False --iterations=500 --train_steps=2006
#
#
#
# COLD START WITH EXISTING RESOURCES
#
# (RE)LAUNCH CLOUD VM AND TPU
# gcloud compute instances list
# gcloud compute tpus list
# gsutil ls
# gsutil ls gs://anders_eu/data
#
# gcloud compute instances start tpu-driver-eur
# gcloud compute tpus start tpu-anders-eur
#
# gcloud compute scp .\mnist.py USERNAME@tpu-driver-eur:.
#
# gcloud ssh...
# Run exports and config zone/region.
#
# gsutil rm gs://anders_eu/output/*
#
# TENSORBOARD
#
# point to output folder, same local and tpu.
#

#
#
# Note: You can either capture a profile or monitor your job; you cannot do both at the same time.
#
# CLOUD TPU TOOLS
# https://cloud.google.com/tpu/docs/cloud-tpu-tools
#
#
# MONITORING YOUR JOB


# gcloud compute ssh anders_topper@tpu-driver-eur --ssh-flag="-L 6006:localhost:6006"


