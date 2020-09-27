mkdir -p task2_checkpoint_electra

echo "Download task2_electra_model.pth from Dropbox"
wget https://www.dropbox.com/s/9j3r1fo39qv454q/electra_task2.pth?dl=1
mv ./electra_task2.pth?dl=1 ./task2_checkpoint_electra/electra_task2.pth

conda install --yes --file requirements.txt

python3 s2s_task2_electra.py $1
