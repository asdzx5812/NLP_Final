mkdir -p checkpoint_electra

echo "Download task1_model.pth from Dropbox"
wget https://www.dropbox.com/s/sv0boti04i61ttc/electra_task1.pth?dl=1
mv ./electra_task1.pth?dl=1 ./checkpoint_electra/electra_task1.pth

conda install --yes --file requirements.txt

python3 s2s_v203_electra.py $1
