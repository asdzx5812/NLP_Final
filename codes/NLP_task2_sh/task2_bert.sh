mkdir -p checkpoint_bert_task2

echo "Download task2_electra_model.pth from Dropbox"
wget https://www.dropbox.com/s/kxu5fgbvvq8mes4/bert_task2.pth?dl=1
mv ./bert_task2.pth?dl=1 ./checkpoint_bert_task2/bert_task2.pth

conda install --yes --file requirements.txt

python3 s2s_task2_bert.py $1


