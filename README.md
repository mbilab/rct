mkdir rct
cd rct/
git init
echo keras > .venv
sudo apt-get install python3-venv
python3 -m venv ~/.virtualenvs/keras
pip install numpy
pip install --upgrade pip
pip install pandas
pip install sklearn
pip install scipy
pip install keras
pip install tensorflow
pip install pyyaml
pip install gensim
pip install nltk
pip freeze
