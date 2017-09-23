# compiled data
stored in ~dirty/rct/tmp/
{tr,te,tte}.ts.pkl # tfidf-sequential, 1D for one word
{tr,te,tte}.c.ts.pkl # tficf-sequential, 10D for one word
  * 'icf' indicates inverse class frequency
  * 9D for nine class tfidf; 1D for overall tfidf

# env
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
pip install datetime
pip freeze
