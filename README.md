# compiled data
stored in ~dirty/rct/tmp/
{tr,te,tte}.dsc-4.pkl # dummy sequence, 1 integer for one word
  * tr: training (3321)
  * te: test
  * tte: true test (336)
  * dsc: dummy sequence with tficf
  * -4: words with tficf less than 1e-4 are removed

# todo
* target variation: only number

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
pip install h5py
pip install xgboost
pip freeze
