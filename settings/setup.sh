
echo "Installing PythonVEnv"
sudo apt install -y python3-venv 

echo "Setting USENV"
python3.6 -m venv usenv

printf "\nexport UNDERSAMPLINGWORKDIR=`dirname $PWD`" >> usenv/bin/activate

source usenv/bin/activate

echo "UNDERSAMPLINGWORKDIR = ${UNDERSAMPLINGWORKDIR}"

pip install --upgrade pip wheel setuptools

pip install -r requirements.txt

deactivate


