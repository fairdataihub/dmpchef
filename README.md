## setup

```bash
git clone https://github.com/fairdataihub/dmpchef.git
cd dmpchef
code .

conda create -n dmpchef python=3.10 -y
conda activate dmpchef

python -m pip install --upgrade pip
pip install -r requirements.txt

python setup.py install
# or (recommended for development)
pip install -e .

uvicorn app:app --reload

Open this link in your browser: http://127.0.0.1:8000/ and test app!

