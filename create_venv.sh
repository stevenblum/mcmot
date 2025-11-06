# Create a new Python virtual environment named .venv2 and install only ultralytics, cvat-sdk, and cvat-cli
python3 --version
cd /home/scblum/Projects/testbed_cv
python3 -m venv-testbed-cv ./.venv-testbed-cv
source .venv/bin/activate
pip install --upgrade pip
pip install ultralytics supervision opencv-python


# Need a Seperate Virtul Environment for CVAT Automatic Annotation
# Because CVAT-CLI has a dependence on an older version of numpy
python3 --version
cd /home/scblum/Projects/testbed_cv
python3 -m venv ./.venv-cvat-aa
source .venv-cvat-aa/bin/activate
pip install --upgrade pip
pip install ultralytics cvat-sdk cvat-cli
