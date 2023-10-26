# arg-sum



# Download the models/datasets
## Sbert
gdown --folder 1-BW987IK_MzNOob1QVWWlZuWHjIsaXfB
## Debate Dataset
gdown 1-8F4b2FxHFrqM9exwELtrPPJXzJMz7Wx
## Bert Model
gdown --folder 10ZzaWaWe3Nd13kZU5ZvA5ua9d9X-IGiZ

unzip './reason-20230826T185023Z-001.zip' -d './'
## ArgKP Dataset
git clone https://github.com/IBM/KPA_2021_shared_task


git clone https://github.com/webis-de/argmining-21-keypoint-analysis-sharedtask-code


python -m spacy download en_core_web_sm

# Run the model
python exp.py
