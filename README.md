
# arg-sum

The implementation for  "Enhancing Argument Summarization via Key Point Clustering: Prioritizing Exhaustiveness and Introducing an Automatic Coverage Evaluation Metric"


## Using our model

### Setting up the environment
* __Python version:__ `python3.6`

* __Dependencies:__ Use the `requirements.txt` file and conda/pip to install all necessary dependencies. E.g., for pip:

		pip install -U pip
		pip install -r requirements.txt 

* __SBERT Model:__ Download the fine-tuned SBERT Model 

		gdown --folder 1-BW987IK_MzNOob1QVWWlZuWHjIsaXfB

* __BERT Model:__ Download the fine-tuned BERT Model 

		gdown --folder 10ZzaWaWe3Nd13kZU5ZvA5ua9d9X-IGiZ
		
* __Debate Dataset:__ Download and unzip the Debate Dataset 

		gdown 1-8F4b2FxHFrqM9exwELtrPPJXzJMz7Wx
		unzip './reason-20230826T185023Z-001.zip' -d './'
		

* __ArgKP Dataset:__ Clone the ArgKP dataset from repository 

		git clone https://github.com/IBM/KPA_2021_shared_task
		git clone https://github.com/webis-de/argmining-21-keypoint-analysis-sharedtask-code



### Generating Summaries
To perform summarization on ArgKP dataset and Debate dataset run the following:

		python exp.py

### Evaluation Metric
To perform evaluation on a summary of ArgKP test set add the summary as a list containing summary of each topic and run the notebook.

# Input and Output Samples
The output of the model given arguments on the topic of child vaccination. 
# Input
The full list of arguments:

[CSV File](https://github.com/IBM/KPA_2021_shared_task/blob/main/test_data/arguments_test.csv)
# Output
- 'Prevents a large number of diseases',
- 'this vaccine could cause unwanted side effects',
- 'Parents should decide what is best for their child.',
- 'Child vaccination should be mandatory to avoid the virus',
- "Child vaccination shouldn't be mandatory because children don't catch the virus",
- 'protecting infants must be a priority for all',
- 'to keep schools safe children must be vaccinated',
- 'because they can have very dangerous reactions to vaccines',
- 'the vaccine provide immunity to  the people  and prevents to contract the disease'
