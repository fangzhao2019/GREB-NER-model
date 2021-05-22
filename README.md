# GREB-NER-model

1.Generate experiment data
python Candidate Entities Graph-ranking Model/dataProcess/raw_data_to_experimental_data.py

2.Generate candidate entities and it's global dependencies for each sample
python Candidate Entities Graph-ranking Model/candidateEntityDataSetCreate.py

3.Copy the data to the folder “Joint Embedding Deep Learning Model\data”

4.Train the model
python Joint Embedding Deep Learning Model\main.py
