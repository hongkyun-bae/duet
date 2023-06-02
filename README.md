## Requirements
Tensorflow 1.9, PyTorch 1.0.0

## Rapid usage
### Training teacher & student models
1. Set ./main_config.json
 
 model_name: CDAE
2. Set ./model_confg/CDAE.json
 
 hidden_dim: 100 (teacher) or 10 (student)
 
 save_output: true (teacher) or false (student)
 
 NOTE: Teacher model is saved to ./data/"data_name"/"model_name".p
3. run main.py

### Knowledge distillation (teacher model required)
1. Set ./main_config.json
 model_name: CDAE_CD
2. Set ./model_confg/CDAE_CD.json
 teacher_dim: 100
 guide: teacher / student
 sampling_method: rank (CD-TG,SG) / random (CD-Base)
3. run main.py

## Hyper-parameter settings
Set ./model_confg/"model_name".json

## Envirionment of experiments settings
Set ./model_confg.json
ex) data
  dataset: "amazon"
