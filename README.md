# Dually Guided Knowledge Distillation (DUET)
![FFF](https://user-images.githubusercontent.com/56680371/143531372-beadf494-1c59-4d06-b5d1-b0a0ec639059.PNG)


## Environments
Our framework including the CF models was implemented with Tensorflow 2.6.
All Experiments were conducted on desktops with 64 GB memory, Intel i9-10900K CPU(3.75 GHz,20M cache), and Nvidia GeForce RTX 3080.
</br>

## Quick usage
### Training teacher \#2
1. Set ./main_config.json<br/>
    model_name: CDAE<br/>
    <br/>
2. Set ./model_config/CDAE.json  <br/>
    hidden_dim: 100<br/>
    save_output: true<br/>
   <br/>
3. Run main.py<br/>
    <br/>
4. Teacher model is saved to ./data/"data_name"/"model_name".p<br/>
</br></br>
### Knowledge Distillation (Teacher model training required)
1. Set ./main_config.json<br/>
    model_name: CDAE_DGKD<br/>
    <br/>
2. Set ./model_config/CDAE_DGKD<br/>
    teacher_dim: 100<br/>
    <br/>
3. Set ./model/CDAE_DGKD.py<br/>
    self.unintFilePath: path for uninteresting file (file containing the result of teacher \#1)<br/>
    self.intFilePath: path for interesting file (file containing the result of teacher \#1)<br/>
    <br/>
4. Run main.py<br/>
</br></br>

## Hyper-parameter settings
Set ./model_config/"model_name".json<br/>
</br>

## Experiment Environment Settings
Set ./model_config.json<br/>
    ex) data --> dataset: "ml1m"<br/>

