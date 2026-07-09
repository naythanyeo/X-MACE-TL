# X-MACE TL Implementation Plan

## 1. Overview Classes
Trying to implement a few more helper classes to make it more convenient for usage, as well as to support transfer learning. Many overlapping features so we don't actually need to re code everything for each strategy, can group into subclasses and mix and match later on. 
The key targets is to be able to 
1) use the classes easily with minimal API complexity 
2) support different TL strategies such as freezing, LORA and multiheaded training 

The following are the planned classes and what they are used for


### 1.1 Trainer
Define a main class called trainer, this class only helps in coordinating the training loop. Ie it will fix the epoch and any strategies for early stopping. This trainer is a legit generic and "dumb" trainer. All the various model architectures etc is handelled by the other classes, intentionally made dumb so it can be general.
Initialised with: 
    - Max Epoch
    - Early stopping (bool), optional
    - Patience (int), required if early stopping is true 
    - Verbose(?)
Training step must input 
    - Training data 
    - Validation data
    - Model 
    - Optimiser 
    - Loss function 
Returns 
    - Trained model 
    - History (?) --> dictionary with all the loss data etc

* Should be able to handle the early stopping / checkpoint to select weights of the minimum validation loss etc. This trainer is mainly for this feature only. 

* Possible API: 
``` python
    base_trainer = mace_trainer(max_epoch=50, 
                                early=True,
                                patience=10)
    trained_model, history = base_trainer.train(train_data, 
                                                valid_data, 
                                                base_model,
                                                optimiser,
                                                loss_fn)
```
### 1.2 Strategies
This strategies class should be a class that mainly takes in a trained model, outputs a new model with suitable weights adjusted. Possible for subclassing? Or maybe not 
But roughly from simpler implementation to harder 
    - Naive Strategy --> No need to implement anything (model is retrained totally)
    - Freezing --> The wrapper will take in the model, set certain layers to be requires grad = False 
    - LORA --> Wrapper takes in model, freezes the weights, adds on new low rank weight matrices, modifies forward function to include the LORA layer etc 
    - Multiheaded training --> need to modify the forward function to change the path based on the dataloader head labels. Later stage implementation 

* Strategies takes in model class, outputs modified model class
* Probably cleaner to be written as a class object so that the individual paramters of eg what layers to freeze or what rank matrix to use can be initialised as base strategy 

* Possible API:
```python
    strat_1 = LORA_strategy(rank=8, 
                            freeze=None, 
                            flexible=["Graph"])
    modified_model = strat_1.apply(trained_model)
    # Then later use the modified model for TL 
    re_trained_model, history = base_trainer.train(train_data, 
                                                   valid_data, 
                                                   modified_model,
                                                   optimiser,
                                                   loss_fn)
```

### 1.3 Multihead Structure 
Multihead training strategy probably implemented the last. Possible to stick with the previous api architecture, but will need to define two main new things?
    - Dataloader 
        - Must have a new specialised one for multiheaded training 
        - Include the label "head" as part of the class
        - "head" will specify eg casscf or caspt2 
    - Outputs
        - Have to save an outputs "heads" as a vector shape [n_graphs, 1]
        - Meant to indicate and label each entry output 
    - Loss fn
        - Take in the ouputs "heads" as a vector also
        - Should assign training weights to the loss function 
        - Eg transfer to CASPT2, higher weightage maybe compare to CASSCF data 

* We need slightly new architecture because the decoder head is probably different for both classes of data, but the encoder or GNN could be trained together (same backbone). At least I think this is what MACE did, joint model for GNN -> latent space use different heads. 
* We can try to look into the latent space for both CASSCF and CASPT2. This is probably the most interesting or next goal, see if it makes more sense to train the head from latent space onwards or before. 
* TBC but maybe we can just implement this as part of the original class? If it doesn't result in too much additional inconveniences, if not can define multihead specific classes for these


## 2. Helper functions 
A few helper functions that are probably good to implement? Just for convenience 

### 2.1 Model caller 
Helper function to load in a default autoencoder model 
Can specify the parameters to change, but if not it will by default be the original paramters that X-MACE uses
Probably useful since for base training this is something that likely won't change? 

### 2.2 Encoder loss
Currently this is very annoying to be defined from user API side 
Probably under autoencoder loss as a separate function to add in the autoencoder to outputs? 
Then under the trainer class can just call this function easily 

### 2.3 Model inspection 
Likely might sample CASSCF from geometric descriptors / latent space / energies? 
Maybe good to have a few helper functions to evaluate the model to output or get descriptors? Eg input the trained model, it will output the autoencoder energies or latent space vector. 
Currently implemented in 04_pca, but maybe can have helpers to include this in source


## 3. Plan 
### 3.1 Priority Updates
1. Draft skeleton first, all the new files and rough comments for the skeleton logic
2. Add in the base trainer first so API is easy to use 
3. Maybe add in a test class also to evaluate on test set?? 
4. Figure out where to include things like E0 basely and z_table into the source
5. Add on basic freezing strategy 
6. LORA, copy the MACE tutorials to see how they do it, check the relevant o3 libraries or layers used 
7. Re plan and skeletal for multi headed transfer learning 

### 3.2 TBC future implementation?? 
1. Optimiser configuration?? tune the learning rate for different layers?? Check what was reported or if this was implemented 
