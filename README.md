# TinyML Offloading and Split Computing

### Contributors: 
- **Fabio Bove:** [fabio.bove.dr@gmail.com]
- **Luca Bedogni** []

# Components Description

## 1. Offloading Tools `[offloading_tools]`
   - Contains the classes of the object involved in the offloading process, such as:
     - **`OffloadingMessage`**
       - Is a message exchanged between the Edge and a Device
       - Features methods to evaluate latency and speed, size and other parameter needed for the Algo computation
     - **`OffloadingDevice`** and **`OffloadingDevicesManager`**:
       - Features different methods to handle connected Devices and track the exchanged messages
     - **`OffloadingCommunicationHandler`**:
       - Takes care of all the previous components: Messages, Devices and the DeviceManager
     - **`OffloadingAlgo`**:
       - It's a class that implements the algorithm used to perform the Offloading operation
       - Evaluates based on 3 possible scenario for the computation: Edge Only, Device Only and Mixed.
       - At the end of the computation return the information about the best offloading layer for a given model, network state and capabilities of the devices involved in the prediction.
     - **`OffloadingManager`**
       - Merges all the object seen until now to expose a simple interface for the Offloading process.
     - **`OffloadingModel`**
       - Manages the inference and evaluation of times and size of each layer, is designed to work with dynamic custom models
       
## 2. Offloading Examples `[offloading-examples]`
   - Contains a class and a module that can be used to simulate all the offloading operations of a fully working environment
     - **`SimpleDevice`**:
       - Allow to send a message, that can be customized, to the edge and logs the response.
     - **`inference_example`**
       - A module that shows how a inference can be performed in a mixed way, both on edge and device
     - **`initialization_example`**
       - A module that shows how the initialization of a model can be done, to evaluate its single layers 
     - **`offloading_evaluation_example`**
       - A module that shows the evaluation process

## 3. Custom Models `[custom_models]`
   - Contains a class that facilitate the usage of neural network models stored locally.
     - **`custom_model`**:
       - It's a class that manages all the aspects related to the model used for computation adn training of a Custom Model.
         _Note: Please make sure that your custom model class includes at least all the methods and attributes used in the given example file._
     - **`./models`**
       - Folder that contains the custom models that can be used in the offloading process.
         Each model needs to have its own folder with the following structure:
         -  `./data`: dataset for the training if needed
         -  `analytics.json`: A file containing the inference time and size of each layer of the model, if not present can be generated using the "model-initialization" api
         -  `<model-name>.keras`: The file of the model
     - **`create_train_model`**
       - A module that shows an example of training and creation of the custom model
     - **model_data**
       - A class that is used as an example, it handles the data manipulation for the given custom model in example.
       - 
## 4. Flask Server `[flask_server]`
- Contains the component needed to expose a web application layer to handle the whole Offloading process via RESTful APIs
- **Available APIs**:
  - `api/offloading/model-initialization`: Initialize the model analytics, for inference times (edge-side)
  - `api/offloading/communication-status`: Shows the status of the communication between devices and the edge
  - `api-offloading/evaluate`: Perform the offloading algorithm for a given model
  - `api/offloading/model-inference`: Perform inference of a given model for the chosen layers

# Getting Started

## Run locally
1. (create and activate a virtual environment)
2. `pip install -e .`
3. (from the root folder)  `python ./src/app.py`

## Run as docker container
1. run the docker-compose file with: `docker-compose up -d --build`

## Run tests
1. (from the src folder: `cd ./src`) `pytest ../`



# Features Under Development:
 ## 1. Neural Network Models:
- Tools to create a simple model for binary classification on images to test the offloading and split computing workflow:
  ~~- **ModelData** (class) `model_data`~~
  ~~- **create_train_model** (module) `create_train_model`~~
  ~~- **ModelManager** (class) `model_manager`: Refactor and Update to support a real prediction~~
  
# Future Improvements:
 - Add examples of different types of models (tensorflow, pythorch, ecc...)