# TinyML Offloading and Split Computing

### Contributors: 
- **Fabio Bove:** [fabio.bove.dr@gmail.com]
- **Luca Bedogni** []
- **Simone Colli** []

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

## 2. Simple Device `[simple-device]`
   - Contains a class and a module that can be used to simulate a device that sends a message to the edge, to perform offloading:
     - **SimpleDevice**:
       - Allow to send a message, that can be customized, to the edge and logs the response.

## 3. NN Model `[nn_model]`
   - Contains a class that facilitate the usage of neural network models stored locally.
     - **ModelManager**:
       - It's a class that manages all the aspects related to the model used for computation, such as: Loading model, Evaluating and Updating inference times and keeping track of its structure (layers number and size).

## 4. Flask Server `[flask_server]`
- Contains the component needed to expose a web application layer to handle the whole Offloading process via RESTful APIs


# Getting Started

## Run locally
1. (create and activate a virtual environment)
2. `pip install -e .`
3. (from the root folder)  `python ./src/app.py`

## Run as docker container
1. run the docker-compose file with: `docker-compose up -d --build`

## Run tests
1. (from the src folder: `cd ./src`) `pytest ../`

