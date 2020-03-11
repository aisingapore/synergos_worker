# Running Dockerised PySyft Worker Node
1) Clone this repository
2) Navigate into the repository
    > `cd /path/to/worker`
3) Build image using the following command(s): 
    > `docker build -t worker:pysyft_demo --label "WebsocketServerWorker" .`
4) Start up the worker node using the following command(s):
    >`docker run -p <host>:<port>:8020 worker:pysyft_demo -H 0.0.0.0 -p 8020 -i <id> -t <train> -e <evaluate> -v`
    
    * host - IP of host machine
    * port - Selected port to route incoming connections into the container
    * id - Participant ID given to machine
    * train - Tag(s) of datasets to be used for model training
    * evaluate - Tag(s) of datasets to be used for evaluation

    **Explanation:**
    
    By default, all PySyft containers, be it TTP or worker, will be set up to run on internal port `8020`.

    Here, we are setting docker up to route any incoming connections/requests on a specified port of the docker host to the internal port `8020`. The container then takes in script variables to log the ID of the machine, as well as which training & validation datasets to use.

5) Connect to bridge network to allow communication with TTP (Required if TTP is dockerised as well - in progress)