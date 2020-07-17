# Running Dockerised PySyft Worker Node

1) Clone this repository

    > <font color='turquoise'>**git clone https://gitlab.int.aisingapore.org/aims/federatedlearning/pysyft_worker.git**</font>

2) Navigate into the repository

    > <font color='turquoise'>**cd ./pysyft_worker**</font>

3) Build image using the following command(s): 

    > <font color='turquoise'>**docker build -t worker:pysyft_demo --label "WebsocketServerWorker" .**</font>

4) Set up the appropriate mountpoint directories.

    A. Datasets

    > <font color='turquoise'>**/datasets <br>&ensp;&ensp;&ensp;&ensp; /tabular <br>&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp; metadata.json <br>&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp; tabular_data.csv <br>&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp; schema.json <br>&ensp;&ensp;&ensp;&ensp; /image <br>&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp; metadata.json <br>&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp; /label_class_0 <br>&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp; image_1.png <br>&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp; image_2.png <br>&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp; image_3.png <br>&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp; /label_class_1 <br>&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp; image_4.png <br>&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp; image_5.png <br>&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp; image_6.png <br>&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp; /label_class_2 <br>&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp; image_7.png <br>&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp; image_8.png <br>&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp; image_9.png <br> &ensp;&ensp;&ensp;&ensp; /text <br>&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp; metadata.json <br>&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp; text_corpus.csv**</font>

    B. Models (Used for SNN only)
    
    > <font color='turquoise'>**/models <br>&ensp;&ensp;&ensp;&ensp; /model_1 <br>&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp; structure.json <br>&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp; hyperparameters.json <br>&ensp;&ensp;&ensp;&ensp; /model_2 <br>&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp; structure.json <br>&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp; hyperparameters.json <br>&ensp;&ensp;&ensp;&ensp; /model_3 <br>&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp; structure.json <br>&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp; hyperparameters.json**</font>

    C. Customised Dependencies
    
    > <font color='turquoise'>**/custom <br>&ensp;&ensp;&ensp;&ensp; /nltk_data <br>&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp; /chunkers <br>&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp; < insert chunker resources here > <br>&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp; /corpora <br>&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp; stopwords.zip <br>&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp; wordnet.zip <br>&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp; /grammars <br>&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp; < insert grammar resources here ><br>&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp; /help <br>&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp; < insert help resources here > <br>&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp; /misc <br>&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp; < insert miscellaneous resources here > <br>&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp; /models <br>&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp; < insert model resources here > <br>&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp; /sentiment <br>&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp; < insert sentiment resources here > <br>&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp; /stemmers <br>&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp; < insert stemmer resources here > <br>&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp; /taggers <br>&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp; < insert tagger resources here > <br>&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp; /tokenizers <br>&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp; punkt.zip <br>&ensp;&ensp;&ensp;&ensp; /spacy <br>&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp; en_core_web_sm-X.X.X.tar.gz <br>&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp; en_core_web_md-X.X.X.tar.gz <br>&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp; en_core_web_lg-X.X.X.tar.gz <br>&ensp;&ensp;&ensp;&ensp; /symspell <br>&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp; XXX_dictionary_XXX.txt <br>&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp; XXX_bigram_XXX.txt**</font>

    D. Outputs
    
    > <font color='turquoise'>**outputs <br>&ensp;&ensp;&ensp;&ensp;< no structural requirements >**</font>

5) Start up the worker node. Start-up commands differ depending on whether or not you are running the REST-RPC grid in standalone mode, or over a distributed network, and are as follows: 

    A. Standalone Grid (i.e. local)

    > <font color='turquoise'>**docker run <br> -v /path/to/datasets:/worker/data <br> -v /path/to/customised/dependencies:/worker/custom <br> -v /path/to/outputs:/worker/outputs <br> --name <worker_id> worker:pysyft_demo**</font>
    


    B. Distributed Servers (i.e. across multiple devices)

    * host - IP of host machine
    * port - Selected port to route incoming connections into the container
    * id - Participant ID given to machine
    * train - Tag(s) of datasets to be used for model training
    * evaluate - Tag(s) of datasets to be used for evaluation

    **Explanation:**
    
    By default, all PySyft containers, be it TTP or worker, will be set up to run on internal port `8020`.

    Here, we are setting docker up to route any incoming connections/requests on a specified port of the docker host to the internal port `8020`. The container then takes in script variables to log the ID of the machine, as well as which training & validation datasets to use.

6) Connect to bridge network to allow communication with TTP (Required if TTP is dockerised as well - in progress)