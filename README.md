# Running Dockerised Synergos Worker

1) **Clone this repository**
    ```bash
    git clone https://github.com/aimakerspace/synergos_worker.git
    ```

2) **Navigate into the repository**
    ```bash
    cd ./synergos_worker
    ```
 
3) **Checkout to stable tag**
    ```bash
    git checkout tags/v0.1.0
    ```
     
3) **Update Submodule**
    ```bash
    git submodule update --init --recursive
    git submodule update --recursive --remote
    ```

3) **Build image using the following command(s):** 
    ```bash
    docker build -t synergos_worker:v0.1.0 --label "WebsocketServerWorker" .
    ```

4) **Set up the appropriate mountpoint directories.**

    Please view [this guide](https://docs.synergos.ai/DatasetStructure.html) on how to organise your dataset for federated training, in Synergos Worker.

5) Start up the worker node. Start-up commands can be reduced depending on whether or not you are running the REST-RPC grid in standalone mode, or over a distributed network. In general, it is as follows: 
    ```bash
    docker run \
        -p <host\>:<f_port\>:5000 \
        -p <host\>:<ws_port\>:8020 \
        -v /path/to/datasets:/worker/data \
        -v /path/to/outputs:/worker/outputs \
        --name <worker_id\> synergos_worker:v0.1.0 \
        --logging_variant basic
    ```

    Let's try to break down what is going on here.

    A. Port Routes
    ```bash
        -p <host\>:<f_port\>:5000
        -p <host\>:<ws_port\>:8020
    ```

    This section maps the incoming connections into the container. 
    
    <!-- In the REST-RPC worker, the 2 main services hosted are:
        
    1. Static REST service

        * Driven by Flask
        * Receives triggers from REST-RPC TTP

    2. Dynamically initialised websocket connection

        * Primarily used by PySyft workers for finetuned federated orchestration.
        * Only active when WSSWs have been initialised by TTP's `"Initialise"` trigger
        * Deactivates when WSSWs have been destroyed by TTP's `"Terminate"` trigger -->

    Glossary:
     
    * host - IP of host machine
    * f_port - Selected port to route incoming HTTP connections for the REST service
    * ws_port - Selected port to route incoming Websocket connections for the PySyft Websocket workers

    B. Volume Mounts
    ```bash
        -v /path/to/datasets:/worker/data \
        -v /path/to/outputs:/worker/outputs
    ```

    Override the internal directories of the containers with the mountable directories you have created.

    C. Launch Examples
    
    You can view the guides for running:
    - [Synergos Basic Standalone Grid i.e. local](https://docs.synergos.ai/BasicRunLocal.html)
    - [Synergos Basic Distributed Grid i.e. across multiple devices](https://docs.synergos.ai/BasicRunDistributed.html)
    - [Synergos Cluster Distributed Grid i.e. across multiple devices](https://docs.synergos.ai/ClusterRunDistributed.html)
    - [Example Datasets and Jupyter Notebooks](https://github.com/aimakerspace/Synergos/tree/master/examples)
    
    <!-- Here are some examples to get you started. 
    
    I. Standalone Grid (i.e. local)

    > <font color='turquoise'>**docker run <br> -v /path/to/datasets:/worker/data <br> -v /path/to/customised/dependencies:/worker/custom <br> -v /path/to/outputs:/worker/outputs <br> --name <worker_id> worker:pysyft_demo**</font>
     
    In a standalone grid, docker's bridge network automatically assigns an IP to each container. This means that each container has a unique IP and thus is not required to perform port routing.

    > To find your container IDs, use either <br>`docker inspect -f '{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}' container_name_or_id` for modern version of docker, or <br>`docker inspect --format '{{ .NetworkSettings.IPAddress }}' container_name_or_id` for the previous versions.

    II. Distributed Servers (i.e. across multiple devices)

    > <font color='turquoise'>**docker run <br><font color='red'>-p <host\>:<f_port\>:5000 <br> -p <host\>:<ws_port\>:8020</font><br><font color='orange'>-v /path/to/datasets:/worker/data <br> -v /path/to/customised/dependencies:/worker/custom <br> -v /path/to/outputs:/worker/outputs</font><br> --name <worker_id> worker:pysyft_demo**</font>

    For a guided tutorial, 
    1. Download worker inputs [here](https://drive.google.com/drive/folders/1hSoOq1z-Lo3w-qUrFbsoPITzIyYWivvD?usp=sharing)
    2. Download test datasets [here](https://drive.google.com/drive/folders/19C9m6XEPHeEMIwmPRajX5-UBNujGOdtM?usp=sharing)
    3. Refer to this [guide](https://gitlab.int.aisingapore.org/aims/federatedlearning/fedlearn-prototype/-/wikis/PySyft/How-to-run-jobs-in-PySyft).
    A. Datasets

    > <font color='turquoise'>**/datasets <br>&ensp;&ensp;&ensp;&ensp; <font color='red'>/tabular <br>&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp; metadata.json <br>&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp; tabular_data.csv <br>&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp; schema.json</font><br>&ensp;&ensp;&ensp;&ensp; <font color='orange'>/image <br>&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp; metadata.json <br>&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp; /label_class_0 <br>&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp; image_1.png <br>&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp; image_2.png <br>&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp; image_3.png <br>&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp; /label_class_1 <br>&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp; image_4.png <br>&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp; image_5.png <br>&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp; image_6.png <br>&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp; /label_class_2 <br>&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp; image_7.png <br>&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp; image_8.png <br>&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp; image_9.png</font> <br> &ensp;&ensp;&ensp;&ensp; <font color='violet'>/text <br>&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp; metadata.json <br>&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp; text_corpus.csv**</font></font>

    Notice that all datasets declared have the same `metadata.json` file? 2 main items must be specified there, namely `datatype` and `operations`. 

    ```
    # Three possible datatypes: ['tabular','image','text']
    
    eg. 
    
    {
        
        "datatype": "tabular",
        "operations": {...}
    }

    Operation options for Tabular data:
        seed: int = 42 
        boost_iter: int = 100
        thread_count: int = None

    Supported Operation options for Image data:
        use_grayscale: bool = True
        use_alpha: bool = False
        use_deepaugment: bool = True

    Supported Operation options for Text data:
        max_df: int = 30000
        max_features: int = 1000
        strip_accents: str = 'unicode'
        keep_html: bool = False
        keep_contractions: bool = False
        keep_punctuations: bool = False
        keep_numbers: bool = False
        keep_stopwords: bool = False
        spellcheck: bool = True
        lemmatize: bool = True
    ```

    Customised preprocessing operations can be specified by participants to augment different declared local datasets for better training efficiency. Note, every dataset declared MUST have a `metadata.json` residing in the same directory.

    I. Tabular Data

    <font color='red'>/tabular <br>&ensp;&ensp;&ensp;&ensp; metadata.json <br>&ensp;&ensp;&ensp;&ensp; tabular_data.csv <br>&ensp;&ensp;&ensp;&ensp; schema.json</font>

    All tabular data MUST be stored as a `.csv` file and have a `schema.json` file declared alongside it containing all the datatype mappings of each feature of the dataset.

    ```
    # Example contents of a schema.json file
    {
        "age": "int32",
        "sex": "category", 
        "cp": "category", 
        "trestbps": "int32", 
        "chol": "int32", 
        "fbs": "category", 
        "restecg": "category", 
        "thalach": "int32", 
        "exang": "category", 
        "oldpeak": "float64", 
        "slope": "category", 
        "ca": "category", 
        "thal": "category", 
        "target": "category"
    }
    ```

    II. Image Data

    <font color='orange'>/image <br>&ensp;&ensp;&ensp;&ensp; metadata.json <br>&ensp;&ensp;&ensp;&ensp; /label_class_0 <br>&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp; image_1.png <br>&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp; image_2.png <br>&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp; image_3.png <br>&ensp;&ensp;&ensp;&ensp;/label_class_1 <br>&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp; image_4.png <br>&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp; image_5.png <br>&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp; image_6.png <br>&ensp;&ensp;&ensp;&ensp; /label_class_2 <br>&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp; image_7.png <br>&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp; image_8.png <br>&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp; image_9.png</font>

    For image data, each image needs to be ordered according to their class labels. Common image types (eg. `.png`, `.gif`, `.jpg` etc.) are supported.

    III. Text Data

    <font color='violet'>/text <br>&ensp;&ensp;&ensp;&ensp; metadata.json <br>&ensp;&ensp;&ensp;&ensp; text_corpus.csv</font>

    Corpora MUST be declared as one or more `.csv` files with only 2 columns availble (i.e. `['text', 'target']`)

    B. Models (Used for SNN only)
    
    > <font color='turquoise'>**/models <br>&ensp;&ensp;&ensp;&ensp; /model_1 <br>&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp; structure.json <br>&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp; hyperparameters.json <br>&ensp;&ensp;&ensp;&ensp; /model_2 <br>&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp; structure.json <br>&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp; hyperparameters.json <br>&ensp;&ensp;&ensp;&ensp; /model_3 <br>&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp; structure.json <br>&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp; hyperparameters.json**</font>

    In Split Neural Networks (SNN), each participant is first expected to train and optimise a local model on the dataset they have declared. 
    
    The model architecture **MUST** be declared in the `"structure.json"`. Here is an example of the achitecture declaration for a simple 2-layer neural network: 
    
    ```
    [
        {
            "activation": "sigmoid",
            "is_input": True,
            "l_type": "Linear",
            "structure": {
                "bias": True,
                "in_features": 15,
                "out_features": 100
            }
        },
        {
            "activation": "sigmoid",
            "is_input": True,
            "l_type": "Linear",
            "structure": {
                "bias": True,
                "in_features": 100,
                "out_features": 1
            }
        }
    ]
    ```

    * `activation` - Any activation function found in the PyTorch's `torch.nn.functional` module
    * `is_input` - Indicates if the current layer is an input layer. If a layer is an input layer, it is considered to be "wobbly" layer, meaning that the in-features may be modified automatically to accomodate changes in input structure post-alignment.
    * `l_type` - Type of layer to be used that can be found in PyTorch's `torch.nn` module. Here, the string specified corresponds to the EXACT layer class name intended to be used (eg. `torch.nn.Conv1` translates to `"Conv1"`)
    * `structure` - Any input parameters accepted in the layer class specified in `l_type`

    If declared properly, the model architecture will be parsed by an internal model-decoding module, after which the resultant model will be loaded into the  WebsocketServerWorker (WSSW).

    The optimal hyperparameter set used to obtain the best model performance **MUST** be declared in the `"structure.json"`. Here's a sample of what a hyperparameter set declaration looks like:

    ```
    {
        "batch_size": 32,
        "rounds": 2,
        "epochs": 1,
        "lr": 0.2,
        "weight_decay": 0.02,
        "mu": 0.15,
        "l1_lambda": 0.2,
        "l2_lambda": 0.3,
        "patience": 2,
        "delta": 0.0001
    }
    ```


    C. Customised Dependencies
    
    > <font color='turquoise'>**/custom <br><font color='red'>&ensp;&ensp;&ensp;&ensp; /nltk_data <br>&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp; /chunkers <br>&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp; < insert chunker resources here > <br>&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp; /corpora <br>&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp; stopwords.zip <br>&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp; wordnet.zip <br>&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp; /grammars <br>&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp; < insert grammar resources here ><br>&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp; /help <br>&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp; < insert help resources here > <br>&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp; /misc <br>&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp; < insert miscellaneous resources here > <br>&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp; /models <br>&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp; < insert model resources here > <br>&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp; /sentiment <br>&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp; < insert sentiment resources here > <br>&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp; /stemmers <br>&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp; < insert stemmer resources here > <br>&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp; /taggers <br>&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp; < insert tagger resources here > <br>&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp; /tokenizers <br>&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp; punkt.zip</font><br>&ensp;&ensp;&ensp;&ensp;<font color='orange'>/spacy <br>&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp; en_core_web_sm-X.X.X.tar.gz <br>&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp; en_core_web_md-X.X.X.tar.gz <br>&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp; en_core_web_lg-X.X.X.tar.gz</font><br>&ensp;&ensp;&ensp;&ensp; <font color='violet'>/symspell <br>&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp; XXX_dictionary_XXX.txt <br>&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp; XXX_bigram_XXX.txt**</font></font>

    For natural language processing mechanisms, the 3 main drivers are:

    1. [NLTK](https://www.nltk.org) - General preprocessing 

        All customisable NLTK backend packages can be found [here](http://www.nltk.org/nltk_data/).

        > Installation - Simply download the .zip resource of your choice and place them in the correct `nltk` sub-folder! (No need to unzip!)

    2. [Spacy](https://spacy.io) - Topic Modelling

        All the latest customisable Spacy dictionaries can be found [here](https://github.com/explosion/spacy-models/releases)

        > Installation - Simply download the .tar.gz distribution package, and place them in the `spacy` directory

    3. [SymSpell](https://github.com/wolfgarbe/SymSpell) - AutoML word spellcheck + correction

        > Installation - 2 core dictionaries that are required for SymSpell can be downloaded from their repository. Download the core frequency dictionary [here](https://raw.githubusercontent.com/mammothb/symspellpy/master/symspellpy/frequency_dictionary_en_82_765.txt). Download the frequency bigram dictionaries [here](https://raw.githubusercontent.com/mammothb/symspellpy/master/symspellpy/frequency_bigramdictionary_en_243_342.txt). For more indepth customisation, visit this [page](https://symspellpy.readthedocs.io/en/latest/users/installing.html)

    [Note: Future customisations will also be inserted here and under go the same manner of installation, so you can expect that any new dependency customisations will be routed here as well.]

    D. Outputs
    
    > <font color='turquoise'>**/outputs <br>&ensp;&ensp;&ensp;&ensp;< no structural requirements >**</font>

    There are no structural requirements for the `outputs` directory. This internal directory is meant to be overridden by your own data directory (refer to section 5 for more information)

6) Payload Submissions

    In order for the TTP to complete the connection, these are 3 payloads that MUST be submitted to the TTP's REST service and they are as follows:

    A. Server Information

        {
            "id": participant_id,   # any user-defined string ID
            "host": <host>,         # IP of server
            "port": <ws_port>,      # Assigned websocket port
            "log_msgs": False,      # Toggles PySyft logging
            "verbose": False,       # Toggles PySyft logging's verbosity
            "f_port": <f_port>      # Assigned REST port
        }

    B. Role Information 

        # 3 possible roles - ['arbiter', 'host', 'guest']
        {"role": "guest"}   

    C. Data Tags

    Data tags are the root filepath tokens that lead to the declared dataset's directory structure. 
    
    For example, if `-v /tabular:/worker/data` was the mount command where the directory structure of `/tabular` is

    ><font color='red'>/tabular<br>&ensp;&ensp;&ensp;&ensp;/train<br>&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp; metadata.json <br>&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp; tabular_data.csv <br>&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp; schema.json<br>&ensp;&ensp;&ensp;&ensp;/evaluate<br>&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp; metadata.json <br>&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp; tabular_data.csv <br>&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp; schema.json<br>&ensp;&ensp;&ensp;&ensp;/predict<br>&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp; metadata.json <br>&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp; tabular_data.csv <br>&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp; schema.json</font>
    
    The corresponding tag payload declarations will be as follows: 

        # Training and evaluation data declaration
        { 
            "train": [["train"]],
            "evaluate": [["evaluate"]]
        } 

        # Participant-driven prediction
        {
            "dockerised": True,
            "tags": {
                project_id: [["predict"]]
        }  -->