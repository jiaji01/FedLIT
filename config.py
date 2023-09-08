CONFIG = {
        "latent_dim": 32,
        "fed_rounds": 6,
        "epochs_per_round": 5,
        'batch_size': 32,
        "early_stop": True,
        "freeze": True, 
        "ablated": True,
        'number': 9,
        "model_name": "number9"       # to be edited

}

PARAMS = {
    "thin":{
        "learning rate": 0.007,
        "sample size": 20,
        "margin": 1,
        "lambda1": 0.2,  # SNL 
        "lambda2":0.5, # contrastive
    },

    "thic":{
        "learning rate": 0.007,
        "sample size": 20,
        "margin": 1,
        "lambda1": 0.2,  # SNL 
        "lambda2":0.5, # contrastive
    },


    "raw":{
        "learning rate": 0.007,
        "sample size": 20,
        "margin": 1,
        "lambda1": 0.2,  # SNL 
        "lambda2":0.5, # contrastive
    },


    "swel":{
        "learning rate": 0.007,
        "sample size": 20,
        "margin": 1,
        "lambda1": 0.2,  # SNL 
        "lambda2":0.5, # contrastive
    },

    "frac":{
        "learning rate": 0.007,
        "sample size": 20,
        "margin": 1,
        "lambda1": 0.2,  # SNL 
        "lambda2":0.5, # contrastive
    },

}

AUG = {
    "thin":{
        "m": 1,
        "n": 5
    },
    "thic":{
        "m": 1,
        "n": 5
    },
    "raw":{
        "m": 1,
        "n": 3
    },
    "swel":{
        "m": 1,
        "n": 5
    },
    "frac":{
        "m": 1,
        "n": 5
    },
}

# AUG = {
#     "thin":{
#         "m": 1,
#         "n": 2
#     },
#     "thic":{
#         "m": 1,
#         "n": 2
#     },
#     "raw":{
#         "m": 1,
#         "n": 1
#     },
#     "swel":{
#         "m": 1,
#         "n": 2
#     },
#     "frac":{
#         "m": 1,
#         "n": 2
#     },
# }