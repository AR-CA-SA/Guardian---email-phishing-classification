from datasetup import processor
import os 
from pathlib import Path




PATHS = {"EMAIL_CLEAN" : Path("cleandataset/email_clean.npz"),
         "EMCL_TRAIN" : Path("cleandataset/email_clean_X.npy"),
         "EMCL_PARQUET" : Path("cleandataset/email_clean_metadata.parquet")}

def main():

    decorator_1 = "="*50
    print(f"{decorator_1} INDEX LOADING {decorator_1}")
    index = processor.loadDataNPZ(importPath = PATHS["EMAIL_CLEAN"])
    print(f"{decorator_1} LOADING TRAINING SET {decorator_1}")
    x_train = processor.loadDataFM(importPath  = PATHS["EMCL_TRAIN"], index = index)
    print(f"{decorator_1} METADATA LOADING {decorator_1}")
    y_target, meta = processor.loadMetaData(importPath = PATHS["EMCL_PARQUET"], X = x_train, index = index)


if __name__ == "__main__":
    main()

    