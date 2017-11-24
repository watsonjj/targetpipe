import pandas as pd
from core import input_path
from tf import *
from tqdm import tqdm


def main():
    store = pd.HDFStore(input_path)
    df = store['df']

    tf_list = [
        TFSamplingCell,
        TFStorageCell,
        TFStorageCellReduced,
        TFStorageCellReducedCompress,
        TFPChip,
        TFBest,
        TFBestCompress,
        TFNothing,
        TFTargetCalib
    ]

    desc = "Looping through TF list"
    for TF in tqdm(tf_list):
        tf = TF()
        tf.create(df)


if __name__ == '__main__':
    main()
