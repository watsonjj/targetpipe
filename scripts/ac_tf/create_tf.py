import pandas as pd
from core import input_path
from tf import TF, child_subclasses
from tqdm import tqdm


def main():
    store = pd.HDFStore(input_path)
    df = store['df']

    tf_list = child_subclasses(TF)

    desc = "Looping through TF list"
    for cls in tqdm(tf_list):
        tf = cls()
        tf.create(df)


if __name__ == '__main__':
    main()
