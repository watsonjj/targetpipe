import pandas as pd
from tqdm import tqdm
from core import pix, core_path as input_path, pix_path as output_path


def main():
    desc = "Reading file"
    chunksize = 100000
    nlines = 1093620302
    nchunks = nlines//chunksize
    tb = pd.read_csv(input_path, iterator=True, chunksize=chunksize)
    df_list = []
    for t in tqdm(tb, total=nchunks, desc=desc):
        df_list.append(t.loc[t['pixel'] == pix])

    df = pd.concat(df_list, ignore_index=True)
    store = pd.HDFStore(output_path)
    store['df'] = df


if __name__ == '__main__':
    main()
