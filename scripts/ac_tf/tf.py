import numpy as np
from tqdm import tqdm, trange
from core import pix_dir, tf_path, pix, pedestal_path
from scipy.interpolate import interp1d, PchipInterpolator
from tables import open_file
from os.path import join
from IPython import embed
import pandas as pd
import target_calib


def child_subclasses(cls):
    family = cls.__subclasses__() + [g for s in cls.__subclasses__()
                                     for g in child_subclasses(s)]
    # children = [g for g in family if not g.__subclasses__()]
    children = [g for g in family if g.fn]
    return children


class TF:
    fn = ""
    cn = ""

    def __init__(self):
        tqdm.write("Created instance of {}".format(self.__class__.__name__))
        self.path = join(pix_dir, self.fn)
        self.step = 8
        self.create_interp = 'linear'
        self.apply_interp = 'linear'
        self.compress = False

    def _prepare_df(self, df):
        tqdm.write(">> Preparing df")
        df[self.cn] = (df['fci'] + df['sample']) % 64
        df_sort = df.sort_values([self.cn, 'vped'])
        return df_sort

    def _get_input(self, df):
        tqdm.write(">> Preparing input")
        n_cell = np.unique(df[self.cn]).size
        vped = np.unique(df['vped'])
        n_vped = vped.size
        index = pd.MultiIndex.from_product([np.arange(n_cell), vped], names=['cell_i', 'vped_i'])
        df_mean = df.groupby([self.cn, 'vped']).mean().reindex(index).reset_index()
        adc = df_mean['adc'].values.reshape([n_cell, n_vped])

        return vped, adc

    def _get_adc_points(self, adc):
        tqdm.write(">> Getting ADC points")
        adc_min = np.nanmin(adc)
        adc_max = np.nanmax(adc)
        n = np.ceil((adc_max - adc_min)/ self.step)
        adc_x = np.linspace(adc_min, adc_max, n)
        return adc_x

    def _create_interpolation(self, adc_x, adc_cell, vped_cell, fill):
        f = interp1d(adc_cell, vped_cell, kind=self.create_interp, fill_value=fill, bounds_error=False)
        return f(adc_x)

    def _get_tf(self, vped, adc, adc_x):
        tqdm.write(">> Building TF")
        n_cell = adc.shape[0]
        tf = np.zeros((adc.shape[0], adc_x.size))
        desc = ">>> Looping through cells"
        for cell in trange(n_cell, desc=desc):
            adc_cell = adc[cell]
            vped_cell = np.copy(vped)

            # Remove nan values from unfilled cells
            notnan = ~np.isnan(adc_cell)
            adc_cell = adc_cell[notnan]
            vped_cell = vped_cell[notnan]

            fill = (vped[0], vped[-1])
            check = True
            while check:
                turn = np.where(np.diff(adc_cell) < 0)[0]
                if turn.size == 0:
                    check = False
                    break
                adc_cell = np.delete(adc_cell, turn)
                vped_cell = np.delete(vped_cell, turn)
                fill = (vped_cell[0], vped_cell[-1])

            tf[cell] = self._create_interpolation(adc_x, adc_cell, vped_cell, fill)
            # if cell == 2846:
            #     embed()
        return tf

    def _save_tf(self, adc_x, tf):
        tqdm.write(">> Saving TF to: {}".format(self.path))
        with open_file(self.path, mode="w", title="ChargeResolutionFile") as f:
            group = f.create_group("/", 'tf'.format(0), '')
            if self.compress:
                tf, offset, scale = self.compress_tf(tf)
                f.create_array(group, 'offset', offset, 'offset')
                f.create_array(group, 'scale', scale, 'scale')
            f.create_array(group, 'adc_x', adc_x, 'adc_x')
            f.create_array(group, 'tf', tf, 'tf')

    def create(self, df):
        tqdm.write("> Creating new TF")
        df_sort = self._prepare_df(df)
        vped, adc = self._get_input(df_sort)
        adc_x = self._get_adc_points(adc)
        tf = self._get_tf(vped, adc, adc_x)
        self._save_tf(adc_x, tf)

    def _load_tf(self):
        tqdm.write(">> Loading TF from: {}".format(self.path))
        with open_file(self.path, mode="r") as f:
            adc_x = f.get_node("/tf", 'adc_x').read()
            tf = f.get_node("/tf", 'tf').read()
            if self.compress:
                offset = f.get_node("/tf", "offset").read()
                scale = f.get_node("/tf", "scale").read()
                tf = self.restore_tf(tf, offset, scale)
        return adc_x, tf

    def _apply(self, x, adc_x, tf, cell):
        fill = (tf[cell, 0], tf[cell, -1])
        f = interp1d(adc_x, tf[cell], kind=self.apply_interp, fill_value=fill, bounds_error=False)
        return f(x)

    def _perform_calibration(self, df, adc_x, tf):
        tqdm.write(">> Performing calibration")
        cells = np.unique(df[self.cn]).astype(np.int)
        cal = np.zeros(df.index.size)
        desc = ">>> Calibrating adc"
        for c in tqdm(cells, desc=desc):
            w = np.where(df[self.cn] == c)[0]
            cal[w] = self._apply(df.iloc[w]['adc'], adc_x, tf, c)
        df['cal'] = cal

    def calibrate(self, df):
        tqdm.write("> Calibrating dataframe")
        df_sort = self._prepare_df(df)
        adc_x, tf = self._load_tf()
        self._perform_calibration(df_sort, adc_x, tf)
        return df_sort

    def calibrate_x(self, x, cell):
        adc_x, tf = self._load_tf()
        return self._apply(x, adc_x, tf, cell)

    @staticmethod
    def compress_tf(tf):
        offset = tf.min()
        tf_offset = tf - offset
        dtype_max = np.iinfo(np.uint16).max
        scale = dtype_max / tf_offset.max()
        tf_scaled = tf_offset * scale
        tf_int = tf_scaled.astype(np.uint16)
        return tf_int, offset, scale

    @staticmethod
    def restore_tf(tf_int, offset, scale):
        tf = (tf_int.astype(np.float) / scale) + offset
        return tf

    def get_tfinput(self, df):
        df_sort = self._prepare_df(df)
        vped, adc = self._get_input(df_sort)
        return vped, adc

    def get_hits(self, df):
        df_sort = self._prepare_df(df)
        n_cell = np.unique(df_sort[self.cn]).size
        vped = np.unique(df_sort['vped'])
        n_vped = vped.size
        index = pd.MultiIndex.from_product([np.arange(n_cell), vped], names=['cell_i', 'vped_i'])
        df_size = df_sort.groupby([self.cn, 'vped']).size().reindex(index)
        hits = df_size.values.reshape([n_cell, n_vped])
        hits[np.isnan(hits)] = 0
        return hits


class TFSamplingCell(TF):
    fn = "tf_samplingcell.h5"
    cn = "samplingcell"

    def __init__(self):
        super().__init__()
        self.step = 8


class TFSamplingCellPerfect(TFSamplingCell):
    fn = "tf_samplingcellperfect.h5"
    cn = "samplingcell"

    def __init__(self):
        super().__init__()
        self.step = 2


class TFStorageCell(TF):
    fn = "tf_storagecell.h5"
    cn = "storagecell"

    def __init__(self):
        super().__init__()
        self.step = 8
        self.create_interp = 'linear'
        self.apply_interp = 'linear'

    def _prepare_df(self, df):
        tqdm.write(">> Preparing df")
        # c = target_calib.GetCellIDArray(df['fci'].values.astype(np.uint16),
        #                                 df['sample'].values.astype(np.uint16))
        # df['cell'] = c
        df[self.cn] = df['cell']
        df_sort = df.sort_values([self.cn, 'vped'])
        return df_sort


class TFStorageCell_60(TFStorageCell):
    fn = "tf_storagecell60.h5"

    def __init__(self):
        super().__init__()
        self.step = 8 * 500/60


class TFStorageCell_150(TFStorageCell):
    fn = "tf_storagecell150.h5"

    def __init__(self):
        super().__init__()
        self.step = 8 * 500/150


class TFStorageCell_300(TFStorageCell):
    fn = "tf_storagecell300.h5"

    def __init__(self):
        super().__init__()
        self.step = 8 * 500/300


class TFStorageCell_500(TFStorageCell):
    fn = "tf_storagecell500.h5"

    def __init__(self):
        super().__init__()
        self.step = 8


class TFStorageCell_1000(TFStorageCell):
    fn = "tf_storagecell1000.h5"

    def __init__(self):
        super().__init__()
        self.step = 8 * 500/1000


class TFStorageCellPedestal(TFStorageCell):
    fn = "tf_storagecellpedestal.h5"

    def create(self, df):
        store = pd.HDFStore(pedestal_path)
        df = pd.concat([df, store['df']], ignore_index=True)
        super().create(df)


class TFStorageCellPedestalZero(TFStorageCellPedestal):
    fn = "tf_storagecellpedestalzero.h5"

    def _get_adc_points(self, adc):
        tqdm.write(">> Getting ADC points")
        adc_min = np.nanmin(adc)
        adc_min = self.step * ((adc_min // self.step) - 1)
        adc_max = np.nanmax(adc)
        adc_max = self.step * ((adc_max // self.step) + 1)
        n = np.ceil((adc_max - adc_min) / self.step)
        adc_x = np.linspace(adc_min, adc_max, n+1)
        return adc_x


# class TFPChip(TFStorageCellReducedCompress):
#     name = "TFPChip"
#     fn = "tf_pchip.h5"
#
#     def __init__(self):
#         super().__init__()
#
#     def _create_interpolation(self, adc_x, adc_cell, vped_cell, fill):
#         f = PchipInterpolator(adc_cell, vped_cell)
#         tf = f(adc_x)
#         oor_low = adc_x < adc_cell.min()
#         oor_high = adc_x > adc_cell.max()
#         tf[oor_low] = fill[0]
#         tf[oor_high] = fill[1]
#         return tf
#
#     def _apply(self, x, adc_x, tf, cell):
#         f = PchipInterpolator(adc_x, tf[cell])
#         oor_low = x < adc_x.min()
#         oor_high = x > adc_x.max()
#         cal = f(x)
#         cal[oor_low] = tf[cell, 0]
#         cal[oor_high] = tf[cell, -1]
#         return cal
#
#
# class TFBest(TFPChip):
#     name = "TFBest"
#     fn = "tf_best.h5"
#
#     def __init__(self):
#         super().__init__()
#         self.compress = False
#         self.n_adc_points = 1000
#
#
# class TFBestCompress(TFBest):
#     name = "TFBestCompress"
#     fn = "tf_bestcompress.h5"
#
#     def __init__(self):
#         super().__init__()
#         self.compress = True
#
#
# class TFNothing(TFSamplingCell):
#     name = "TFNothing"
#     fn = "tf_nothing.h5"
#
#     def create(self, df):
#         pass
#
#     def calibrate(self, df):
#         tqdm.write("> Calibrating dataframe")
#         df_sort = self._prepare_df(df)
#
#         cells = np.unique(df_sort[self.cn]).astype(np.int)
#         cal = np.zeros(df_sort.index.size)
#         vpeds = np.unique(df_sort['vped']).astype(np.int)
#         desc = ">>> Calibrating adc"
#         for c in tqdm(cells, desc=desc):
#             w = np.where(df_sort[self.cn] == c)[0]
#             df_cell = df_sort.iloc[w]
#             df_lv = df_cell.loc[df_cell['vped'] == vpeds[-1]]
#             scale = vpeds[-1] / df_lv['adc'].mean()
#             cal[w] = df_cell['adc'] * scale
#         df_sort['cal'] = cal
#
#         return df_sort
#
#
# class TFTargetCalib(TFSamplingCell):
#     name = "TFTargetCalib"
#     fn = "tf_targetcalib.h5"
#
#     def create(self, df):
#         pass
#
#     def calibrate(self, df):
#         tqdm.write("> Calibrating dataframe")
#         calibrator = target_calib.Calibrator('', tf_path)
#
#         df_sort = self._prepare_df(df)
#
#         cells = np.unique(df_sort[self.cn]).astype(np.int)
#         cal = np.zeros(df_sort.index.size, dtype=np.float32)
#         vpeds = np.unique(df_sort['vped']).astype(np.int)
#         desc = ">>> Calibrating adc"
#         for c in tqdm(cells, desc=desc):
#             w = np.where(df_sort[self.cn] == c)[0]
#             df_cell = df_sort.iloc[w]
#             cell = df_cell.iloc[0]['cell']
#             blockphase = int(cell % 32)
#             row = (cell // 32) % 8
#             column = (cell // 32) // 8
#             block = int(column * 8 + row)
#             adc = np.ascontiguousarray(df_cell['adc'].values, dtype=np.float32)
#             cal_c = np.zeros(adc.size, dtype=np.float32)
#             calibrator.ApplyArray(adc, cal_c, 0, pix, block, blockphase)
#             cal[w] = cal_c
#         df_sort['cal'] = cal
#
#         return df_sort
