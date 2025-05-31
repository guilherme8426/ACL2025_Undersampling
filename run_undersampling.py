from src.main.python.utils.general import get_data, get_splits, checkpoint_splits, translate_train_idxinfold
from src.main.python.utils.arguments_phase0 import arguments
from src.main.python.utils.save_results_phase0 import save_results
import argparse
from datetime import datetime
import time
import numpy as np
import gc
import io
import os
import pandas as pd
from collections import Counter
# from src.main.python.undersampling import cnn, enn, ncr, oss, raus, tl, e2sc, renn, nearmiss, allknn, iht, cc_nn, sbc, obu, rr_01, e2sc_novo, rr_02
from src.main.python.undersampling import cnn, enn, ncr, oss, tl, renn, nearmiss, allknn, iht, cc_nn, sbc, obu, e2sc_us, ubr, akcs, enu

import socket

import logging
import logging.config

logging.config.fileConfig('settings/logging.conf', defaults={'logfilename': f'resources/logs/{socket.gethostname()}.log'})
logger = logging.getLogger(__name__)

def get_selector(method: str, args: dict, fold: int):

    if method == 'akcs':           return akcs.AKCS()
    if method == 'cnn':            return cnn.CNN()
    if method == 'enn':            return enn.ENN()
    if method == 'ncr':            return ncr.NCR()
    if method == 'oss':            return oss.OSS()
    if method == 'tl':             return tl.TL()
    if method == 'e2sc_us':        return e2sc_us.E2SC_US(alphaMode="logistic_regression")
    if method == 'renn':           return renn.RENN()
    if method == 'allknn':         return allknn.ALLKNN()
    if method == 'nearmiss_1':     return nearmiss.NEARMISS(version=1)
    if method == 'nearmiss_2':     return nearmiss.NEARMISS(version=2)
    if method == 'nearmiss_3':     return nearmiss.NEARMISS(version=3)
    if method == 'ubr':            return ubr.UBR()
    if method == 'iht':            return iht.IHT()
    if method == 'cc_nn':          return cc_nn.CC_NN()
    if method == 'sbc':            return sbc.SBC()
    if method == 'obu':            return obu.OBU()
    if method == 'enub':           return enu.ENU(estrategia="ENUB")
    if method == 'enut':           return enu.ENU(estrategia="ENUT")
    if method == 'enuc':           return enu.ENU(estrategia="ENUC")
    if method == 'enur':           return enu.ENU(estrategia="ENUR")

    return None


def get_selection(X, y, fold, args):

    method = args.method

    total = Counter(y)
    logger.debug(f"Total instances number: {total}")

    selector = get_selector(method, args, fold)

    selector.fit(X, y)

    logger.info("Result: ", Counter(y[selector.sample_indices_]))

    return selector.sample_indices_


def main():

    gc.collect()

    args, info = arguments()
    logger.info(str(args))

    print(f"{args.splitdir}/split_{args.folds}.pkl")
    splits_df = get_splits(f"{args.splitdir}/split_{args.folds}.pkl")

    splits_to_save = {c: [] for c in splits_df.columns if c.endswith("idxs")}
       
    for f in range(args.folds):
    #for f in range(1):

        logger.info("Fold {}".format(f))
        print("Fold {}".format(f))
        
        splits_to_save['test_idxs'].append(splits_df.loc[f].test_idxs)


        X_train, y_train, _, _, _ = get_data(args.inputdir, f)
        t = len(y_train)

        ti = time.time()

        idxs_docs = get_selection(X_train, y_train, f, args)

        s = len(y_train[idxs_docs])
        r = (t-s)/t

        info['time_for_reduce'].append(time.time() - ti)
        print(info['time_for_reduce'])
        info['original_len'].append(t)
        info['reduced_len'].append(s)
        info['reducion'].append(r)

        splits_to_save['train_idxs'].append(idxs_docs)

    logger.info(f"time: {np.mean(info['time_for_reduce'])}")
    logger.info(f"time std: {np.std(info['time_for_reduce'])}")
    logger.info(f"reducion: {np.mean(info['reducion'])}")
    logger.info(f"reducion std: {np.std(info['reducion'])}")

    splits_to_save_df = pd.DataFrame(data=splits_to_save)

    filename = f"{args.outputdir}/split_{args.folds}_{args.method}_idxinfold.pkl"

    checkpoint_splits(
        splits_df=splits_to_save_df,
        filename = filename
    )

    splits_to_save_df_traslated = translate_train_idxinfold(
        splits_to_save_df, splits_df)

    checkpoint_splits(
        splits_df=splits_to_save_df_traslated,
        filename=filename.replace("_idxinfold", "")
    )

    if args.save:
        save_results(args, info)
    
    print("END")
    exit()


if __name__ == '__main__':
    main()
