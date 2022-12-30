import keras
from sklearn import metrics
import pandas as pd
import time

benchmark = ["raw", "highlevel", "all"]
usecols = []
dropout = [0, 0.01, 0.5]

# Column to load according to benchmark
# data range is 0 to 28 for Higgs.
# 21 low-level features & 7 high-level features
# and the column 0 is for the label
all_n = 28  # total features
raw_n = 21  # raw features
high_n = 7  # highlevel features

# DELETE if using the original data from UCI
# usecols = [i + 1 for i in usecols]
print(benchmark)

if __name__ == '__main__':

    for bench in benchmark:
        # list(range(a, b+1)) returns an array of [a, ..., b]
        # usecols points to the columns to use
        # column 0 is always loaded
        if bench is "raw":
            # [1, .., 21] ===> 21 items
            # ===> [0,1, ..., 21]
            usecols = [0] + list(range(1, raw_n + 1))
            input_n = raw_n
        elif bench is "highlevel":
            # [22, ..., 28] ===> 7 items
            # [0,22, ..., 28]
            usecols = [0] + list(range(raw_n + 1, all_n + 1))
            input_n = high_n
        elif bench is "all":
            # [1, ..., 28] ===> 28 items
            # [0,1, ..., 28]
            usecols = [0] + list(range(1, all_n + 1))
            input_n = all_n

        print("benchmark = {}\n".format(bench))

        # Data
        print("Loading data")
        t = time.time()
        tables = []
        datas = pd.read_csv(filepath_or_buffer=r"../../datasets/HIGGS.csv.gz",
                            low_memory=True, compression="gzip", usecols=usecols,
                            na_filter=False)

        print("usecols : {}".format(usecols))


        print("Loading Time : {:6.6}s".format(time.time() - t))

        x = datas.iloc[:, 1:].as_matrix()
        y = datas.iloc[:, 0].as_matrix()

        for do in dropout:
            print("dropout = {}\n".format(do))

            # Model
            # model_path = sys.argv[1]
            model_path = r'../../process/saves/HIGGS/' + \
                         'model_HIGGS_layers4_Epoch20_width128_do{do}_{bench}.h5'.format(bench=bench, do=do)
            model = keras.models.load_model(filepath=model_path)

            # evaluate
            # ev = model.evaluate(x, y)
            # print('evaluate ', ev[0], ev[1])

            # predict
            pred = model.predict(x, verbose=1)[:, 0]

            # Compute AUC-ROC (Area Under Curve - Receiver Operating Characteristics)
            # fpr (false positive recall), tpr (true positive recall)
            fpr, tpr, thresholds = metrics.roc_curve(y, pred)
            auc = metrics.auc(fpr, tpr)

            output_path = r'../../process/saves/HIGGS/' + \
                          'ROC_model_HIGGS_layers4_Epoch20_width128_do{do}_{bench}.csv'.format(bench=bench, do=do)
            df = pd.DataFrame(dict(tpr=tpr, fpr=fpr))
            df.to_csv(path_or_buf=output_path, sep=',', index=False)
            print('\n AUC:', auc)
