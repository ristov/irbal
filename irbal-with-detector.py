import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import pandas as pd
import numpy as np

from modAL.models import ActiveLearner
from modAL.uncertainty import uncertainty_sampling

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.neighbors import KNeighborsClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV

from catboost import CatBoostClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier, IsolationForest, AdaBoostClassifier

from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

from river import drift

from functools import partial

from glob import glob

from time import process_time

import argparse
import sys
import re


n_estimators = 100

monthly_pool_size = 10000
monthly_points = 1000

daily_pool_size = 300
daily_points = 30

query_strategy = uncertainty_sampling

drop_column_list = ['UNIXtime', 'Timestamp', 'SignatureText', 
                    'SignatureID', 'ExtIP', 'IntIP', 'Label']

try:
    HDDMA = drift.binary.HDDMA
except AttributeError:
    HDDMA = drift.binary.HDDM_A

############################## functions ##############################

def build_classifier(i):

    if classifier == "lr":

        clf = make_pipeline(StandardScaler(), 
                            LogisticRegression(max_iter=1000,
                                               random_state=i, 
                                               class_weight='balanced'))

    elif classifier == "knn":

        clf = make_pipeline(StandardScaler(), 
                            KNeighborsClassifier(n_neighbors=50,
                                                 weights='distance'))

    elif classifier == "svm":

        # according to scikit-learn documentation, probability=True
        # setting for SVC() is deprecated and slow, and it is recommended
        # to use CalibratedClassifierCV(estimator=base_clf, ensemble=False)
        # for getting probabilites, as SVC(probability=True) implements
        # the same method internally
        #
        # clf = make_pipeline(StandardScaler(),
        #                     SVC(kernel='linear', probability=True,
        #                         random_state=i, class_weight='balanced'))

        base_clf = LinearSVC(max_iter=10000, 
                             random_state=i, 
                             class_weight='balanced')

        clf = make_pipeline(StandardScaler(), 
                            CalibratedClassifierCV(estimator=base_clf,
                                                   ensemble=False))

    elif classifier == "cb":

        clf = CatBoostClassifier(thread_count=1,
                                 verbose=False,
                                 n_estimators=n_estimators,
                                 random_seed=i,
                                 auto_class_weights='Balanced')

    elif classifier == "ab":

        base_clf = DecisionTreeClassifier(max_depth=1, 
                                          class_weight='balanced')

        clf = AdaBoostClassifier(estimator=base_clf,
                                 n_estimators=n_estimators, 
                                 random_state=i)

    else:

        clf = RandomForestClassifier(n_estimators=n_estimators, 
                                     random_state=i,
                                     class_weight='balanced')

    return clf


def build_drift_detector(detector_type):

    if detector_type == "adwin":

        return drift.ADWIN(delta=0.001)

    elif detector_type == "fhddm":

        return drift.binary.FHDDM(confidence_level=0.001)

    elif detector_type == "hddma":

        return HDDMA(drift_confidence=0.001)

    else:

        return None


def build_supervised(k, i, modeldata=None):

    file = filelist[k]

    print("Training supervised model on", file, "with rand", i, 
          file = sys.stderr)

    training_set = pd.read_csv(file)

    X_train = training_set.drop(columns=drop_column_list)

    y_train = training_set['Label']

    if modeldata != None:

        indices = modeldata['past_file_indices']

        for past_file in (filelist[j] for j in indices):

            past_data = pd.read_csv(past_file)

            print("Appending", len(past_data), "points from", past_file, 
                  "to training data", file = sys.stderr)

            X_temp = past_data.drop(columns=drop_column_list)

            y_temp = past_data['Label']

            X_train = pd.concat([X_train, X_temp], ignore_index=True)

            y_train = pd.concat([y_train, y_temp], ignore_index=True)

    else:

        indices = []

    indices.append(k)

    print("Building supervised model on", len(indices), "files with", 
          len(X_train), "points", file = sys.stderr)

    clf = build_classifier(i) 

    model = clf.fit(X_train, y_train)

    return { 'model' : model, 'past_file_indices' : indices,
             'fp_drift_detector' : build_drift_detector(detector),  
             'fn_drift_detector' : build_drift_detector(detector) }


def build_random(k, i, modeldata=None):

    file = filelist[k]

    print("Training random model on", file, "with rand", i, file = sys.stderr)

    training_set = pd.read_csv(file)

    rng = np.random.default_rng(i)

    print("Sampling", points_from_pool, "points from", file, file = sys.stderr)

    idx = rng.choice(training_set.index, points_from_pool, replace=False)
    training_set = training_set[training_set.index.isin(idx)]

    X_train = training_set.drop(columns=drop_column_list)

    y_train = training_set['Label']

    if modeldata != None:

        indices = modeldata['past_file_indices']

        for past_file in (filelist[j] for j in indices):

            past_data = pd.read_csv(past_file)

            idx = rng.choice(past_data.index, points_from_pool, replace=False)
            past_data = past_data[past_data.index.isin(idx)]

            print("Appending", len(past_data), "points from", past_file, 
                  "to training data", file = sys.stderr)

            X_temp = past_data.drop(columns=drop_column_list)

            y_temp = past_data['Label']

            X_train = pd.concat([X_train, X_temp], ignore_index=True)

            y_train = pd.concat([y_train, y_temp], ignore_index=True)

    else:

        indices = []

    indices.append(k)

    temp = y_train[y_train == 0]

    if len(temp) == 0:
        print("No points with label 0 in data", file = sys.stderr)
        return None
    else:
        print(len(temp), "points with label 0 in data", file = sys.stderr)

    temp = y_train[y_train == 1]

    if len(temp) == 0:
        print("No points with label 1 in data", file = sys.stderr)
        return None
    else:
        print(len(temp), "points with label 1 in data", file = sys.stderr)

    print("Building random model on", len(indices), "files with", 
          len(X_train), "points", file = sys.stderr)

    clf = build_classifier(i) 

    model = clf.fit(X_train, y_train)

    return { 'model' : model, 'past_file_indices' : indices,
             'fp_drift_detector' : build_drift_detector(detector),  
             'fn_drift_detector' : build_drift_detector(detector) }


def build_altrad(k, i, modeldata=None):

    file = filelist[k]

    print("Training altrad model on", file, "with rand", i, file = sys.stderr)

    training_set = pd.read_csv(file)

    rng = np.random.default_rng(i)

    # if the model has not been provided to this function, 
    # build the seed that will be used for creating a new model

    if modeldata == None:

        idx = rng.choice(training_set.index, seed_size, replace=False)
        seed = training_set[training_set.index.isin(idx)]
        training_set = training_set.drop(idx)

        temp = seed[seed['Label'] == 0]

        if len(temp) == 0:
            print("No points with label 0 in seed", file = sys.stderr)
            return None
        else:
            print(len(temp), "points with label 0 in seed", file = sys.stderr)

        temp = seed[seed['Label'] == 1]

        if len(temp) == 0:
            print("No points with label 1 in seed", file = sys.stderr)
            return None
        else:
            print(len(temp), "points with label 1 in seed", file = sys.stderr)

        print("Building new altrad model on", file, "with seed of", 
              seed_size, "points", file = sys.stderr)

        X_seed = seed.drop(columns=drop_column_list).to_numpy()
        y_seed = seed['Label'].to_numpy()

        clf = build_classifier(i) 

        model = ActiveLearner(
            estimator=clf,
            query_strategy=query_strategy,
            X_training=X_seed, y_training=y_seed
        )

        indices = []

    else:

        model = modeldata['model']
        indices = modeldata['past_file_indices']

    # build the pool

    if pool_size > 0:
        idx = rng.choice(training_set.index, pool_size, replace=False)
        pool = training_set[training_set.index.isin(idx)]
    else:
        pool = training_set

    temp = pool[pool['Label'] == 0]

    if len(temp) == 0:
        print("No points with label 0 in pool", file = sys.stderr)
        return None
    else:
        print(len(temp), "points with label 0 in pool", file = sys.stderr)

    temp = pool[pool['Label'] == 1]

    if len(temp) == 0:
        print("No points with label 1 in pool", file = sys.stderr)
        return None
    else:
        print(len(temp), "points with label 1 in pool", file = sys.stderr)

    # update the model which was either provided to this function or 
    # created from the seed, using the points from the pool

    print("Updating existing altrad model on", file, "with", 
          points_from_pool, "points from pool of", len(pool), "points", 
          file = sys.stderr)

    X_pool = pool.drop(columns=drop_column_list).to_numpy()
    y_pool = pool['Label'].to_numpy()

    for j in range(points_from_pool):
        
        index, X_instance = model.query(X_pool)
        y_instance = y_pool[index]
    
        model.teach(X_instance, y_instance)

        X_pool = np.delete(X_pool, index, axis=0)
        y_pool = np.delete(y_pool, index, axis=0)

    indices.append(k)

    return { 'model' : model, 'past_file_indices' : indices,
             'fp_drift_detector' : build_drift_detector(detector),  
             'fn_drift_detector' : build_drift_detector(detector) }


def build_alrank(name, k, i, modeldata=None):

    file = filelist[k]

    print("Training", name, "model on", file, "with rand", i, file = sys.stderr)

    training_set = pd.read_csv(file)

    rng = np.random.default_rng(i)

    if name == "al-ifanom":

        temp = training_set.drop(columns=drop_column_list)

        clf = IsolationForest(random_state=i, contamination=0.02).fit(temp)

        training_set['ifscore'] = clf.score_samples(temp)

        training_set = training_set.sort_values('ifscore').drop(columns='ifscore')

    elif name == "al-sigfreq":

        training_set['freq'] = training_set['SignatureID'].map(training_set['SignatureID'].value_counts())

        training_set = training_set.sort_values('freq').drop(columns='freq')

    else:
        training_set = training_set.sort_values(by=['SignatureMatchesPerDay'])

    if headsize == 0:

        num_significant = int(proportion * pool_size)

        if modeldata == None:
            num_significant += int(proportion * seed_size)

    else:
        num_significant = int(headsize * len(training_set))

    head = training_set.head(n=num_significant)
    tail = training_set.tail(n=-num_significant)

    # if the model has not been provided to this function, 
    # build the seed that will be used for creating a new model

    if modeldata == None:

        num_significant = int(proportion * seed_size)

        if len(head) < num_significant:
            print("Too few significant points for the seed, including all", 
                  len(head), "points", file = sys.stderr)
            num_significant = len(head)

        idx = rng.choice(tail.index, seed_size - num_significant, replace=False)
        seed0 = tail[tail.index.isin(idx)]
        tail = tail.drop(idx)

        idx = rng.choice(head.index, num_significant, replace=False)
        seed1 = head[head.index.isin(idx)]
        head = head.drop(idx)

        seed = pd.concat([seed0, seed1], ignore_index=True)

        temp = seed[seed['Label'] == 0]

        if len(temp) == 0:
            print("No points with label 0 in seed", file = sys.stderr)
            return None
        else:
            print(len(temp), "points with label 0 in seed", file = sys.stderr)

        temp = seed[seed['Label'] == 1]

        if len(temp) == 0:
            print("No points with label 1 in seed", file = sys.stderr)
            return None
        else:
            print(len(temp), "points with label 1 in seed", file = sys.stderr)

        print("Building new", name, "model on", file, "with seed of", 
              seed_size, "points", file = sys.stderr)

        X_seed = seed.drop(columns=drop_column_list).to_numpy()
        y_seed = seed['Label'].to_numpy()

        clf = build_classifier(i) 

        model = ActiveLearner(
            estimator=clf,
            query_strategy=query_strategy,
            X_training=X_seed, y_training=y_seed
        )

        indices = []

    else:

        model = modeldata['model']
        indices = modeldata['past_file_indices']

    # build the pool

    num_significant = int(proportion * pool_size)

    if len(head) < num_significant:
        print("Too few significant points for the pool, including all", 
              len(head), "points", file = sys.stderr)
        num_significant = len(head)

    idx = rng.choice(tail.index, pool_size - num_significant, replace=False)
    pool0 = tail[tail.index.isin(idx)]

    idx = rng.choice(head.index, num_significant, replace=False)
    pool1 = head[head.index.isin(idx)]

    pool = pd.concat([pool0, pool1], ignore_index=True)

    temp = pool[pool['Label'] == 0]

    if len(temp) == 0:
        print("No points with label 0 in pool", file = sys.stderr)
        return None
    else:
        print(len(temp), "points with label 0 in pool", file = sys.stderr)

    temp = pool[pool['Label'] == 1]

    if len(temp) == 0:
        print("No points with label 1 in pool", file = sys.stderr)
        return None
    else:
        print(len(temp), "points with label 1 in pool", file = sys.stderr)

    # update the model which was either provided to this function or 
    # created from the seed, using the points from the pool

    print("Updating existing", name, "model on", file, "with", 
          points_from_pool, "points from pool of", pool_size, "points", 
          file = sys.stderr)

    X_pool = pool.drop(columns=drop_column_list).to_numpy()
    y_pool = pool['Label'].to_numpy()

    for j in range(points_from_pool):
        
        index, X_instance = model.query(X_pool)
        y_instance = y_pool[index]
    
        model.teach(X_instance, y_instance)

        X_pool = np.delete(X_pool, index, axis=0)
        y_pool = np.delete(y_pool, index, axis=0)

    indices.append(k)

    return { 'model' : model, 'past_file_indices' : indices,
             'fp_drift_detector' : build_drift_detector(detector),  
             'fn_drift_detector' : build_drift_detector(detector) }


def build_aloutl(k, i, modeldata=None):

    file = filelist[k]

    print("Training aloutl model on", file, "with rand", i, file = sys.stderr)

    training_set = pd.read_csv(file)

    rng = np.random.default_rng(i)

    scas0 = training_set[training_set['SCAS'] == 0]
    scas1 = training_set[training_set['SCAS'] == 1]

    # if the model has not been provided to this function, 
    # build the seed that will be used for creating a new model

    if modeldata == None:

        num_outliers = int(proportion * seed_size)

        if len(scas1) < num_outliers:
            print("Too few SCAS outliers for the seed, including all", 
                  len(scas1), "outliers", file = sys.stderr)
            num_outliers = len(scas1)

        idx = rng.choice(scas0.index, seed_size - num_outliers, replace=False)
        seed0 = scas0[scas0.index.isin(idx)]
        scas0 = scas0.drop(idx)

        idx = rng.choice(scas1.index, num_outliers, replace=False)
        seed1 = scas1[scas1.index.isin(idx)]
        scas1 = scas1.drop(idx)

        seed = pd.concat([seed0, seed1], ignore_index=True)

        temp = seed[seed['Label'] == 0]

        if len(temp) == 0:
            print("No points with label 0 in seed", file = sys.stderr)
            return None
        else:
            print(len(temp), "points with label 0 in seed", file = sys.stderr)

        temp = seed[seed['Label'] == 1]

        if len(temp) == 0:
            print("No points with label 1 in seed", file = sys.stderr)
            return None
        else:
            print(len(temp), "points with label 1 in seed", file = sys.stderr)

        print("Building new aloutl model on", file, "with seed of", 
              seed_size, "points", file = sys.stderr)

        X_seed = seed.drop(columns=drop_column_list).to_numpy()
        y_seed = seed['Label'].to_numpy()

        clf = build_classifier(i) 

        model = ActiveLearner(
            estimator=clf,
            query_strategy=query_strategy,
            X_training=X_seed, y_training=y_seed
        )

        indices = []

    else:

        model = modeldata['model']
        indices = modeldata['past_file_indices']

    # build the pool

    num_outliers = int(proportion * pool_size)

    if len(scas1) < num_outliers:
        print("Too few SCAS outliers for the pool, including all", 
              len(scas1), "outliers", file = sys.stderr)
        num_outliers = len(scas1)

    idx = rng.choice(scas0.index, pool_size - num_outliers, replace=False)
    pool0 = scas0[scas0.index.isin(idx)]

    idx = rng.choice(scas1.index, num_outliers, replace=False)
    pool1 = scas1[scas1.index.isin(idx)]

    pool = pd.concat([pool0, pool1], ignore_index=True)

    temp = pool[pool['Label'] == 0]

    if len(temp) == 0:
        print("No points with label 0 in pool", file = sys.stderr)
        return None
    else:
        print(len(temp), "points with label 0 in pool", file = sys.stderr)

    temp = pool[pool['Label'] == 1]

    if len(temp) == 0:
        print("No points with label 1 in pool", file = sys.stderr)
        return None
    else:
        print(len(temp), "points with label 1 in pool", file = sys.stderr)

    # update the model which was either provided to this function or 
    # created from the seed, using the points from the pool

    print("Updating existing aloutl model on", file, "with", 
          points_from_pool, "points from pool of", pool_size, "points", 
          file = sys.stderr)

    X_pool = pool.drop(columns=drop_column_list).to_numpy()
    y_pool = pool['Label'].to_numpy()

    for j in range(points_from_pool):
        
        index, X_instance = model.query(X_pool)
        y_instance = y_pool[index]
    
        model.teach(X_instance, y_instance)

        X_pool = np.delete(X_pool, index, axis=0)
        y_pool = np.delete(y_pool, index, axis=0)

    indices.append(k)

    return { 'model' : model, 'past_file_indices' : indices,
             'fp_drift_detector' : build_drift_detector(detector),  
             'fn_drift_detector' : build_drift_detector(detector) }


def detect_drift(modeldata, y_test, y_pred):

    if y_test.shape != y_pred.shape:
        raise ValueError("Input arrays have different shapes")

    if len(y_test) != len(y_pred):
        raise ValueError("Input arrays have different lengths")

    data = pd.DataFrame({ "Label": y_test, "Prediction": y_pred })

    pred_positives = data[data['Prediction'] == 1]

    drift_fp = 0
    
    for _,row in pred_positives.iterrows():
        
        if row['Label'] == 0:
            modeldata['fp_drift_detector'].update(1)
        else:
            modeldata['fp_drift_detector'].update(0)

        if modeldata['fp_drift_detector'].drift_detected:
            drift_fp += 1

    positives = data[data['Label'] == 1]

    drift_fn = 0
    
    for _,row in positives.iterrows():
            
        if row['Prediction'] == 0:
            modeldata['fn_drift_detector'].update(1)
        else:
            modeldata['fn_drift_detector'].update(0)

        if modeldata['fn_drift_detector'].drift_detected:
            drift_fn += 1

    return [drift_fp, drift_fn]


def print_results(prefix, timestamp, labels, results):

    precision_list = []
    recall_list = []
    f1_list = []

    for iter in range(iterations):

        precision = precision_score(labels[iter], results[iter])
        recall = recall_score(labels[iter], results[iter])
        f1 = f1_score(labels[iter], results[iter])

        precision_list.append(precision)
        recall_list.append(recall)
        f1_list.append(f1)

        tn, fp, fn, tp = confusion_matrix(labels[iter], results[iter]).ravel()

        print(prefix, timestamp, "Iteration", iter+1, 
              "Precision", precision, "Recall", recall, "F1-score", f1, 
              "TN", tn, "FP", fp, "FN", fn, "TP", tp,
              "Total", tn+fp+fn+tp)

    print(prefix, timestamp, "Avg_precision", np.mean(precision_list), 
                             "stddev", np.std(precision_list))
    print(prefix, timestamp, "Avg_recall", np.mean(recall_list), 
                             "stddev", np.std(recall_list))
    print(prefix, timestamp, "Avg_f1-score", np.mean(f1_list), 
                             "stddev", np.std(f1_list))


def nonnegint_check(arg):
    try:
        i = int(arg)
    except ValueError:    
        raise argparse.ArgumentTypeError("Must be an integer number")
    if i < 0:
        raise argparse.ArgumentTypeError("Argument must be >= 0")
    return i


def posint_check(arg):
    try:
        i = int(arg)
    except ValueError:    
        raise argparse.ArgumentTypeError("Must be an integer number")
    if i < 1:
        raise argparse.ArgumentTypeError("Argument must be >= 1")
    return i


def proportion_check(arg):
    try:
        f = float(arg)
    except ValueError:    
        raise argparse.ArgumentTypeError("Must be a floating point number")
    if f < 0 or f > 1:
        raise argparse.ArgumentTypeError("Argument must be >= 0 and <= 1")
    return f

############################## main program ##############################

start_time = process_time()

build_map = { "supervised" : build_supervised,
              "random": build_random,
              "al-trad" : build_altrad, 
              "al-ifanom" : partial(build_alrank, "al-ifanom"),
              "al-sigfreq" : partial(build_alrank, "al-sigfreq"),
              "al-sigmatch" : partial(build_alrank, "al-sigmatch"),
              "al-outlier" : build_aloutl }

# parse command line arguments

parser = argparse.ArgumentParser()

parser.add_argument("--datadir", required=True, help="data directory (files in the directory must follow the name format dataset-YYYY-MM.csv or dataset-YYYY-MM-DD.csv)")

parser.add_argument("--classifier", default="rf", choices=["rf", "cb", "ab", "lr", "svm", "knn"])

parser.add_argument("--strategy", required=True, choices=["supervised", "random", "al-trad", "al-ifanom", "al-sigfreq", "al-sigmatch", "al-outlier"])

parser.add_argument("--detector", required=True, choices=["adwin", "fhddm", "hddma"], help="concept drift detector type; the data from the first N timeslots is used to create the initial model (N is set with --window option), and the training data from the following timeslots is included only if concept drift detector fires for them")

parser.add_argument("--proportion", type=proportion_check, default=0.9, help="proportion of informative training data points in seed and pool ([0.0..1.0])")

parser.add_argument("--headsize", type=proportion_check, default=0.05, help="size of informativeness ranking head ([0.0..1.0])")

parser.add_argument("--window", default=1, type=nonnegint_check, help="window size (non-negative integer)")

parser.add_argument("--seed", type=posint_check, default=100, help="seed size (positive integer)")

parser.add_argument("--pool", type=nonnegint_check, help="pool size (non-negative integer), 0 denotes using entire training dataset as pool")

parser.add_argument("--samples", type=posint_check, help="number of points sampled from pool (positive integer)")

parser.add_argument("--iterations", type=posint_check, default=20, help="number of experiment iterations (positive integer)")

parser.add_argument("--randinit", type=nonnegint_check, default=0, help="initial random number generator seed (non-negative integer)")

args = parser.parse_args()

# set global variables from command line arguments

datadir = args.datadir

classifier = args.classifier

if classifier == "lr" or classifier == "knn" or classifier == "svm":
    drop_column_list.extend(['Proto', 'ExtPort', 'IntPort'])

strategy = args.strategy

if strategy in build_map:
    trainfunc = build_map[strategy]
else:
    print("No training function found for strategy", strategy, 
          file = sys.stderr)
    sys.exit(0)
    
detector = args.detector

proportion = args.proportion
headsize = args.headsize

seed_size = args.seed

iterations = args.iterations
randinit = args.randinit

# the following arguments are set to None if not provided on commandline

window = args.window

pool_size = args.pool
points_from_pool = args.samples

# check the dataset files

filelist = sorted(glob(datadir + '/dataset-*.csv'))

daily_regex = re.compile(r'dataset-(?P<date>(?P<month>[0-9]{4}-[0-9]{2})-(?P<dayofmonth>[0-9]{2}))\.csv$')
monthly_regex = re.compile(r'dataset-(?P<month>[0-9]{4}-[0-9]{2})\.csv$')

daily_files = list(filter(daily_regex.search, filelist))
monthly_files = list(filter(monthly_regex.search, filelist))

if len(daily_files) > 0 and len(monthly_files) > 0:

    print("Data file names must either follow the dataset-YYYY-MM.csv or dataset-YYYY-MM-DD.csv format and not both", file = sys.stderr)

    sys.exit(0)

elif len(daily_files) > 0:

    print("Data file names are following the dataset-YYYY-MM-DD.csv format", file = sys.stderr)

    if pool_size == None:
        pool_size = daily_pool_size

    if points_from_pool == None:
        points_from_pool = daily_points

    filelist = daily_files

    daily_datasets = True

elif len(monthly_files) > 0:

    print("Data file names are following the dataset-YYYY-MM.csv format", file = sys.stderr)

    if pool_size == None:
        pool_size = monthly_pool_size

    if points_from_pool == None:
        points_from_pool = monthly_points

    filelist = monthly_files

    daily_datasets = False

else:

    print("Data file names are not following the dataset-YYYY-MM.csv or dataset-YYYY-MM-DD.csv format", file = sys.stderr)

    sys.exit(0)


prediction_time = 0

rfmodel = []

training_needed = []

labels = []
results = []

daily_labels = []
daily_results = []

for iter in range(iterations):

    rfmodel.append(None)

    training_needed.append(False)

    labels.append(None)
    results.append(None)

    daily_labels.append(None)
    daily_results.append(None)


for k in range(len(filelist) - 1):

    training_file = filelist[k]
    test_file = filelist[k+1]

    if daily_datasets:

        timeinfo = daily_regex.search(test_file)
        timestamp = timeinfo.group('month')
        daily_timestamp = timeinfo.group('date')

        if k < len(filelist) - 2:

            next_file = filelist[k+2]

            timeinfo = daily_regex.search(next_file)

            if timeinfo.group('dayofmonth') == "01":
                monthly_report = True
            else:
                monthly_report = False

        else:
            monthly_report = True

    else:

        monthly_report = True

        timeinfo = monthly_regex.search(test_file)
        timestamp = timeinfo.group('month')

    print("Processing test file", test_file, file = sys.stderr)

    test_set = pd.read_csv(test_file)

    if strategy == "al-trad" or strategy == "al-ifanom" or strategy == "al-sigfreq" or strategy == "al-sigmatch" or strategy == "al-outlier":
        X_test = test_set.drop(columns=drop_column_list).to_numpy()
        y_test = test_set['Label'].to_numpy()
    else:
        X_test = test_set.drop(columns=drop_column_list)
        y_test = test_set['Label']

    iter = 0
    rand = randinit

    while iter < iterations:

        if k == 0:
            model = trainfunc(k, rand)

        elif k < window:
            model = trainfunc(k, rand, rfmodel[iter])

        elif training_needed[iter] == True:
            model = trainfunc(k, rand, rfmodel[iter])

        else:
            model = rfmodel[iter]

        if model == None:

            print("Training on", training_file, 
                  "failed for iteration", iter+1, file = sys.stderr)
            rand += 1
            continue

        else:
            print("After processing training data file", training_file, 
                  "for iteration", iter+1, "day", k + 1, ",",
                  "the model is using training data from", 
                  len(model['past_file_indices']), "files", file = sys.stderr)

        rfmodel[iter] = model

        time1 = process_time()

        result = rfmodel[iter]['model'].predict(X_test)

        time2 = process_time()

        prediction_time += (time2 - time1)

        if daily_datasets:

            daily_labels[iter] = y_test
            daily_results[iter] = result

            if results[iter] is None:
                labels[iter] = y_test
                results[iter] = result
            else:
                labels[iter] = np.concatenate((labels[iter], y_test))
                results[iter] = np.concatenate((results[iter], result))

        else:

            labels[iter] = y_test
            results[iter] = result

        training_needed[iter] = False

        drift_info = detect_drift(rfmodel[iter], y_test, result)

        if drift_info[0] > 0:

            print("After evaluating the model on test file", test_file,
                  "for iteration", iter+1,
                  "FP drift was detected", drift_info[0], "times",
                  file = sys.stderr)

            training_needed[iter] = True

        if drift_info[1] > 0:

            print("After evaluating the model on test file", test_file,
                  "for iteration", iter+1,
                  "FN drift was detected", drift_info[1], "times",
                  file = sys.stderr)

            training_needed[iter] = True

        iter += 1
        rand += 1

    if daily_datasets:

        print_results("Day", daily_timestamp, daily_labels, daily_results)

    if monthly_report:

        print_results("Month", timestamp, labels, results)

        for iter in range(iterations):

            labels[iter] = None
            results[iter] = None


end_time = process_time()

print("Total CPU time: ", end_time - start_time, file = sys.stderr)
print("Prediction CPU time: ", prediction_time, file = sys.stderr)
