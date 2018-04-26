import os
import numpy as np
import shutil

from logger import Logger as Log

def load_result_file(file):
    arr = np.load(file)

    D = dict([(k, arr[k]) for k in arr.keys()])

    return D

def load_config(cfgfile):
    """ Parses a configuration file """

    cfgf = open(cfgfile,'r')
    cfg = {}
    for l in cfgf:
        ps = [p.strip() for p in l.split(':')]
        if len(ps)==2:
            try:
                cfg[ps[0]] = float(ps[1])
            except ValueError:
                cfg[ps[0]] = ps[1]
                if cfg[ps[0]] == 'False':
                    cfg[ps[0]] = False
                elif cfg[ps[0]] == 'True':
                    cfg[ps[0]] = True
    cfgf.close()
    return cfg

def load_single_result(result_dir):
    if Log.VERBOSE:
        print 'Loading %s...' % result_dir

    config_path = '%s/config.txt' % result_dir
    has_config = os.path.isfile(config_path)
    if not has_config:
        print 'WARNING: Could not find config.txt for %s. Skipping.' % os.path.basename(result_dir)
        config = None
    else:
        config = load_config(config_path)

    train_path = '%s/result.npz' % result_dir
    test_path = '%s/result.test.npz' % result_dir

    has_test = os.path.isfile(test_path)

    try:
        train_results = load_result_file(train_path)
    except:
        'WARNING: Couldnt load result file. Skipping'
        return None

    n_rep = np.max([config['repetitions'], config['experiments']])

    if len(train_results['pred'].shape) < 4 or train_results['pred'].shape[2] < n_rep:
        print 'WARNING: Experiment %s appears not to have finished. Skipping.' % result_dir
        return None

    if has_test:
        test_results = load_result_file(test_path)
    else:
        test_results = None

    return {'train': train_results, 'test': test_results, 'config': config}

def load_single_reps(rep_dir):
    if Log.VERBOSE:
        print 'Loading %s...' % rep_dir

    train_path = '%s/reps.npz' % rep_dir
    test_path = '%s/reps.test.npz' % rep_dir

    has_test = os.path.isfile(test_path)

    try:
        train_reps = load_result_file(train_path)

    except:
        'WARNING: Couldnt load reps file. Skipping'
        return None
    
    if has_test:
        test_reps = load_result_file(test_path)
    else:
        test_reps = None

    return {'train': train_reps, 'test': test_reps}

def get_exp_dirs(output_dir):
    files = ['%s/%s' % (output_dir, f) for f in os.listdir(output_dir)]
    exp_dirs = [f for f in files if os.path.isdir(f)
                    if os.path.isfile('%s/result.npz' % f)]
    return exp_dirs

def load_results(output_dir):
    if Log.VERBOSE:
        print 'Loading results from %s...' % output_dir

    ''' Detect results structure '''
    # Single result
    if os.path.isfile('%s/results.npz' % output_dir):
        #@TODO: Implement
        pass

    # Multiple results
    exp_dirs = get_exp_dirs(output_dir)

    if Log.VERBOSE:
        print 'Found %d experiment configurations.' % len(exp_dirs)

    # Load each result folder
    results = []
    for dir in exp_dirs:
        dir_result = load_single_result(dir)
        if dir_result is not None:
            results.append(dir_result)

    return results

def del_nan_results(output_dir, del_idx):
    if len(del_idx) != 0:
        if Log.VERBOSE:
            print 'Deleting NaN results from %s...' % output_dir

        ''' Detect results structure '''
        # Single result
        if os.path.isfile('%s/results.npz' % output_dir):
            #@TODO: Implement
            pass

        # Multiple results
        files = ['%s/%s' % (output_dir, f) for f in os.listdir(output_dir)]
        exp_dirs = [f for f in files if os.path.isdir(f)
                        if os.path.isfile('%s/result.npz' % f)]

        # Delete all NaN result folders
        for ind, dir in enumerate(exp_dirs):
            if ind in del_idx:
                print 'Deleting %d...' % (ind+1)
                shutil.rmtree(dir)

    return

def load_data(datapath):
    """ Load dataset """
    arr = np.load(datapath)
    xs = arr['x']

    HAVE_TRUTH = False
    SPARSE = False

    if len(xs.shape)==1:
        SPARSE = True

    ts = arr['t']
    yfs = arr['yf']
    try:
        es = arr['e']
    except:
        es = None
    try:
        ate = np.mean(arr['ate'])
    except:
        ate = None
    try:
        ymul = arr['ymul'][0,0]
        yadd = arr['yadd'][0,0]
    except:
        ymul = 1
        yadd = 0
    try:
        ycfs = arr['ycf']
        mu0s = arr['mu0']
        mu1s = arr['mu1']
        HAVE_TRUTH = True
    except:
        print 'Couldn\'t find ground truth. Proceeding...'
        ycfs = None; mu0s = None; mu1s = None

    # import csv
    # with open('tt.csv', 'wb') as csvfile:
    #     writer = csv.writer(csvfile)
    #     writer.writerows(ts)
    # with open('yf.csv', 'wb') as csvfile:
    #     writer = csv.writer(csvfile)
    #     writer.writerows(yfs)

    # i = 10
    # j = 21
    # print yfs[i,j], ycfs[i,j], ts[i,j], mu1s[i,j], mu0s[i,j]
    # exit()

    # ate_trn=[]
    # ateHat_trn=[]
    # for exp in range(100):
    #     ate_trn.append(np.mean(mu1s[:,exp] - mu0s[:,exp]))
    #     ateHat_trn.append(np.sum(np.multiply(yfs[:,exp], ts[:,exp]))/np.sum(ts[:,exp]) - np.sum(np.multiply(yfs[:,exp], 1-ts[:,exp]))/np.sum(1-ts[:,exp]))
    # print np.mean(ate_trn), np.std(ate_trn)
    # print np.mean(ateHat_trn), np.std(ateHat_trn)
    # print ate
    
    data = {'x':xs, 't':ts, 'e':es, 'yf':yfs, 'ycf':ycfs, \
            'mu0':mu0s, 'mu1':mu1s, 'ate':ate, 'YMUL': ymul, \
            'YADD': yadd, 'ATE': ate.tolist(), 'HAVE_TRUTH': HAVE_TRUTH, \
            'SPARSE': SPARSE}

    return data

def load_reps(output_dir):
    if Log.VERBOSE:
        print 'Loading representations from %s...' % output_dir

    ''' Detect results structure '''
    # Single result
    if os.path.isfile('%s/reps.npz' % output_dir):
        # Multiple results
        files = ['%s/%s' % (output_dir, f) for f in os.listdir(output_dir)]
        exp_dirs = [f for f in files if os.path.isdir(f)
                        if os.path.isfile('%s/reps.npz' % f)]

        if Log.VERBOSE:
            print 'Found %d experiment configurations.' % len(exp_dirs)

        # Load each result folder
        reps = []
        for dir in exp_dirs:
            dir_reps = load_single_reps(dir)
            if dir_reps is not None:
                reps.append(dir_reps)

        return reps
    
    else:
        return None
