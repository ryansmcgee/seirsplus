## Support to run

import sys

import os
p = os.path.dirname(os.path.abspath(__file__))
sys.path.append(p)


from models import *
from .networks import *
from sim_loops import *
from utilities import *
import collections

import multiprocessing as mp
import pickle

import networkx
import argparse

def pack(f,*args,getelement = "", **kwds):
    """"Pack a function evaluation into a symbolic representations we can easily 'pickle' """
    return ("eval"+str(getelement),f.__name__, args,kwds)

def packfirst(f,*args,**kwds):
    return pack(f,*args,getelement=0,**kwds)

def unpack(O):
    """Unpack and evaluate expression"""
    K = len("eval") # this is four of course but just in case we change things later
    if isinstance(O,tuple) and (len(O)>1) and (O[0][:K]=="eval"):
        f = globals()[O[1]]
        res =  f(*O[2],**O[3])
        if len(O[0]) > K:
            i = int(O[0][K:])
            return res[i]
        return res
    return O

def run(model_params,run_params, extra, keep_model = False):
    """Run an execution with given parameters"""
    MP = { key: unpack(val) for key,val in model_params.items() }
    RP = { key: unpack(val) for key,val in run_params.items() }
    if not MP['G_Q']:
        MP['G_Q'] = networkx.classes.function.create_empty_copy(MP["G"]) # default quarantine graph is empty 
    desc=  dict(extra)
    desc.update({key : str(val) for key,val in model_params.items() })
    desc.update({key : str(val) for key,val in run_params.items() })
    model = ExtSEIRSNetworkModel(**MP)
    hist = collections.OrderedDict()
    run_tti_sim(model, history=hist, **RP)
    df, summary =  hist2df(hist,**desc)
    m = model if keep_model else None
    return df,summary, m

def run_(T):
    # single parameter version of run - returns only summary with an additional "model"
    T[1]["verbose"] = False # no printouts when running in parallel
    df, summary,model =  run(T[0],T[1],T[2],T[3])
    summary["model"] = model
    return summary

def parallel_run(to_do, realizations= 1, keep_in = 0):
    """Get list of pairs (MP,RP, extra) of model parameters to run,  run each given number of realizations in parallel
    Among all realizations we keep.
    Extra is extra fields for logging and grouping purposes"""
    print("Preparing list to run", flush=True)
    run_list = [(M,R,E, r < keep_in) for r in range(realizations) for M,R,E in to_do]
    print(f"We have {mp.cpu_count()} CPUs")
    pool = mp.Pool(mp.cpu_count())
    print(f"Starting execution of {len(run_list)} runs", flush=True)
    # rows = list(map(single_exec, run_list))
    rows = list(pool.map(run_, run_list))
    print("done", flush=True)
    pool.close()
    df = pd.DataFrame(rows)
    return df

def save_to_file(L,filename = 'torun.pickle'):
    """Save list of (MP,RP) pairs to run"""
    with open(filename, 'wb') as handle:
        pickle.dump(L, handle, protocol=pickle.HIGHEST_PROTOCOL)

def read_from_file(prefix= 'data'):
    i = 1
    chunks = []
    while os.path.exists(prefix+"_"+str(i)+".zip"):
        print("Loading chunk "+ str(i), flush=True)
        chunks.append(pd.read_pickle(prefix+"_"+str(i)+".zip"))
        i += 1
    return pd.concat(chunks)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--torun", default = "torun.pickle", help="File name of list to run")
    parser.add_argument("--realizations", default = 5, type=int, help="Number of realizations")
    parser.add_argument("--savename", default="data", help="File name to save resulting data (with csv and zip extensions)")
    args = parser.parse_args()
    print("Arguments:")
    for arg in vars(args):
        print(arg,":", getattr(args, arg))
    print("Loading torun", flush=True)
    with open(args.torun, 'rb') as handle:
        torun = pickle.load(handle)
    print("Loaded", flush=True)
    data = parallel_run(torun, args.realizations)
    print("Saving csv", flush=True)
    data.to_csv(args.savename+'.csv')
    chunk_size = 100000
    print("Saving split parts", flush=True)
    i = 1
    for start in range(0, data.shape[0], chunk_size):
        print(f"Saving pickle {i}", flush = True)
        temp = data.iloc[start:start + chunk_size]
        fname = args.savename+"_"+str(i)+".zip"
        temp.to_pickle(fname)
        i += 1
    fname = args.savename + "_" + str(i) + ".zip"
    if os.path.exists(fname): # so there is no confusion that this was the last part
        os.remove(fname)
    print("Done", flush=True)




if __name__ == "__main__":
    # execute only if run as a script
    main()

