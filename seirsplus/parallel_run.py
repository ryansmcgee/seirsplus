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
import random

import pickle
import inspect
import networkx
import argparse
import string
from networkx import *


try:
    from p_tqdm import p_umap # https://github.com/swansonk14/p_tqdm
except ImportError:
    print("Please install p_tqdm package via pip install p_tqdm")
    print("Preparing code for parallel run will work, but running the script will not")
    def p_umap(*L):
        raise Exception("Package p_tqdm not found")



class Defer:
    """Class for deferring computation
    Defer(f,positional and keyword arguments) stores f's name and the arguments for later evaluation"""
    def __init__(self,f,*args,**kwds):
        self.f_name = f.__name__
        self.args = args
        self.kwds = kwds

    def __str__(self):
        res = self.f_name+"("
        res += ", ".join([str(a) for a in self.args])
        if self.args and self.kwds:
            res+=", "
        if self.kwds:
            res += ", ".join([str(k)+"="+str(v) for k,v in self.kwds.items()])
        return res

    def eval(self):
        f = globals()[self.f_name]
        args_ = [unpack(a) for a in self.args]
        kwds_ = {k:unpack(v) for k,v in self.kwds.items()}
        return f(*args_,**kwds_)

def unpack(O):
    if  "Defer" in type(O).__name__:
        return O.eval()
    return O


def generate_workplace_contact_network_(*args,**kwds):
    """Helper function to produce graph + isolation groups"""
    G, cohorts, teams = generate_workplace_contact_network(*args,**kwds)
    return G, list(teams.values())

def generate_workplace_contact_network_deferred(*args,**kwds):
    """Returns deferred execution of functoin to generate return graph and isolation groups"""
    return Defer(generate_workplace_contact_network_,*args,**kwds)



def run(params_, keep_model = False):
    """Run an execution with given parameters"""
    params = { key: unpack(val) for key,val in params_.items() }
    # replace key a value pair of form (k1,k2,k3):(v1,v2,v3) with k1:v1,k2:v2,k3:v3 etc..
    # useful if several keys depend on the same deferred computation
    for key in list(params.keys()):
        if isinstance(key,tuple):
            L = params[key]
            if not isinstance(L,(list,tuple)):
                raise Exception("L is of type " +str(type(L)) + " and not tuple (L= " + str(L) +")")
            if len(L) != len(key):
                raise Exception("Key" + str(key) + "should have same length as value" + str(L))
            for i,subkey in enumerate(key):
                params[subkey] = L[i]
            del params[key]

    if ('G_Q' not in params) or (not params['G_Q']):
        params['G_Q'] = networkx.classes.function.create_empty_copy(params["G"]) # default quarantine graph is empty
    desc= { "run_id" : random.choices(string.ascii_lowercase,k=8) } # unique id to help in aggregating
    model_params = {}
    run_params = {}
    for k, v in params.items():
        if k in inspect.signature(ExtSEIRSNetworkModel).parameters:
            model_params[k] = v
        elif k in inspect.signature(run_tti_sim).parameters:
            run_params[k] = v
        else:
            desc[k] = v
    desc.update({key : make_compact(val) for key,val in model_params.items() })
    desc.update({key : make_compact(val) for key,val in run_params.items() })
    if ("verbose" in params_) and params_["verbose"]:
        print("Parameters :", desc)
    model = ExtSEIRSNetworkModel(**model_params)
    hist = collections.OrderedDict()
    run_tti_sim(model, history=hist, **run_params)
    df, summary =  hist2df(hist,**desc)
    m = model if keep_model else None
    return df,summary, m

def run_(T):
    # single parameter version of run - returns only summary with an additional "model"
    T[0]["verbose"] = False # no printouts when running in parallel
    df, summary,model =  run(T[0],T[1])
    summary["model"] = model
    return summary.to_dict()

def parallel_run(to_do, realizations= 1, keep_in = 0):
    """Get list of dictionaries of model and run parameters to run,  run each given number of realizations in parallel
    Among all realizations we keep."""
    print("Preparing list to run", flush=True)
    run_list = [(D, r < keep_in) for r in range(realizations) for D in to_do]
    #print(f"We have {mp.cpu_count()} CPUs")
    #pool = mp.Pool(mp.cpu_count())
    print("Starting execution of " +str(len(run_list)) +" runs", flush=True)
    rows = list(p_umap(run_,run_list))
    #rows = list(pool.map(run_, run_list))
    print("done", flush=True)
    df = pd.DataFrame(rows)
    return df

def save_to_file(L,filename = 'torun.pickle'):
    """Save list of  parameter dictionaries  to run"""
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
        print("Saving pickle " + str(i), flush = True)
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

