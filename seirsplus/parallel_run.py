## Support to run

import sys

import os
p = os.path.dirname(os.path.abspath(__file__))
sys.path.append(p)


from models import *
from networks import *
from sim_loops import *
from utilities import *
import collections

import multiprocessing as mp
import pickle

import networkx



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
            i = int(O[K:])
            return res[i]
        return res
    return O

def run(model_params,run_params, keep_model = False):
    """Run an execution with given parameters"""
    MP = { key: unpack(val) for key,val in model_params.items() }
    RP = { key: unpack(val) for key,val in run_params.items() }
    if not MP['G_Q']:
        MP['G_Q'] = networkx.classes.function.create_empty_copy(MP["G"]) # default quarantine graph is empty

    desc=  {key : str(val) for key,val in model_params.items() }
    desc.update({key : str(val) for key,val in run_params.items() })
    model = ExtSEIRSNetworkModel(**MP)
    hist = collections.OrderedDict()
    run_tti_sim(model, hist=hist, **RP)
    df, sum =  hist2df(hist,desc)
    return df,sum, model if keep_model else None

def run_(T):
    # single parameter version of run - returns only summary with an additional "model"
    df, sum,model =  run(T[0],T[1],T[2])
    T[1]["verbose"] = False # no printouts when running in parallel
    sum["model"] = model
    return sum

def parallel_run(to_do, realizations= 1, keep_in = 0):
    """Get list of triples (MP,RP) of model parameters to run,  run each given number of realizations in parallel
    Among all realizations we keep"""
    print("Preparing list to run", flush=True)
    run_list = [(T[0],T[1], r < keep_in) for r in range(realizations) for T in to_do]
    print(f"We have {mp.cpu_count()} CPUs")
    pool = mp.Pool(mp.cpu_count())
    print(f"Starting execution of {len(run_list)} runs", flush=True)
    # rows = list(map(single_exec, run_list))
    rows = list(pool.map(run_, run_list))
    print("done", flush=True)
    pool.close()
    df = pd.DataFrame(rows)
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--torun", default = "torun.pickle", help="File name of list to run")
    parser.add_argument("--realizations", default = 5, help="Number of realizations")
    parser.add_argument("--savename", default="data", help="File name to save resulting data (with csv and zip extensions)")
    args = parser.parse_args()
    print("Loading torun", flush=True)
    with open(args.torun, 'rb') as handle:
        torun = pickle.load(handle)
    print("Loaded", flush=True)
    data = parallel_run(torun, args.realizations)
    print("Saving csv", flush=True)
    data.to_csv(args.savename+'.csv')
    chunk_size = 100000
    if data.shape[0] > chunk_size:
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
    else:
        print("Saving data")
        data.to_pickle(args.savename+".zip")
    print("Done", flush=True)




if __name__ == "__main__":
    # execute only if run as a script
    main()

