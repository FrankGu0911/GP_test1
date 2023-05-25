import os,sys,shutil
from multiprocessing import Pool
from tqdm import tqdm


def process(route):
    if os.path.exists(os.path.join(route, "rgb_full")):
        shutil.rmtree(os.path.join(route, "rgb_full"))
    if os.path.exists(os.path.join(route, "measurements_full")):
        shutil.rmtree(os.path.join(route, "measurements_full"))

if __name__=='__main__':
    list_file = sys.argv[1]
    routes = []
    for line in open(list_file, "r").readlines():
        routes.append(line.strip())
    with Pool(24) as p:
        r = list(tqdm(p.imap(process, routes), total=len(routes)))