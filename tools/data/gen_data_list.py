import os,sys,argparse

parser = argparse.ArgumentParser(description="Generate Datalist For carla dataset")
parser.add_argument(
    '-n',
    '--withnum',
    action='store_true',
    default=False,
    help='Generate Datalist with frame numbers, like "weather-7/data/routes_town06_long_w7_11_28_18_28_35/ 1062"'
    )
parser.add_argument(
    '-r',
    '--dataroot',
    required=True,
    help='Root directory of carla dataset',
    type=str
    )

def GenDataList(dataroot:str,withnum:bool=False):
    if withnum:
        routes = []
        for i in range(14):
            subs = os.listdir(os.path.join(data_root, "weather-%d" % i, "data"))
            for sub in subs:
                route_dir = os.path.join(data_root, "weather-%d" % i, "data", sub)
                if not os.path.isdir(route_dir):
                    continue
                frames = len(os.listdir(os.path.join(route_dir, "measurements_full")))
                routes.append((route_dir,frames))
        datafile = os.path.join(data_root,"dataset_index.txt")
        with open(datafile, "w") as f:
            for route in routes:
                f.write("%s %d\n" % route)
    else:
        routes = []
        for i in range(14):
            subs = os.listdir(os.path.join(data_root, "weather-%d" % i, "data"))
            for sub in subs:
                if not os.path.isdir(os.path.join(data_root, "weather-%d" % i, "data", sub)):
                    continue
                routes.append(os.path.join(data_root, "weather-%d" % i, "data", sub))
        with open("routes_list.txt", "w") as f:
            for route in routes:
                f.write("%s\n" % route)

if __name__ == '__main__':
    args = parser.parse_args()
    data_root = args.dataroot
    withnum = args.withnum
    GenDataList(data_root,withnum)
