import json
import os
import matplotlib.pyplot as plt
import numpy as np

OUTFILE = "statistics.json"

def extract_from_log(save=False):
    logs = ["output_pic1/log.txt", "output_pic2/log.txt", "output_pic3/log.txt"]
    objs = []
    for idx, log in enumerate(logs):
        with open(log, "r") as lf:
            obj = {}
            for line in lf:
                if "time" in obj.keys():
                    objs.append(obj)
                    obj = {}
                
                tokens = line.split()

                if "Processing" in line:
                    obj["r"] = int(tokens[3].replace("r=", "").replace(",", ""))
                    obj["c"] = int(tokens[4].replace("c=", "").replace(",", ""))
                    obj["spatial"] = True if tokens[8].lower() == "true" else False
                    obj["pic"] = tokens[1].replace(".\\", "")
                elif "Skipped" in line:
                    obj["skipped"] = float(tokens[1].replace("%", ""))/100.0
                elif "Time" in line:
                    obj["time"] = float(tokens[2])
            objs.append(obj)
    
    if save:
        with open(OUTFILE, "w") as outf:
            json.dump(objs, outf, indent=2)

    return objs

def plot_varying(param, objs):
    rs = [4, 8, 16, 32]
    cs = [4, 8, 16]
    dims = ["3D", "5D"]
    means = []
    deviations = []
    if param == "r":
        plt.figure()
        plt.xlabel("r")
        plt.ylabel("Seconds")
        for c in cs:
            static_c = [x for x in objs if x["c"] == c]
            means_c = []
            deviations_c = []
            for r in rs:
                static_c_r = [x for x in static_c if x["r"] == r]
                times = [x["time"] for x in static_c_r]
                means_c.append(np.mean(times))
                deviations_c.append(np.std(times))
            means.append(means_c)
            deviations.append(deviations_c)
        plt.errorbar(rs, means[0], yerr=deviations[0], c="green", fmt="--o", label="c=4", elinewidth=.8, capsize=4)
        plt.errorbar(rs, means[1], yerr=deviations[1], c="yellow", fmt="--o", label="c=16", elinewidth=.8, capsize=4)
        plt.errorbar(rs, means[2], yerr=deviations[2], c="red", fmt="--o", label="c=32", elinewidth=.8, capsize=4)
        #plt.show()
    elif param == "c":
        plt.figure()
        plt.xlabel("c")
        plt.ylabel("Seconds")
        for r in rs:
            static_r = [x for x in objs if x["r"] == r]
            means_r = []
            deviations_r = []
            for c in cs:
                static_r_c = [x for x in static_r if x["c"] == c]
                times = [x["time"] for x in static_r_c]
                means_r.append(np.mean(times))
                deviations_r.append(np.std(times))
            means.append(means_r)
            deviations.append(deviations_r)
        plt.errorbar(cs, means[0], yerr=deviations[0], c="red", fmt="--o", label="r=4", elinewidth=.8, capsize=4)
        plt.errorbar(cs, means[1], yerr=deviations[1], c="orange", fmt="--o", label="r=8", elinewidth=.8, capsize=4)
        plt.errorbar(cs, means[2], yerr=deviations[2], c="yellow", fmt="--o", label="r=16", elinewidth=.8, capsize=4)
        plt.errorbar(cs, means[3], yerr=deviations[3], c="green", fmt="--o", label="r=32", elinewidth=.8, capsize=4)
        #plt.show()
    elif param == "d":
        pics = [1, 2, 3]
        plt.figure()
        plt.xlabel("Picture")
        plt.xticks(pics)
        plt.ylabel("Seconds")
        means.append([])
        means.append([])
        deviations.append([])
        deviations.append([])
        for pic in pics:
            static_pic = [x for x in objs if x["pic"] == "pic{}.jpg".format(pic)]
            dim3 = [x["time"] for x in static_pic if x["spatial"] == False]
            dim5 = [x["time"] for x in static_pic if x["spatial"] == True]
            means[0].append(np.mean(dim3))
            means[1].append(np.mean(dim5))
            deviations[0].append(np.std(dim3))
            deviations[1].append(np.std(dim5))
        plt.errorbar(pics, means[0], yerr=deviations[0], c="blue", fmt="--o", label="3D", elinewidth=.8, capsize=4)
        plt.errorbar(pics, means[1], yerr=deviations[1], c="red", fmt="--o", label="5D", elinewidth=.8, capsize=4)
    legend = plt.legend(loc="upper center", shadow=True)

def plot_opt_vs_unopt():
    with open("statistics_opt.json", "r") as optf:
        opts = json.load(optf)
    with open("statistics_unopt.json", "r") as unoptf:
        unopts = json.load(unoptf)

    pics = [1, 2, 3]
    means = [[], []]
    dev = [[], []]
    plt.figure()
    plt.xlabel("Picture")
    plt.xticks(pics)
    plt.ylabel("Seconds")
    for pic in pics:
        opt = [x["time"] for x in [y for y in opts if y["pic"] == "pic{}.jpg".format(pic)]]
        unopt = [x["time"] for x in [y for y in unopts if y["pic"] == "pic{}.jpg".format(pic)]]
        means[0].append(np.mean(opt))
        means[1].append(np.mean(unopt))
        dev[0].append(np.std(opt))
        dev[1].append(np.std(unopt))

    print(np.array(means[0])/np.array(means[1]))
    #plt.errorbar(pics, means[0], yerr=dev[0], c="blue", fmt="--o", label="optimized", elinewidth=.8, capsize=4)
    #plt.errorbar(pics, means[1], yerr=dev[1], c="red", fmt="--o", label="unoptimized", elinewidth=.8, capsize=4)
    #legend = plt.legend(loc="upper center", shadow=True)

if __name__ == "__main__":
    #objs = extract_from_log()
    #plot_varying("r", objs)
    #plot_varying("c", objs)
    #plot_varying("d", objs)
    plot_opt_vs_unopt()
    #plt.show()