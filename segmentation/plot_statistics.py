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
        plt.errorbar(rs, means[0], yerr=deviations[0], c="green", fmt="--o", label="c=4")
        plt.errorbar(rs, means[1], yerr=deviations[1], c="yellow", fmt="--o", label="c=16")
        plt.errorbar(rs, means[2], yerr=deviations[2], c="red", fmt="--o", label="c=32")
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
        plt.errorbar(cs, means[0], yerr=deviations[0], c="red", fmt="--o", label="r=4")
        plt.errorbar(cs, means[1], yerr=deviations[1], c="orange", fmt="--o", label="r=8")
        plt.errorbar(cs, means[2], yerr=deviations[2], c="yellow", fmt="--o", label="r=16")
        plt.errorbar(cs, means[3], yerr=deviations[3], c="green", fmt="--o", label="r=32")
        #plt.show()
    legend = plt.legend(loc="upper center", shadow=True)


if __name__ == "__main__":
    objs = extract_from_log(True)
    plot_varying("r", objs)
    plot_varying("c", objs)
    plt.show()