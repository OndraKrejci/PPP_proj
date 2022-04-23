#! /usr/bin/env python3
import csv
import matplotlib.pyplot as plt

suffixes=["_p2p", "_p2p_seq_IO", "_p2p_par_IO" ,"_rma", "_rma_seq_IO", "_rma_par_IO"] 

ploth_width  = 4 
ploth_height = 3
legend_font_size = 6

x_axis_points = [1, 16, 32, 64, 128]
#x_axis_points = [1,2,4]
data = []
with open('run_full_mpi_2d_out.csv', newline='') as csvfile:
    csvreader = csv.reader(csvfile, delimiter=';')
    data = list(csvreader)

# print(data)
# exit()

data_dict = {}
headerlen = 1

for i in range(0,len(data)-headerlen):
    suffix = suffixes[i % 6]
    row = data[i + headerlen]
    datakey = str(row[4]) + suffix
    iterationtime = float(row[13])
    if not datakey in data_dict.keys():
        data_dict[datakey] = []
    data_dict[datakey].append(iterationtime)

for comm_type in ['p2p', 'rma']:

    # scaling
    plt.figure(figsize=[ploth_width, ploth_height])
    for key, data_series in data_dict.items():
        plot_marker = ""
        if key.find('256')  != -1:
            plot_marker = "o"
        if key.find('512')  != -1:
            plot_marker = "v"
        if key.find('1024') != -1:
            plot_marker = "s"
        if key.find('2048') != -1:
            plot_marker = "X"
        if key.find('4096') != -1:
            plot_marker = "d"
        
        if key.find(comm_type) != -1:
            scaling_values      = data_series
            plt.plot(x_axis_points[0:len(scaling_values)], scaling_values, marker=plot_marker, label=key)

    plt.xlabel('Number of cores')
    plt.ylabel('Iteration time [ms]')
    plt.yscale('log')
    plt.xscale('log')
    plt.legend( prop={'size': legend_font_size})
    plt.grid(True)
    plt.gca().set_position([0, 0, 1, 1])
    plt.savefig("ppp_scaling_mpi_" + comm_type + ".svg")

    # speedup
    plt.figure(figsize=[ploth_width, ploth_height])
    for key, data_series in data_dict.items():
        plot_marker = ""
        if key.find('256')  != -1:
            plot_marker = "o"
        if key.find('512')  != -1:
            plot_marker = "v"
        if key.find('1024') != -1:
            plot_marker = "s"
        if key.find('2048') != -1:
            plot_marker = "X"
        if key.find('4096') != -1:
            plot_marker = "d"
        if key.find(comm_type) != -1:
            scaling_values      = data_series
            speedup_values      = [data_series[0]/data_point for data_point in data_series]

            plt.plot(x_axis_points[0:len(speedup_values)], speedup_values, marker=plot_marker, label=key)

    plt.legend(prop={'size': legend_font_size})
    plt.xlabel('Number of cores')
    plt.ylabel('Speedup')
    plt.grid(True)
    plt.gca().set_position([0, 0, 1, 1])
    plt.savefig("ppp_speedup_mpi_" + comm_type + ".svg")

    # efficiency
    plt.figure(figsize=[ploth_width, ploth_height])
    for key, data_series in data_dict.items():
        plot_marker = ""
        if key.find('256')  != -1:
            plot_marker = "o"
        if key.find('512')  != -1:
            plot_marker = "v"
        if key.find('1024') != -1:
            plot_marker = "s"
        if key.find('2048') != -1:
            plot_marker = "X"
        if key.find('4096') != -1:
            plot_marker = "d"
        if key.find(comm_type) != -1:
            scaling_values      = data_series
            speedup_values      = [data_series[0]/data_point for data_point in data_series]
            efficiency_values = []
            for i in range(0,len(speedup_values)):
                efficiency_values.append(speedup_values[i]/x_axis_points[i]*100)

            plt.plot(x_axis_points[0:len(efficiency_values)], efficiency_values, marker=plot_marker, label=key)

            
    plt.xlabel('Number of cores')
    plt.ylabel('Efficiency [%]')
    plt.legend(prop={'size': legend_font_size})
    plt.grid(True)
#plt.show()
    plt.gca().set_position([0, 0, 1, 1])
    plt.savefig("ppp_efficiency_mpi_" + comm_type + ".svg")



x_axis_points = [1, 2*9, 4*9, 8*9, 16*9, 32*9]
#x_axis_points = [1,2,4]
data = []
with open('run_full_hybrid_2d_out.csv', newline='') as csvfile:
    csvreader = csv.reader(csvfile, delimiter=';')
    data = list(csvreader)


data_dict = {}


for i in range(0,len(data)-headerlen):
    suffix = suffixes[i % 6]
    row = data[i + headerlen]
    datakey = str(row[4]) + suffix
    iterationtime = float(row[13])
    if not datakey in data_dict.keys():
        data_dict[datakey] = []
        data_dict[datakey].append(iterationtime)
    else:
        data_dict[datakey].append(iterationtime)


for comm_type in ['p2p', 'rma']:

    # scaling
    plt.figure(figsize=[ploth_width, ploth_height])
    for key, data_series in data_dict.items():
        plot_marker = ""
        if key.find('256')  != -1:
            plot_marker = "o"
        if key.find('512')  != -1:
            plot_marker = "v"
        if key.find('1024') != -1:
            plot_marker = "s"
        if key.find('2048') != -1:
            plot_marker = "X"
        if key.find('4096') != -1:
            plot_marker = "d"
        
        if key.find(comm_type) != -1:
            scaling_values      = data_series
            plt.plot(x_axis_points[0:len(scaling_values)], scaling_values, marker=plot_marker, label=key)

    plt.xlabel('Number of cores')
    plt.ylabel('Iteration time [ms]')
    plt.yscale('log')
    plt.xscale('log')
    plt.legend(prop={'size': legend_font_size})
    plt.grid(True)
    plt.gca().set_position([0, 0, 1, 1])
    plt.savefig("ppp_scaling_hybrid_" + comm_type + ".svg")

    # speedup
    plt.figure(figsize=[ploth_width, ploth_height])
    for key, data_series in data_dict.items():
        plot_marker = ""
        if key.find('256')  != -1:
            plot_marker = "o"
        if key.find('512')  != -1:
            plot_marker = "v"
        if key.find('1024') != -1:
            plot_marker = "s"
        if key.find('2048') != -1:
            plot_marker = "X"
        if key.find('4096') != -1:
            plot_marker = "d"
        if key.find(comm_type) != -1:
            scaling_values      = data_series
            speedup_values      = [data_series[0]/data_point for data_point in data_series]

            plt.plot(x_axis_points[0:len(speedup_values)], speedup_values, marker=plot_marker, label=key)

    plt.legend(prop={'size': legend_font_size})
    plt.xlabel('Number of cores')
    plt.ylabel('Speedup')
    plt.grid(True)
    plt.gca().set_position([0, 0, 1, 1])
    plt.savefig("ppp_speedup_hybrid_" + comm_type + ".svg")

    # efficiency
    plt.figure(figsize=[ploth_width, ploth_height])
    for key, data_series in data_dict.items():
        plot_marker = ""
        if key.find('256')  != -1:
            plot_marker = "o"
        if key.find('512')  != -1:
            plot_marker = "v"
        if key.find('1024') != -1:
            plot_marker = "s"
        if key.find('2048') != -1:
            plot_marker = "X"
        if key.find('4096') != -1:
            plot_marker = "d"
        if key.find(comm_type) != -1:
            scaling_values      = data_series
            speedup_values      = [data_series[0]/data_point for data_point in data_series]
            efficiency_values = []
            for i in range(0,len(speedup_values)):
                efficiency_values.append(speedup_values[i]/x_axis_points[i]*100)

            plt.plot(x_axis_points[0:len(efficiency_values)], efficiency_values, marker=plot_marker, label=key)

            
    plt.xlabel('Number of cores')
    plt.ylabel('Efficiency [%]')
    plt.legend(prop={'size': legend_font_size})
    plt.grid(True)
    plt.gca().set_position([0, 0, 1, 1])
    plt.savefig("ppp_efficiency_hybrid_" + comm_type + ".svg")


x_axis_points = [1, 2*9, 4*9, 8*9, 16*9, 32*9]
#x_axis_points = [1,2,4]
data = []
with open('run_full_hybrid_1d_out.csv', newline='') as csvfile:
    csvreader = csv.reader(csvfile, delimiter=';')
    data = list(csvreader)


data_dict = {}

for i in range(0,len(data)-headerlen):
    suffix = suffixes[i % 6]
    row = data[i + headerlen]
    datakey = str(row[4]) + suffix
    iterationtime = float(row[13])
    if not datakey in data_dict.keys():
        data_dict[datakey] = []
        data_dict[datakey].append(iterationtime)
    else:
        data_dict[datakey].append(iterationtime)

for comm_type in ['p2p', 'rma']:

    # scaling
    plt.figure(figsize=[ploth_width, ploth_height])
    for key, data_series in data_dict.items():
        plot_marker = ""
        if key.find('256')  != -1:
            plot_marker = "o"
        if key.find('512')  != -1:
            plot_marker = "v"
        if key.find('1024') != -1:
            plot_marker = "s"
        if key.find('2048') != -1:
            plot_marker = "X"
        if key.find('4096') != -1:
            plot_marker = "d"
        
        if key.find(comm_type) != -1:
            scaling_values      = data_series
            plt.plot(x_axis_points[0:len(scaling_values)], scaling_values, marker=plot_marker, label=key)

    plt.xlabel('Number of cores')
    plt.ylabel('Iteration time [ms]')
    plt.yscale('log')
    plt.xscale('log')
    plt.legend(prop={'size': legend_font_size})
    plt.grid(True)
    plt.gca().set_position([0, 0, 1, 1])
    plt.savefig("ppp_scaling_hybrid_1D_" + comm_type + ".svg")

    # speedup
    plt.figure(figsize=[ploth_width, ploth_height])
    for key, data_series in data_dict.items():
        plot_marker = ""
        if key.find('256')  != -1:
            plot_marker = "o"
        if key.find('512')  != -1:
            plot_marker = "v"
        if key.find('1024') != -1:
            plot_marker = "s"
        if key.find('2048') != -1:
            plot_marker = "X"
        if key.find('4096') != -1:
            plot_marker = "d"
        if key.find(comm_type) != -1:
            scaling_values      = data_series
            speedup_values      = [data_series[0]/data_point for data_point in data_series]

            plt.plot(x_axis_points[0:len(speedup_values)], speedup_values, marker=plot_marker, label=key)

    plt.legend(prop={'size': legend_font_size})
    plt.xlabel('Number of cores')
    plt.ylabel('Speedup')
    plt.grid(True)
    plt.gca().set_position([0, 0, 1, 1])
    plt.savefig("ppp_speedup_hybrid_1D_" + comm_type + ".svg")

    # efficiency
    plt.figure(figsize=[ploth_width, ploth_height])
    for key, data_series in data_dict.items():
        plot_marker = ""
        if key.find('256')  != -1:
            plot_marker = "o"
        if key.find('512')  != -1:
            plot_marker = "v"
        if key.find('1024') != -1:
            plot_marker = "s"
        if key.find('2048') != -1:
            plot_marker = "X"
        if key.find('4096') != -1:
            plot_marker = "d"
        if key.find(comm_type) != -1:
            scaling_values      = data_series
            speedup_values      = [data_series[0]/data_point for data_point in data_series]
            efficiency_values = []
            for i in range(0,len(speedup_values)):
                efficiency_values.append(speedup_values[i]/x_axis_points[i]*100)

            plt.plot(x_axis_points[0:len(efficiency_values)], efficiency_values, marker=plot_marker, label=key)

            
    plt.xlabel('Number of cores')
    plt.ylabel('Efficiency [%]')
    plt.legend(prop={'size': legend_font_size})
    plt.grid(True)
    plt.gca().set_position([0, 0, 1, 1])
    plt.savefig("ppp_efficiency_hybrid_1D_" + comm_type + ".svg")