from numpy.core.fromnumeric import argmax
import numpy as np
from scipy.stats import pearsonr

import pyqtgraph as pg
from pyqtgraph.Qt import QtGui, QtCore

import os
import time
import argparse
from io import StringIO

import csv
import json

import numpy as np
from scipy import signal
import pandas as pd



COLUMN = [
    "type",
    "id",
    "mac",
    "rssi",
    "rate",
    "sig_mode",
    "mcs",
    "bandwidth",
    "smoothing",
    "not_sounding",
    "aggregation",
    "stbc",
    "fec_coding",
    "sgi",
    "noise_floor",
    "ampdu_cnt",
    "channel",
    "secondary_channel",
    "local_timestamp",
    "ant",
    "sig_len",
    "rx_state",
    "len",
    "first_word",
    "data"]

# only plot useful subcarrier channel
'''
select_list = []
select_list += [i for i in range(5, 31)]
select_list += [i for i in range(33, 58)]
select_list += [i for i in range(66, 122)]
select_list += [i for i in range(123, 191)]
select_list.remove(128)

Combined_Channels = [122, 121, 120, 117, 118, 119]
default_channels = [122, 93,  121, 94,  
                    120, 117, 118, 119, 
                    95,  96,  97,  98,  
                    99,  100, 101, 102, 
                    103, 105, 106, 104]
'''
select_list = []
select_list += [i for i in range(96, 122)]
select_list += [i for i in range(123, 144)]
select_list.remove(128)

Combined_Channels = [41, 40, 39, 36, 37, 38]
default_channels = [41, 12, 40, 13, 
                    39, 36, 37, 38, 
                    14, 15, 16, 17, 
                    18, 19, 20, 21, 
                    22, 24, 25, 23]



def move_avg(a,n,mode="same"):
    return(np.convolve(a, np.ones((n,))/n, mode=mode))


def parse_csi_data(strings: str, num: int):
    """ parse csi data from string beginning with "CSI_DATA"

    :strings: string beginning with "CSI_DATA"
    :num: the elements number in strings
    :returns: List[COLUMN]

    """
    f = StringIO(strings)
    csv_reader = csv.reader(f)
    csi_data = next(csv_reader)

    if len(csi_data) != num:
        raise ValueError(f"element number is not equal {num}")

    try:
        json.loads(csi_data[-1])
    except json.JSONDecodeError:
        raise ValueError(f"data is not incomplete")
        
    return csi_data





if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description="monitor RSSI and CSI from Wifi")
    parser.add_argument('-S', '--src', dest='src_file', action='store',
                        help="console_test logfile")
    parser.add_argument('-F', '--folder', dest='log_folder', action='store', default='.',
                        help="monitor file change in the given folder")   
    parser.add_argument('-T', '--timer-interval', dest='timer_interval', default='100',
                        help='time interval (milliseconds) for UI update')                                         
    args = parser.parse_args()
    print(args)

    src_file = args.src_file

    if not src_file:
        # get interesting files
        files = [f for f in os.listdir(args.log_folder)
                if os.path.isfile(f) and os.path.splitext(f)[1].lower() == ".txt"]
        modtimes = [os.path.getmtime(f) for f in files]             
        idx_latest = np.argmax(modtimes)
        src_file = files[idx_latest]    

    mod_time = os.path.getmtime(src_file)    

    lines_done = 0
    plot_len = 1600

    # raw data from log
    id_list = []
    time_list = []
    rssi_list = []
    csi_list = []

    # filtered data
    winsize_hampel = 5
    hampeled_num = 0
    rssi_array = np.array([])
    csi_array = np.zeros((0,len(select_list)))


    # refer
    # https://github.com/MichaelisTrofficus/hampel_filter/blob/master/src/hampel.py
    def hampel(sig, window_size=5, n=3):

        x = np.asarray(sig)
        assert len(x.shape) == 1

        # do nothing if there is not enough data in x
        if len(x) <= window_size * 2:
            return x, []

        # the full window length
        window_len = window_size * 2 + 1

        # Constant scale factor, which depends on the distribution
        # In this case, we assume normal distribution
        k = 1.4826     
        outliers = []

        for i in range(len(x) - window_size * 2):
            # for the data in the window center
            idx = i + window_size
            window = x[i:i+window_len]
            window_median = np.median(window)
            window_sigma = k * np.median(np.abs(window - window_median))

            is_outlier = np.abs(window[window_size] - window_median) >= (n * window_sigma)

            if is_outlier:
                outliers.append(idx)
                x[idx] = window_median

        return x, outliers     



    def parse_new_log():

        global mod_time, lines_done, hampeled_num
        global id_list, time_list, rssi_list, csi_list
        global rssi_array, csi_array

        # check modified time before parsing
        # if os.path.getmtime(src_file) == mod_time:
        #     return lines_done
        # else:
        #     mod_time = os.path.getmtime(src_file)

        f_src = open(src_file, 'r', encoding='UTF-8')

        # skip lines which have been parsed
        for _ in np.arange(lines_done):
            next(f_src)

        while True:
            line = f_src.readline()
            if not line:
                #print('No more line')
                break    

            lines_done += 1    

            index = line.find(('CSI_DATA'))
            if index != -1:
                try:
                    row = parse_csi_data(line[index:], 25)
                    log_id = int(row[1])

                    if log_id in id_list:
                        print('Log already recorded!', log_id)
                        continue

                    time_stamp = int(row[18])
                    rssi_value = int(row[3])
                    csi_raw = json.loads(row[-1])

                    if len(csi_raw) != 384:
                        print(f'Skipped due to mismatching CSI format of {len(csi_raw)} numbers')
                        continue

                    id_list.append(log_id)
                    rssi_list.append(rssi_value)
                    time_list.append(time_stamp)

                    # TODO: phase is ignored due to only one antenna
                    csi_value = [complex(int(csi_raw[2*n]), int(csi_raw[2*n + 1]))
                                    for n in range(len(csi_raw)//2)]
                    csi_value = np.abs(csi_value)
                    csi_value = [csi_value[i] for i in select_list]
                    csi_list.append(csi_value)

                except ValueError:
                    print('It is not a valid CSI data!')
                    continue

        # hampel filtered
        if len(rssi_list) > hampeled_num + winsize_hampel * 2:

            todo_num = len(rssi_list) - (hampeled_num + winsize_hampel * 2)

            rssi_pre = np.asarray(rssi_list[-winsize_hampel:])
            if len(rssi_array) > winsize_hampel:
                rssi_pre = rssi_array[-winsize_hampel:]
            rssi_todo = np.hstack([rssi_pre, rssi_list[winsize_hampel+hampeled_num:]])
            rssi_hampeled = hampel(rssi_todo, window_size=winsize_hampel, n=3)[0]
            rssi_array = np.hstack([ rssi_array, rssi_hampeled[winsize_hampel:-winsize_hampel] ])

            #print('rssi_todo', rssi_todo)
            #print('rssi_hampeled',  rssi_hampeled[winsize_hampel:-winsize_hampel])
            #print('new rssi_array', rssi_array)

            csi_pre = np.asarray(csi_list)[-winsize_hampel:,:]
            if len(rssi_array) > winsize_hampel:
                csi_pre = csi_array[-winsize_hampel:,:]
            csi_todo = np.vstack([csi_pre, np.asarray(csi_list)[winsize_hampel+hampeled_num:,:]])

            csi_hampeled = []
            for ch in range(csi_todo.shape[1]):
                ch_hampeled = hampel(csi_todo[:,ch], window_size=winsize_hampel, n=3)[0]
                csi_hampeled.append(ch_hampeled)    
            csi_hampeled = np.asarray(csi_hampeled).T
            csi_array = np.vstack([csi_array, csi_hampeled[winsize_hampel:-winsize_hampel,:]])

            # print(f'filtered {todo_num} data!')
            hampeled_num += todo_num       
        
        f_src.close()

        return lines_done


    app = pg.mkQApp("CSI Monitor")
    #mw = QtGui.QMainWindow()
    #mw.resize(800,800)

    win = pg.GraphicsLayoutWidget(show=True, title="CSI Monitor Window")
    win.resize(900,600)
    win.setWindowTitle('Wifi-CSI')

    # Enable antialiasing for prettier plots
    pg.setConfigOptions(antialias=True)

    ROW, COLUMN = 5, 4
    plot_nums = ROW*COLUMN


    # plots = []
    curves = []
    channels = []


    def plot_curve(win, title, pen='y'):
        plot = win.addPlot(title=title)
        plot.enableAutoRange('xy', True)
        curve = plot.plot(pen=pen)
        return curve

    # RSSI
    curve = plot_curve(win, 'RSSI - Magnitude', 
                       pen=pg.mkPen('r', width=2))
    curves.append(curve)

    curve = plot_curve(win, 'RSSI - Frequency', 
                       pen=pg.mkPen('r', width=2))
    curves.append(curve)    

    # combined CSI
    curve = plot_curve(win, 'CSI - Magnitude', 
                    pen=pg.mkPen('g', width=2))
    curves.append(curve)    

    curve = plot_curve(win, 'CSI - Frequency', 
                    pen=pg.mkPen('g', width=2))
    curves.append(curve)        
    
    # CSI channels
    for r in range(ROW):
        win.nextRow()     
        for c in range(COLUMN):
            curve = plot_curve(win, f'CSI-{COLUMN*r+c+1}')
            curves.append(curve)


    def update_plot():
        global curves, channels

        # read rssi and csi in real-time
        parse_new_log()

        t = time_list[-plot_len:]

        rssi_plot = rssi_array[-plot_len:]
        rssi_curve = np.ones(plot_len) * rssi_plot[0]
        rssi_curve[-len(rssi_plot):] = rssi_plot
        curves[0].setData(rssi_curve)

        csi_plot = csi_array[-plot_len:, Combined_Channels].mean(axis=1)
        csi_plot = signal.medfilt(csi_plot,5)
        csi_plot = move_avg(csi_plot, 7, 'valid')
        
        csi_curve = np.ones(plot_len) * csi_plot[0]
        csi_curve[-len(csi_plot):] = csi_plot
        curves[2].setData(csi_curve)

        if len(channels)==0:
            if len(rssi_array) > plot_len:
                # default
                # channels = default_channels

                # using std from CSI data itself
                # csi_std = csi_data.std(axis=0)
                # channels = np.argsort(csi_std)[-plot_nums:]

                # pearsonr with RSSI
                # corr_values = []
                # for ch in range(csi_data.shape[1]):
                #     corr, p_value = pearsonr(rssi_plot[-plot_len:], csi_data[-plot_len:, c])
                #     print(ch, corr, p_value)
                #     corr_values.append(corr)                
                # channels = np.argsort(np.abs(corr_values))[-plot_nums:]

                # or random?
                channels = np.random.choice(a=csi_array.shape[1], size=plot_nums, replace=False)

                print(channels)
                

        else:
            rssi_f, rssi_Pxx_den = signal.welch(rssi_curve, 10e3, nperseg=1024)
            curves[1].setData(rssi_Pxx_den)
            curves[1].setLogMode(True, False)

            csi_f, csi_Pxx_den = signal.welch(csi_curve, 10e3, nperseg=1024)
            curves[3].setData(csi_Pxx_den)
            curves[3].setLogMode(True, False)   

            for i in range(plot_nums):
                csi_curve = csi_array[-plot_len:, channels[i]]
                #data = signal.medfilt(data,5)
                #data = move_avg(data, 11, 'valid')
                curves[i+4].setData(csi_curve)

    timer = QtCore.QTimer()
    timer.timeout.connect(update_plot)
    timer.start(50) # milliseconds

    pg.mkQApp().exec_()