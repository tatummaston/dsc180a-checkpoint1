import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

run_200_300 = pd.read_csv('data/20211026T184846_200-300-iperf.csv')
run_200_1000 = pd.read_csv('data/20211026T184846_200-1000-iperf.csv')
run_200_25000 = pd.read_csv('data/20211026T184846_200-25000-iperf.csv')
run_200_5000 = pd.read_csv('data/20211026T184847_200-5000-iperf.csv')

figure(figsize=(12, 6), dpi=80)
plt.plot(run_200_300['Time'],run_200_300['1->2Bytes'])
plt.xlabel('Time (s)')
plt.ylabel('Bytes')
plt.title('Bytes per second -- 1->2Bytes (200/300)')
plt.show()

figure(figsize=(12, 6), dpi=80)
plt.plot(run_200_1000['Time'],run_200_1000['1->2Bytes'])
plt.xlabel('Time (s)')
plt.ylabel('Bytes')
plt.title('Bytes per second -- 1->2Bytes (200/1000)')
plt.show()

figure(figsize=(12, 6), dpi=80)
plt.plot(run_200_25000['Time'],run_200_25000['1->2Bytes'])
plt.xlabel('Time (s)')
plt.ylabel('Bytes')
plt.title('Bytes per second -- 1->2Bytes (200/25000)')
plt.show()

figure(figsize=(12, 6), dpi=80)
plt.plot(run_200_5000['Time'],run_200_5000['1->2Bytes'])
plt.xlabel('Time (s)')
plt.ylabel('Bytes')
plt.title('Bytes per second -- 1->2Bytes (200/5000)')
plt.show()

figure(figsize=(12, 6), dpi=80)
plt.plot(run_200_300['Time'],run_200_300['2->1Bytes'])
plt.xlabel('Time (s)')
plt.ylabel('Bytes')
plt.title('Bytes per second -- 2->1Bytes (200/300)')
plt.show()

figure(figsize=(12, 6), dpi=80)
plt.plot(run_200_1000['Time'],run_200_1000['2->1Bytes'])
plt.xlabel('Time (s)')
plt.ylabel('Bytes')
plt.title('Bytes per second -- 2->1Bytes (200/1000)')
plt.show()

figure(figsize=(12, 6), dpi=80)
plt.plot(run_200_25000['Time'],run_200_25000['2->1Bytes'])
plt.xlabel('Time (s)')
plt.ylabel('Bytes')
plt.title('Bytes per second -- 2->1Bytes (200/25000)')
plt.show()

figure(figsize=(12, 6), dpi=80)
plt.plot(run_200_5000['Time'],run_200_5000['2->1Bytes'])
plt.xlabel('Time (s)')
plt.ylabel('Bytes')
plt.title('Bytes per second -- 2->1Bytes (200/5000)')
plt.show()


figure(figsize=(12, 6), dpi=80)
plt.plot(run_200_300['Time'],run_200_300['2->1Bytes'])
plt.plot(run_200_1000['Time'],run_200_1000['2->1Bytes'])
plt.plot(run_200_25000['Time'],run_200_25000['2->1Bytes'])
plt.plot(run_200_5000['Time'],run_200_5000['2->1Bytes'])
plt.xlabel('Time (s)')
plt.ylabel('Bytes')
plt.title('Bytes per second -- 2->1Bytes (200/300)')
plt.show()

def pckt_count(a):
    a = a.split(';')
    a = [int(i) for i in a[:-1]]
    return len(a)

run_200_300['pckt_ct'] = run_200_300['packet_times'].apply(pckt_count)


figure(figsize=(12, 6), dpi=80)
plt.plot(run_200_300['Time'], run_200_300['pckt_ct'])
plt.xlabel('Time (s)')
plt.ylabel('Bytes')
plt.title('Packets per second -- 1->2Bytes (200/300)')
plt.show()

run_200_5000['packet_size_total'] = run_200_5000['1->2Pkts'] + run_200_5000['2->1Pkts']
run_200_300['packet_size_total'] = run_200_300['1->2Pkts'] + run_200_300['2->1Pkts']
run_200_1000['packet_size_total'] = run_200_1000['1->2Pkts'] + run_200_1000['2->1Pkts']
run_200_25000['packet_size_total'] = run_200_25000['1->2Pkts'] + run_200_25000['2->1Pkts']
plt.show()

figure(figsize=(12, 6), dpi=80)
plt.plot(run_200_5000['Time'], run_200_5000['packet_size_total'])
plt.xlabel('Time (s)')
plt.ylabel('Bytes')
plt.title('Packet Length per second -- 1->2Bytes (200/5000)')
plt.show()

figure(figsize=(12, 6), dpi=80)
plt.plot(run_200_300['Time'], run_200_300['packet_size_total'])
plt.xlabel('Time (s)')
plt.ylabel('Bytes')
plt.title('Packet Length per second -- 1->2Bytes (200/300)')
plt.show()


ps = []
for i in run_200_5000.packet_sizes[:10]:
    a = i.split(';')
    a = [int(i) for i in a[:-1]]
    ps = ps + a
ps = ps[:-1]

pt = []
for i in run_200_5000.packet_times[:10]:
    a = i.split(';')
    a = [int(i) for i in a[:-1]]
    pt = pt + a

 diff_list = []
for x, y in zip(pt[0::], pt[1::]):
    diff_list.append(y-x)

pt = diff_list


figure(figsize=(12, 6), dpi=80)
plt.plot(pt)
plt.show()

