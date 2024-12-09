
# This file is mainly used for preliminary partitioning of TOF datasets with different energy thresholds
from my_library import *
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import warnings

# close all warnings
warnings.filterwarnings("ignore")
NE213A_data = pd.read_pickle('NE213A_data.pkl')

# The flight time range of neutron gamma
nToFRange_NE213A = [28.1e-9, 60e-9]
gTOFRange = [0.5e-9, 6e-9]

# Select energy threshold
energy_threshold_low = 2
energy_threshold_up = 6.105

# filter Data by using QDC and energy
PS_cut = 0.3
QDC_threshold_low = search_qdc_threshold(energy_threshold_low, data=NE213A_data, ToFRange=nToFRange_NE213A)
NE213A_data['PS'] = (NE213A_data['qdc_lg_ch1'] - NE213A_data['qdc_sg_ch1']) / NE213A_data['qdc_lg_ch1']
Gamma = NE213A_data.query(f'tof_ch2 > {gTOFRange[0]} and tof_ch2 < {gTOFRange[1]} and qdc_lg_ch1 > {QDC_threshold_low}')

# Gamma = Gamma.query(f'PS < {PS_cut}')
Neutron = NE213A_data.query(
    f'tof_ch2 > {nToFRange_NE213A[0]} and tof_ch2 < {nToFRange_NE213A[1]} and tofE_ch2 > {energy_threshold_low} and tofE_ch2 < {energy_threshold_up} and qdc_lg_ch1 > {QDC_threshold_low}')


# DF_plot_No_NPG
N_PS = np.array(Neutron['PS'])
G_PS = np.array(Gamma['PS'])
fig, ax = plt.subplots(figsize=(10, 6))
ax.hist(N_PS, bins=30, color='b', edgecolor='black', alpha=0.7, label='Neutron', density=True)
ax.hist(G_PS, bins=30, color='r', edgecolor='black', alpha=0.7, label='Gamma', density=True)
ax.set_title('Neutron and Gamma PS Histogram(Include NPG)')
ax.set_xlabel('PS')
ax.set_ylabel('Yield')
ax.set_yscale('log')
ax.legend()
plt.tight_layout()
plt.show()

# Eliminate the impact of NPG
NPG = Neutron.query(f'PS < {PS_cut}')
Neutron = Neutron.query(f'PS >= {PS_cut}')
Gamma = pd.concat([NPG, Gamma])
Gamma.reset_index(drop=True, inplace=True)
Neutron.reset_index(drop=True, inplace=True)
gamma_signal = np.array(Gamma['samples_ch1'])
gamma_signal = -(np.vstack(gamma_signal))
neutron_signal = np.array(Neutron['samples_ch1'])
neutron_signal = -(np.vstack(neutron_signal))

# filter abnormal data
filter_gamma = []
filter_neutron = []
def filter_signal(signal, filter_store, position, Crop_threshold):
    NUM = signal.shape[0]
    for i in range(NUM):
        s_signal = signal[i, :]
        if np.max(s_signal[position:-1]) <= Crop_threshold:
            filter_store.append(s_signal)
    return filter_store
filter_gamma_signal = np.array(filter_signal(gamma_signal, filter_gamma, 300, 300)).T
filter_neutron_signal = np.array(filter_signal(neutron_signal, filter_neutron, 300, 300)).T
np.save('H:\\Lund_dataset\\nppp-master\\nppp-master\\MY_datasetloader\\NE213A_1.5Mev\\gamma_1.5Mev', abs(filter_gamma_signal))
np.save('H:\\Lund_dataset\\nppp-master\\nppp-master\\MY_datasetloader\\NE213A_1.5Mev\\neutron_1.5Mev', abs(filter_neutron_signal))
