import os
import pandas as pd
from func import *
import statsmodels.api as sm


class exp_poisson_glm():
    '''
    exponential poisson GLM
    𝜆(t)=exp(kx(t)+b) 
    𝜆: spike rate
    x(t): position up to time t
    k,b : parameters to be fitted
    '''
    def __init__(self) -> None:
        pass


    def fit(self,spikes,design_mat):
        '''
        '''
        self.pGLM_all_model = sm.GLM(endog=spikes, exog=design_mat,
                         family=sm.families.Poisson())
        self.pGLM_all_results = self.pGLM_all_model.fit(max_iter=100, tol=1e-6, tol_criterion='params')
        return self.pGLM_all_results

if __name__=="__main__":
    import matplotlib.pyplot as plt

    all_data_dir='Modules/data/alldata/'
    datalist=os.listdir(all_data_dir)

    data_dir=all_data_dir+datalist[1] # load data
    sample_name=data_dir.split('/')[-1]
    print(sample_name)

    position,spikes=data_loader(data_dir)

    binned_position=bin_pos(position)
    time_bin_size=1/3 #second
    num_time_bins,num_cells = spikes.shape

    ntfilt=25
    nthist=20
    neuron_idx=32

    design_mat_all=design_matrix_encoder(binned_position,spikes[:,-30:],ntfilt,nthist)

    pGLM_all_model = sm.GLM(endog=spikes[:,neuron_idx], exog=design_mat_all,
                            family=sm.families.Poisson()) # assumes 'log' link.
    pGLM_all_results = pGLM_all_model.fit(max_iter=50, tol=1e-6, tol_criterion='params')

    pGLM_all_const = pGLM_all_results.params[0]
    pGLM_all_filt = pGLM_all_results.params[1:ntfilt+1] # stimulus filter
    pGLM_all_hist_filt = pGLM_all_results.params[ntfilt+1:] # all cells spike history filter
    pGLM_all_hist_filt = np.reshape(pGLM_all_hist_filt, (nthist,30), order='F')
    rate_pred_all = np.exp(pGLM_all_const + design_mat_all[:,1:] @ pGLM_all_results.params[1:])
    print(rate_pred_all)

    # ----------Make plot
    iiplot = np.arange(3000)
    ttplot = iiplot*time_bin_size
    tth = np.arange(-1*nthist,0)*time_bin_size

    plt.subplot(211) # Plot spike history filter
    plt.plot(tth,tth*0,'k--')
    cs = ['r', 'orange', 'purple','g','b']
    for i in np.arange(5):
        plt.plot(tth,pGLM_all_hist_filt[:,i], c=cs[i], label='from ' + str(i+31))
    plt.legend(loc='upper left')
    plt.title(f'coupling filters: into cell {str(neuron_idx+1)}')
    plt.xlabel('time before spike (s)')
    plt.ylabel('weight')

    plt.subplot(212)
    markerline,_,_ = plt.stem(ttplot, spikes[:,neuron_idx][iiplot], linefmt='k-', basefmt='k-', label='spikes')
    plt.setp(markerline, 'markerfacecolor', 'none')
    plt.setp(markerline, 'markeredgecolor', 'k')
    plt.plot(ttplot, rate_pred_all[iiplot], c='purple', label='coupled-GLM')
    plt.legend(loc='upper left')
    plt.xlabel('time (s)')
    plt.title('spikes and rate predictions')
    plt.ylabel('spike count / bin')
    plt.tight_layout()
    plt.show()

