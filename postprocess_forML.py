#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 10 13:18:38 2022
Last modified: 20230511
@author: Daniel N. Blaschke

© 2023. Triad National Security, LLC. All rights reserved.

This program was produced under U.S. Government contract 89233218CNA000001 for Los Alamos

National Laboratory (LANL), which is operated by Triad National Security, LLC for the U.S.

Department of Energy/National Nuclear Security Administration. All rights in the program are.

reserved by Triad National Security, LLC, and the U.S. Department of Energy/National Nuclear

Security Administration. The Government is granted for itself and others acting on its behalf a

nonexclusive, paid-up, irrevocable worldwide license in this material to reproduce, prepare.

derivative works, distribute copies to the public, perform publicly and display publicly, and to permit.

others to do so.
"""
## requires python 3.9 or higher (because read_data_multipleruns() uses union operator for dicts)

import os
import sys
# from shutil import copyfile
# import glob
from scipy.signal import medfilt, find_peaks
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
fntsize=11
    
def resampledata(data,targetindices):
    '''resamples irregularly spaced data within a pandas dataframe with just one column according 
    to an array (or list) of target index values.'''
    fillnan = pd.DataFrame(np.repeat(float('nan'),len(targetindices)),\
                           index=pd.Index(targetindices),columns=data.columns)
    fillnan.index.name = data.index.name
    combined = pd.concat([data,fillnan]).sort_index().interpolate('slinear')
    return combined[~combined.index.duplicated(keep='first')].loc[list(targetindices)]

def filterdata(data):
    '''apply a median filter'''
    return pd.DataFrame(medfilt(data.to_numpy()[:,0],5),index=data.index,columns=data.columns)

def read_data(foldername,resample=False,ntimestamps=201,skip_pos=None,filterporos=False,filtervelo=False):
    '''Reads raw data files for surface velocity and porosity (naming scheme velocity_i.0.csv, porosity_i.0.csv)
    and returns pandas data frames. TEPLA parameters are from file design.csv.
    If "resample" is set to "True", then the velocities dataframe will be resampled using "timestamps".'''
    print(f"reading {foldername} ...")
    ## read surface velocities first:
    design = pd.read_csv(os.path.join(foldername,"design.csv"))
    velocities = []
    porosities = []
    for i in range(len(design)):
        velocities.append(pd.read_csv(os.path.join(foldername,f"velocity_{i}.csv"),index_col=0))
        porosities.append(pd.read_csv(os.path.join(foldername,f"porosity_{i}.csv"),index_col=0))
    targetthickness = porosities[0].index[-1]
    metadata = {"TTARG":targetthickness,'positions':porosities[0].index}
    if resample:
        print("resampling data ...")
        for i in range(len(velocities)):
            timestamps = np.linspace(velocities[i].index[0],velocities[i].index[-1],ntimestamps)
            velocities[i] = resampledata(velocities[i], timestamps)
            metadata['timestamps'] = pd.Index(timestamps,name='time[us]')
        if skip_pos is not None:
            for i in range(len(porosities)):
                porosities[i] = porosities[i][::skip_pos]
            metadata['positions'] = porosities[0].index[::skip_pos]
    if filtervelo:
        print("applying median filter to velocities ...")
        for i in range(len(velocities)):
            tmp_velo = velocities[i].copy()
            velocities[i] = pd.DataFrame(medfilt(tmp_velo.to_numpy()[:,0],5),index=tmp_velo.index,columns=tmp_velo.columns)
    if filterporos:
        print("applying median filter to porosities ...")
        for i in range(len(porosities)):
            tmp_poros = porosities[i].copy()
            porosities[i] = pd.DataFrame(medfilt(tmp_poros.to_numpy()[:,0],5),index=tmp_poros.index,columns=tmp_poros.columns)
    return velocities,porosities,design,metadata

def read_data_multipleruns(listoffolders,resample=False,ntimestamps=201,skip_pos=None,filterporos=False,filtervelo=False):
    '''calls read_data() on all elements of <listoffolders> and merges the output into one big dataset'''
    tupleofdata = []
    no_entries = 0
    for i, fol in enumerate(listoffolders):
        tupleofdata.append(read_data(fol,resample,ntimestamps,skip_pos,filterporos=filterporos,filtervelo=filtervelo))
        if i==0:
            velocities = tupleofdata[i][0]
            porosities = tupleofdata[i][1]
        elif i>0:
            velocities += tupleofdata[i][0]
            porosities += tupleofdata[i][1]
        no_entries += len(tupleofdata[i][0])
    
    design = pd.concat([des[2] for des in tupleofdata])
    design.reset_index(inplace=True,drop=True)
    metadata = tupleofdata[0][3]
    for i in range(len(tupleofdata)-1):
        metadata |= tupleofdata[i+1][3] ## union/update operator introduced in python 3.9
    return velocities,porosities,design,metadata
    
def write_data(velocities,porosities,design,metadata=None,targetfolder="./MLdata"):
    '''takes output of read_data() as input (potentially after filtering) and writes those data to csv files'''
    cwd = os.getcwd()
    if not os.path.exists(targetfolder):
        os.mkdir(targetfolder)
    os.chdir(targetfolder)
    design.to_csv("design.csv",index=False)
    for i in range(len(design)):
        velocities[i].to_csv(f"velocity_{i}.csv")
        porosities[i].to_csv(f"porosity_{i}.csv")
    os.chdir(cwd)

def combine_data(velocities,porosities,design,metadata=None):
    '''takes output of read_data() as input and combines it into one big data frame'''
    x_porosity = [f"porosity{i}" for i in range(len(porosities[0]))]
    t_velocity = [f"velocity{i}" for i in range(len(velocities[0]))]
    for i in range(len(porosities)):
        porosities[i].columns = pd.Index([f"porosity{i}"])
    for i in range(len(velocities)):
        velocities[i].columns = pd.Index([f"velocity{i}"])
    allporosities = pd.concat(porosities,axis=1).T.reset_index(drop=True)
    allporosities.columns = x_porosity
    allvelocities = pd.concat(velocities,axis=1).T.reset_index(drop=True)
    allvelocities.columns = t_velocity
    alldata = pd.concat([allporosities,allvelocities,design.reset_index(drop=True)],axis=1)
    return alldata

def write_data_onebigfile(dataset,targetfolder="./MLdata",hdf=False):
    '''takes output of combine_data() or reduce_data() as input and writes those data to one big csv file'''
    cwd = os.getcwd()
    if not os.path.exists(targetfolder):
        os.mkdir(targetfolder)
    os.chdir(targetfolder)
    if hdf:
        dataset.to_hdf('porosity_velocity_all.hdf5','porosity_velocity_all',index=False)
    else:
        dataset.to_csv('porosity_velocity_all.csv',index=False)
    os.chdir(cwd)

def reduce_data(velocities,porosities,design,metadata,reduce_velocity=True,plotn=[],include_minpor=False):
    '''takes output of read_data() and generates a reduced dataset by extracting features from the
       porosity and surface velocity data.'''
    TTARG = metadata['TTARG']
    # Nthresholds = [0.25,0.5,0.75]
    Nthresholds = [0.1,0.3,0.5,0.7,0.9]
    if include_minpor:
        # cols = ['min_poros','max_poros','spread25','spread50','spread75'] ## min_poros very hard to measure, makes no sense to train with it
        cols = ['min_poros','max_poros','spread10','spread30','spread50','spread70','spread90']
        porosityfeatures = pd.DataFrame(np.zeros((len(porosities),len(Nthresholds)+2)),columns=cols)
    else:
        # cols = ['max_poros','spread25','spread50','spread75']
        cols = ['max_poros','spread10','spread30','spread50','spread70','spread90']
        porosityfeatures = pd.DataFrame(np.zeros((len(porosities),len(Nthresholds)+1)),columns=cols)
    for n in range(len(porosities)):
        currentdata = porosities[n]
        currentdata = currentdata[currentdata.columns[0]] ## convert dataframe to series
        maxdata = currentdata.max()
        scatter_x = [currentdata.idxmax()]
        scatter_y = [maxdata]
        if include_minpor:
            mindata = currentdata.min()
            porosityfeatures.iloc[n,0] = mindata
            porosityfeatures.iloc[n,1] = maxdata
            ctr=2
            scatter_x.append(currentdata.idxmin())
            scatter_y.append(mindata)
        else:
            mindata = 0
            porosityfeatures.iloc[n,0] = maxdata
            ctr=1
        for ithr,nthr in enumerate(Nthresholds):
            threshold = mindata + nthr*(maxdata - mindata) ## (don't) offset by initial porosity phi0
            x0 = None
            x1 = None
            lendat = len(currentdata)
            for i in range(lendat):
                if currentdata.iloc[i] >= threshold and x0 is None:
                    x0 = i
                if x1 is None and currentdata.iloc[-1-i] >= threshold:
                    x1 = lendat-1-i
                if x0 is not None and x1 is not None:
                    break
            porosityfeatures.iloc[n,ctr+ithr] = (x1-x0)*TTARG/lendat
            scatter_x.append(currentdata.index[x0])
            scatter_x.append(currentdata.index[x1])
            scatter_y.append(threshold)
            scatter_y.append(threshold)
        if n in plotn:
            porplot = currentdata.plot(xlabel='Position [cm]',ylabel='Porosity',figsize=(5.5, 4.5))
            porplot.scatter(scatter_x,scatter_y,color="orange")
            for l in range(len(Nthresholds)):
                porplot.plot([scatter_x[-2*l-1],scatter_x[-2*l-2]],[scatter_y[-2*l-1],scatter_y[-2*l-2]],\
                             linestyle='dashed',color='gray')
            porplot.get_figure().savefig(f'porosity{n}.pdf', format='pdf',bbox_inches='tight')
            plt.close()
    if reduce_velocity:
        velcols = ['vel_localmax1','vel_localmin1','vel_localmax2','vel_localmin2','vel_localmax3','vel_localmin3']
        lenvelo = len(velocities)
        velocityfeatures = pd.DataFrame(np.zeros((lenvelo,len(velcols))),columns=velcols) #.replace(0,np.nan)
        for n in range(lenvelo):
            currentdata = velocities[n]
            currentdata = currentdata[currentdata.columns[0]] ## convert dataframe to series
            # maxidx = find_peaks(currentdata,prominence=3.e-4,height=3.e-3,width=int(lenvelo/200),distance=int(lenvelo/44))[0][:4] ## find indices of first 4 local maxima (one extra for finding minima between those maxima)
            maxidx = find_peaks(currentdata,prominence=3.e-4,height=3.e-3,distance=5)[0][:4] ## find indices of first 4 local maxima (one extra for finding minima between those maxima)
            # x0label=currentdata.idxmax()
            maxval = currentdata.iloc[maxidx[0]]
            # x0 = currentdata.index.get_loc(x0label)
            if len(maxidx)>1:
                x1label = currentdata.iloc[maxidx[0]:maxidx[1]].idxmin()
            else:
                x1label = currentdata.iloc[maxidx[0]:].idxmin()
            minval = currentdata.loc[x1label]
            x1 = currentdata.index.get_loc(x1label)
            velocityfeatures.iloc[n,0] = maxval
            velocityfeatures.iloc[n,1] = minval
            # velocityfeatures.iloc[n,2] = velocities[n].index[x1] ## time in us
            # minlabel = x1label
            x3 = []
            for i in range(min(int(len(velcols)/2-1),len(maxidx)-1)):
                # maxlabel = currentdata.loc[minlabel:].idxmax()
                maxval2 = currentdata.iloc[maxidx[1+i]]
                # x2 = currentdata.index.get_loc(maxlabel)
                if len(maxidx)>i+2:
                    minlabel = currentdata.iloc[maxidx[1+i]:maxidx[2+i]].idxmin()
                else:
                    minlabel = currentdata.iloc[maxidx[1+i]:].idxmin()
                minval2 = currentdata.loc[minlabel]
                x3.append(currentdata.index.get_loc(minlabel))
                velocityfeatures.iloc[n,2*(i+1)] = maxval2
                velocityfeatures.iloc[n,2*(i+1)+1] = minval2
            if n in plotn:
                velplot = (1e4*currentdata).plot(ylabel=r'Free surface velocity [m/s]',xlabel=r'Time$\,$[$\mu$s]',figsize=(5.5, 4.5))
                if len(maxidx)>2:
                    scatter_x = [currentdata.index[maxidx[0]],currentdata.index[x1],currentdata.index[maxidx[1]],currentdata.index[x3[0]],currentdata.index[maxidx[2]],currentdata.index[x3[1]]]
                    scatter_y = 1e4*velocityfeatures.iloc[n,:]
                else:
                    scatter_x = [currentdata.index[maxidx[0]],currentdata.index[x1],currentdata.index[maxidx[1]],currentdata.index[x3[0]]]
                    scatter_y = 1e4*velocityfeatures.iloc[n,:4]
                # print(f"{maxidx=}", len(currentdata.index))
                # print(scatter_x)
                # print(scatter_y)
                velplot.scatter(scatter_x,scatter_y,color="orange")
                velplot.get_figure().savefig(f'velocity_{n}.pdf', format='pdf',bbox_inches='tight')
                plt.close()
        reduceddata = pd.concat([porosityfeatures,velocityfeatures,design.reset_index(drop=True)],axis=1)
    else:
        dataset = combine_data(velocities,porosities,design)
        vel_start_index = dataset.columns.get_loc('velocity0')
        reduceddata = pd.concat([porosityfeatures,dataset.iloc[:,vel_start_index:]],axis=1)
    return reduceddata

    
if __name__ == '__main__':
    if len(sys.argv) == 2:
        listoffolders = sys.argv[1]
    elif len(sys.argv) > 2:
        listoffolders = sys.argv[1:]
    else:
        print(f"Usage: {sys.argv[0]} <target_folder(s)>\n"\
            "aborting.")
        sys.exit()
    
    plotn = [0,10,20,123,1023]
    
    if isinstance(listoffolders,list):
        velocities,porosities,design,metadata = read_data_multipleruns(listoffolders,resample=True,ntimestamps=201,filterporos=True,filtervelo=True)
    else:
        velocities,porosities,design,metadata = read_data(listoffolders,resample=True,ntimestamps=201,filterporos=True,filtervelo=True)
    dataset = combine_data(velocities,porosities,design)
        
    write_data_onebigfile(dataset,targetfolder=".",hdf=False)
    write_data(velocities,porosities,design,targetfolder="MLdata")
    
    #### generate reduced dataset:
    print("extracting/generating reduced data ...")
    reduceddata= reduce_data(velocities, porosities, design, metadata,plotn=plotn,include_minpor=True)
    write_data_onebigfile(reduceddata,targetfolder="reducedMLdata")
    
    
    