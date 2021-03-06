# Formatting Guide
# Use yapf for PEP8 style guide (yapf file --style pep8 -i --in-place)
# Use blank lines sparingly, shows change in logic/focus
# Variable names should be the first five letters of the description
# Comment Code Accordingly
# Anything with ### above and below needs to be updated
# Any graphing function should begin with show
# All functions should begin with show, return, find, create, or other specific verb

### Other Notes ###

# Change Docstrings to reflect: https://www.python.org/dev/peps/pep-0257/#:~:text=The%20docstring%20for%20a%20function,called%20(all%20if%20applicable).

# Consider changing y_data to y_true and predi to y_data to reflect industry standards.

# Update cut times to reflect the portion of the curve that is cut off due to longer
# than standard length

###

# Standard libraries
import sys
import math
import random
import time
import os
import glob

# Third-party imports
import numpy as np
import pandas as pd
import statistics
import scipy

import matplotlib.pyplot as plt
import matplotlib.font_manager
from matplotlib import rc
import seaborn as sns

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import backend as K

import sklearn.metrics as metrics

from datetime import datetime
from pytz import timezone

import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

import warnings

# Project imports
import ephesus
import troia
import tdpy
from const import *

# Do NOT use latex for matplotlib plots
rc('text', usetex=False)
# Plot as many plots as speficied
plt.rcParams.update({'figure.max_open_warning': 0})


def retur_secto_files(secto_range=[1,27], remov_dupli=False, path=f'{raw_tess_path}'):
    '''Returns the file names for sectors of fits data'''
    file_names = []
    # Loops through directories to find fits files
    for sector in range(secto_range[0], secto_range[1]):
        secto_numbe = str(sector)
        if len(secto_numbe) != 2:
            secto_numbe = f'0{secto_numbe}'
        for file in os.listdir(f'{path}sector-{secto_numbe}'):
            if file.endswith('.fits'):
                file_names.append([file, find_TIC_ID(file)])

    file_names_np = np.array(file_names, dtype=object)
    if not remov_dupli:
        return file_names_np[:, 0].tolist()
    # Removes duplicates across sectors
    uniqu_files = (
        file_names_np[np.unique(file_names_np[:, 1], return_index=True)[1],
                      0]).tolist()
    return uniqu_files


def find_TIC_ID(filen):
    '''Find TIC ID from fits filename.'''
    hyphe = []
    for i in range(len(filen)):
        if filen[i] == '-':
            hyphe.append(i)
    return int(filen[hyphe[1] + 1:hyphe[2]])


def find_TIC_secto(filen):
    '''Finds TIC sector from fits filename.'''
    hyphe = []
    for i in range(len(filen)):
        if filen[i] == '-':
            hyphe.append(i)
    return int(filen[hyphe[0] + 2:hyphe[1]])


def retur_fits_data(file, path=f'{raw_tess_path}', featu=infor):
    '''Import a light curve from a fits file.'''
    secto_numbe = str(find_TIC_secto(file))
    if len(secto_numbe) != 2:
        secto_numbe = f'0{secto_numbe}'
    light_curve = ephesus.read_tesskplr_file(f'{path}sector-{secto_numbe}/{file}')[0][:, 0:2].tolist()
    
    # Insert cell for dictionary containing features
    light_curve.append(['infor', 0])

    # Populate basic features
    if len(featu) > 0:
        light_curve[-1][1] = featu.copy()
        light_curve[-1][1]['file_name'] = file
        light_curve[-1][1]['tic_id'] = find_TIC_ID(file)
        light_curve[-1][1]['curve_type'] = 'Light Curve'
        
    return light_curve


def creat_confu_matri(data,
                      predi,
                      cutof=0.5,
                      detec_type='plane_moon_cut_injec'):
    '''Returns a list of numpy arrays containing true positives, \
true negatives, false positives, and false negatives.'''
    # Turn predictions into binary predictions based on the cutoff
    binar_predi = predi > cutof

    true_false = []
    for j in range(len(data)):
        true_false.append(data[j, -1,
                               1]['plane_moon_cut_injec'] == binar_predi[j])
    true_posit = []
    true_negat = []
    false_posit = []
    false_negat = []

    # Create the confusion matrix lists
    for i in range(len(data)):
        if true_false[i]:
            if data[i, -1, 1][detec_type]:
                true_posit.append(i)
            else:
                true_negat.append(i)
        else:
            if data[i, -1, 1][detec_type]:
                false_negat.append(i)
            else:
                false_posit.append(i)

    true_posit = np.array(true_posit)
    true_negat = np.array(true_negat)
    false_posit = np.array(false_posit)
    false_negat = np.array(false_negat)
    return [true_posit, true_negat, false_posit, false_negat]


def show_confu_matri(data,
                     predi,
                     cutof=0.5,
                     featu=None,
                     bins=1,
                     detec_type='plane_moon_cut_injec',
                     ignor_strin_none=True,
                     save_figur_path='',
                     width=6,
                     heigh=5,
                     font_size=1.3):
    '''Shows a confusion matrix for the given data and predictions.'''
    if featu is not None and bins > 1:
        binne_data = bin_data(data, featu, bins, ignor_strin_none)
        # Set the number of rows and columns for the graphic
        numbe_colum = 3
        if len(binne_data) == 2:
            numbe_colum = 2
        numbe_rows = math.ceil(len(binne_data) / numbe_colum)
        sns.set(font_scale=font_size)
        plt.figure(figsize=(width * numbe_colum, heigh * numbe_rows))

        for i in range(len(binne_data)):
            plt.subplot(numbe_rows, numbe_colum, i + 1)
            # Mark TP, TN, FP, FN
            marke_data = creat_confu_matri(
                data[binne_data[i][:, 0].astype(int)], predi, cutof,
                detec_type)
            confu_matrix = np.array([[len(marke_data[1]),
                                      len(marke_data[2])],
                                     [len(marke_data[3]),
                                      len(marke_data[0])]])
            sns.heatmap(confu_matrix, annot=True, fmt='g')
            plt.title(
                f'{forma_names[featu]}: {binne_data[i][0, 1]}-{binne_data[i][-1, 1]}'
            )
            plt.xlabel('Predictions')
            plt.ylabel('Actual')
    else:
        plt.figure(figsize=(width, heigh))
        # Mark TP, TN, FP, FN
        marke_data = creat_confu_matri(data, predi, cutof, detec_type)
        confu_matrix = np.array([[len(marke_data[1]),
                                  len(marke_data[2])],
                                 [len(marke_data[3]),
                                  len(marke_data[0])]])
        sns.heatmap(confu_matrix, annot=True, fmt='g')
        sns.set_style('white')
        plt.title(f'Confusion Matrix')
        plt.xlabel('Predictions')
        plt.ylabel('Actual')

    plt.tight_layout()
    if save_figur_path:
        plt.savefig(f'{save_figur_path}confu_matrix-{float(time.time())}.pdf')
    plt.show()


def bin_data(data,
             featu,
             bins=1,
             equal_width_bins=True,
             ignor_strin_none=True):
    '''Returns a list of numpy arrays containing the index \
and value of binned ascending numerical data or based on their string feature.'''

    # Bin the data based on the spread of the feature, not the
    # occurence rate of a feature
    if equal_width_bins:
        unord_data = []
        # Extract the index and feature
        for i in range(len(data)):
            if isinstance(data[i, -1, 1][featu], (float, int)):
                unord_data.append([i, data[i, -1, 1][featu]])
        # Raise error if no data is returned
        if not len(unord_data):
            raise RuntimeError(f'No quantitative data provided. If trying to \
bin based on a qualitative feature, you must set equal_width_bins to false.')

        # Convert the unordered data to an array
        unord_data = np.array(unord_data)
        # Sort the unordered data in ascending order based on the feature
        index_order = np.argsort(unord_data[:, 1])
        order_infor = unord_data[index_order]

        # Determin the bin length through splitting the min and max feature values
        featu_min = min(order_infor[:, 1])
        featu_max = max(order_infor[:, 1])
        bin_lengt = (featu_max - featu_min) / bins
        binne_data = []

        # Bin all the data except the last bin
        start_index = 0
        for i in range(0, bins - 1):
            for ii in range(start_index, len(order_infor)):
                if order_infor[ii, 1] > (i + 1) * bin_lengt + featu_min:
                    binne_data.append(order_infor[start_index:ii])
                    start_index = ii
                    break
        # Add the last bin
        binne_data.append(order_infor[start_index:])
        # Return binned data, list of arrays in index, feature format
        return binne_data

    # Bin the data based on the occurence rate of a feature
    binne_data = []
    strin_featu = []
    for i in range(len(data)):
        # Bin based on string name
        if isinstance(data[i, -1, 1][featu], (str)):
            if ignor_strin_none:
                curre_featu = data[i, -1, 1][featu]
                if curre_featu != 'none':
                    # If curre_featu is not already in strin_featu, add curre_featu
                    if curre_featu not in strin_featu:
                        strin_featu.append(curre_featu)
                        binne_data.append([[i, curre_featu]])
                    # Otherwise add curre_featu to other same strin_featu
                    else:
                        for ii in range(len(strin_featu)):
                            if strin_featu[ii] == curre_featu:
                                binne_data[ii].append([i, curre_featu])
            else:
                curre_featu = data[i, -1, 1][featu]
                if curre_featu not in strin_featu:
                    strin_featu.append(curre_featu)
                    binne_data.append([[i, curre_featu]])
                else:
                    for ii in range(len(strin_featu)):
                        if strin_featu[ii] == curre_featu:
                            binne_data[ii].append([i, curre_featu])
    if len(strin_featu):
        # Convert binned data to type object, then indexes to integers
        for ii in range(len(binne_data)):
            binne_data[ii] = np.array(binne_data[ii]).astype(object)
        for ii in range(len(binne_data)):
            for iii in range(len(binne_data[ii])):
                binne_data[ii][iii][0] = int(binne_data[ii][iii][0])
        # Return binned string data
        return binne_data

    # Bin data based on a numerical feature
    numer_featu = []
    for i in range(len(data)):
        # Find all values of feature
        if isinstance(data[i, -1, 1][featu], (int, float)):
            numer_featu.append([i, data[i, -1, 1][featu]])
        elif data[i, -1, 1][featu] is not None and \
        data[i, -1, 1][featu].lower() != 'none':
            raise TypeError(
                f'Feature must be numerical or string, currently {type(data[i, -1, 1][feature])}.'
            )

    # Sort data
    numer_featu = np.array(numer_featu)
    if not len(numer_featu):
        raise TypeError(
            f'No suitable data to bin. Current data type is {type(data[i, -1, 1][featu])}.'
        )
    order_infor = np.argsort(numer_featu[:, 1])
    order_infor = numer_featu[order_infor]
    if bins < 1:
        raise ValueError(
            f'The number of bins must be greater than 0. Currently: {bins}.')
    if bins == 1:
        return [np.array(order_infor)]

    lengt_bins = len(order_infor) // bins
    # Bin the data
    for i in range(0, len(order_infor) - lengt_bins, lengt_bins):
        binne_data.append(order_infor[i:(i + lengt_bins)])
    # Add remainder data to last binned array
    if not len(order_infor) % lengt_bins:
        binne_data.append(order_infor[(i + lengt_bins):])
    else:
        binne_data[-1] = order_infor[(i):]
    # Return binned data, list of arrays in index, feature format
    return binne_data


def show_tpr_fpr(data,
                 predi,
                 cutof,
                 featu,
                 bins=35,
                 overp=False,
                 warn_missi_fp=True,
                 detec_type='plane_moon_cut_injec',
                 save_figur_path='',
                 width=20,
                 heigh=5):
    '''Shows how a change in a numerical feature changes the \
TPR and FPR in two plots or overlayed in one.'''
    # infor is an array containing the feature, true positive status, true negative status,
    # and false positive status (only contains relevant data)
    infor = []
    # Missing false positives that are not included since they have no moon
    # and therefore no moon feature
    missi_fp = 0
    # Binary prediction
    binar_predi = predi > cutof
    for i in range(len(data)):
        curre_featu = data[i, -1, 1][detec_type]
        if not isinstance(curre_featu,
                          (int, float)) and curre_featu is not None:
            raise TypeError(f'Feature must be an integer or float, currently: {type(curre_featu)}')
        if data[i, -1, 1][detec_type] == binar_predi[i]:
            if curre_featu:
                if data[i, -1, 1][featu] is not None:
                    infor.append([data[i, -1, 1][featu], True, False, False])
            else:
                if data[i, -1, 1][featu] is not None:
                    infor.append([data[i, -1, 1][featu], False, True, False])
        elif not data[i, -1, 1][detec_type] and binar_predi[i]:
            if data[i, -1, 1][featu] is not None:
                infor.append([data[i, -1, 1][featu], False, False, True])
            else:
                missi_fp += 1
    if warn_missi_fp:
        warnings.warn(f'Unaccounted for false positives: {missi_fp}')
    infor = np.array(infor)
    # Sort infor to prepare for binning
    order_infor = np.argsort(infor[:, 0])
    infor = infor[order_infor]
    # pr containts tpr, fpr
    pr = []
    # bound contains the min and max values of the feature on the bounds
    bound = []
    lengt_inter = len(infor) // (bins)
    remai = len(infor) - lengt_inter * bins
    for i in range(0, len(infor) - remai, lengt_inter):
        bound.append(infor[i + lengt_inter // 2][0])
        tp = 0
        tn = 0
        fp = 0
        for ii in range(i, i + lengt_inter):
            if ii < len(infor):
                tp += infor[ii][1]
                tn += infor[ii][2]
                fp += infor[ii][3]
        print(tp, tn, fp)
        pr.append([tp / (tp + fp), fp / (tn + fp)])
    pr = np.array(pr)
    if overp:
        plt.figure(figsize=(width, heigh))
        plt.plot(bound,
                 pr[:, 0],
                 linewidth=1.8,
                 c='navy',
                 label='True Positive Rate')
        plt.plot(bound,
                 pr[:, 1],
                 linewidth=1.8,
                 c='red',
                 label='False Positive Rate')
        plt.title(f'{featu} compared with TPR and FPR', fontsize=18)
        plt.xlabel(f'{featu}', fontsize=14)
        plt.ylabel('Rate', fontsize=14)
        plt.legend()
    else:
        plt.figure(figsize=(width, heigh * 2))
        plt.subplot(2, 1, 1)
        plt.plot(bound, pr[:, 0], linewidth=1.8, c='navy')
        plt.xlabel(f'{forma_names[featu]}', fontsize=14)
        plt.ylabel('True Positive Rate', fontsize=14)

        plt.subplot(2, 1, 2)
        plt.plot(bound,
                 pr[:, 1],
                 linewidth=1.8,
                 c='red',
                 label='False Positive Rate')
        plt.xlabel(f'{forma_names[featu]}', fontsize=14)
        plt.ylabel('False Positive Rate', fontsize=14)

    plt.tight_layout()
    if save_figur_path:
        plt.savefig(f'{save_figur_path}TPR_FPR-{float(time.time())}.pdf')
    plt.show()


def send_task_comple_email(subje='',
                           messa='',
                           high_prior=False,
                           recie='rfstatusupdates@gmail.com',
                           high_prior_recie='fradkin.rom@gmail.com',
                           sende='rfstatusupdates@gmail.com'):
    '''Sends an email informing the receiver that a task is complete.'''
    if subje=='' and messa=='':
        raise ValueError('"subje" and "messa" cannot both be empty strings.')
    email = MIMEMultipart()
    email['Subject'] = subje
    email['From'] = sende
    email.attach(MIMEText(messa))
    if high_prior:
        email['To'] = high_prior_recie
    else:
        email['To'] = recie
    serve = smtplib.SMTP('localhost')
    serve.sendmail(sende, recie, email.as_string())
    serve.quit()


def find_neare_index(data, value):
    '''Finds the nearest index in an array to the given value'''
    data = np.array(data)
    index = (np.abs(data - value)).argmin()
    return index


def find_signa_start_stop_index(signa):
    '''Finds the indexes of the beginning and ending of each transit.'''
    signa_index = np.where(signa < 1)[0]
    signa_start_stops = []
    curre_start_stop = [signa_index[0]]
    for i in range(1, len(signa_index) - 1):
        # Ignore single value transit occurences
        if signa_index[i - 1] + 1 != signa_index[i] and \
        signa_index[i + 1] - 1 != signa_index[i]:
            pass
        else:
            # Find start
            if signa_index[i] - signa_index[i - 1] > 1 and \
            not len(curre_start_stop):
                curre_start_stop.append(signa_index[i])
            # Find stop
            elif signa_index[i + 1] - signa_index[i] > 1 and \
            len(curre_start_stop):
                curre_start_stop.append(signa_index[i])
            # Reset start stop
            if len(curre_start_stop) == 2:
                signa_start_stops.append(curre_start_stop)
                curre_start_stop = []
    curre_start_stop.append(signa_index[-1])
    signa_start_stops.append(curre_start_stop)

    # Remove extraneous single point signal times
    i = 0
    while i < len(signa_start_stops):
        if len(signa_start_stops[i]) == 1:
            signa_start_stops.pop(i)
        else:
            i += 1
    return np.array(signa_start_stops)


def find_start(curve):
    '''Finds the start of a curve, ignoring padding.'''
    for i in range(len(curve) - 1):
        if curve[i, 0]:
            return i
    warnings.warn('Array contains only [0,0] and information.', RuntimeWarning)
    return -1


def injec_signa(curve,
                plane_max_numbe=1,
                moon_max_numbe=1,
                type_orbit_archi='planmoon',
                separ_plane_moon=True,
                anima_path=None):
    '''Returns an injected light curve with exoplanets and exomoons.'''
    
    if plane_max_numbe < 1:
        raise ValueError(f"The maximum number of planets must be \
greater than 1. Currently, it's {plane_max_numbe} and {moon_max_numbe} respectively."
                         )
    elif type_orbit_archi == 'planmoon' and moon_max_numbe < 1:
        raise ValueError(f"The maximum number of moons must be \
greater than 1 for a planet moon injection. Currently, it's {moon_max_numbe}."
                             )

    # Increase the number of dimensions if only a single curve is provided
    if len(curve.shape) == 2:
        curve = [curve]

    for curre_curve in curve:
        # If stellar radius or stellar mass is missing, do not inject
        # If TOI or EB, do not inject
        if curre_curve[-1, 1]['stell_radiu'] is None or \
curre_curve[-1, 1]['stell_mass'] is None or curre_curve[-1, 1]['toi'] or \
curre_curve[-1, 1]['eb']:
            continue

        if curre_curve[-1, 1]['curve_type'] != 'Light Curve':
            warnings.warn(
                "Trying to inject an injected curve.",
                RuntimeWarning)
            continue
            
        time_axis = curre_curve[find_start(curre_curve):-1, 0].astype(float)
        time_min = np.amin(time_axis)
        time_max = np.amax(time_axis)

        # Dictionary of conversion factors
        dictfact = ephesus.retr_factconv()

        # Star
        stell_radiu = curre_curve[-1, 1]['stell_radiu']
        stell_mass = curre_curve[-1, 1]['stell_mass']

        # Planets
        plane_numbe = np.random.randint(1, plane_max_numbe + 1)
        plane_index = np.arange(plane_numbe)
        plane_perio = np.empty(plane_numbe)
        plane_epoch = np.empty(plane_numbe)
        plane_incli_radia = np.empty(plane_numbe)
        plane_incli_degre = np.empty(plane_numbe)
        # Assign planet feature values
        for i in plane_index:
            plane_perio[i] = np.random.random() * 5 + 3
            plane_incli_radia[i] = np.random.random() * 0.03
            plane_incli_degre[i] = 180. / np.pi * np.arccos(
                plane_incli_radia[i])
        plane_epoch = tdpy.icdf_self(np.random.rand(plane_numbe), time_min,
                                     time_max)
        plane_eccen = 0
        plane_sin_w = 0
        if type_orbit_archi == 'planmoon' or type_orbit_archi == 'plan':
            plane_radiu = np.empty(plane_numbe)
            for i in plane_index:
                plane_radiu[i] = tdpy.icdf_powr(np.random.rand(1), 1, 12, 1.8)[0]
            plane_mass = ephesus.retr_massfromradi(plane_radiu)
            plane_densi = plane_mass / (4/3 * np.pi * plane_radiu**3)

        # Approximate total mass of the system
        total_mass = stell_mass
        plane_smax = ephesus.retr_smaxkepl(plane_perio, total_mass)
        rsma = None

        # Moon characteristics
        if type_orbit_archi.endswith('moon'):
            moon_numbe = np.empty(plane_numbe, dtype=int)
            moon_radiu = [[] for i in plane_index]
            moon_mass = [[] for i in plane_index]
            moon_densi = [[] for i in plane_index]
            moon_perio = [[] for i in plane_index]
            moon_epoch = [[] for i in plane_index]
            moon_smax = [[] for i in plane_index]
            moon_index = [[] for i in plane_index]
            # Hill radius of the planet
            radihill = ephesus.retr_radihill(plane_smax,
                                             plane_mass / dictfact['msme'],
                                             stell_mass)
            # Maximum semi-major axis of the moons
            moon_max_smax = 0.5 * radihill
            total_mass = np.sum(plane_mass) / dictfact['msme']
            for i in plane_index:
                arry = np.arange(1, moon_max_numbe + 1)
                prob = arry**(-2.)
                prob /= np.sum(prob)
                moon_numbe[i] = np.random.choice(arry, p=prob)
                moon_index[i] = np.arange(moon_numbe[i])
                moon_smax[i] = np.empty(moon_numbe[i])
                moon_radiu[i] = tdpy.icdf_powr(
                    np.random.rand(moon_numbe[i]), 0.2, 0.7, 1.8) * plane_radiu[i]
                moon_mass[i] = ephesus.retr_massfromradi(moon_radiu[i])
                moon_densi[i] = moon_mass[i] / (4/3 * np.pi * moon_radiu[i]**3)
                moon_min_smax = ephesus.retr_radiroch(
                    plane_radiu[i], plane_densi[i], moon_densi[i])
                for ii in moon_index[i]:
                    moon_smax[i][ii] = tdpy.icdf_powr(
                        np.random.rand(), moon_min_smax[ii],
                        moon_max_smax[i], 2.)
                moon_perio[i] = ephesus.retr_perikepl(
                    moon_smax[i], total_mass)
                moon_epoch[i] = tdpy.icdf_self(
                    np.random.rand(moon_numbe[i]), time_min, time_max)
            # Check to make sure realistic semi-major axis length
            if (moon_smax[i] > plane_smax[i] / 1.2).any():
                continue
            # Not used in simulation at the moment
            moon_eccen = 0
            moon_sin_w = 0
            moon_incli_radia = 0
        else:
            moon_perio = moon_epoch = moon_mass = moon_radiu \
= moon_incli_radia = moon_eccen = moon_sin_w = moon_densi = None
            moon_numbe = 0
            rsma = ephesus.retr_rsma(plane_radiu, stell_radiu, plane_smax)

        # Simulation settings
        trape_trans = True
        plane_type = 'plan'
        type_limb_darke = 'none'
        linea_limb_darke_coeff = 0.2
        quadr_limb_darke_coeff = 0.2

        # Create animation
        anima_name = ''
        if anima_path is not None:
            anima_name = f'{curre_curve[-1, 1]["tic_id"]}-{int(time.time())}'

        # Generate signal
        relat_flux_dicti = ephesus.retr_rflxtranmodl(
            time_axis,
            stell_radiu,
            plane_perio,
            plane_epoch,
            inclcomp=plane_incli_degre,
            massstar=stell_mass,
            radicomp=plane_radiu,
            masscomp=plane_mass,
            perimoon=moon_perio,
            epocmoon=moon_epoch,
            radimoon=moon_radiu,
            typecomp=plane_type,
            eccecomp=plane_eccen,
            sinwcomp=plane_sin_w,
            eccemoon=moon_eccen,
            sinwmoon=moon_sin_w,
            typelmdk=type_limb_darke,
            coeflmdklinr=linea_limb_darke_coeff,
            booltrap=trape_trans,
            coeflmdkquad=quadr_limb_darke_coeff,
            pathanim=anima_path,
            strgextn=anima_name,
            boolcompmoon=separ_plane_moon)

        relat_flux = relat_flux_dicti['rflx']
        
        # plane_trans_durat = calcu_trans_lengt(relat_flux_dicti['rflxcomp'])
        # print(plane_trans_durat)
        
        # Find the planet transit length for snr calculations
        try:
            print('here')
            curre_curve[-1, 1]['plane_trans_durat'] = calcu_trans_lengt(relat_flux_dicti['rflxcomp'])
            # print(plane_trans_durat)
        except KeyError:
            # plane_trans_durat = None
            print('here1')
            pass
        
        relat_flux_time = deter_signa_times(relat_flux, curre_curve)

        # Add signal to curve
        curre_curve[find_start(curre_curve):-1, 1] += relat_flux - 1
        plane_densi *= 5.972E27 * 1 / (4 / 3 * np.pi * (6.371E8 ** 3)) # conversion from earth mass/radius to g/cm^3
        if moon_densi is not None:
            for ii in range(len(moon_densi)):
                moon_densi[ii] *= 5.972E27 * 1 / (4 / 3 * np.pi * (6.371E8 ** 3))
            
        # Add characteristics to each curve's dictionary
        curre_curve[-1, 1]['max_ampli'] = -(min(relat_flux) - 1)
        curre_curve[-1, 1]['plane_numbe'] = plane_numbe
        curre_curve[-1, 1]['curve_injec'] = True
        curre_curve[-1, 1]['type_limb_darke'] = type_limb_darke
        curre_curve[-1, 1]['linea_limb_darke_coeff'] = linea_limb_darke_coeff
        curre_curve[-1, 1]['quadr_limb_darke_coeff'] = quadr_limb_darke_coeff
        curre_curve[-1, 1]['trape_trans'] = trape_trans
        curre_curve[-1, 1]['curve_type'] = 'Injected Curve'
        curre_curve[-1, 1]['injec_times'] = relat_flux_time
        # Turned off for memory purposes 
        curre_curve[-1, 1]['signa'] = relat_flux
        curre_curve[-1, 1]['type_orbit_archi'] = type_orbit_archi
        curre_curve[-1, 1]['plane_type'] = plane_type

        # Multiplanetary-moon systems
        if plane_numbe > 1 or moon_numbe > 1:
            curre_curve[-1, 1]['plane_epoch'] = plane_epoch
            curre_curve[-1, 1]['plane_perio'] = plane_perio
            curre_curve[-1, 1]['plane_radiu'] = plane_radiu
            curre_curve[-1, 1]['plane_mass'] = plane_mass
            curre_curve[-1, 1]['plane_incli'] = plane_incli_degre
            curre_curve[-1, 1]['plane_eccen'] = plane_eccen
            curre_curve[-1, 1]['plane_sin_w'] = plane_sin_w
            curre_curve[-1, 1]['plane_densi'] = plane_densi
            curre_curve[-1, 1]['moon_epoch'] = moon_epoch
            curre_curve[-1, 1]['moon_perio'] = moon_perio
            curre_curve[-1, 1]['moon_radiu'] = moon_radiu
            curre_curve[-1, 1]['moon_mass'] = moon_mass
            curre_curve[-1, 1]['moon_incli'] = moon_incli_radia
            curre_curve[-1, 1]['moon_eccen'] = moon_eccen
            curre_curve[-1, 1]['moon_sin_w'] = moon_sin_w
            curre_curve[-1, 1]['moon_densi'] = moon_densi
        # Single planet and moon systems
        elif plane_numbe:
            # Convert planet and moon numbers from lists/arrays to floats
            curre_curve[-1, 1]['plane_epoch'] = plane_epoch[0]
            curre_curve[-1, 1]['plane_perio'] = plane_perio[0]
            curre_curve[-1, 1]['plane_radiu'] = plane_radiu[0]
            curre_curve[-1, 1]['plane_mass'] = plane_mass[0]
            curre_curve[-1, 1]['plane_incli'] = plane_incli_degre[0]
            # Convert from sun radii to earth radii
            curre_curve[-1, 1]['ratio_plane_stell_radiu'] = plane_radiu[0] / (
                stell_radiu * 109.076)
            curre_curve[-1, 1]['plane_sin_w'] = plane_sin_w
            curre_curve[-1, 1]['plane_eccen'] = plane_eccen
            curre_curve[-1, 1]['plane_sin_w'] = plane_sin_w
            curre_curve[-1, 1]['plane_densi'] = plane_densi[0]

            if moon_numbe:
                curre_curve[-1, 1]['moon_epoch'] = moon_epoch[0][0]
                curre_curve[-1, 1]['moon_perio'] = moon_perio[0][0]
                curre_curve[-1, 1]['moon_radiu'] = moon_radiu[0][0]
                curre_curve[-1, 1]['moon_mass'] = moon_mass[0][0]
                curre_curve[-1, 1]['moon_numbe'] = moon_numbe[0]
                curre_curve[-1, 1]['ratio_moon_plane_radiu'] = moon_radiu[0][
                    0] / plane_radiu[0]
                curre_curve[-1,
                            1]['ratio_moon_stell_radiu'] = moon_radiu[0][0] / (
                                stell_radiu * 109.076)
                curre_curve[-1, 1]['moon_incli'] = moon_incli_radia
                curre_curve[-1, 1]['moon_eccen'] = moon_eccen
                curre_curve[-1, 1]['moon_sin_w'] = moon_sin_w
                curre_curve[-1, 1]['moon_densi'] = moon_densi[0][0]
                if separ_plane_moon:
                    relat_flux_plane = relat_flux_dicti['rflxcomp']
                    relat_flux_moon = relat_flux_dicti['rflxmoon']
                    relat_flux_plane_time = deter_signa_times(relat_flux_plane, curre_curve)
                    relat_flux_moon_time = deter_signa_times(relat_flux_moon, curre_curve)
                    curre_curve[-1, 1]['plane_signa'] = relat_flux_plane
                    curre_curve[-1, 1]['moon_signa'] = relat_flux_moon
                    curre_curve[-1, 1]['plane_signa_time'] = relat_flux_plane_time
                    curre_curve[-1, 1]['moon_signa_time'] = relat_flux_moon_time
            else:
                curre_curve[-1, 1]['moon_epoch'] = moon_epoch
                curre_curve[-1, 1]['moon_perio'] = moon_perio
                curre_curve[-1, 1]['moon_radiu'] = moon_radiu
                curre_curve[-1, 1]['moon_mass'] = moon_mass
                curre_curve[-1, 1]['moon_incli'] = moon_incli_radia
                curre_curve[-1, 1]['moon_eccen'] = moon_eccen
                curre_curve[-1, 1]['moon_sin_w'] = moon_sin_w
                curre_curve[-1, 1]['moon_numbe'] = moon_numbe
    if len(curve) - 1:
        return curve
    # Return only the curve if one curve is provided
    # Used for multiprocessing
    return curve[0]


def mark_TOI(curve, TOI_ID):
    '''Mark TESS Objects of Interest (TOIs) through a comparison \
between TOI ID and TIC ID.'''
    for i in range(len(curve)):
        curve_ID = curve[i, -1, 1]['tic_id']
        curve[i, -1, 1]['toi'] = bool(len(np.where(TOI_ID == curve_ID)[0]))


def retur_rando_sampl(numbe_of_sampl, data_lengt):
    '''Returns a numpy array containing a specified number of random indexes.'''
    if numbe_of_sampl == -1:
        return range(data_lengt)
    return np.random.choice(data_lengt, numbe_of_sampl, replace=False)


class resto_best_valid_accur(keras.callbacks.Callback):
    '''Restores the best validation accuracy at the end of training.'''
    def __init__(self):
        super().__init__()
        self.best_weight = None
        self.highe_valid_accur = 0

    def on_epoch_end(self, epoch, logs=None):
        curre_valid_accur = logs.get('val_accuracy')
        if np.greater(curre_valid_accur, self.highe_valid_accur):
            self.highe_valid_accur = curre_valid_accur
            self.best_weight = self.model.get_weights()

    def on_train_end(self, logs=None):
        print(f"\nRestoring model weights from the \
end of the best epoch ({self.highe_valid_accur:.1%}).")
        self.model.set_weights(self.best_weight)


class email_train_progr(keras.callbacks.Callback):
    '''Sends the validation accuracy of the model at the end of training.'''
    def __init__(self):
        super().__init__()
        # Why do I keep track of the validation accuracy? Wonderful question.
        # For whatever reason, on_train_end, the keys from the logs no longer
        # exist in this version of keras. Therefore, keeping track of validation 
        # after each epoch is the work around.
        self.highe_valid_accur = 0
        self.curre_epoch_numbe = 0
        self.preci = 0
        self.recal = 0
   
    def on_epoch_end(self, epoch, logs=None):
        self.curre_epoch_numbe += 1
        curre_valid_accur = logs.get('val_accuracy')
        if self.highe_valid_accur < curre_valid_accur:
            self.highe_valid_accur = curre_valid_accur
            self.preci = logs.get('val_preci')
            self.recal = logs.get('val_recal')
    
    def on_train_end(self, logs=None):
        infor = pd.read_csv(f'{main_path}rnns/tuner_backg_infor.csv')
        statu = 'Consistent'
        if self.highe_valid_accur > infor['best_accur'][0]:
            tuner_messa_backg_infor(setup=False, best_accur=self.highe_valid_accur,
                               best_preci=self.preci, best_recal=self.recal, 
                               best_trial_numbe=(infor['curre_trial_numbe'][0] + 1), curre_accur=self.highe_valid_accur,
                               curre_preci=self.preci, curre_recal=self.recal, curre_epoch=self.curre_epoch_numbe)
            if infor['curre_trial_numbe'][0] == 0:
                statu = 'Initial'
            else:
                statu = 'Improvement'
        else:
            tuner_messa_backg_infor(setup=False, curre_accur=self.highe_valid_accur,
                               curre_preci=self.preci, curre_recal=self.recal, curre_epoch=self.curre_epoch_numbe)   
            
        infor = pd.read_csv(f'{main_path}rnns/tuner_backg_infor.csv')
        send_task_comple_email(f'{statu} -- {infor["curre_trial_numbe"][0]} / {infor["total_trial_numbe"][0]}', tuner_messa_updat(infor))
        
        
def retur_predi_true_false(predi, y_data):
    '''Returns a numpy array containing whether each prediction was true or false.'''
    return np.array(abs(predi - y_data) < 0.5)


def norma_data(data, zero_one=True, zero=False, one=False):
    '''Normalizes the data from 0 to 1, around 0, or around 1.'''
    if zero_one + zero + one > 1:
        if zero_one and zero:
            raise ValueError(
                f'You can only normalize the data to one value range. \
You currently have zero to one set to {zero_one} and around zero set to {zero}.'
            )
        elif zero_one and one:
            raise ValueError(
                f'You can only normalize the data to one value range. \
You currently have zero to one set to {zero_one} and around one set to {one}.')
        if zero and one:
            raise ValueError(
                f'You can only normalize the data to one value range. \
You currently have zero set to {zero} and around one set to {one}.')
    # Normalize from 0 to 1
    if zero_one:
        return (data - min(data)) / (max(data) - min(data))
    # Center around 0
    elif zero:
        return data - statistics.mean(data)
    # Center around 1
    elif one:
        return data - statistics.mean(data) + 1
    raise ValueError('You must select a range to normalize the data on. \
Currently, none are selected.')


def show_roc(predi, y_data, cutof, save_figur_path='', width=16, heigh=5):
    '''Show a receiver operating characteristic curve.'''
    # Calculate fpr, tpr, and thresholds
    fpr, tpr, thres = metrics.roc_curve(y_data, predi.ravel())
    # Find the index in threshold closest to cutoff
    index = find_neare_index(thres, cutof)
    # Calculate the area under the roc curve
    auc = metrics.roc_auc_score(y_data, predi)

    plt.figure(figsize=(width, heigh))
    # Create a line showing no learning
    plt.plot([0, 1], [0, 1], 'k--')
    # Plot the provided cutoff point
    plt.plot(fpr[index],
             tpr[index],
             'go',
             color='red',
             label=f'Cutoff: {cutof}')
    # Plot the fpr and tpr
    plt.plot(fpr, tpr, label=f'AUC: {auc:.4}')
    # Font characteristics
    title_font = {'size': '16', 'color': 'black', 'weight': 'normal'}
    axes_font = {'size': '14'}

    plt.xlabel('False Positive Rate', **axes_font)
    plt.ylabel('True Positive Rate', **axes_font)
    plt.title('ROC Curve', **title_font)
    plt.legend()

    plt.tight_layout()
    if save_figur_path:
        plt.savefig(f'{save_figur_path}roc_curve-{float(time.time())}.pdf')
    plt.show()


def show_preci_recal(predi,
                     y_data,
                     cutof,
                     save_figur_path='',
                     width=16,
                     heigh=5):
    '''Show a precision recall curve, without a feature.'''
    # Calculate precision, recall, and thresholds
    preci, recal, thres = metrics.precision_recall_curve(y_data, predi.ravel())
    # Find the index in threshold closest to cutoff
    index = find_neare_index(thres, cutof)
    # Calculate the area under the pr curve
    auc = metrics.auc(recal, preci)

    plt.figure(figsize=(width, heigh))
    # Plot the provided cutoff point
    plt.plot(recal[index],
             preci[index],
             'go',
             color='red',
             label=f'Cutoff: {cutof}')
    # Plot the precision and recall
    plt.plot(recal, preci, label=f'AUC: {auc:.4}')
    # Font characteristics
    title_font = {'size': '16', 'color': 'black', 'weight': 'normal'}
    axes_font = {'size': '14', 'color': 'blue'}

    plt.xlabel('Recall', **axes_font)
    plt.ylabel('Precision', **axes_font)
    plt.title('Precision Recall Curve', **title_font)
    plt.legend()

    plt.tight_layout()
    if save_figur_path:
        plt.savefig(f'{save_figur_path}pr_curve-{float(time.time())}.pdf')
    plt.show()


def inser_inter_spot(curve, max_time_gap, caden=2):
    '''Inserts a string of \'inter_spot\' into the curve at the \
provided cadence (mins) given that the gap between the two points \
is less than the max gap time (mins) for the flux value.'''
    # Convert max time gap and cadence from mins to days
    max_time_gap /= 1440
    caden /= 1440
    # Convert to list if numpy array, since faster to work with
    if isinstance(curve, np.ndarray):
        curve = curve.tolist()

    i = 0
    # - 2 since need to ignore information as well
    while i < len(curve) - 2:
        # Check if there is a gap and if it less than the max_time_gap
        if curve[i + 1][0] - curve[i][0] > (2 * caden) and \
curve[i + 1][0] - curve[i][0] < max_time_gap:
            # Insert 'inter_spot'
            curve.insert(i + 1, [curve[i][0] + caden, 'inter_spot'])
        else:
            i += 1
    return np.array(curve)


def cut_curve(curve, max_time_gap, min_lengt, stand_lengt):
    '''Cuts the curve at each time slot greater than the max \
gap time. Deletes curves less than the length of min length. Cuts \
curves in multiple peices that are greater than or equal to the standard \
length.'''
    # Convert max_time_gap from minutes to days
    max_time_gap /= 1440

    cuts = []
    cut_index = []
    start = 0
    # - 3 to account for information and i + 1
    for i in range(1, len(curve) - 3):
        if curve[i + 1][0] - curve[i][0] > max_time_gap:
            cut_index.append([start, i])
            start = i + 1
    # Add final cut, given that it is not just one point
    if curve[-2][0] - curve[-3][0] < max_time_gap:
        cut_index.append([start, len(curve) - 1])
    infor = curve[-1][1]
    
    for curre_index in cut_index:
        curre_cut = curve[curre_index[0]:curre_index[1]]
        if len(curre_cut) > min_lengt:
            if len(curre_cut) <= stand_lengt:
                # Copy information
                curre_cut.append(['infor', infor.copy()])
#                 curre_cut[-1][1]['cut_start_index'] = curre_index[0]
                cuts.append(np.array(curre_cut).astype(object))
            else:
                # If the cut is greater than the standard length,
                # cut the curve into multiple peices of standard
                # length
                for i in range(0, len(curre_cut), stand_lengt - 1):
                    curre_cut_cut = curre_cut[i:i + stand_lengt - 1]
                    curre_cut_cut.append(['infor', infor.copy()])
#                     curre_cut_cut[-1][1]['cut_start_index'] = curre_index[0] + i
                    if min_lengt < len(curre_cut_cut) <= stand_lengt:
                        cuts.append(np.array(curre_cut_cut).astype(object))
    # A list of numpy arrays
    return cuts


def inter_curve(curve, splin_type):
    '''Interpolate a spline to the \'inter_spot\' points within the curve.'''
    # Create the x and y values for the spline
    x = curve[np.where(curve[:-1, 1] != 'inter_spot')[0], 0].astype(float)
    x = x[find_start(curve):]
    y = curve[np.where(curve[:-1, 1] != 'inter_spot')[0], 1].astype(float)
    y = y[find_start(curve):]
    # Create a function of type splin_type
    inter = scipy.interpolate.interp1d(x,
                                       y,
                                       kind=splin_type,
                                       fill_value='extrapolate')
    # Find values of 'inter_spot' in the function and find their value in the
    # interpolation. Then set the 'inter_spot' value to the interpolated value
    for i in range(len(curve) - 1):
        if curve[i, 1] == 'inter_spot':
            curve[i, 1] = inter(curve[i, 0])
    return curve


def retur_curve_injec_statu(cut):
    '''Returns the status of whether a curve contains an injection.'''
    injec_times = cut[-1, 1]['injec_times']
    if injec_times is None or len(injec_times) == 1:
        return False
    # Find start and end of cut
    start_time = cut[find_start(cut), 0]
    end_time = cut[-2, 0]
    for i in range(len(injec_times)):
        try:
            if start_time < injec_times[i, 0] < injec_times[i, 1] < end_time:
                return True
        # Return false if cut contains only [0,0]
        except TypeError:
            return False
    return False


def retur_TIC_ID_index(data, tic_id, ignor_warni=False):
    '''Returns the index of a TIC ID in a given dataset.'''
    for i in range(len(data)):
        if data[i, -1, 1]['tic_id'] == tic_id:
            return i
    if not ignor_warni:
        warnings.warn('Data does not contain TIC ID.', RuntimeWarning)
    return None


def retur_curve_color(curve):
    '''Returns the color of a curve based on this key:\
Light Curve: Orange
Injected Planet: Brown
Injected Planet and Moon: Navy
Detrended Planet: Grey
Detrended Planet and Moon: Red
Detrended Light Curve: Dark Slate Blue
Padded Cut Planet: Magenta
Padded Cut Planet and Moon: Green
Padded Cut Light Curve: Purple
'''
    if curve[-1, 1]['curve_type'] != 'Light Curve':
        if curve[-1, 1]['cut_numbe'] is not None:
            if curve[-1, 1]['plane_moon_cut_injec']:
                return 'green'
            elif curve[-1, 1]['plane_cut_injec']:
                return 'magenta'
            return 'purple'
        elif curve[-1, 1]['curve_type'] == 'Detrended Curve':
            if curve[-1, 1]['type_orbit_archi'] == 'planmoon':
                return 'red'
            elif curve[-1, 1]['type_orbit_archi'] == 'plan':
                return 'grey'
            return 'darkslateblue'
        elif curve[-1, 1]['curve_type'] == 'Injected Curve':
            if curve[-1, 1]['type_orbit_archi'] == 'planmoon':
                return 'navy'
            elif curve[-1, 1]['type_orbit_archi'] == 'plan':
                return 'brown'
        return 'orange'


def retur_title(curve,
                featu,
                uniqu_featu='',
                ignor_zeros=True,
                ignor_simul_featu_uninj=True,
                inclu_TIC_ID=True,
                detec_type='plane_moon_cut_injec'):
    '''Returns a title containing the features.'''
    infor = curve[-1, 1]
    title = ''
    numbe_featu = 0
    # The name of simulated features start with these keywords
    # These are to be ignored if ignore the simulated feature for uninjected
    # curves is turned on
    simul_name_start = ['plane', 'moon_', 'ratio']
    if uniqu_featu:
        title = f'{uniqu_featu}, '
        numbe_featu += 1
    if inclu_TIC_ID:
        title = f'{title}{forma_names["tic_id"]}: {infor["tic_id"]},'
        numbe_featu += 1
    for curre_featu in featu:
        simul_featu_flag=False
        if isinstance(infor[curre_featu], (int, str, np.int64)):
            # Use else statement for cut_numbe because cut_numbe can be 0
            if ignor_zeros and curre_featu != 'cut_numbe':
                if infor[curre_featu]:
                    if ignor_simul_featu_uninj and not infor[detec_type]:
                        for simul_chara in simul_name_start:
                            if curre_featu[:5] == simul_chara:
                                simul_featu_flag = True
                        if simul_featu_flag:
                            numbe_featu += 1 
                        else:
                            title = f'{title} {forma_names[curre_featu]}: {infor[curre_featu]},'
                            numbe_featu += 1                                
                    else:
                        title = f'{title} {forma_names[curre_featu]}: {infor[curre_featu]},'
                        numbe_featu += 1
            else:
                title = f'{title} {forma_names[curre_featu]}: {infor[curre_featu]},'
                numbe_featu += 1
        elif isinstance(infor[curre_featu], (float, np.float64)):
            if ignor_zeros:
                if infor[curre_featu]:
                    if ignor_simul_featu_uninj and not infor[detec_type]:
                        for simul_chara in simul_name_start:
                            if curre_featu[:5] == simul_chara:
                                simul_featu_flag = True
                        if simul_featu_flag:
                            numbe_featu += 1 
                        else:
                            # Round after 4 digits
                            title = f'{title} {forma_names[curre_featu]}: {infor[curre_featu]:.4},'
                            numbe_featu += 1
                    else:
                        title = f'{title} {forma_names[curre_featu]}: {infor[curre_featu]:.4},'
                        numbe_featu += 1
            else:
                title = f'{title} {forma_names[curre_featu]}: {infor[curre_featu]:.4},'
                numbe_featu += 1
        # Create a line break every 50 characters
        if not numbe_featu % 3:
            title = f'{title}\n'
    # Remove the last comma, and anything after it
    return title[:len(title) - title[::-1].find(',') - 1]


def show_curve(data,
               start_stop_tic_id=[],
               featu=[],
               highl_injec=False,
               highl_cuts=False,
               show_signa=False,
               ignor_zeros=True,
               save_figur_path=None,
               save_figur_name=None,
               figur_chara={'figsize': [15, 5]},
               title_chara={},
               x_chara={},
               y_chara={},
               legen_chara={}):
    '''Shows curves from start to stop index values, or a \
single TIC ID (if cuts, will show all). Includes features and can \
highlight injection times.'''
    if len(start_stop_tic_id) == 2:
        start = start_stop_tic_id[0]
        stop = start_stop_tic_id[1]
    elif len(start_stop_tic_id) == 1:
        start = retur_TIC_ID_index(data, start_stop_tic_id[0])
        if start is None:
            raise RuntimeError(
                f'TIC ID {start_stop_tic_id[0]} is not in the dataset.')
        for i in range(start + 1, len(data)):
            if retur_TIC_ID_index(data[i:], start_stop_tic_id[0],
                                  True) is None:
                stop = i
                break
    else:
        raise ValueError(
            f'Start stop index must be in [start, stop] or [TIC ID] format. \
Currently: {start_stop_tic_id}.')
    # Increase the height of the figure based on
    # the number of curves if the figure size is
    # specified
    try:
        figur_chara['figsize'][1] *= (stop - start)
    except:
        warnings.warn('Figure size not specified.', RuntimeWarning)
    figur, axes = plt.subplots((stop - start), 1, **figur_chara)
    # Create a list of axes to be accessed if only one curve
    if stop - start == 1:
        axes = [axes]
    remov_axes = []
    for i in range(start, stop):
        if show_signa:
            cut_start_index = data[i, -1, 1]['cut_start_index']
            if data[i, -1, 1]['cut_start_index'] is None:
                cut_start_index = find_start(data[i]) - data[i, -1,
                                                             1]['initi_paddi']
            ### Consider adding way to show signal cut, not just entire signal
            if data[i, -1,
                    1]['signa'] is not None and data[i, -1,
                                                     1]['cut_numbe'] is None:
                axes[i - start].scatter(data[i, find_start(data[i]):-1, 0],
                                        data[i, -1, 1]['signa'],
                                        s=0.4,
                                        c=retur_curve_color(data[i]))
            ###
            else:
                # Create a list of axes to remove
                remov_axes.append(i)
        else:
            axes[i - start].scatter(data[i, find_start(data[i]):-1, 0],
                                    data[i, find_start(data[i]):-1, 1],
                                    s=0.4,
                                    c=retur_curve_color(data[i]))
        title = f'{retur_title(data[i], featu, "", ignor_zeros)}'
        axes[i - start].set_xlabel('Time [BJD]', **x_chara)
        axes[i - start].set_ylabel('Relative Flux', **y_chara)
        # Highlight injection times
        if highl_injec:
            if data[i, -1, 1]['plane_cut_injec']:
                injec_label_name = 'Planet'
                if data[i, -1, 1]['injec_times'] is not None and np.size(data[i, -1, 1]['injec_times']):
                    for injec_times in data[i, -1, 1]['injec_times']:
                        if data[i, find_start(data[i]),0] < injec_times[0] < injec_times[1] < data[i, -2,0]:
                            axes[i - start].axvspan(injec_times[0] - 0.05,
                                                    injec_times[1] + 0.05,
                                                    facecolor='green',
                                                    alpha=0.22,
                                                    label=injec_label_name)
                            injec_label_name = None
                            axes[i - start].legend(**legen_chara)            
            elif data[i, -1, 1]['plane_moon_cut_injec']:
                if data[i, -1, 1]['plane_signa_time'] is not None and np.size(data[i, -1, 1]['plane_signa_time']):
                    injec_label_name = 'Planet'
                    for injec_times in data[i, -1, 1]['plane_signa_time']:
                        if data[i, find_start(data[i]),0] < injec_times[0] < injec_times[1] < data[i, -2,0]:
                            axes[i - start].axvspan(injec_times[0],
                                                    injec_times[1],
                                                    facecolor='red',
                                                    alpha=0.2,
                                                    label=injec_label_name)
                            injec_label_name = None
                            axes[i - start].legend(**legen_chara)
                            
                if data[i, -1, 1]['moon_signa_time'] is not None and \
                np.size(data[i, -1, 1]['moon_signa_time']):
                    injec_label_name = 'Moon'
                    for injec_times in data[i, -1, 1]['moon_signa_time']:
                        if data[i, find_start(data[i]),0] < injec_times[0] < injec_times[1] < data[i, -2,0]:
                            axes[i - start].axvspan(injec_times[0],
                                                    injec_times[1],
                                                    facecolor='blue',
                                                    alpha=0.2,
                                                    label=injec_label_name)
                            injec_label_name = None
                            axes[i - start].legend(**legen_chara)
        # Highlight cut times
        if highl_cuts and data[i, -1, 1]['cut_numbe'] is None and \
data[i, -1, 1]['cut_times'] is not None:
            cuts_color = ['blue', 'orange']
            color_switch = 0
            for cut_times in data[i, -1, 1]['cut_times']:
                color_switch += 1
                axes[i - start].axvspan(cut_times[0],
                                        cut_times[1],
                                        facecolor=cuts_color[color_switch % 2],
                                        alpha=0.07)
            title = f'{title}\nCuts Appear in Alternating Highlights'

        axes[i - start].set_title(title, **title_chara)

    for delet in remov_axes:
        ### Figure out way to hide extra whitespace from deleted axes
        figur.delaxes(axes[delet - start])
        ###

    figur.tight_layout()
    if save_figur_path:
        if save_figur_name is not None:
            figur.savefig(f'{save_figur_path}{save_figur_name}.pdf')
        else:
            figur.savefig(f'{save_figur_path}curves-{int(time.time())}.pdf')

    figur.show()

    # Reset the figsize because running the code can change the parameter
    figur_chara['figsize'] = [15, 5]


def log_predi_infor(
        data,
        predi,
        path,
        model_name,
        datas_name,
        first_colum=['predi', 'plane_moon_cut_injec', 'plane_cut_injec'],
        ignor_featu=['signa']):
    '''Logs prediction data and features to given path.'''
    # Create list of desired features
    featu = ['predi']
    for curre_featu in data[0, -1, 1]:
        if curre_featu not in ignor_featu:
            featu.append(curre_featu)
    # Create array of data by featu filled with np.nan
    predi_infor = np.full((len(data), len(featu)), np.NaN).astype(object)

    for i in range(len(data)):
        # Set prediction values
        predi_infor[i, 0] = predi[i]
        for ii in range(1, len(featu)):
            if data[i, -1, 1][featu[ii]] is not None:
                predi_infor[i, ii] = data[i, -1, 1][featu[ii]]

    log_infor = pd.DataFrame(predi_infor, columns=featu)

    # Calculate accuracy
    accur = sum(retur_predi_true_false((predi > 0.5), predi)) / len(predi)
    # Create metadata
    infor = f'"model_file":{model_name}, "accur":{accur}, "train_for":"Exomoons", \
"train_on":{datas_name}, "notes":"Accuracy is calculated with a cutoff of 0.5"'

    metad = np.full(len(featu), np.NaN).astype(object)
    metad[0] = 1

    # Insert metadata spot to the first row and sort by predi
    log_infor.loc[-1] = metad
    log_infor.sort_values(by=['predi'], ascending=False, inplace=True)
    # Put first_colum in front
    log_infor = log_infor[
        first_colum +
        [colum for colum in log_infor if colum not in first_colum]]
    # Insert metadata information
    log_infor.iloc[0, 0] = infor

    # Log csv to path
    log_infor.to_csv(f'{path}{datas_name}_{model_name}_{time.time()}.csv')


def show_predi_compa(data,
                     predi,
                     cutof,
                     start_stop,
                     statu,
                     featu=[],
                     detec_type='plane_moon_cut_injec',
                     save_figur_path='',
                     width=20,
                     heigh=5):
    '''Shows a side by side comparison of predictions based on \
a desired status.'''
    binar_predi = predi > cutof
    tp = []
    fp = []
    tn = []
    fn = []
    # Find tp, fp, tn, fn
    for i in range(len(data)):
        if statu == 'posit':
            if data[i, -1, 1][detec_type] and binar_predi[i]:
                tp.append(i)
            elif not data[i, -1, 1][detec_type] and binar_predi[i]:
                fp.append(i)
        elif statu == 'negat':
            if not data[i, -1, 1][detec_type] and not binar_predi[i]:
                tn.append(i)
            elif data[i, -1, 1][detec_type] and not binar_predi[i]:
                fn.append(i)
        elif statu == 'true':
            if data[i, -1, 1][detec_type] and binar_predi[i]:
                tp.append(i)
            elif not data[i, -1, 1][detec_type] and not binar_predi[i]:
                tn.append(i)
        elif statu == 'false':
            if not data[i, -1, 1][detec_type] and binar_predi[i]:
                fp.append(i)
            elif data[i, -1, 1][detec_type] and not binar_predi[i]:
                fn.append(i)
        else:
            raise ValueError(
                f'statu must be posit, negat, true, or, false. Currently: {statu}.'
            )

    rows = (start_stop[1] - start_stop[0])
    figur, axes = plt.subplots(rows, 2, figsize=(width, heigh * rows))

    if statu == 'negat':
        for i in range(start_stop[0], start_stop[1]):
            # Delete extra subplots
            if i - start_stop[0] == len(tn) or i - start_stop[0] == len(fn):
                for ii in range(i + start_stop[0], start_stop[1]):
                    figur.delaxes(axes[ii, 0])
                    figur.delaxes(axes[ii, 1])
                break
            axes[i - start_stop[0],
                 0].scatter(x=data[tn[i], find_start(data[tn[i]]):-1, 0],
                            y=data[tn[i], find_start(data[tn[i]]):-1, 1],
                            c=retur_curve_color(data[tn[i]]),
                            s=1.2)
            axes[i - start_stop[0],
                 0].set_title(f'{retur_title(data[tn[i]], featu)}')
            axes[i - start_stop[0],
                 1].scatter(x=data[fn[i], find_start(data[fn[i]]):-1, 0],
                            y=data[fn[i], find_start(data[fn[i]]):-1, 1],
                            c=retur_curve_color(data[fn[i]]),
                            s=1.2)
            axes[i - start_stop[0],
                 1].set_title(f'{retur_title(data[fn[i]], featu)}')
            axes[i - start_stop[0], 0].set_xlabel('Days [BJD]')
            axes[i - start_stop[0], 0].set_ylabel('Relative Flux')
            axes[i - start_stop[0], 1].set_xlabel('Days [BJD]')
            axes[i - start_stop[0], 1].set_ylabel('Relative Flux')
        figur.suptitle('True Negative and False Negative', fontsize=22, y=1)

    if statu == 'posit':
        for i in range(start_stop[0], start_stop[1]):
            if i - start_stop[0] == len(tp) or i - start_stop[0] == len(fp):
                for ii in range(i + start_stop[0], start_stop[1]):
                    figur.delaxes(axes[ii, 0])
                    figur.delaxes(axes[ii, 1])
                break
            axes[i - start_stop[0],
                 0].scatter(x=data[tp[i], find_start(data[tp[i]]):-1, 0],
                            y=data[tp[i], find_start(data[tp[i]]):-1, 1],
                            c=retur_curve_color(data[tp[i]]),
                            s=1.2)
            axes[i - start_stop[0],
                 0].set_title(f'{retur_title(data[tp[i]], featu)}')
            axes[i - start_stop[0],
                 1].scatter(x=data[fp[i], find_start(data[fp[i]]):-1, 0],
                            y=data[fp[i], find_start(data[fp[i]]):-1, 1],
                            c=retur_curve_color(data[fp[i]]),
                            s=1.2)
            axes[i - start_stop[0],
                 1].set_title(f'{retur_title(data[fp[i]], featu)}')
            axes[i - start_stop[0], 0].set_xlabel('Days [BJD]')
            axes[i - start_stop[0], 0].set_ylabel('Relative Flux')
            axes[i - start_stop[0], 1].set_xlabel('Days [BJD]')
            axes[i - start_stop[0], 1].set_ylabel('Relative Flux')
        figur.suptitle('True Positive and False Positive', fontsize=22, y=1)

    if statu == 'false':
        for i in range(start_stop[0], start_stop[1]):
            if i - start_stop[0] == len(fp) or i - start_stop[0] == len(fn):
                for ii in range(i + start_stop[0], start_stop[1]):
                    figur.delaxes(axes[ii, 0])
                    figur.delaxes(axes[ii, 1])
                break
            axes[i - start_stop[0],
                 0].scatter(x=data[fp[i], find_start(data[fp[i]]):-1, 0],
                            y=data[fp[i], find_start(data[fp[i]]):-1, 1],
                            c=retur_curve_color(data[fp[i]]),
                            s=1.2)
            axes[i - start_stop[0],
                 0].set_title(f'{retur_title(data[fp[i]], featu)}')
            axes[i - start_stop[0],
                 1].scatter(x=data[fn[i], find_start(data[fn[i]]):-1, 0],
                            y=data[fn[i], find_start(data[fn[i]]):-1, 1],
                            c=retur_curve_color(data[fn[i]]),
                            s=1.2)
            axes[i - start_stop[0],
                 1].set_title(f'{retur_title(data[fn[i]], featu)}')
            axes[i - start_stop[0], 0].set_xlabel('Days [BJD]')
            axes[i - start_stop[0], 0].set_ylabel('Relative Flux')
            axes[i - start_stop[0], 1].set_xlabel('Days [BJD]')
            axes[i - start_stop[0], 1].set_ylabel('Relative Flux')
        figur.suptitle('False Positive and False Negative', fontsize=22, y=1)

    if statu == 'true':
        for i in range(start_stop[0], start_stop[1]):
            if i - start_stop[0] == len(tp) or i - start_stop[0] == len(tn):
                for ii in range(i + start_stop[0], start_stop[1]):
                    figur.delaxes(axes[ii, 0])
                    figur.delaxes(axes[ii, 1])
                break
            axes[i - start_stop[0],
                 0].scatter(x=data[tp[i], find_start(data[tp[i]]):-1, 0],
                            y=data[tp[i], find_start(data[tp[i]]):-1, 1],
                            c=retur_curve_color(data[tp[i]]),
                            s=1.2)
            axes[i - start_stop[0],
                 0].set_title(f'{retur_title(data[tp[i]], featu)}')
            axes[i - start_stop[0],
                 1].scatter(x=data[tn[i], find_start(data[tn[i]]):-1, 0],
                            y=data[tn[i], find_start(data[tn[i]]):-1, 1],
                            c=retur_curve_color(data[tn[i]]),
                            s=1.2)
            axes[i - start_stop[0],
                 1].set_title(f'{retur_title(data[tn[i]], featu)}')
            axes[i - start_stop[0], 0].set_xlabel('Days [BJD]')
            axes[i - start_stop[0], 0].set_ylabel('Relative Flux')
            axes[i - start_stop[0], 1].set_xlabel('Days [BJD]')
            axes[i - start_stop[0], 1].set_ylabel('Relative Flux')
        figur.suptitle('True Positive and True Negative', fontsize=22, y=1)

    figur.tight_layout()
    if save_figur_path:
        figur.savefig(f'{save_figur_path}predi_compa-{int(time.time())}.pdf')
    figur.show()


def calcu_area_under_curve(data):
    '''Calculates the area under a nondifferentiable function using trapezoidal \
subregions.'''
    # data must be a 2D numpy array in the form of [[x1, y1], [x2, y2]...]
    area = 0
    # Calculate trapezoid areas
    for i in range(len(data) - 1):
        # Height * (Base1 + Base2) / 2
        area += (data[i + 1, 0] - data[i, 0]) * (data[i, 1] +
                                                 data[i + 1, 1]) / 2
    return area


def show_featu_preci_recal(data,
                           predi,
                           y_data,
                           cutof,
                           featu=None,
                           bins=20,
                           equal_width_bins=True,
                           stand_axis=True,
                           trans_type='plane_moon_cut_injec',
                           save_figur_path='',
                           figur_chara={'figsize': [15, 5]},
                           title_chara={'size': 16},
                           x_chara={'size': 14},
                           y_left_chara={'c': 'blue','size': 14},
                           y_left_uncer_chara={'color': 'blue', 'alpha': 0.2},
                           y_right_chara={'c': 'green','size': 14},
                           y_right_uncer_chara={'color': 'green', 'alpha': 0.2},
                           legen={'loc': 'upper left'}):
    '''Show precision and recall as a function of a quantitative feature.'''
    # Plot standard precision and recall curve
    if featu is None or featu == 'predi' or data is None:
        show_preci_recal(predi, y_data, cutof, save_figur_path, width, heigh)
    # Plot precision and recall as a function of a numerical feature
    else:
        binne_data = bin_data(data, featu, bins, equal_width_bins)
        preci = []
        recal = []

        # Used for calculating the false positive correction
        total_tp = 0
        total_fp = 0
        total_lengt = 0
        for i in range(len(binne_data)):
            for ii in range(len(binne_data[i])):
                curre_curve_index = binne_data[i][ii][0].astype(int)
                injec_statu = data[curre_curve_index, -1, 1][trans_type]
                predi_float = predi[curre_curve_index]
                total_tp += injec_statu and predi_float > cutof
                total_fp += not injec_statu and predi_float > cutof
                total_lengt += 1
        # Total number of missing false positives
        missi_fp = sum(predi > cutof) - total_tp - total_fp
        
        for i in range(len(binne_data)):
            tp = 0
            fp = 0
            fn = 0
            for ii in range(len(binne_data[i])):
                # Keep track of true positives, false positives, and false negatives
                curre_curve_index = binne_data[i][ii][0].astype(int)
                injec_statu = data[curre_curve_index, -1, 1][trans_type]
                predi_float = predi[curre_curve_index]
                tp += injec_statu and predi_float > cutof
                fp += not injec_statu and predi_float > cutof
                fn += injec_statu and predi_float < cutof
            
            # Add a correction to the false positives since many are not injected
            # so they may not have the feature used in binning.
            # Therefore, I proportionaly add the number of missing false positives
            # to the false positives to account for this disparity.
            # This is not a true representative of how many false positives are in
            # each bin, but is close enough for our purposes
            fp += missi_fp * len(binne_data[i]) / total_lengt
            
            # If divide by 0, ignore
            try:
                if tp or fp:
                    # Calculate precision
                    preci.append(
                        [np.mean(binne_data[i][:, 1]), tp / (tp + fp), 1 / (tp + fp)**(0.5)])
            except:
                pass
            try:
                if tp or fn:
                    # Calculate the recall
                    recal.append(
                        [np.mean(binne_data[i][:, 1]), tp / (tp + fn), 1 / (tp + fn)**(0.5)])
            except:
                pass
        # Convert precision and recall from lists to numpy arrays
        preci = np.array(preci)
        recal = np.array(recal)
        # Create a random guess baseline
        basel = np.array([[recal[0, 0], sum(y_data)/len(y_data)], [recal[-1, 0], sum(y_data)/len(y_data)]])
        # Create a plot
        figur, axes1 = plt.subplots(1, 1, **figur_chara)
        axes2 = axes1.twinx()

        axes1.plot(preci[:, 0], preci[:, 1], c=y_left_chara['c'])
        axes1.fill_between(preci[:, 0], preci[:, 1] + preci[:, 2], preci[:, 1] - preci[:, 2], **y_left_uncer_chara)
        axes2.plot(recal[:, 0], recal[:, 1], c=y_right_chara['c'])
        axes1.fill_between(recal[:, 0], recal[:, 1] + recal[:, 2], recal[:, 1] - recal[:, 2], **y_right_uncer_chara)
        axes1.plot(basel[:, 0], basel[:, 1], c='black', ls='--', label='Baseline')

        # Standardize the axis values from 0 to 1
        if stand_axis:
            axes1.set_ylim([0, 1])
            axes2.set_ylim([0, 1])

        # Set labels
        axes1.set_ylabel('Precision', **y_left_chara)
        axes2.set_ylabel('Recall', **y_right_chara)
        axes1.legend()
        axes1.legend(**legen)
        axes1.set_xlabel(f'{forma_names[featu]}', **x_chara)
        # Remove units from title
        axes1.set_title(
            f'Precision and Recall as a Function of {forma_names[featu].split(" [")[0]}',
            **title_chara)

        figur.tight_layout()
        if save_figur_path:
            plt.savefig(f'{save_figur_path}pr_curve-{int(time.time())}.pdf')
        figur.show()

def retur_most_recen(folde_path):
    '''Returns the name of the most recent file.'''
    if folde_path[-1] != '/':
        folde_path = f'{folde_path}/'
    files_in_folde = glob.glob(f'{folde_path}*')
    return max(files_in_folde, key=os.path.getctime).split('/')[-1]


def secon_to_hours_minut_secon(secon):
    '''Converts seconds to hours, minutes, and seconds.'''
    hours = int(secon // 3600)
    minut = int((secon - 3600 * hours) // 60)
    secon = int((secon - 3600 * hours - 60 * minut) // 1)
    return f'{hours:02}:{minut:02}:{secon:02}'


def remov_TOI(data):
    '''Removes TOIs from the dataset.'''
    toi_index = []
    for i in range(len(data)):
        if data[i, -1, 1]['toi']:
            toi_index.append(i)
    return np.delete(data, toi_index, axis=0)


def chang_moon_injec(curve):
    '''Adds or removes the moon signal from an injected curve.'''
    # Check to see whether the moon signal should be added or removed
    statu = 'add'
    for i in range(len(curve)):
        if curve[i, -1, 1]['plane_moon_cut_injec']:
            statu = 'remov'
            break
            
    if statu == 'add':
        for i in range(len(curve)):
            if curve[i, -1, 1]['unmod_plane_moon_cut_injec']:
                cut_start_index = curve[i, -1, 1]['cut_start_index'] - curve[i, -1, 1]['initi_paddi']
                cut_lengt = len(curve[i, find_start(curve[i]): -1])
                curve[i, find_start(curve[i]):-1, 1] += curve[i, -1, 1]['moon_signa'] \
[cut_start_index:cut_start_index + cut_lengt] - 1
                curve[i, -1, 1]['plane_moon_cut_injec'] = True
        return 'Added'
    else:
        for i in range(len(curve)):
            if curve[i, -1, 1]['unmod_plane_moon_cut_injec']:
                cut_start_index = curve[i, -1, 1]['cut_start_index'] - curve[i, -1, 1]['initi_paddi']
                cut_lengt = len(curve[i, find_start(curve[i]): -1])
                curve[i, find_start(curve[i]):-1, 1] -= curve[i, -1, 1]['moon_signa'] \
[cut_start_index:cut_start_index + cut_lengt] - 1
                curve[i, -1, 1]['plane_moon_cut_injec'] = False
        return 'Removed'
            
        
def binar_searc(data, targe, low=0, high=None):
    '''Binary search algorithm, returns index of target or None if target \
is not in the dataset.'''
    if high is None:
        high = len(data)
    targe_index = (low + high) // 2
    if data[targe_index] == targe:
        return targe_index    
    elif high > low:
        if data[targe_index] > targe:
            return binar_searc(data, targe, low, targe_index - 1)
        elif data[targe_index] < targe:
            return binar_searc(data, targe, targe_index + 1, high)
    else:
        return None
    
def show_histo(data,
               featu,
               bins=20,
               featu_min=-np.inf,
               featu_max=np.inf,
               trans_type='plane_moon_cut_injec',
               ignor_strin_none=True,
               save_figur_path='',
               save_figur_as='.pdf',
               line_histo_chara={'histtype': 'step', 'color': 'mediumblue'},
               backg_histo_chara={'alpha': 0.3, 'color': 'mediumblue'},
               bar_graph_chara={'color':['red', 'blue', 'brown']},
               figur_chara={'figsize': [15, 5]},
               title_chara={'size': 16},
               x_chara={'size': 14},
               y_chara={'size': 14}):

    featu_data = []
    quant = None
    for i in range(len(data)):
        curre_data = data[i, -1, 1][featu]
        if curre_data is not None:
            if isinstance(curre_data, (int, float)):
                featu_data.append(curre_data)
                quant = True
            elif isinstance(curre_data, (str)):
                featu_data = bin_data(data, featu, bins, False, ignor_strin_none)
                quant = False
            else:
                raise TypeError(f'Feature must be an int, float, or a string. Currently \
{type(curre_data)}.')
      
    plt.figure(**figur_chara)
    plt.xlabel(f'{forma_names[featu]}', **x_chara)
    plt.ylabel(f'Occurences', **y_chara)
    
    if quant:
        plt.hist(featu_data, bins, **line_histo_chara)
        plt.hist(featu_data, bins, **backg_histo_chara)
        plt.title(f'Histogram of {forma_names[featu].split(" [")[0]}', **title_chara)
        if save_figur_path:
            filen = f'{save_figur_path}histo-{featu}-{int(time.time())}'
            plt.savefig(f'{filen}{save_figur_as}')
    else:
        heigh = []
        for i in range(len(featu_data)):
            heigh.append(len(featu_data[i]))
        plt.bar(featu_data[:, 0], heigh, **bar_graph_chara)
        plt.title(f'Bar Graph of {forma_names[featu].split(" [")[0]}', **title_chara)
        if save_figur_path:
            filen = f'{save_figur_path}bar-{featu}-{int(time.time())}'
            plt.savefig(f'{filen}{save_figur_as}')
    plt.show()       
    
    
def count(start=1):
    '''Counts the number of occurences using a generator.'''
    count = start
    while True:
        yield count
        count += 1
        
def tuner_messa_updat(infor):
    curre_time_elaps = secon_to_hours_minut_secon(infor['curre_time'][0] - infor['previ_time'][0])
    total_time_elaps = secon_to_hours_minut_secon(infor['curre_time'][0] - infor['start_time'][0])
    estim_finis_time_unix = time.time() + ((infor['curre_time'][0] - infor['start_time'][0]) / \
infor['curre_trial_numbe'][0]) * (infor['total_trial_numbe'][0] - infor['curre_trial_numbe'][0])
    estim_finis_time_human = str(datetime.fromtimestamp(estim_finis_time_unix, timezone('US/Central'))).split('.')[0]
    messa = \
f'''\
Trial Number: {infor['curre_trial_numbe'][0]} / {infor['total_trial_numbe'][0]}
Accuracy, Precision, Recall: {infor['curre_accur'][0]:.2%}, {infor['curre_preci'][0]:.2%}, {infor['curre_recal'][0]:.2%}
Epochs Utilized: {infor['curre_epoch'][0]} / {infor['total_epoch_numbe'][0]}
Time Elapsed: {curre_time_elaps}

Total Time Elapsed: {total_time_elaps}
Estimated Finish: {estim_finis_time_human} CT

Best Trial Number: {infor['best_trial_numbe'][0]}
A, P, R: {infor['best_accur'][0]:.2%}, {infor['best_preci'][0]:.2%}, {infor['best_recal'][0]:.2%}

Baseline A, P, R: {infor['basel_accur'][0]:.2%}, {infor['basel_preci'][0]:.2%}, {infor['basel_recal'][0]:.2%}
Trial Start Time: {infor['start_time'][0]}
(All Statistics Are Validation Results)'''
    return messa

def tuner_messa_backg_infor(setup=True, total_trial_numbe=None, total_epoch_numbe=None,
                           basel_accur=None, basel_preci=None, basel_recal=None, best_accur=None,
                           best_preci=None, best_recal=None, best_trial_numbe=None, curre_accur=None,
                           curre_preci=None, curre_recal=None, curre_epoch=None):
    if setup:
        initi_time = time.time()
        infor = pd.DataFrame({'start_time': [initi_time], 
                            'previ_time': [initi_time],
                            'curre_time': [initi_time],
                            'total_trial_numbe': [total_trial_numbe],
                            'total_epoch_numbe': [total_epoch_numbe],
                            'basel_accur': [basel_accur],
                            'basel_preci': [basel_preci],
                            'basel_recal': [basel_recal],
                            'best_accur': [0],
                            'curre_trial_numbe': [0]})
    else:
        infor = pd.read_csv(f'{main_path}rnns/tuner_backg_infor.csv')
        if best_accur is not None:
            infor['best_accur'] = best_accur
            infor['best_preci'] = best_preci
            infor['best_recal'] = best_recal
            infor['best_accur'] = best_accur
            infor['best_trial_numbe'] = best_trial_numbe
        infor['curre_accur'] = curre_accur
        infor['curre_preci'] = curre_preci
        infor['curre_recal'] = curre_recal
        infor['curre_epoch'] = curre_epoch
        infor['curre_trial_numbe'] += 1
        infor['previ_time'] = infor['curre_time']
        infor['curre_time'] = time.time()

    infor.to_csv(f'{main_path}rnns/tuner_backg_infor.csv', index=False)
    
def recal(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall_keras = true_positives / (possible_positives + K.epsilon())
    return recall_keras


def preci(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision_keras = true_positives / (predicted_positives + K.epsilon())
    return precision_keras

def curre_time(timez='US/Central'):
    '''Returns the current day and time in a human readable format.'''
    return datetime.now(timezone(timez)).strftime("%d-%m-%Y_%H:%M:%S")

class custo_model_check(keras.callbacks.Callback):
    '''Restores the best validation accuracy at the end of training.'''
    def __init__(self, path):
        super().__init__()
        self.highe_valid_accur = 0
        self.path = path

    def on_epoch_end(self, epoch, logs=None):
        curre_valid_accur = logs.get('val_accuracy')
        if np.greater(curre_valid_accur, self.highe_valid_accur):
            self.highe_valid_accur = curre_valid_accur
            
    def on_train_end(self, logs=None):
        infor = pd.read_csv(f'{main_path}rnns/tuner_backg_infor.csv')
        self.model.save(f'{self.path}t:{infor["curre_trial_numbe"][0]}_va:{self.highe_valid_accur:.3}.h5')
        
        
def retur_tp_tn_fp_fn(predi, y_true, cutof=0.5):
    '''Classifies each curve as a true/false positive/negative. \
Returns a list of list of indexes in the order of tp, tn, fp, fn.'''
    if len(predi) != len(y_true):
        raise ValueError(
            f'The length of the prediction and y_true arrays must be the same. Currently \
{len(predi)} and {len(y_true)} respectively.')
    tp = []
    tn = []
    fp = []
    fn = []
    for i in range(len(predi)):
        if predi[i] > cutof and y_true[i]:
            tp.append(i)
        elif predi[i] < cutof and not y_true[i]:
            tn.append(i)
        elif predi[i] > cutof and not y_true[i]:
            fp.append(i)
        else:
            fn.append(i)
    return [tp, tn, fp, fn]


def graph_tp_tn_fp_fn(predi,
                      full_datas,
                      max_numbe_of_curve=50,
                      cutof=0.5,
                      featu=[],
                      highl_injec=False,
                      highl_cuts=False,
                      show_signa=False,
                      ignor_zeros=True,
                      save_figur_path=None,
                      figur_chara={'figsize': [15, 5]},
                      title_chara={},
                      x_chara={},
                      y_chara={},
                      legen_chara={},
                      detec_type='plane_moon_cut_injec'):
    '''Creates indivdual true/false positive/negative graphs.'''
    class_type = ['tp', 'tn', 'fp', 'fn']
    if save_figur_path is not None:
        # Create the folders to save the different graphs
        try:
            os.mkdir(f'{save_figur_path}/graph')
        except FileExistsError:
            pass
        for curre_type in class_type:
            try:
                os.mkdir(f'{save_figur_path}/graph/{curre_type}')
            except FileExistsError:
                pass

    # Create a y_true dataset from the full information dataset
    y_true = []
    for i in range(len(full_datas)):
        y_true.append(full_datas[i, -1, 1][detec_type])
    # Classify the curves into true/false positive/negative
    tp_tn_fp_fn_index = retur_tp_tn_fp_fn(predi, y_true, cutof)

    for i in range(len(class_type)):
        curre_numbe_save_curve = max_numbe_of_curve
        if len(tp_tn_fp_fn_index[i]) < max_numbe_of_curve:
            curre_numbe_save_curve = len(tp_tn_fp_fn_index[i])
        for ii in range(curre_numbe_save_curve):
            curre_figur_name = f"{class_type[i]}_{full_datas[tp_tn_fp_fn_index[i][ii], -1, 1]['tic_id']}"
            show_curve(
                full_datas,
                start_stop_tic_id=[
                    tp_tn_fp_fn_index[i][ii], tp_tn_fp_fn_index[i][ii] + 1
                ],
                featu=featu,
                highl_injec=highl_injec,
                highl_cuts=highl_cuts,
                show_signa=show_signa,
                ignor_zeros=ignor_zeros,
                save_figur_path=f'{save_figur_path}/graph/{class_type[i]}/',
                save_figur_name=curre_figur_name,
                figur_chara=figur_chara,
                title_chara=title_chara,
                x_chara=x_chara,
                y_chara=y_chara,
                legen_chara=legen_chara)
            
            
def deter_signa_times(signa_curve, injec_curve):
    '''Determines the start and end injection times based on a signal and injected curve.'''
    # Remove possible first value edge case
    if signa_curve[0] != 1 and signa_curve[1] == 1:
        signa_curve[0] = 1

    # Find signal times
    if len(np.where(signa_curve < 1)[0]) > 1:
        signa_curve_index = find_signa_start_stop_index(signa_curve)
        signa_curve_time = []
        for i in range(len(signa_curve_index)):
            signa_curve_time.append([
                injec_curve[find_start(injec_curve) + signa_curve_index[i][0], 0], 
                injec_curve[find_start(injec_curve) + signa_curve_index[i][1], 0]])
    else:
        signa_curve_time = [[]]
    signa_curve_time = np.array(signa_curve_time)

    return signa_curve_time

def calcu_trans_lengt(curve):
    '''Calculates the length of the transit in days from a relative flux curve'''
    # list of all possible transit lengths
    trans_lengt = []
    curre_lengt = 0
    for i in range(len(curve)):
        if curve[i] < 1:
            curre_lengt += 1
        elif curre_lengt > 0:
            trans_lengt.append(curre_lengt)
            curre_lengt = 0
            
    # return the longest (i.e. most complete) transit in days
    max_trans_lengt = max(trans_lengt)
    trans_lengt_minut = (max_trans_lengt - 1) * 2 / 1440
    return trans_lengt_minut

# def calcu_rms(curve):
#     '''Calculate the RMS of a curve.'''
#     # find the data indexes within the curve
#     useab_infor = np.where(curve[:-1, 1] != 0)[0]
#     # Find the sum of the squares (ie x subscript i squared)
#     total = np.sum(np.square(curve[useab_infor, 1]))
#     # multiply sum of squares by 1 / n
#     # and take the square root
#     return np.sqrt(len(useab_infor) * total)

def calcu_rms(curve):
    '''Calculate the RMS of a curve.'''
    from sklearn.metrics import mean_squared_error
    
    # find the data indexes within the curve
    useab_infor = np.where(curve[:-1, 1] != 0)[0]
    curve_actua = curve[useab_infor, 1]
    curve_corre = np.ones(len(useab_infor))
    rms = mean_squared_error(curve_actua, curve_corre, squared=False)
    return rms