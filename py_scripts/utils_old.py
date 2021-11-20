main_path = '/home/rfradkin/xom/'
tess_data = '/scratch/data/tess/lcur/spoc/raws/'
tess_metad = '/data/scratch/data/tess/meta/'
xom_data = '/data/scratch/xomoons/'

# import sys
# import numpy as np
# import pandas as pd
# import math
# import random
# import time
# import os 
# import matplotlib.pyplot as plt
import ephesus
import troia
import tdpy
# import tensorflow as tf
# import seaborn as sns
# from tensorflow import keras
# from tensorflow.keras import layers
# from sklearn.metrics import precision_recall_curve
# from sklearn.metrics import roc_curve
# import sklearn.metrics as metrics
# from sklearn.metrics import auc
# import matplotlib.pyplot as plt
# import matplotlib.font_manager
# import smtplib
# from email.mime.image import MIMEImage
# from email.mime.multipart import MIMEMultipart
plt.rcParams.update({'figure.max_open_warning': 0})
np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning) 

# def sect_file(sector_range, path):
#     file_names = []
#     for sector in range(sector_range[0], sector_range[1] + 1):
#         secto_numbe = str(sector)
#         if len(secto_numbe) != 2:
#             secto_numbe = f'0{secto_numbe}'        
#         for file in os.listdir(f'{path}sector-{secto_numbe}'):
#             if file.endswith('.fits'): 
#                 file_names.append([file, find_TIC_id(file)])
#     file_names_np = np.array(file_names, dtype = object)
#     unique_files = (file_names_np[np.unique(file_names_np[:, 1], return_index = True)[1], 0]).tolist()
#     return unique_files

# def lc_data(curve_names, information, path): 
#     file_list = []
#     for file in curve_names:
#         secto_numbe = str(find_sector(file))
#         if len(secto_numbe) != 2:
#             secto_numbe = f'0{secto_numbe}'
#         file_list.append(ephesus.read_tesskplr_file(f'{path}sector-{secto_numbe}/{file}')[0][:, 0:2])
#     curves = tf.keras.preprocessing.sequence.pad_sequences(file_list, padding = 'post', dtype = float)
#     curves = curves.astype(object)
#     curves = np.insert(curves, len(curves[0]), np.zeros((1, 2), dtype = object), axis = 1)
#     curves[:, len(curves[0]) - 1, 0] = 'infor'
#     if len(information) > 0:
#         for i in range(len(curves)):
#             curves[i, len(curves[0]) - 1, 1] = information.copy()
#             curves[i, -1, 1]['file_name'] = curve_names[i]
#             curves[i, -1, 1]['tic_id'] = find_TIC_id(curve_names[i])
#             curves[i, -1, 1]['curve_type'] = 'Light Curve'
#     return curves

# def find_TIC_id(TIC):
#     hyphens = []
#     for i in range(len(TIC)):
#         if TIC[i] == '-':
#             hyphens.append(i)
#     return int(TIC[hyphens[1] + 1:hyphens[2]])

# def find_sector(TIC):
#     hyphens = []
#     for i in range(len(TIC)):
#         if TIC[i] == '-':
#             hyphens.append(i)
#     return int(TIC[hyphens[0] + 2:hyphens[1]])

# def injections_start_stop_index(injections):
#     tf_injections = np.where(injections < 1)[0]
#     injection_start_stops = []
#     current_start_stop = [tf_injections[0]]
    
#     for i in range(1, len(tf_injections)  - 1):
#         if tf_injections[i - 1] + 1 != tf_injections[i] and tf_injections[i + 1] - 1 != tf_injections[i]:
#             pass
#         else:
#             if tf_injections[i] - tf_injections[i - 1] > 1 and len(current_start_stop) == 0:
#                 current_start_stop.append(tf_injections[i])
#             elif tf_injections[i + 1] - tf_injections[i] > 1 and len(current_start_stop) == 1:
#                 current_start_stop.append(tf_injections[i])
#             if len(current_start_stop) == 2:
#                 injection_start_stops.append(current_start_stop)
#                 current_start_stop = []
            
#     current_start_stop.append(tf_injections[-1])
#     injection_start_stops.append(current_start_stop)
    
#     i = 0
#     while i < len(injection_start_stops):
#         if len(injection_start_stops[i]) == 1:
#             injection_start_stops.pop(i)
#         else:
#             i += 1
#     return np.array(injection_start_stops)

# def index_to_time(indexes, curve):
#     if len(curve.shape) != 2:
#         print('Shape does not match')
#         return
#     times = []
#     for i in range(len(indexes)):
#         current_times = []
#         for j in range(len(indexes[i])):
#             current_times.append(curve[indexes[i,j], 0])
#         times.append(current_times)
#     return np.array(times)

# def injec(light_curve, plane_max_numbe = 1, moon_max_numbe = 1, type_orbit_archi='planmoon', anima_path=None):

#     if plane_max_numbe < 1 or moon_max_numbe < 1:
#         raise ValueError(f"The maximum number of companions and moons \
# must be greater than 1. Currently, it's {plane_max_numbe} and {moon_max_numbe} respectively.")
    
#     injec_curve = np.copy(light_curve)
    
#     if injec_curve.ndim == 2:
#         injec_curve = np.expand_dims(injec_curve, axis=0)
        
#     for k in range(len(injec_curve)):
#         injec_curve[k, -1, 1] = injec_curve[k, -1, 1].copy()
    
#     for modif_light_curve in injec_curve:
        
#         if modif_light_curve[-1, 1]['star_radiu'] is None or modif_light_curve[-1, 1]['star_mass'] is None and modif_light_curve[-1, 1]['curve_type'] == 'Light Curve':
#             continue
        
#         time = modif_light_curve[:find(modif_light_curve, 'end'), 0].astype(float)
#         time_min = np.amin(time)
#         time_max = np.amax(time)

#         # dictionary of conversion factors
#         dictfact = ephesus.retr_factconv()

#         # host
#         ## radius of the star [R_S]
#         star_radiu = modif_light_curve[-1, 1]['star_radiu']
#         ## total mass [M_S]
#         star_mass = modif_light_curve[-1, 1]['star_mass']

#         # companions
#         ## number of companions
#         compa_numbe = np.random.randint(1, plane_max_numbe + 1)
#         compa_index = np.arange(compa_numbe)

#         compa_perio = np.empty(compa_numbe)
#         compa_epoch = np.empty(compa_numbe)
#         compa_incli_radia = np.empty(compa_numbe)
#         compa_incli_degre = np.empty(compa_numbe)

#         for k in compa_index:
#             compa_perio[k] = np.random.random() * 5 + 2
#             compa_incli_radia[k] = np.random.random() * 0.04
#             compa_incli_degre[k] = 180. / np.pi * np.arccos(compa_incli_radia[k])
            
#         compa_epoch = tdpy.icdf_self(np.random.rand(compa_numbe), time_min, time_max)
#         compa_eccen  = 0
#         compa_sin_w = 0

#         if type_orbit_archi == 'planmoon' or type_orbit_archi == 'plan':
#             ## planet properties
#             ## radii [R_E]
#             compa_radiu = np.empty(compa_numbe)
#             for l in compa_index:
#                 compa_radiu[l] = 10 * np.random.random() + 1
#             ## masses [M_E]
#             compa_mass = ephesus.retr_massfromradi(compa_radiu)
#             ## densities [d_E]
#             compa_densi = compa_mass / compa_radiu**3

#         # approximate total mass of the system
#         total_mass = star_mass
    
#         ## semi-major axes [AU] 
#         compa_smax = ephesus.retr_smaxkepl(compa_perio, total_mass)
        
#         # sum of radii divided by the semi-major axis
#         rsma = None
#         if type_orbit_archi.endswith('moon'):

#             moon_numbe = np.empty(compa_numbe, dtype=int)
#             moon_radiu = [[] for j in compa_index]
#             moon_mass = [[] for j in compa_index]
#             moon_densi = [[] for j in compa_index]
#             moon_perio = [[] for j in compa_index]
#             moon_epoch = [[] for j in compa_index]
#             moon_smax = [[] for j in compa_index]
#             moon_index = [[] for j in compa_index]

#             # Hill radius of the companion
#             radihill = ephesus.retr_radihill(compa_smax, compa_mass / dictfact['msme'], star_mass)
#             # maximum semi-major axis of the moons 
#             moon_max_smax = 0.5 * radihill

#             total_mass = np.sum(compa_mass) / dictfact['msme'] + star_mass
            
#             for j in compa_index:
#                 # number of moons
#                 arry = np.arange(1, moon_max_numbe + 1)
#                 prob = arry**(-2.)
#                 prob /= np.sum(prob)
#                 moon_numbe[j] = np.random.choice(arry, p=prob)
#                 moon_index[j] = np.arange(moon_numbe[j])
#                 moon_smax[j] = np.empty(moon_numbe[j])
#                 # properties of the moons
#                 ## radii [R_E]
#                 moon_radiu[j] = tdpy.icdf_powr(np.random.rand(moon_numbe[j]), 0.15, 0.6, 2.)
                
#                 ## mass [M_E]
#                 moon_mass[j] = ephesus.retr_massfromradi(moon_radiu[j])
#                 ## densities [d_E]
#                 moon_densi[j] = moon_mass[j] / moon_radiu[j]**3
#                 # minimum semi-major axes
#                 moon_min_smax = ephesus.retr_radiroch(compa_radiu[j], compa_densi[j], moon_densi[j])
#                 # semi-major axes of the moons
#                 for jj in moon_index[j]:
#                     moon_smax[j][jj] = tdpy.icdf_powr(np.random.rand(), moon_min_smax[jj], moon_max_smax[j], 2.)
                    
#                 # orbital period of the moons
#                 moon_perio[j] = ephesus.retr_perikepl(moon_smax[j], total_mass)
                
#                 # mid-transit times of the moons
#                 moon_epoch[j] = tdpy.icdf_self(np.random.rand(moon_numbe[j]), time_min, time_max)
            
#             if (moon_smax[j] > compa_smax[j] / 1.2).any():
#                 continue
                
#             moon_eccen = 0
#             moon_sin_w = 0
#             moon_incli_radia = 0 
            
#         else:
#             moon_perio = moon_epoch = moon_mass = moon_radiu = moon_incli_radia = moon_numbe = moon_eccen = moon_sin_w = None
#             rsma = ephesus.retr_rsma(compa_radiu, star_radiu, compa_smax)
        
#         trape_trans = False
#         compa_type = 'plan'
#         type_limb_darke = 'none'
#         linea_limb_darke_coeff = 0.2
#         quadr_limb_darke_coeff = 0.2
        
#         anima_name = ''
#         if anima_path is not None:
#             if type_orbit_archi.endswith('moon'):
#                 for j in compa_index:
#                     anima_name += f'{j+1}-{moon_numbe[j]}_'
#             else:
#                 anima_name = f'{len(compa_index)}'          
#             anima_name = f'{modif_light_curve[-1, 1]["tic_id"]}_{anima_name[:-1]}'
            
#         # generate light curve
#         relat_flux = ephesus.retr_rflxtranmodl(time, star_radiu, compa_perio, compa_epoch, inclcomp=compa_incli_degre, \
#                             massstar=star_mass, radicomp=compa_radiu, masscomp=compa_mass, \
#                             perimoon=moon_perio, epocmoon=moon_epoch, radimoon=moon_radiu, typecomp=compa_type, \
#                             eccecomp=compa_eccen, sinwcomp=compa_sin_w, eccemoon=moon_eccen, sinwmoon=moon_sin_w, \
#                             typelmdk=type_limb_darke, coeflmdklinr=linea_limb_darke_coeff, booltrap=trape_trans, \
#                             coeflmdkquad=quadr_limb_darke_coeff, rsma=rsma, pathanim=anima_path, \
#                             strgextn=anima_name
#                             )

#         if relat_flux[0] != 1 and relat_flux[1] == 1:
#             relat_flux[0] = 1

#         if len(np.where(relat_flux < 1)[0]) > 1:

#             relat_flux_index = injections_start_stop_index(relat_flux)
#             relat_flux_time = index_to_time(relat_flux_index, modif_light_curve)
#         else:
#             relat_flux_time = [[]]
            
#         modif_light_curve[:find(modif_light_curve, 'end'), 1] += relat_flux - 1
        
#         modif_light_curve[-1, 1]['max_ampli'] = (min(relat_flux) - 1) * -1000
#         modif_light_curve[-1, 1]['compa_type'] = compa_type
#         modif_light_curve[-1, 1]['compa_epoch'] = compa_epoch
#         modif_light_curve[-1, 1]['compa_perio'] = compa_perio    
#         modif_light_curve[-1, 1]['compa_radiu'] = compa_radiu
#         modif_light_curve[-1, 1]['compa_mass'] = compa_mass                                                                 
#         modif_light_curve[-1, 1]['compa_incli_degre'] = compa_incli_degre     
#         modif_light_curve[-1, 1]['compa_eccen'] = compa_eccen
#         modif_light_curve[-1, 1]['compa_sin_w'] = compa_sin_w
#         modif_light_curve[-1, 1]['moon_epoch'] = moon_epoch
#         modif_light_curve[-1, 1]['moon_perio'] = moon_perio    
#         modif_light_curve[-1, 1]['moon_radiu'] = moon_radiu
#         modif_light_curve[-1, 1]['moon_mass'] = moon_mass                                                                 
#         modif_light_curve[-1, 1]['moon_incli'] = moon_incli_radia     
#         modif_light_curve[-1, 1]['moon_eccen'] = moon_eccen
#         modif_light_curve[-1, 1]['moon_sin_w'] = moon_sin_w  
#         modif_light_curve[-1, 1]['type_limb_darke'] = type_limb_darke                                                                
#         modif_light_curve[-1, 1]['linea_limb_darke_coeff'] = linea_limb_darke_coeff     
#         modif_light_curve[-1, 1]['quadr_limb_darke_coeff'] = quadr_limb_darke_coeff
#         modif_light_curve[-1, 1]['trape_trans'] = trape_trans                                                                                                                                 
#         modif_light_curve[-1, 1]['curve_type'] = 'Injected Curve'     
#         modif_light_curve[-1, 1]['injec_times'] = relat_flux_time   
#         modif_light_curve[-1, 1]['signal'] = relat_flux 
#         modif_light_curve[-1, 1]['type_orbit_archi'] = type_orbit_archi 
#         modif_light_curve[-1, 1]['moon_numbe'] = moon_numbe
#         modif_light_curve[-1, 1]['compa_numbe'] = compa_numbe
      
#     return injec_curve



# def inje(ligh_curv, id_radi_mass, max_numb_plan = 1, max_numb_moon = 1, step = 2):

#     inje_curv = np.copy(ligh_curv)
#     if max_numb_plan > 1:
#         numb_plan_all = np.random.randint(1, max_numb_plan + 1, len(ligh_curv))
#     else:
#         numb_plan_all = np.ones(len(ligh_curv))
        
#     if max_numb_moon > 1:
#         numb_moon_all = np.random.randint(1, max_numb_moon + 1, len(ligh_curv))
#     else:
#         numb_moon_all = np.ones(len(ligh_curv))

#     for i in range(0, len(ligh_curv), step):

#         if str(id_radi_mass[i, 1]) != 'nan' and str(id_radi_mass[i, 2]) != 'nan':

#             plan_peri = []
#             plan_radi = []
#             plan_epoc = []
#             plan_cosi = []

#             for j in range(int(numb_plan_all[i])):

#                 plan_peri.append(np.random.random() * 2 + 2)
#                 plan_radi.append(np.random.random() * 19 + 1)
#                 plan_epoc.append(np.random.random() * (ligh_curv[i, :find(ligh_curv[i], 'end'), 0][-1] - ligh_curv[i, 0, 0]))
#                 plan_cosi.append(0)#plan_cosi.append(np.random.random() * 0.03)

#             plan_peri = np.array(plan_peri)
#             plan_radi = np.array(plan_radi)
#             plan_epoc = np.array(plan_epoc)
#             plan_cosi = np.array(plan_cosi)

#             semi_major = ephesus.retr_smaxkepl(plan_peri, id_radi_mass[i, 2])
#             rsma = ephesus.retr_rsma(plan_radi, id_radi_mass[i, 1], semi_major)

#             plan_inje = ephesus.retr_rflxtranmodl((ligh_curv[i, :find(ligh_curv[i], 'end'), 0]), plan_peri, plan_epoc, 
#                                              plan_radi, id_radi_mass[i, 1], rsma, plan_cosi, ecce = 0., sinw = 0., booltrap = False)
            
#             if plan_inje[0] != 1 and plan_inje[1] == 1:
#                 plan_inje[0] = 1

#             inje_curv[i, :find(inje_curv[i], 'end'), 1] += plan_inje - 1

#             inje_time = 0        

#             if len(np.where(plan_inje < 1)[0]) > 1:

#                 plan_inde = injections_start_stop_index(plan_inje)
#                 plan_inje_time = index_to_time(plan_inde, ligh_curv[i])
                
                
#             moon_peri = np.zeros((int(numb_plan_all[i]), int(numb_moon_all[i])))
#             moon_radi = np.zeros((int(numb_plan_all[i]), int(numb_moon_all[i])))
#             moon_epoc = np.zeros((int(numb_plan_all[i]), int(numb_moon_all[i])))
#             moon_cosi = np.zeros((int(numb_plan_all[i]), int(numb_moon_all[i])))
                
#             for k in range(int(numb_plan_all[i])):

#                 for j in range(int(numb_moon_all[i])):
# #MAKE SURE EXOMOON RADIUS IS LESS THAN THAT OF EXOPLANET
#                     moon_peri[k, j] = plan_peri[k]
#                     moon_radi[k, j] = np.random.random() * 6 + 1
#                     moon_epoc[k, j] = plan_epoc[k] + 0.25 * np.random.random() - 0.125
#                     moon_cosi[k, j] = (0)#moon_cosi.append(np.random.random() * 0.03)
# #                     print(moon_peri, moon_radi, moon_epoc, moon_cosi)
#                     semi_major = ephesus.retr_smaxkepl(moon_peri[k, j], id_radi_mass[i, 2])
#                     rsma = ephesus.retr_rsma(moon_radi[k, j], id_radi_mass[i, 1], semi_major)

#                 moon_inje = ephesus.retr_rflxtranmodl((ligh_curv[i, :find(ligh_curv[i], 'end'), 0]), moon_peri[k], moon_epoc[k], 
#                                                  moon_radi[k], id_radi_mass[i, 1], rsma, moon_cosi[k], ecce = 0., sinw = 0., 
#                                                  booltrap = False)

# #                     inje_curv[i, :find(inje_curv[i], 'end'), 1] += moon_inje - 1

#                 fron_back_inde_leng = 1440

#                 plan_inje_inde_no_add = np.where(plan_inje < 1)[0]
#                 plan_inje_inde = []
# #                     print(plan_inje_inde_no_add)
#                 for l in range(plan_inje_inde_no_add[0] - fron_back_inde_leng, plan_inje_inde_no_add[0]):
#                     plan_inje_inde.append(l)

#                 for l in range(1, len(plan_inje_inde_no_add) - 1):
#                     if plan_inje_inde_no_add[l - 1] + 1 != plan_inje_inde_no_add[l]:
#                         for m in range(plan_inje_inde_no_add[l] - fron_back_inde_leng, plan_inje_inde_no_add[l]):
#                             plan_inje_inde.append(m)
#                     elif plan_inje_inde_no_add[l] != plan_inje_inde_no_add[l + 1] - 1:
#                         for m in range(plan_inje_inde_no_add[l] + 1, plan_inje_inde_no_add[l] + fron_back_inde_leng + 1):
#                             plan_inje_inde.append(m)
#                     else:
#                         plan_inje_inde.append(plan_inje_inde_no_add[l])

# #                     print(plan_inje_inde)       
#                 moon_inje_inde = np.where(moon_inje < 1)[0]
#                 moon_true_inje = np.intersect1d(plan_inje_inde, moon_inje_inde)
#                 if len(moon_true_inje) == 0:
#                     print(i)
# #                     print(i)
# #                     print(moon_true_inje)

#                 for l in moon_true_inje:
#                     inje_curv[i, l, 1] += moon_inje[l] - 1

# #                     inje_curv[i, :find(inje_curv[i], 'end'), 1] += moon_inje - 1
                    
#             inje_curv[i, find(ligh_curv[i], 'max amplitude'), 1] = (min(plan_inje) - 1) * -1000
#             inje_curv[i, find(ligh_curv[i], 'epoch'), 1] = plan_epoc       
#             inje_curv[i, find(ligh_curv[i], 'period'), 1] = plan_peri     
#             inje_curv[i, find(ligh_curv[i], 'star radius'), 1] = id_radi_mass[i, 1] 
#             inje_curv[i, find(ligh_curv[i], 'planet radius'), 1] = plan_radi   
#             inje_curv[i, find(ligh_curv[i], 'star mass'), 1] = id_radi_mass[i, 2]    
#             inje_curv[i, find(ligh_curv[i], 'inclination'), 1] = np.degrees(np.arccos(plan_cosi))        
#             inje_curv[i, find(ligh_curv[i], 'number of planets'), 1] = numb_plan_all[i]      
#             inje_curv[i, find(ligh_curv[i], 'curve type'), 1] = 'Injected Curve'      
#             inje_curv[i, find(ligh_curv[i], 'injection times'), 1] = plan_inje_time
#             inje_curv[i, find(ligh_curv[i], 'curve injected'), 1] = True 
        
#     return inje_curv

# def remo_TOI(ligh_curv_np, TOI_id):
    
#     TOI_id = np.sort(TOI_id)
#     ligh_curv = []
    
#     for i in range(len(ligh_curv_np)):

#         ligh_curv_id = find_TIC_id(ligh_curv_np[i, find(ligh_curv_np[i], 'TIC id'), 1])
#         if ligh_curv_id != TOI_id[np.searchsorted(TOI_id, ligh_curv_id)]:
#             ligh_curv.append(ligh_curv_np[i])
                        
#     return np.array(ligh_curv)

# def mark_toi(light_curve, toi_id):
    
#     toi_id = np.sort(toi_id)
#     for i in range(len(light_curve)):

#         light_curve_id = light_curve[i, -1, 1]['tic_id']
#         if light_curve_id == toi_id[np.searchsorted(toi_id, light_curve_id)]:
#             light_curve[i, -1, 1]['toi'] = True
#         else:
#             light_curve[i, -1, 1]['toi'] = False
                        

# def TIC_to_index(TIC, curves, information = None):
#     if len(curves.shape) == 3:
#         for i in range(len(curves)):
#             if find_TIC_id(curves[i][find(curves[i], 'TIC id'), 1]) == TIC:
#                 return i  
#     elif len(curves.shape) == 2:
#         for i in range(len(curves)):
#             if find_TIC_id(curves[find(curves, 'TIC id'), 1]) == TIC:
#                 return i   
#     elif len(curves.shape) == 1:
#         if(information is None):
#             print(f'Shape is 1: Information Needed.')
#             return            
#         for i in range(len(curves)):
#             if find_TIC_id(curves[find(curves, 'TIC id', Information)]) == TIC:
#                 return i
            
#     print(f'Index Not Found.')
#     return

# def list_contains(possible_features, desired_features):
#     contains_list = []
#     found = False
#     for feature in possible_features:
#         for current_feature in desired_features:
#             if feature.lower() == current_feature.lower():
#                 contains_list.append(True)
#                 found = True
#         if not found:
#             contains_list.append(False)
#         found = False
    
#     return contains_list  

# def find(curve, characteristic, ignore_padding = False, information = None):
# #When characteristic = 'end', returns last value index (not last value index + 1)    
#     value = None
#     end = None
    
#     if characteristic.lower() == 'start':
#         if len(curve.shape) == 2:
#             for i in range(len(curve)):
#                 if curve[i, 0] != 0:
#                     return i   
#         elif len(curve.shape) == 1:
#             for i in range(len(curve)):
#                 if curve[i] != 0:
#                     return i
#         else:
#             print(f'Light curve start not found.')
#             return
    
#     if not ignore_padding:
#         if len(curve.shape) == 2:
#             for i in range(len(curve) - 1, -1, -1):
#                 if type(curve[i, 0]) == float and curve[i, 0] != 0:
#                     end = i
#                     break
#                 if end is None:
#                     for i in range(len(curve) - 1, -1, -1):
#                         if type(curve[i, 0]) == float:
#                             end = i - 1
#                             break
#         elif len(curve.shape) == 1:
#             for i in range(len(curve)):
#                 if curve[i] == 0:
#                     end = i
#                     break
#                 if end is None:
#                     for i in range(len(curve)):
#                         if type(curve[i]) == np.str_ or type(curve[i]) == str:
#                             end = i - 1    
#                             break
#         else:
#             print(f'Light curve end not found.')
#             return    
#     else:
#         if len(curve.shape) == 2:
#             for i in range(len(curve) - 1, -1, -1):
#                 if type(curve[i, 0]) == float:
#                     end = i
#                     break
#         elif len(curve.shape) == 1:
#             for i in range(len(curve)):
#                 if type(curve[i]) == np.str_ or type(curve[i]) == str:
#                     end = i - 1 
#                     break
#         else:
#             print(f'Light curve end not found.')
#             return
        
#     if characteristic.lower() == 'end':
#         return end
    
#     if len(curve.shape) == 2:
#         for i in range(len(curve) - 1, end, -1):
#             if((type(curve[i, 0]) == np.str_ or type(curve[i, 0]) == str) and \
#                curve[i, 0].split(' [')[0].lower() == characteristic.lower()):
#                 return i
#     elif len(curve.shape) == 1:
#         if information is None:
#             return
#         for i in range(len(information)):
#             if information[::-1][i].split(' [')[0].lower() == characteristic.lower():
#                 return len(curve) - i - 1
    
#     print(f'Characteristic ({characteristic}) not found.')
#     return

def time_difference(curve, mins):
    days = mins/1440
    breaks = []
    if len(curve.shape) == 2:
        for i in range(find(curve, 'end') - 1):
            if curve[i + 1, 0] - curve[i, 0] > days:
                breaks.append(i + 1)
        return breaks
    else:
        print('Curve not accepted. Check curve shape.')
    return -1

# def cut_curve(curve, max_gap_time, min_cut_time, standard_length, cadence = 2):
#     breaks = time_difference(curve, max_gap_time)
#     breaks.insert(0, 0)
#     breaks.append(find(curve, 'end') + 1)
#     i = 1
#     while i < len(breaks):
#         if breaks[i] - breaks[i - 1] > standard_length:
#             midpoint = (breaks[i] + breaks[i - 1]) // 2
#             breaks.insert(i, midpoint)
#         else:
#             i += 1
            
#     broken_curves = []
    
#     if type(breaks) != int:
#         for i in range(len(breaks) - 1):
#             infor_copy = np.expand_dims(['infor', curve[-1, 1].copy()], axis=0)
#             current_cut = np.concatenate((curve[breaks[i]: breaks[i + 1]], infor_copy))
#             broken_curves.append(current_cut)            
            
#     min_index = min_cut_time / cadence

#     short_broken_curves = []
#     for i in range(len(broken_curves)):
#         if len(broken_curves[i]) > min_index:
#             short_broken_curves.append(broken_curves[i])

#     return short_broken_curves

def cut_is_injected(cut):
    injected_times = cut[-1, 1]['injec_times']
    if injected_times is None:
        return False
    start_time = cut[0, 0]
    end_time = cut[find(cut, 'end'), 0]
    for i in range(len(injected_times)):
#         print(f'start_time {start_time}')
#         print(f'injected_times[{i}, 0] {injected_times[i, 0]}')
        if start_time < injected_times[i, 0] < end_time and start_time < injected_times[i, 0] < end_time:
            return True
    return False

# def insert_nans(_curve, max_time, cadence = 2.5):
#     curve = _curve[np.where(_curve[:find(_curve, 'end') + 1, 1] > 0)[0]].tolist()
#     background_statistics = len(_curve) - find(_curve, 'end', True) - 1
#     i = 0
#     while i < len(curve) - background_statistics:
#         if curve[i + 1][0] - curve[i][0] > cadence/1440:
#             if curve[i + 1][0] - curve[i][0] < max_time/1440:
#                 curve.insert(i + 1, [curve[i][0] + cadence/1440, np.nan])
#             else:
#                 i += 1
#         else:
#             i += 1
#     df = pd.DataFrame(np.array(curve, dtype = float), columns = ['Time', 'Flux'])
#     return df

def cut_and_pad(curves, type_orbit_archi, max_gap_time = 10, min_cut_time = 1440, spline_type = 'akima', standard_length = 1900):
    list_cuts = []
    interpolated_curves = []
    for i in range(len(curves)): 
        df = insert_nans(curves[i], max_gap_time)
        df = df.assign(Flux = df.Flux.interpolate(method = spline_type))
        interpolated_curves.append(df)

        inter_curve = np.concatenate((df.to_numpy(dtype = object), np.expand_dims(curves[i][-1], axis=0)))
        cuts = cut_curve(inter_curve, \
                         max_gap_time, min_cut_time, standard_length)
        for j in range(len(cuts)):
#             print(f'cuts {j} {cuts[j]}')
            cuts[j][-1, 1]['cut_numbe'] = j
            cuts[j][-1, 1]['fill_spline_type'] = spline_type
#             print(cuts[j][-1, 1]['injec_type'])
            if cuts[j][-1, 1]['type_orbit_archi'] == type_orbit_archi:
#                 print('here')
#                 print(cuts[j][-1, 1]['injec_times'])
                cuts[j][-1, 1]['cut_injec'] = cut_is_injected(cuts[j])
            else:
                cuts[j][-1, 1]['cut_injec'] = False
            list_cuts.append(cuts[j].tolist())
    return tf.keras.preprocessing.sequence.pad_sequences(list_cuts, maxlen = (standard_length + 1), dtype = object)

# def rand_samp(num_of_samp, data_len):
#     if num_of_samp == -1:
#         return range(data_len)
#     return np.random.choice(data_len, num_of_samp, replace = False)

# def normalize_data(data):
#     return (data - data.min()) / (data.max() - data.min())

# class restore_best_val_accuracy(keras.callbacks.Callback):
#     def __init__(self, patience=0):
#         super().__init__()
#         self.best_weights = None
#         self.best_acc = 0

#     def on_epoch_end(self, epoch, logs=None):
#         current = logs.get("val_accuracy")
#         if np.greater(current, self.best_acc):
#             self.best_acc = current
#             self.best_weights = self.model.get_weights()

#     def on_train_end(self, logs=None):
#         print(f"\nRestoring model weights from the end of the best epoch ({self.best_acc:.2%}).")
#         self.model.set_weights(self.best_weights)
        
# def true_false(pred, y_test):
#     return np.array(abs(pred - y_test) < 0.5)

def horizontal_subplot_curve_plotter(curves, features, pred_status = [], ignore_padding = True):
    
    color_list = ['blue', 'red', 'yellow', 'purple', 'orange']
    fig, ax = plt.subplots(1, len(curves), figsize=(25, 5))
    
    initial = 1 
    for j in range(len(curves)):
        if len(pred_status) == 0:
            title = f'TIC ID: {curves[j][-1, 1]["tic_id"]},'
        else:
            k = -1
            if pred_status[j].find('True') != -1:
                k = 0
            elif pred_status[j].find('False') != -1:
                k = 1
            title = f'Prediction Status: {pred_status[j]}, TIC ID: {curves[j][-1, 1]["tic_id"]},'
            initial += 1
        for i in range(len(features)):
            if features[i].lower() == 'tic_id':
                features.pop(i)
                break
#         print(curves[j])
        for i in range(len(features)):
            if (i + initial) % 3 == 0:
                title = f'{title}\n'
            if i == len(features) - 1:
#                 print(curves[j][-1, 1])
                if type(curves[j][-1, 1][features[i]]) == str:
                    try:
                        title = f'{title} {features[i]}: {curves[j][-1, 1][features[i]]}'
                    except:
                        pass
                else:
                    try:
                        title = f'{title} {features[i]}: {float(curves[j][-1, 1][features[i]]):.4}'
                    except:
                        pass
            else:
                if type(curves[j][-1, 1][features[i]]) == str:
                    try:
                        title = f'{title} {features[i]}: {curves[j][-1, 1][features[i]]},'
                    except:
                        pass
                else:
                    try:    
                        title = f'{title} {features[i]}: {float(curves[j][-1, 1][features[i]]):.4},'
                    except:
                        pass

        if ignore_padding:
            x_axis = curves[j][np.where(curves[j][:find(curves[j], 'end')] > 0)[0], 0]
            flux = curves[j][np.where(curves[j][:find(curves[j], 'end')] > 0)[0], 1]
            
        else:
            x_axis = curves[j][:find(curves[j], 'end'), 0]
            flux = curves[j][:find(curves[j], 'end'), 1]
        if k == -1:
            k = j
        ax[j].set_title(title, wrap = True)
        ax[j].set_xlabel('Time (TBJD)')
        ax[j].set_ylabel('Flux')
        ax[j].scatter(x_axis, flux, s = 0.5, c = color_list[k])
        
def sing_curv_plot(curve, features, color, save_figu = False, save_figu_path = '', figu_name = '', ignore_padding = True):
    
    initial = 1
    title = f'TIC ID: {find_TIC_id(curve[find(curve, "tic id"), 1])}'
    for i in range(len(features)):
        if features[i].lower() == 'tic id':
            features.pop(i)
            break

    for i in range(len(features)):
        
        if (i + initial) % 8 == 0:
            title = f'{title}\n'
        if i == len(features) - 1:
            if type(curve[find(curve, features[i]), 1]) == str:
                title = f'{title} {curve[find(curve, features[i]), 0]}: {curve[find(curve, features[i]), 1]}'
            else:
                title = f'{title} {curve[find(curve, features[i]), 0]}: {float(curve[find(curve, features[i]), 1]):.4}'
        else:
            if type(curve[find(curve, features[i]), 1]) == str:
                title = f'{title} {curve[find(curve, features[i]), 0]}: {curve[find(curve, features[i]), 1]},'
            else:
                title = f'{title} {curve[find(curve, features[i]), 0]}: {float(curve[find(curve, features[i]), 1]):.4},'      
            
    if ignore_padding:
        flux = curve[np.where(curve[:find(curve, 'end')] > 0)[0], 1]
        x_axis = curve[np.where(curve[:find(curve, 'end')] > 0)[0], 0]
    else:
        flux = curve[:, 1]
        x_axis = curve[: 0]
    front_time = (x_axis[0] // 100) * 100
    x_axis = ((x_axis - front_time))
    plt.rc('axes', titlesize=10) 
    plt.gcf().set_size_inches(25, 5)
    plt.scatter(x_axis, flux, s = 0.5, c = color)
    plt.title(title)
    plt.xlabel(f'Time (BJD - {int(front_time)})')
    plt.ylabel('Flux')
    if save_figu:
        if figu_name == '':
            plt.savefig(f"{save_figu_path}{float(time.time())}.pdf")
        else:
             plt.savefig(f"{save_figu_path}{figu_name}.pdf")
    plt.show()
    
def show_curv(curv, star, stop, feat, show_inje = True, save_figu = False, save_figu_path = '', ignore_padding = True):
    
    colo = ['green', 'purple', 'orange']
    figu_name = ''
    
    for i in range(star, stop):
        
        if save_figu:
            figu_name = curv[i, find(curv[i], 'TIC id'), 1]
            
        if show_inje:
            if curv[i, find(curv[i], 'Cut Injected'), 1] or (curv[i, find(curv[i], 'Curve Injected'), 1] 
                                                             and curv[i, find(curv[i], 'Cut Number'), 1] == -1):
                show_inje_curv_plot(curv[i], feat, colo[1], save_figu, save_figu_path, figu_name,
                                    widt = 20, heig = 5, ignore_padding = True)
            else:
                show_inje_curv_plot(curv[i], feat, colo[2], save_figu, save_figu_path, figu_name,
                                    widt = 20, heig = 5, ignore_padding = True)
        else:
            sing_curv_plot(curv[i], feat, colo[0], save_figu, save_figu_path, figu_name, ignore_padding)

def show_pred_resu(curves, prediction_type, features, true_false, curve_range, ignore_padding = True):
    true = np.where(true_false == 1)[0]
    false = np.where(true_false == 0)[0]
    
    injected = []
    uninjected = []
    for i in range(len(curves)):
        if curves[i, -1, 1]['cut_injec']:
            injected.append(i)
        else:
            uninjected.append(i)
    injected = np.array(injected)
    uninjected = np.array(uninjected)
#     print(injected)
#     injected = np.where(curves[:, -1, 1]['cut_injected'] == 1)[0]
#     uninjected = np.where(curves[:, -1, 1] == 0)[0]

    true_positive = np.intersect1d(injected, true)
    false_positive = np.intersect1d(uninjected, false)
    true_negative = np.intersect1d(uninjected, true)
    false_negative = np.intersect1d(injected, false)
    
    for i in range(curve_range[0], curve_range[1]):
        if prediction_type.lower() == 'true':
            horizontal_subplot_curve_plotter([curves[true_positive[i]], curves[true_negative[i]]], features, 
                                             ['True Positive', 'True Negative'], ignore_padding)
        elif prediction_type.lower() == 'false':
            horizontal_subplot_curve_plotter([curves[false_positive[i]], curves[false_negative[i]]], features, 
                                             ['False Positive', 'False Negative'], ignore_padding)
        elif prediction_type.lower() == 'positive':
            try:
                horizontal_subplot_curve_plotter([curves[true_positive[i]], curves[false_positive[i]]], features, 
                                                 ['True Positive', 'False Positive'], ignore_padding)
            except IndexError:
                pass
        elif prediction_type.lower() == 'negative':
            horizontal_subplot_curve_plotter([curves[true_negative[i]], curves[false_negative[i]]], features, 
                                             ['True Negative', 'False Negative'], ignore_padding)
        else:
            raise ValueError(f'Prediction type ({prediction_type}) not accepted. Prediction type must true, false, positive, or negative.')
            
def show_inje_curv_plot(curve, features, color, save_figu = False, save_figu_path = '', figu_name = '', 
                        widt = 20, heig = 5, ignore_padding = True):
    
    initial = 1
    if len(features) > 0:
        title = f'TIC ID: {curve[-1, 1]["tic_id"]},'
    else:
        title = f'TIC ID: {curve[-1, 1]["tic_id"]}'
    for i in range(len(features)):
        if features[i].lower() == 'tic_id':
            features.pop(i)
            break
            
    title_font = {'fontname':'Times New Roman', 'size':'14', 'color':'black', 'weight':'normal'}
    axis_font = {'fontname':'Times New Roman', 'size':'14'}

    for i in range(len(features)):
        
        if (i + initial) % 8 == 0:
            title = f'{title}\n'
        if i == len(features) - 1:
            if type(curve[-1, 1][features[i]]) == str:
                title = f'{title} {curve[find(curve, features[i]), 0]}: {curve[-1, 1][features[i]]}'
            else:
                title = f'{title} {curve[find(curve, features[i]), 0]}: {float(curve[-1, 1][features[i]]):.3}'
        else:
            if type(curve[-1, 1][features[i]]) == str:
                title = f'{title} {curve[find(curve, features[i]), 0]}: {curve[-1, 1][features[i]]},'
            else:
                title = f'{title} {curve[find(curve, features[i]), 0]}: {float(curve[-1, 1][features[i]]):.3},'  
 
    if ignore_padding:
        flux = curve[np.where(curve[:find(curve, 'end')] > 0)[0], 1]
        x_axis = curve[np.where(curve[:find(curve, 'end')] > 0)[0], 0]
    else:
        flux = curve[:, 1]
        x_axis = curve[: 0]
       
    front_time = (x_axis[0] // 100) * 100
    x_axis -= front_time    
    plt.figure(figsize = (widt, heig))
#     over_poin = []
    if curve[find(curve, 'Curve Injected'), 1]:
#         for i in range(len(curve[find(curve, 'Injection Times'), 1])):
#             for j in range(len(curve[:find(curve, 'end')])):
#                 if curve[find(curve, 'Injection Times'), 1][i][0] < curve[j, 0] < curve[find(curve, 'Injection Times'), 1][i][1]:
#                     over_poin.append(curve[j])
#         over_poin = np.array(over_poin, dtype = float)
#         over_poin[:, 0] -= front_time
        first_time = True
        for k in range(len(curve[find(curve, 'Injection Times'), 1])):
            if x_axis[0] < curve[find(curve, 'Injection Times'), 1][k, 0] - front_time and x_axis[-1] > curve[find(curve, 'Injection Times'), 1][k, 1] - front_time:
                if first_time:
                    plt.axvspan(curve[find(curve, 'Injection Times'), 1][k, 0] - front_time - 0.05, curve[find(curve, 'Injection Times'), 1][k, 1] - front_time + 0.05, 0, max(flux), facecolor='black', alpha = 0.3, label = 'Injected')
                    first_time = False
                else:
                    plt.axvspan(curve[find(curve, 'Injection Times'), 1][k, 0] - 0.05 - front_time, curve[find(curve, 'Injection Times'), 1][k, 1] + 0.05 - front_time, 0, max(flux), facecolor='black', alpha = 0.3)
    
#     title = f'True Positive, {title}'
#     title = f'False Positive, TIC ID: {find_TIC_id(curve[find(curve, "tic id"), 1])}'
    plt.rc('axes', titlesize=10) 
    plt.scatter(x_axis, flux, s = 0.5, c = color, label = 'Uninjected')
    plt.title(title, **title_font)
    plt.xlabel(f'Time (BJD - {int(front_time)})', **axis_font)
    plt.ylabel('Flux', **axis_font)
    plt.xticks(size = 13)
    plt.yticks(size = 13)
    plt.legend()
    if save_figu:
        if figu_name == '':
            plt.savefig(f"{save_figu_path}{float(time.time())}.pdf")
        else:
             plt.savefig(f"{save_figu_path}{figu_name}.pdf")
    plt.show()

# def confu_matr(bina_pred, y_test, sav_fig = False, widt = 5, heig = 5): 
#     confusion_matrix = metrics.confusion_matrix(y_true = y_test, y_pred = bina_pred)
#     confusion_dataframe = pd.DataFrame(confusion_matrix)
#     sns.heatmap(confusion_dataframe, annot = True, fmt = 'g')
#     plt.title('Confusion Matrix')
#     plt.xlabel('Predictions')
#     plt.ylabel('Actual')
#     if sav_fig:
#         plt.savefig(f"{float(time.time())}.pdf")
#     plt.show()
    
# def roc_curv(pred, y_true, cuto, sav_fig = False, widt = 20, heig = 5):
#     fpr_keras, tpr_keras, thresholds_keras = roc_curve(y_true, pred.ravel())
#     idx = find_nearest(thresholds_keras, cuto)
#     auc_keras = auc(fpr_keras, tpr_keras)
#     plt.figure(figsize = (widt, heig))
#     plt.plot([0, 1], [0, 1], 'k--')
#     plt.scatter(y = tpr_keras[idx],x = fpr_keras[idx] , color = 'red', label = f'Cutoff: {cuto}')
#     plt.plot(fpr_keras, tpr_keras, label = 'AUC: {:.4f}'.format(auc_keras))
#     title_font = {'fontname':'Times New Roman', 'size':'16', 'color':'black', 'weight':'normal'} 
#     axis_font = {'fontname':'Times New Roman', 'size':'14'}
#     plt.xlabel('False positive rate', **axis_font)
#     plt.ylabel('True positive rate', **axis_font)
#     plt.title('ROC curve', **title_font)
#     plt.legend()
#     if sav_fig:
#         plt.savefig(f"{float(time.time())}.pdf")
#     plt.show()
    
# def pr_curv(pred, y_true, cuto, sav_fig = False, widt = 20, heig = 5):
#     no_skill = len(y_true[y_true==1]) / len(y_true)
#     plt.figure(figsize = (widt, heig))
#     plt.plot([0, 1], [no_skill, no_skill], 'k--')
#     precision, recall, thresholds_keras = precision_recall_curve(y_true, pred)
#     idx = find_nearest(thresholds_keras, cuto)
#     plt.scatter(y = precision[idx],x = recall[idx] , color = 'red', label = f'Cutoff: {cuto}')
#     auc_score = auc(recall, precision)
#     title_font = {'fontname':'Times New Roman', 'size':'16', 'color':'black', 'weight':'normal'} 
#     axis_font = {'fontname':'Times New Roman', 'size':'14'}
#     plt.text(y = precision[idx]-0.1,x = recall[idx]-0.15, s = f'Precision: {precision[idx]:.3}, Recall: {recall[idx]:.3}', **axis_font)
#     plt.plot(recall, precision, label=f'AUC: {auc_score:.4}')
#     plt.xlabel('Recall', **axis_font)
#     plt.ylabel('Precision', **axis_font)
#     plt.title('Precision-Recall curve', **title_font)
#     plt.legend(prop={'size': 15})
#     if sav_fig:
#         plt.savefig(f"{float(time.time())}.pdf")
#     plt.show()
    
# def model_information(model_history):
#     print(model_history.history.keys())
#     plt.plot(model_history.history['accuracy'])
#     plt.title('model accuracy')
#     plt.ylabel('accuracy')
#     plt.xlabel('epoch')
#     plt.show()
#     plt.plot(modelHistory.history['loss'])
#     plt.title('model loss')
#     plt.ylabel('loss')
#     plt.xlabel('epoch')
#     plt.legend(['train', 'validation'], loc='upper left')
#     plt.show()
    
# def find_neare_index(data, value):
#     '''Finds the nearest index in an array to the given value'''
#     data = np.array(data)
#     index = (np.abs(data - value)).argmin()
#     return index

# def task_comple_email(task, sende='romfradkin22@gmail.com', recie='fradkin.rom@gmail.com'):
#     msg = MIMEMultipart()
#     msg['Subject'] = task
#     msg['From'] = sende
#     msg['To'] = recie
#     s = smtplib.SMTP('localhost')
#     s.sendmail(sende, recie, msg.as_string())
#     s.quit()