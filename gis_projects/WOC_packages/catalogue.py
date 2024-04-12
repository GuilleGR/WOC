# -*- coding: utf-8 -*-
"""
Created on Fri Mar 22 09:09:44 2024

@author: Guillermo Rilova
"""
import pandas as pd
from types import SimpleNamespace

# Catalogue location
filename = r"C:\Users\Guillermo Rilova\OneDrive - Greengo Energy\Documents\Wind\DEVELOPMENT\R&D\gis_projects\WOC_packages\WTG_catalogue.xlsx"


def load_wtg_P(sheet_PC):
    tmp =  pd.read_excel(filename, sheet_PC)
    return tmp.P.to_numpy()

def load_wtg_CT(sheet_CT):
    tmp =  pd.read_excel(filename, sheet_CT)
    return tmp.CT.to_numpy()

def load_wtg_WS(sheet_PC):
    tmp =  pd.read_excel(filename, sheet_PC)
    return tmp.WS.to_numpy()

def load_wtg(wtg_model):
    tmp_PC =  load_wtg_P(wtg_model+'-PC')
    tmp_CT =  load_wtg_CT(wtg_model+'-CT')
    tmp_WS =  load_wtg_WS(wtg_model+'-PC')
    d = {"WS": tmp_WS,"Power": tmp_PC,"CT": tmp_CT}
    dict_wtg = SimpleNamespace(**d)
    return dict_wtg

def list_wtg(pattern=None):
    xl = pd.ExcelFile(filename)
    ls_names = xl.sheet_names 
    if not pattern:
        ls_wtg = list(set([(i.split('-')[0]) for i in ls_names]))
    else:
        ls_wtg = list(set([(i.split('-')[0]) for i in ls_names if pattern in i]))
    return ls_wtg
    
