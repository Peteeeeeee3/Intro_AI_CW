# -*- coding: utf-8 -*-
"""
Created on Wed Dec 15 12:28:09 2021

@author: Peter & Xavier
"""

import panda as pd

#load dataset
intl_spi_data = pd.load_cvs("spi_global_rankings_intl.csv")

qualidied_teams = ["Qatar", "Germany", "Denmark", "Brazil", "France", "Belgium",
                   "Croatia", "Spain", "Serbia", "England", "Switzerland", "Netherlands",
                   "Argentina"]

confederations = ["AFC", "CAF", "CONCACAF", "CONMEBOL", "OFC", "UEFA"]
