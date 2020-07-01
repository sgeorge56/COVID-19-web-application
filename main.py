# That's an impressive list of imports.
import numpy as np
from numpy import linalg
from numpy.linalg import norm
from scipy.spatial.distance import squareform, pdist
import statistics
import json
import math
import statsmodels.api as sm
# We import sklearn.
import sklearn
from sklearn.manifold import TSNE
from sklearn.datasets import load_digits
from sklearn.preprocessing import scale

# We'll hack a bit with the t-SNE code in sklearn 0.15.2.
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.manifold.t_sne import (_joint_probabilities,
                                    _kl_divergence)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
from keras.utils import to_categorical
from keras.optimizers import SGD
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import sklearn
import io
import base64

from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import time
import numpy as np
from numpy import linalg
from numpy.linalg import norm
from scipy.spatial.distance import squareform, pdist
from sklearn.datasets import fetch_openml
from sklearn.manifold import TSNE

from flask import Flask, render_template, request, Response, url_for
from flask_sqlalchemy import SQLAlchemy
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import tensorflow as tf

import sqlite3
import dask.dataframe as dd
import plotly.graph_objs as go
from plotly.offline import plot  #Importing all the required libraries and dependencies

app = Flask(__name__)
app.config[
    'SQLALCHEMY_DATABASE_URL'] = 'sqlite:///mnt/C/Users/sgeorge/OneDrive - Imprivata/Desktop/Final Year Project - Python Files/shon.db'
db = SQLAlchemy(app)



@app.route('/',methods=['POST', 'GET'])
def index():
    return render_template("main.html")  # routing to the render.html page

@app.route('/start',methods=['POST', 'GET'])
def startpage():
    return render_template("landing.html")


@app.route('/upload', methods=['POST', 'GET'])  # function to run algorithm and use POST to display
def upload():
    ALLOWED_EXTENSIONS = ['xls', 'xlsx']
    ALLOWED_CSV = ['csv']
    file = request.files['file']
    file2 = request.files['file2']
    if file.filename == "" and file2.filename == "":
        conn = sqlite3.connect('database.sqlite')
        train_df = pd.read_sql('select * from TABLE1', conn)
        df_phyl = pd.read_sql('select * from TABLE2', conn)
        country1 = request.form.get("country1")
        country2 = request.form.get("country2")
    else:
        country1 = request.form.get("country1")
        country2 = request.form.get("country2")
        if file.filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS:
            train_df = pd.read_excel(file)
            df_phyl = pd.read_excel(file2)
        elif file.filename.rsplit('.', 1)[1].lower() in ALLOWED_CSV:
            train_df = pd.read_csv(file)
            df_phyl = pd.read_csv(file2)
    conn = sqlite3.connect('database.sqlite')
    train_df.to_sql('TABLE1', conn, if_exists='replace', index=False)
    df_phyl.to_sql('TABLE2', conn, if_exists='replace', index=False)
    feat_cols = []
    join_df = pd.DataFrame()
    last_df = pd.DataFrame()
    pos_df = pd.DataFrame()
    final_df = pd.DataFrame()

    rlist1 = []
    rlist2 = []
    mat_df = train_df.loc[train_df['countriesAndTerritories'] == country1]
    join_df = mat_df[['dateRep','day','month','cases']]
    join_df = join_df[join_df.cases != 0]
    join_df = join_df[join_df.month != 12]
    join_df = join_df[join_df['cases'].notna()]
    join_df.sort_values(["month", "day"], ascending=True, inplace=True)
    r0_list = join_df['month'].unique().tolist()

    for n in r0_list:
        last_df = join_df.loc[join_df['month'] == n]
        last_df['logcases'] = np.log(last_df.cases)

        X = last_df.day
        X = sm.add_constant(X)
        y = last_df.logcases
        mod = sm.OLS(y,X)
        res = mod.fit()
        r = res.params
        rval1 = math.exp(r.day)
        rlist1.append(rval1)
        print(rlist1)

    next_df = train_df.loc[train_df['countriesAndTerritories'] == country2]
    pos_df = next_df[['dateRep', 'day', 'month', 'cases']]
    pos_df = pos_df[pos_df.cases != 0]
    pos_df = pos_df[pos_df.month != 12]
    pos_df = pos_df[pos_df['cases'].notna()]
    pos_df.sort_values(["month", "day"], ascending=True, inplace=True)
    r1_list = pos_df['month'].unique().tolist()

    for n in r1_list:
        final_df = pos_df.loc[pos_df['month'] == n]
        final_df['logcases'] = np.log(final_df.cases)

        X = final_df.day
        X = sm.add_constant(X)
        y = final_df.logcases
        mod = sm.OLS(y, X)
        res = mod.fit()
        r = res.params
        rval2 = math.exp(r.day)
        rlist2.append(rval2)
        print(rlist2)
    fig = plt.figure(figsize=(5, 5))
    line1, = plt.plot(r0_list, rlist1,label = str(country1))
    line2, = plt.plot(r1_list, rlist2,label = str(country2))
    plt.title("Comparison of the R0 Value for selected countries")
    plt.xlabel("Months")
    plt.ylabel("R0 Value")
    plt.legend(handles=[line1,line2])
    img8 = io.BytesIO()
    plt.savefig(img8, format='png')
    plt.close()
    img8.seek(0)
    plot_urlr = base64.b64encode(img8.getvalue()).decode('utf8')


    #################################### Mapping spread of Different Virus Strands - India ####################################

    indmakelist = []
    ind_clean_run = {"age": {"?": 0}}
    ind_rslt_df = df_phyl.loc[df_phyl['country'] == 'India']
    ind_rslt_df = ind_rslt_df[ind_rslt_df.age != '?']
    for indvalue in ind_rslt_df['age']:
        indmakelist.append(int(indvalue))
    for indelem in indmakelist:
        if indelem != 0:
            indminimumage = min(indmakelist)
            indmaximumage = max(indmakelist)
            indaverageage = statistics.mean(indmakelist)
        else:
            print("there is missing values")
    indstrainlist = []
    indlengthlist = []
    indexposurelist = []
    indgencounter = 0

    for indlength in ind_rslt_df['length']:
        indlengthlist.append(int(indlength))
    indnum = indlengthlist[0]
    for indstrain in ind_rslt_df['strain']:
        indstrainlist.append(indstrain)
    for indgenlength in indlengthlist:
        indgenlenfreq = indlengthlist.count(indgenlength)
        if (indgenlenfreq > indgencounter):
            indgencounter =  indgenlenfreq
            indnum = indgenlength
            ind_index = indlengthlist.index(indgenlength)
    indfreqgenome = indnum
    indmostfreqstr = indstrainlist[ind_index]

    indgencounter = 0
    indnumber = indlengthlist[0]
    for indexposure in ind_rslt_df['country_exposure']:
        indexposurelist.append(indexposure)
    for indcountry in indexposurelist:
        if indcountry != 'India':
            indexposurefreq = indexposurelist.count(indcountry)
            if (indexposurefreq > indgencounter):
                indgencounter = indexposurefreq
                indnumber = indcountry
    indcountryexpos = indnumber






####################################################  Mapping spread of Different Virus Strands - China #######################################
    chinmakelist = []
    chin_clean_run = {"age": {"?": 0}}
    chin_rslt_df = df_phyl.loc[df_phyl['country'] == 'China']
    chin_rslt_df = chin_rslt_df[chin_rslt_df.age != '?']
    for chinvalue in chin_rslt_df['age']:
        chinmakelist.append(int(chinvalue))
    for chinelem in chinmakelist:
        if chinelem != 0:
            chinminimumage = min(chinmakelist)
            chinmaximumage = max(chinmakelist)
            chinaverageage = statistics.mean(chinmakelist)
        else:
            print("there is missing values")
    chinstrainlist = []
    chinlengthlist = []
    chinexposurelist = []
    chingencounter = 0

    for chinlength in chin_rslt_df['length']:
        chinlengthlist.append(int(chinlength))
    chinnum = chinlengthlist[0]
    for chinstrain in chin_rslt_df['strain']:
        chinstrainlist.append(chinstrain)
    for chingenlength in chinlengthlist:
        chingenlenfreq = chinlengthlist.count(chingenlength)
        if (chingenlenfreq > chingencounter):
            chingencounter = chingenlenfreq
            chinnum = chingenlength
            chin_index = chinlengthlist.index(chingenlength)
    chinfreqgenome = chinnum
    chinmostfreqstr = chinstrainlist[chin_index]

    chingencounter = 0
    chinnumber = chinlengthlist[0]
    for chinexposure in chin_rslt_df['country_exposure']:
        chinexposurelist.append(chinexposure)
    for chincountry in chinexposurelist:
        if chincountry != 'China':
            chinexposurefreq = chinexposurelist.count(chincountry)
            if (chinexposurefreq > chingencounter):
                chingencounter = chinexposurefreq
                chinnumber = chincountry
    chincountryexpos = chinnumber

    ####################################################  Mapping spread of Different Virus Strands - USA #######################################
    usmakelist = []
    us_clean_run = {"age": {"?": 0}}
    us_rslt_df = df_phyl.loc[df_phyl['country'] == 'USA']
    us_rslt_df = us_rslt_df[us_rslt_df.age != '?']
    for usvalue in us_rslt_df['age']:
        usmakelist.append(int(usvalue))
    for uselem in usmakelist:
        if uselem != 0:
            usminimumage = min(usmakelist)
            usmaximumage = max(usmakelist)
            usaverageage = statistics.mean(usmakelist)
        else:
            print("there is missing values")
    usstrainlist = []
    uslengthlist = []
    usexposurelist = []
    usgencounter = 0

    for uslength in us_rslt_df['length']:
        uslengthlist.append(int(uslength))
    usnum = uslengthlist[0]
    for usstrain in us_rslt_df['strain']:
        usstrainlist.append(usstrain)
    for usgenlength in uslengthlist:
        usgenlenfreq = uslengthlist.count(usgenlength)
        if (usgenlenfreq > usgencounter):
            usgencounter = usgenlenfreq
            usnum = usgenlength
            us_index = uslengthlist.index(usgenlength)
    usfreqgenome = usnum
    usmostfreqstr = usstrainlist[us_index]

    usgencounter = 0
    usnumber = uslengthlist[0]
    for usexposure in us_rslt_df['country_exposure']:
        usexposurelist.append(usexposure)
    for uscountry in usexposurelist:
        if uscountry != 'USA':
            usexposurefreq = usexposurelist.count(uscountry)
            if (usexposurefreq > usgencounter):
                usgencounter = usexposurefreq
                usnumber = uscountry
    uscountryexpos = usnumber


######################################################## Mapping the spread of COVID-19 UK ##########################################
    ukmakelist = []
    uk_clean_run = {"age": {"?": 0}}
    uk_rslt_df = df_phyl.loc[df_phyl['country'] == 'United Kingdom']
    uk_rslt_df = uk_rslt_df[uk_rslt_df.age != '?']
    for ukvalue in uk_rslt_df['age']:
        ukmakelist.append(int(ukvalue))
    for ukelem in ukmakelist:
        if ukelem != 0:
            ukminimumage = min(ukmakelist)
            ukmaximumage = max(ukmakelist)
            ukaverageage = statistics.mean(ukmakelist)
        else:
            print("there is missing values")
    ukstrainlist = []
    uklengthlist = []
    ukexposurelist = []
    ukgencounter = 0

    for uklength in uk_rslt_df['length']:
        uklengthlist.append(int(uklength))
    uknum = uklengthlist[0]
    for ukstrain in uk_rslt_df['strain']:
        ukstrainlist.append(ukstrain)
    for ukgenlength in uklengthlist:
        ukgenlenfreq = uklengthlist.count(ukgenlength)
        if (ukgenlenfreq > ukgencounter):
            ukgencounter = ukgenlenfreq
            uknum = ukgenlength
            uk_index = uklengthlist.index(ukgenlength)
    ukfreqgenome = uknum
    ukmostfreqstr = ukstrainlist[uk_index]

    ukgencounter = 0
    uknumber = uklengthlist[0]
    for ukexposure in uk_rslt_df['country_exposure']:
        ukexposurelist.append(ukexposure)
    for ukcountry in ukexposurelist:
        if ukcountry != 'United Kingdom':
            ukexposurefreq = ukexposurelist.count(ukcountry)
            if (ukexposurefreq > ukgencounter):
                ukgencounter = ukexposurefreq
                uknumber = ukcountry
    ukcountryexpos = uknumber

    print(ukminimumage)
    print(ukmaximumage)
    print(ukaverageage)
    print(ukfreqgenome)
    print(ukcountryexpos)

    ######################################################## Mapping the spread of COVID-19 Italy ##########################################

    itmakelist = []
    it_clean_run = {"age": {"?": 0}}
    it_rslt_df = df_phyl.loc[df_phyl['country'] == 'Italy']
    it_rslt_df = it_rslt_df[it_rslt_df.age != '?']
    for itvalue in it_rslt_df['age']:
        itmakelist.append(int(itvalue))
    for itelem in itmakelist:
        if itelem != 0:
            itminimumage = min(itmakelist)
            itmaximumage = max(itmakelist)
            itaverageage = statistics.mean(itmakelist)
        else:
            print("there is missing values")
    itstrainlist = []
    itlengthlist = []
    itexposurelist = []
    itgencounter = 0

    for itlength in it_rslt_df['length']:
        itlengthlist.append(int(itlength))
    itnum = itlengthlist[0]
    for itstrain in it_rslt_df['strain']:
        itstrainlist.append(itstrain)
    for itgenlength in itlengthlist:
        itgenlenfreq = itlengthlist.count(itgenlength)
        if (itgenlenfreq > itgencounter):
            itgencounter = itgenlenfreq
            itnum = itgenlength
            it_index = itlengthlist.index(itgenlength)
    itfreqgenome = itnum
    itmostfreqstr = itstrainlist[it_index]

    itgencounter = 0
    itnumber = itlengthlist[0]
    for itexposure in it_rslt_df['country_exposure']:
        itexposurelist.append(itexposure)
    for itcountry in itexposurelist:
        if itcountry != 'Italy':
            itexposurefreq = itexposurelist.count(itcountry)
            if (itexposurefreq > itgencounter):
                itgencounter = itexposurefreq
                itnumber = itcountry
    itcountryexpos = itnumber

    ######################################################## Mapping the spread of COVID-19 Australia ##########################################
    aumakelist = []
    au_clean_run = {"age": {"?": 0}}
    au_rslt_df = df_phyl.loc[df_phyl['country'] == 'Australia']
    au_rslt_df = au_rslt_df[au_rslt_df.age != '?']
    for auvalue in au_rslt_df['age']:
        aumakelist.append(int(auvalue))
    for auelem in aumakelist:
        if auelem != 0:
            auminimumage = min(aumakelist)
            aumaximumage = max(aumakelist)
            auaverageage = statistics.mean(aumakelist)
        else:
            print("there is missing values")
    austrainlist = []
    aulengthlist = []
    auexposurelist = []
    augencounter = 0

    for aulength in au_rslt_df['length']:
        aulengthlist.append(int(aulength))
    aunum = aulengthlist[0]
    for austrain in au_rslt_df['strain']:
        austrainlist.append(austrain)
    for augenlength in aulengthlist:
        augenlenfreq = aulengthlist.count(augenlength)
        if (augenlenfreq > augencounter):
            augencounter = augenlenfreq
            aunum = augenlength
            au_index = aulengthlist.index(augenlength)
    aufreqgenome = aunum
    aumostfreqstr = austrainlist[au_index]

    augencounter = 0
    aunumber = aulengthlist[0]
    for auexposure in au_rslt_df['country_exposure']:
        auexposurelist.append(auexposure)
    for aucountry in auexposurelist:
        if aucountry != 'Australia':
            auexposurefreq = auexposurelist.count(aucountry)
            if (auexposurefreq > augencounter):
                augencounter = auexposurefreq
                aunumber = aucountry
    aucountryexpos = aunumber

    ######################################################## Mapping the spread of COVID-19 Brazil ##########################################
    brmakelist = []
    br_clean_run = {"age": {"?": 0}}
    br_rslt_df = df_phyl.loc[df_phyl['country'] == 'Brazil']
    br_rslt_df = br_rslt_df[br_rslt_df.age != '?']
    for brvalue in br_rslt_df['age']:
        brmakelist.append(int(brvalue))
    for brelem in brmakelist:
        if brelem != 0:
            brminimumage = min(brmakelist)
            brmaximumage = max(brmakelist)
            braverageage = statistics.mean(brmakelist)
        else:
            print("there is missing values")
    brstrainlist = []
    brlengthlist = []
    brexposurelist = []
    brgencounter = 0

    for brlength in br_rslt_df['length']:
        brlengthlist.append(int(brlength))
    brnum = brlengthlist[0]
    for brstrain in br_rslt_df['strain']:
        brstrainlist.append(brstrain)
    for brgenlength in brlengthlist:
        brgenlenfreq = brlengthlist.count(brgenlength)
        if (brgenlenfreq > brgencounter):
            brgencounter = brgenlenfreq
            brnum = brgenlength
            br_index = brlengthlist.index(brgenlength)
    brfreqgenome = brnum
    brmostfreqstr = brstrainlist[br_index]

    brgencounter = 0
    brnumber = brlengthlist[0]
    for brexposure in br_rslt_df['country_exposure']:
        brexposurelist.append(brexposure)
    for brcountry in brexposurelist:
        if brcountry != 'Brazil':
            brexposurefreq = brexposurelist.count(brcountry)
            if (brexposurefreq > brgencounter):
                brgencounter = brexposurefreq
                brnumber = brcountry
    brcountryexpos = brnumber

    ######################################################## Mapping the spread of COVID-19 France ##########################################

    frmakelist = []
    fr_clean_run = {"age": {"?": 0}}
    fr_rslt_df = df_phyl.loc[df_phyl['country'] == 'France']
    fr_rslt_df = fr_rslt_df[fr_rslt_df.age != '?']
    for frvalue in fr_rslt_df['age']:
        frmakelist.append(int(frvalue))
    for frelem in frmakelist:
        if frelem != 0:
            frminimumage = min(frmakelist)
            frmaximumage = max(frmakelist)
            fraverageage = statistics.mean(frmakelist)
        else:
            continue
    frstrainlist = []
    frlengthlist = []
    frexposurelist = []
    frgencounter = 0

    for frlength in fr_rslt_df['length']:
        frlengthlist.append(int(frlength))
    frnum = frlengthlist[0]
    for frstrain in fr_rslt_df['strain']:
        frstrainlist.append(frstrain)
    for frgenlength in frlengthlist:
        frgenlenfreq = frlengthlist.count(frgenlength)
        if (frgenlenfreq > frgencounter):
            frgencounter = frgenlenfreq
            frnum = frgenlength
            fr_index = frlengthlist.index(frgenlength)
    frfreqgenome = frnum
    frmostfreqstr = frstrainlist[fr_index]

    frgencounter = 0
    frnumber = frlengthlist[0]
    for frexposure in fr_rslt_df['country_exposure']:
        frexposurelist.append(frexposure)
    for frcountry in frexposurelist:
        if frcountry != 'France':
            frexposurefreq = frexposurelist.count(frcountry)
            if (frexposurefreq > frgencounter):
                frgencounter = frexposurefreq
                frnumber = frcountry
    frcountryexpos = frnumber

    ############################################################COVID 19 Spread Russia ##################################################################
    rumakelist = []
    ru_clean_run = {"age": {"?": 0}}
    ru_rslt_df = df_phyl.loc[df_phyl['country'] == 'Russia']
    ru_rslt_df = ru_rslt_df[ru_rslt_df.age != '?']
    for ruvalue in ru_rslt_df['age']:
        rumakelist.append(int(ruvalue))
    for ruelem in rumakelist:
        if ruelem != 0:
            ruminimumage = min(rumakelist)
            rumaximumage = max(rumakelist)
            ruaverageage = statistics.mean(rumakelist)
        else:
            print("there is missing values")
    rustrainlist = []
    rulengthlist = []
    ruexposurelist = []
    rugencounter = 0

    for rulength in ru_rslt_df['length']:
        rulengthlist.append(int(rulength))
    runum = rulengthlist[0]
    for rustrain in ru_rslt_df['strain']:
        rustrainlist.append(rustrain)
    for rugenlength in rulengthlist:
        rugenlenrueq = rulengthlist.count(rugenlength)
        if (rugenlenrueq > rugencounter):
            rugencounter = rugenlenrueq
            runum = rugenlength
            ru_index = rulengthlist.index(rugenlength)
    rurueqgenome = runum
    rumostrueqstr = rustrainlist[ru_index]

    rugencounter = 0
    runumber = rulengthlist[0]
    for ruexposure in ru_rslt_df['country_exposure']:
        ruexposurelist.append(ruexposure)
    for rucountry in ruexposurelist:
        if rucountry != 'Russia':
            ruexposurerueq = ruexposurelist.count(rucountry)
            if (ruexposurerueq > rugencounter):
                rugencounter = ruexposurerueq
                runumber = rucountry
    rucountryexpos = runumber

    ############################################################COVID 19 Spread Canada ##################################################################
    camakelist = []
    ca_clean_can = {"age": {"?": 0}}
    ca_rslt_df = df_phyl.loc[df_phyl['country'] == 'Canada']
    ca_rslt_df = ca_rslt_df[ca_rslt_df.age != '?']
    for cavalue in ca_rslt_df['age']:
        camakelist.append(int(cavalue))
    for caelem in camakelist:
        if caelem != 0:
            caminimumage = min(camakelist)
            camaximumage = max(camakelist)
            caaverageage = statistics.mean(camakelist)
        else:
            print("there is missing values")
    castrainlist = []
    calengthlist = []
    caexposurelist = []
    cagencounter = 0

    for calength in ca_rslt_df['length']:
        calengthlist.append(int(calength))
    canum = calengthlist[0]
    for castrain in ca_rslt_df['strain']:
        castrainlist.append(castrain)
    for cagenlength in calengthlist:
        cagenlencaeq = calengthlist.count(cagenlength)
        if (cagenlencaeq > cagencounter):
            cagencounter = cagenlencaeq
            canum = cagenlength
            ca_index = calengthlist.index(cagenlength)
    cafreqgenome = canum
    camostfreqstr = castrainlist[ca_index]

    cagencounter = 0
    canumber = calengthlist[0]
    for caexposure in ca_rslt_df['country_exposure']:
        caexposurelist.append(caexposure)
    for cacountry in caexposurelist:
        if cacountry != 'Canada':
            caexposurecaeq = caexposurelist.count(cacountry)
            if (caexposurecaeq > cagencounter):
                cagencounter = caexposurecaeq
                canumber = cacountry
    cacountryexpos = canumber

    ############################################################COVID 19 Spread South Africa ##################################################################
    samakelist = []

    sa_rslt_df = df_phyl.loc[df_phyl['country'] == 'South Africa']
    sa_rslt_df = sa_rslt_df[sa_rslt_df.age != '?']
    for savalue in sa_rslt_df['age']:
        samakelist.append(int(savalue))
    for saelem in samakelist:
        if saelem != 0:
            saminimumage = min(samakelist)
            samaximumage = max(samakelist)
            saaverageage = statistics.mean(samakelist)
        else:
            print("there is missing values")
    sastrainlist = []
    salengthlist = []
    saexposurelist = []
    sagencounter = 0

    for salength in sa_rslt_df['length']:
        salengthlist.append(int(salength))
    sanum = salengthlist[0]
    for sastrain in sa_rslt_df['strain']:
        sastrainlist.append(sastrain)
    for sagenlength in salengthlist:
        sagenlenfreq = salengthlist.count(sagenlength)
        if (sagenlenfreq > sagencounter):
            sagencounter = sagenlenfreq
            sanum = sagenlength
            sa_index = salengthlist.index(sagenlength)
    safreqgenome = sanum
    samostfreqstr = sastrainlist[sa_index]

    sagencounter = 0
    sanumber = salengthlist[0]
    for saexposure in sa_rslt_df['country_exposure']:
        saexposurelist.append(saexposure)
    for sacountry in saexposurelist:
        if sacountry != 'South Africa':
            saexposurefreq = saexposurelist.count(sacountry)
            if (saexposurefreq > sagencounter):
                sagencounter = saexposurefreq
                sanumber = sacountry
    sacountryexpos = sanumber

    # create a dataframe with all training data except the target column
    train_new = train_df.loc[train_df['countriesAndTerritories'] == country1, ['day', 'month', 'cases']]
    test_X = train_df.loc[train_df['countriesAndTerritories'] == country2, ['day', 'month', 'cases']]

    # check that the target variable has been removed
    train_new.head()

    # create a dataframe with only the target column
    train_y = train_df.loc[train_df['countriesAndTerritories'] == country1, ['deaths']]


    # create model
    model_mc = Sequential()

    # get number of columns in training data
    n_cols = train_new.shape[1]

    # add model layers
    model_mc.add(Dense(2000, activation='relu', input_shape=(n_cols,)))
    model_mc.add(Dense(2000, activation='relu'))
    model_mc.add(Dense(2000, activation='relu'))
    model_mc.add(Dense(1))

    # compile model using mse as a measure of model performance
    model_mc.compile(optimizer='adam', loss='mean_squared_error')
    time_start = time.time()
    # set early stopping monitor so the model stops training when it won't improve anymore
    early_stopping_monitor = EarlyStopping(patience=3)
    model_mc.fit(train_new, train_y, validation_split=0.2, epochs=30, callbacks=[early_stopping_monitor])

    test_y_predictions = model_mc.predict(test_X)
    fig = plt.figure(figsize=(7, 6))
    rect = [0.1, 0.1, 0.8, 0.8]
    xticks = range(0,120,10)
    a1 = plt.axes(rect)  # Create subplot, rect = [left, bottom, width, height] in normalized (0, 1) units
    a1.yaxis.tick_left()  # Use ticks only on left side of plot
    a1.set_xticklabels(xticks[::-1])
    line6, = plt.plot(train_df.loc[train_df['countriesAndTerritories'] == country2, ['deaths']], color='orange', label = "Observed Deaths")
    plt.ylabel('Observed')
    plt.xlabel('Days')

    a2 = plt.axes(rect, frameon=False)  # frameon, if False, suppress drawing the figure frame
    a2.yaxis.tick_right()
    line7, = plt.plot(test_y_predictions, color='indigo', label = "model predicted")
    a2.yaxis.set_label_position('right')
    plt.ylabel('Predicted')
    a2.set_xticks([])
    a2.set_title("Comparison of Observed vs Model predicted deaths by country ")
    a2.legend(handles = [line6,line7])

    img = io.BytesIO()
    plt.savefig(img, format='png')
    plt.close()
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode('utf8')


    return render_template('landing.html', plot_url=plot_url, plot_urlr = plot_urlr, indminimumage = indminimumage,
                           indmaximumage = indmaximumage, indaverageage = indaverageage, indmostfreq = indfreqgenome, indmostfreqstr = indmostfreqstr,
                           indcountryexpos = indcountryexpos, chinminimumage = chinminimumage,
                           chinmaximumage = chinmaximumage, chinaverageage = chinaverageage, chinmostfreq = chinfreqgenome, chinmostfreqstr = chinmostfreqstr,
                           chincountryexpos = chincountryexpos,usminimumage = usminimumage,
                           usmaximumage = usmaximumage, usaverageage = usaverageage, usmostfreq = usfreqgenome, usmostfreqstr = usmostfreqstr,
                           uscountryexpos = uscountryexpos, ukminimumage = ukminimumage,
                           ukmaximumage = ukmaximumage, ukaverageage = ukaverageage, ukmostfreq = ukfreqgenome, ukmostfreqstr = ukmostfreqstr,
                           ukcountryexpos = ukcountryexpos,itminimumage = itminimumage,
                           itmaximumage = itmaximumage, itaverageage = itaverageage, itmostfreq = itfreqgenome, itmostfreqstr = itmostfreqstr,
                           itcountryexpos = itcountryexpos, auminimumage = auminimumage,
                           aumaximumage = aumaximumage, auaverageage = auaverageage, aumostfreq = aufreqgenome, aumostfreqstr = aumostfreqstr,
                           aucountryexpos = aucountryexpos, brminimumage = brminimumage,
                           brmaximumage = brmaximumage, braverageage = braverageage, brmostfreq = brfreqgenome, brmostfreqstr = brmostfreqstr,
                           brcountryexpos = brcountryexpos, frminimumage = frminimumage,
                           frmaximumage = frmaximumage, fraverageage = fraverageage, frmostfreq = frfreqgenome, frmostfreqstr = frmostfreqstr,
                           frcountryexpos = frcountryexpos, ruminimumage = ruminimumage,
                           rumaximumage = rumaximumage, ruaverageage = ruaverageage, rumostrueq = rurueqgenome, rumostrueqstr = rumostrueqstr,
                           rucountryexpos = rucountryexpos, caminimumage = caminimumage,
                           camaximumage = camaximumage, caaverageage = caaverageage, camostfreq = cafreqgenome, camostfreqstr = camostfreqstr,
                           cacountryexpos = cacountryexpos, saminimumage = saminimumage,
                           samaximumage = samaximumage, saaverageage = saaverageage, samostfreq = safreqgenome, samostfreqstr = samostfreqstr,
                           sacountryexpos = sacountryexpos)


###########################################  TSNE CLASSIFICATION ALGORITHM ##############################################
@app.route('/prediction', methods=['GET', 'POST'])
def test_link():
    return render_template("tsne.html")


@app.route('/tsne', methods=['POST', 'GET'])
def tsne():
    count = 0
    Age = 0
    Pressure = 0
    list = []
    list1 = []
    ALLOWED_EXTENSIONS = ['xls', 'xlsx']
    ALLOWED_CSV = ['csv']
    ALLOWED_ZIP = ['zip']


    file = request.files['file']
    if file.filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS:
        TextFileReader = pd.read_excel(file, nrows=20000, chunksize=1000, skiprows=1,
                                     low_memory=False)  # the number of rows per chunk
    elif file.filename.rsplit('.', 1)[1].lower() in ALLOWED_CSV:
        TextFileReader = pd.read_csv(file, nrows=20000, chunksize=1000, skiprows=1,
                                     low_memory=False)
    elif file.filename.rsplit('.', 1)[1].lower() in ALLOWED_ZIP:
        TextFileReader = pd.read_csv(file, compression='zip', nrows=20000, chunksize=1000, skiprows=1,
                                     low_memory=False)



    feat_cols = []
    dfList = []
    for df in TextFileReader:
        dfList.append(df)
    df = pd.concat(dfList, sort=False)  # Using Pandas dataframe to concatenate the chunks
    for col in df.columns:
        feat_cols.append(col)
    df['label'] = df.iloc[:, -1]  # Allows the access of last column in pandas dataframe

    np.random.seed(42)  # setting seed allows code to have repeatable outputs
    rndperm = np.random.permutation(df.shape[0])

    plt.gray()
    fig = plt.figure(figsize=(5, 5))
    for i in range(0, 15):
        ax = fig.add_subplot(3, 5, i + 1, title="Digit: {}".format(str(df.loc[rndperm[i], 'label'])))
        ax.matshow(df.loc[rndperm[i], feat_cols].values.reshape((28, 28)).astype(float))  # allows for plot to show
    img2 = io.BytesIO()
    plt.savefig(img2, format='png')
    plt.close()
    img2.seek(0)
    plot_url2 = base64.b64encode(img2.getvalue()).decode('utf8')

    # Now creating the t-SNE model
    N = 1000  # taking 1000 values as a subset     # setting a small subset
    df_subset = df.loc[rndperm[:N], :].copy()  # accesing the subset of the dataframe and assignment
    data_subset = df_subset[feat_cols].values  # assigning column values to data_subset

    time_start = time.time()  # setting the time variable
    tsne = TSNE(n_components=2, verbose=1,  # setting parameters for TSNE class
                perplexity=40, n_iter=300)

    tsne_results = tsne.fit_transform(data_subset)

    print('t-sne done! Time elapsed: {} seconds'.format(time.time() - time_start))  # shows elapsed time for process

    df_subset['tsne-2d-one'] = tsne_results[:, 0]
    df_subset['tsne-2d-two'] = tsne_results[:, 1]
    plt.figure(figsize=(5, 5))
    sns.scatterplot(
        x="tsne-2d-one", y="tsne-2d-two", palette=sns.color_palette("hls", 10), hue=df.iloc[:, -1],
        data=df_subset,
        legend="full",
        alpha=0.3
    )
    img1 = io.BytesIO()
    plt.savefig(img1, format='png')
    plt.close()
    img1.seek(0)
    plot_url1 = base64.b64encode(img1.getvalue()).decode('utf8')

    return render_template('tsne.html', plot_url=plot_url1, plot_urlx=plot_url2)


##################################################### Model Classifier - Coronavirus ######################################################
@app.route('/classification', methods=['GET', 'POST'])
def classification_link():
    return render_template("classification.html")


@app.route('/classifier', methods=['POST', 'GET'])
def classifier():
    file = request.files['file']
    train_df_2 = pd.read_csv(file)  # the number of rows per chunk
    feat_cols = []
    dfList = []

    for col in train_df_2.columns:
        feat_cols.append(col)

    # create a dataframe with all training data except the target column


    train_new_2 = train_df_2[["patient_age_quantile",
                              "hematocrit", "hemoglobin",
                              "platelets", "mean_platelet_volume", "red_blood_cells", "lymphocytes",
                              "mean_corpuscular_hemoglobin_concentration_mchc",
                              "leukocytes", "basophils",
                              "mean_corpuscular_hemoglobin_mch", "eosinophils", "mean_corpuscular_volume_mcv",
                              "monocytes",
                              "red_blood_cell_distribution_width_rdw", "serum_glucose"]].dropna(axis='rows',
                                                                                                thresh=16)  # cleaning the dataframe off Null or empty values ## axis='columns',thresh=threshold

    normalized_df1 = (train_new_2 - train_new_2.min()) / (train_new_2.max() - train_new_2.min())

    test_X_2 = train_df_2[["patient_age_quantile",
                           "hematocrit", "hemoglobin",
                           "platelets", "mean_platelet_volume", "red_blood_cells", "lymphocytes",
                           "mean_corpuscular_hemoglobin_concentration_mchc",
                           "leukocytes", "basophils",
                           "mean_corpuscular_hemoglobin_mch", "eosinophils", "mean_corpuscular_volume_mcv", "monocytes",
                           "red_blood_cell_distribution_width_rdw", "serum_glucose"]].dropna(axis='rows', thresh=16)

    normalized_df2 = (test_X_2 - test_X_2.min()) / (test_X_2.max() - test_X_2.min())

    # check that the target variable has been removed

    # create a dataframe with only the target column
    train_y_3 = pd.DataFrame()
    clean_run = {"sars_cov_2_exam_result": {"negative": 0, "positive": 1}}
    lister = list(train_new_2.index)
    for num in lister:
        train_y_2 = pd.DataFrame(train_df_2.iloc[[num]].sars_cov_2_exam_result,
                                 columns=["sars_cov_2_exam_result"], index=[num])
        train_y_3 = train_y_3.append(train_y_2)

    train_y_3.replace(clean_run, inplace=True)
    train_y_4 = to_categorical(train_y_3)

    # view dataframe

    print(test_X_2.tail(20))
    print(train_y_3.tail(20))

    # create model
    model_2 = Sequential()

    # get number of columns in training data
    n_cols_2 = normalized_df1.shape[1]

    # add model layers
    model_2.add(Dense(1000, activation='relu', input_shape=(n_cols_2,)))
    model_2.add(Dense(1000, activation='relu'))
    model_2.add(Dense(1000, activation='relu'))
    model_2.add(Dense(2, activation='softmax'))

    # compile model using mse as a measure of model performance

    model_2.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
    time_start = time.time()
    # set early stopping monitor so the model stops training when it won't improve anymore
    early_stopping_monitor = EarlyStopping(patience=3)
    model_2.fit(normalized_df1, train_y_4, epochs=30, validation_split=0.2, verbose=1)

    test_y_classifications = model_2.predict(normalized_df2)
    positive = []
    negative = []
    counter1 = 0
    counter2 = 0
    counter3 = 0
    counter4 = 0
    mainlist = []

    print(test_y_classifications)
    print(type(test_y_classifications))

    for elem in range(len(test_y_classifications)):
        if abs(test_y_classifications[elem][0]) > 0.5 and abs(test_y_classifications[elem][1]) < 0.5:
            negative.append(elem)
            counter1 += 1

        elif abs(test_y_classifications[elem][0]) < 0.5 and abs(test_y_classifications[elem][1]) > 0.5:
            positive.append(elem + 2)
            counter2 += 1
        else:
            print("Classification issue")

    print("negative numbers are %d" % (counter1))
    print("positive numbers are %d" % (counter2))

    for num in range(len(train_y_4)):
        if train_y_4[num][0] == 1 and train_y_4[num][1] == 0:
            counter3 += 1
        elif train_y_4[num][0] == 0 and train_y_4[num][1] == 1:
            counter4 += 1
        else:
            print("Classification issue")

    print("original negative numbers are %d" % (counter3))
    print("original positive numbers are %d" % (counter4))

    while ((counter2 / counter4) * 100 < 57):
        positive = []
        negative = []
        mainlist = []
        counter1 = 0
        counter2 = 0
        counter3 = 0
        counter4 = 0
        model_2.fit(normalized_df1, train_y_4, epochs=30, validation_split=0.2, verbose=1)
        test_y_classifications = model_2.predict(normalized_df2)
        for elem in range(len(test_y_classifications)):
            if test_y_classifications[elem][0] > 0.5 and test_y_classifications[elem][1] < 0.5:
                negative.append(elem)
                counter1 += 1
            elif test_y_classifications[elem][0] < 0.5 and test_y_classifications[elem][1] > 0.5:
                positive.append(elem)
                counter2 += 1
            else:
                print("Classification issue")
        print("negative numbers are %d" % (counter1))
        print("positive numbers are %d" % (counter2))

        for num in range(len(train_y_4)):
            if train_y_4[num][0] == 1 and train_y_4[num][1] == 0:
                counter3 += 1
            elif train_y_4[num][0] == 0 and train_y_4[num][1] == 1:
                counter4 += 1
            else:
                print("Classification issue")
        print("original negative numbers are %d" % (counter3))
        print("original positive numbers are %d" % (counter4))



    reduced_df = train_df_2[["sars_cov_2_exam_result", "patient_age_quantile",
                             "hematocrit", "hemoglobin",
                             "red_blood_cells",
                             "monocytes"
                             ]].dropna().loc[
    train_df_2['sars_cov_2_exam_result'] == 'positive']  # creating a new dataframe

    df_num = reduced_df.select_dtypes(include=[np.number])  # picking out only the numerical values and not categorical
    df_norm = (df_num - df_num.min()) / (df_num.max() - df_num.min())  # using Min-Max normalisation
    reduced_df[df_norm.columns] = df_norm  # combining the categorical value with the normalized values

    boxplot = reduced_df.boxplot(column=["patient_age_quantile",
                                         "hematocrit", "hemoglobin",
                                         "red_blood_cells",
                                         "monocytes"], by=["sars_cov_2_exam_result"])  # Creation of matplotlib boxplot
    img5 = io.BytesIO()
    plt.savefig(img5, format='png')
    plt.close()
    img5.seek(0)
    plot_url4 = base64.b64encode(img5.getvalue()).decode('utf8')





    positives = (counter2, counter4)
    negatives = (counter1, counter3)
    n = 2
    width = 0.35
    ind = np.arange(n)
    plt.bar(ind, positives, width, label='positive')
    plt.bar(ind + width, negatives, width, label='negative')

    plt.ylabel('Cases')
    plt.title('Model plot of Cases')

    plt.xticks(ind + width / 2, ('model predicted', 'actual'))
    plt.legend(loc='best')

    img3 = io.BytesIO()
    plt.savefig(img3, format='png')
    plt.close()
    img3.seek(0)
    plot_url = base64.b64encode(img3.getvalue()).decode('utf8')


    a_dict = {}
    data = []
    negative_df = pd.DataFrame()
    modDfObj = pd.DataFrame()
    for val in positive:
        dfObj = pd.DataFrame(test_X_2.iloc[[val]].values,
                             columns=["patient_age_quantile",
                                      "hematocrit", "hemoglobin",
                                      "platelets", "mean_platelet_volume", "red_blood_cells", "lymphocytes",
                                      "mean_corpuscular_hemoglobin_concentration_mchc",
                                      "leukocytes", "basophils",
                                      "mean_corpuscular_hemoglobin_mch", "eosinophils", "mean_corpuscular_volume_mcv",
                                      "monocytes",
                                      "red_blood_cell_distribution_width_rdw", "serum_glucose"], index=[val])
        modDfObj = modDfObj.append(dfObj)
        data.append(train_df_2.iloc[[val]].patient_id.values)
    modDfObj.insert(0, 'Patient ID', data)

    train_df_2["sars_cov_2_exam_result"].replace({"negative": 0, "positive": 1},inplace=True)
    corr_df = train_df_2[["sars_cov_2_exam_result","patient_age_quantile",
                              "hematocrit", "hemoglobin",
                              "platelets", "mean_platelet_volume", "red_blood_cells", "lymphocytes",
                              "mean_corpuscular_hemoglobin_concentration_mchc",
                              "leukocytes", "basophils",
                              "mean_corpuscular_hemoglobin_mch", "eosinophils", "mean_corpuscular_volume_mcv",
                              "monocytes",
                              "red_blood_cell_distribution_width_rdw", "serum_glucose"]].dropna()
    plt.figure(figsize=(10, 9))
    sns.set(font_scale=0.7)
    sns.heatmap(corr_df.corr(), annot=True)
    plt.yticks(rotation=0)
    img4 = io.BytesIO()
    plt.savefig(img4, format='png')
    plt.close()
    img4.seek(0)
    plot_url3 = base64.b64encode(img4.getvalue()).decode('utf8')

    return render_template('classification.html', plot_url=plot_url, plot_url3 = plot_url3, plot_url4 = plot_url4, name=classifier, data=modDfObj.to_html())

########################################## Model Classifier  - Diabetes Detection in Patients #########################
@app.route('/diabetes_classification', methods=['GET', 'POST'])
def diabetes_classification_link():
    return render_template("diabetes_classification.html")


@app.route('/diabetes_classifier', methods=['POST', 'GET'])
def diabetes_classifier():
    file = request.files['file']
    ALLOWED_EXTENSIONS = ['xls', 'xlsx']
    ALLOWED_CSV = ['csv']
    feat_cols = []
    dfList = []

    if file.filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS:
        train_df_2 = pd.read_excel(file)
    elif file.filename.rsplit('.', 1)[1].lower() in ALLOWED_CSV:
        train_df_2 = pd.read_csv(file)

    # create a dataframe with all training data except the target column

    train_new_2 = train_df_2.drop(columns=['diabetes'])
    test_X_2 = train_df_2[['pregnancies', 'glucose', 'diastolic', 'triceps', 'insulin', 'bmi', 'dpf', 'age']]

    # check that the target variable has been removed
    print(test_X_2.head())

    # create a dataframe with only the target column
    train_y_2 = to_categorical(train_df_2.diabetes)

    # view dataframe

    # create model
    model_2 = Sequential()

    # get number of columns in training data
    n_cols_2 = train_new_2.shape[1]

    # add model layers
    model_2.add(Dense(1000, activation='relu', input_shape=(n_cols_2,)))
    model_2.add(Dense(1000, activation='relu'))
    model_2.add(Dense(1000, activation='relu'))
    model_2.add(Dense(2, activation='softmax'))

    # compile model using mse as a measure of model performance
    model_2.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    time_start = time.time()
    # set early stopping monitor so the model stops training when it won't improve anymore
    early_stopping_monitor = EarlyStopping(patience=3)
    model_2.fit(train_new_2, train_y_2, epochs=30, validation_split=0.2, callbacks=[early_stopping_monitor])

    test_y_classifications = model_2.predict(test_X_2)
    positive = []
    negative = []
    counter1 = 0
    counter2 = 0
    counter3 = 0
    counter4 = 0
    mainlist = []

    print(test_y_classifications)
    print(type(test_y_classifications))
    for elem in range(len(test_y_classifications)):
        if test_y_classifications[elem][0] > 0.5 and test_y_classifications[elem][1] < 0.5:
            negative.append(elem)
            counter1 += 1

        elif test_y_classifications[elem][0] < 0.5 and test_y_classifications[elem][1] > 0.5:
            positive.append(elem + 2)
            counter2 += 1
        else:
            print("Classification issue")

    print("negative numbers are %d" % (counter1))
    print("positive numbers are %d" % (counter2))

    for num in range(len(train_y_2)):
        if train_y_2[num][0] == 1 and train_y_2[num][1] == 0:
            counter3 += 1
        elif train_y_2[num][0] == 0 and train_y_2[num][1] == 1:
            counter4 += 1
        else:
            print("Classification issue")

    print("original negative numbers are %d" % (counter3))
    print("original positive numbers are %d" % (counter4))

    while (abs((counter4 - counter2) / counter4) * 100 > 11):
        positive = []
        negative = []
        mainlist = []
        counter1 = 0
        counter2 = 0
        counter3 = 0
        counter4 = 0
        model_2.fit(train_new_2, train_y_2, epochs=30, validation_split=0.2, callbacks=[early_stopping_monitor])
        test_y_classifications = model_2.predict(test_X_2)
        for elem in range(len(test_y_classifications)):
            if test_y_classifications[elem][0] > 0.5 and test_y_classifications[elem][1] < 0.5:
                negative.append(elem)
                counter1 += 1
            elif test_y_classifications[elem][0] < 0.5 and test_y_classifications[elem][1] > 0.5:
                positive.append(elem)
                counter2 += 1
            else:
                print("Classification issue")
        print("negative numbers are %d" % (counter1))
        print("positive numbers are %d" % (counter2))

        for num in range(len(train_y_2)):
            if train_y_2[num][0] == 1 and train_y_2[num][1] == 0:
                counter3 += 1
            elif train_y_2[num][0] == 0 and train_y_2[num][1] == 1:
                counter4 += 1
            else:
                print("Classification issue")
        print("original negative numbers are %d" % (counter3))
        print("original positive numbers are %d" % (counter4))

    positives = (counter2, counter4)
    negatives = (counter1, counter3)
    n = 2
    width = 0.35
    ind = np.arange(n)
    plt.bar(ind, positives, width, label='positive')
    plt.bar(ind + width, negatives, width, label='negative')

    plt.ylabel('Cases')
    plt.title('Model plot of Cases')

    plt.xticks(ind + width / 2, ('model predicted', 'actual'))
    plt.legend(loc='best')

    img3 = io.BytesIO()
    plt.savefig(img3, format='png')
    plt.close()
    img3.seek(0)
    plot_url = base64.b64encode(img3.getvalue()).decode('utf8')
    a_dict = {}
    data = []
    negative_df = pd.DataFrame()
    modDfObj = pd.DataFrame()
    for val in positive:
        dfObj = pd.DataFrame(test_X_2.iloc[[val]].values,
                             columns=['pregnancies', 'glucose', 'diastolic', 'triceps', 'insulin', 'bmi', 'dpf', 'age'],
                             index=[val])
        modDfObj = modDfObj.append(dfObj)
    print(modDfObj)

    return render_template('diabetes_classification.html', plot_url=plot_url, name=classifier, data=modDfObj.to_html())




if __name__ == "__main__":
    app.run(debug=True)
