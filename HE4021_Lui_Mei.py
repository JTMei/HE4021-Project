import pandas as pd
import statsmodels.api as sm
import numpy as np
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.api import VAR
from statsmodels.tsa.vector_ar import vecm
from statsmodels.tsa.stattools import grangercausalitytests
from sklearn.metrics import mean_squared_error
import matplotlib 

###--------------------------------------------Import and clean data-----------------------------------------------------

#------------------------------------------------------Retail-----------------------------------------------------------

#read data sets the first 8 lines are useless info
sk_s = pd.read_excel(r'C:\DOC\NTU Year3 Sem2\MH4021 Adv Econometrics\Project\ds\Steak, sirloin, USDA Choice, boneless.xlsx',
                   skiprows=9,  dtype={'Year': int})
sk_r = pd.read_excel(r'C:\DOC\NTU Year3 Sem2\MH4021 Adv Econometrics\Project\ds\Steak, round, USDA Choice, boneless.xlsx',
                   skiprows=9,  dtype={'Year': int})
rd_r = pd.read_excel(r'C:\DOC\NTU Year3 Sem2\MH4021 Adv Econometrics\Project\ds\Round roast, USDA Choice, boneless.xlsx',
                   skiprows=9,  dtype={'Year': int})
ck_r = pd.read_excel(r'C:\DOC\NTU Year3 Sem2\MH4021 Adv Econometrics\Project\ds\Chuck roast, USDA Choice, boneless.xlsx',
                   skiprows=9,  dtype={'Year': int})
oth = pd.read_excel(r'C:\DOC\NTU Year3 Sem2\MH4021 Adv Econometrics\Project\ds\All Uncooked Other Beef (Excluding Veal).xlsx',
                   skiprows=9,  dtype={'Year': int})
gr = pd.read_excel(r'C:\DOC\NTU Year3 Sem2\MH4021 Adv Econometrics\Project\ds\All uncooked ground beef.xlsx',
                   skiprows=9,  dtype={'Year': int})
roa = pd.read_excel(r'C:\DOC\NTU Year3 Sem2\MH4021 Adv Econometrics\Project\ds\All Uncooked Beef Roasts.xlsx',
                   skiprows=9,  dtype={'Year': int})
sk = pd.read_excel(r'C:\DOC\NTU Year3 Sem2\MH4021 Adv Econometrics\Project\ds\All Uncooked Beef Steaks.xlsx',
                   skiprows=9,  dtype={'Year': int})

#lists to run iterations
#list of og datasets
#lists of col name
df_list = [sk_s, sk_r, rd_r, ck_r, oth, gr, roa, sk]
var_name_list = ['Steak, sirloin','Steak, round','Round roast', 'Chuck roast',
                 'All Uncooked Other Beef (Excluding Veal)', 'All uncooked ground beef', 'All Uncooked Beef Roasts',
                 'All Uncooked Beef Steaks']

#define a fucntion to cleadn data
def magic_clean (df,val_name):
    
    #generate month list since monthly data
    month_list= []
    for i in range(1,13):
        month_list.append(df.columns[i])
    
    #combine dfs
    df = pd.melt(df, id_vars='Year', value_vars=month_list,
            var_name='mon', value_name=val_name)
    
    #gen year-mon and set as time sereies data
    df=df.astype({'Year':'string'})
    df['Month']=df['Year']+df['mon']
    df.set_index(pd.to_datetime(df['Month'],format='%Y%b'),
             inplace=True)
    df.set_index(df.index.to_period('M'),inplace=True)
    df.drop(columns=['Year','mon','Month'],inplace=True)
    
    #sort by time and drop nan column
    df.sort_values(by='Month',inplace=True)
    df.drop(index=['2022-03','2022-04','2022-05','2022-06',
        '2022-07','2022-08','2022-09',
        '2022-10','2022-11', '2022-12'],inplace=True)
    return df

# use loop to match col with df
for i in range(0,8):
    df_list[i]=magic_clean(df_list[i], var_name_list[i])

retail_prices =  pd.concat(df_list,axis=1)

#there was a Nan in 2020 April, replace it with the mean of 2020-03 and 2020-05
retail_prices=retail_prices.fillna((retail_prices.loc['2020-03','All Uncooked Other Beef (Excluding Veal)']+
                     retail_prices.loc['2020-05','All Uncooked Other Beef (Excluding Veal)'])/2)


#for debugging
print (retail_prices.head(10))


#------------------------------------------------------Wholesale-----------------------------------------------------------

#import wholesale price
wholesale_prices = pd.read_excel(r'C:\DOC\NTU Year3 Sem2\MH4021 Adv Econometrics\Project\ds\WholesalePrices_new.xlsx',
                                 'Historical',
                                 skiprows=3,
                                dtype={'Period': str},
                                index_col='Period')

#keep the only two prices we want 
wholesale_prices.rename(columns={"Dollar/cwt": "Boxed beef cutout choice", "Dollar/cwt.1": "Boxed beef cutout select"},inplace=True)
wholesale_prices = wholesale_prices[['Boxed beef cutout choice','Boxed beef cutout select']]

#rename/convert to monthly for index
wholesale_prices.index.rename('Month',inplace=True)
wholesale_prices.set_index(pd.to_datetime(wholesale_prices.index).to_period('M'),
        inplace=True)

#for debugging
print (wholesale_prices.head(10))


#-----------------------------------------------Merge retail and wholesale and manipulate the ag_df----------------------------
# ag is the aggregated dataframe containing both wholesale and retail
ag_df =  pd.concat([retail_prices, wholesale_prices],axis=1)

#drop 2022-02 because no wholesale data
ag_df.drop(index=['2022-02'], inplace=True)


#----------------------------------------------------Adf test-----------------------------------------------

# define a function for doing adf test across AIC/BIC and for constant (c), constant and linear(ct), and constant,linear, quardratic trend(ctt)

#return a dictionary
def adf_test(df):
    adf_dic = {' ':['p-value']}
    for i in df.columns:
        for j in ['AIC','BIC']:
            for x in ['c','ct','ctt']:
                adftest = adfuller(df[i], autolag=j, regression=x)
                adf_dic[i+','+j+','+x]=[adftest[1]]
    return adf_dic

#define a function to quickly check if all p-values are smaller than 1%
def check_adf(df):
    df_mask = df.mask(df>0.01)
    if df_mask.isna().sum()[0]==0:
        print ('H0 is rejected in all tests.')
    else:
        print('H0 cannot be rejected in all test.')

#gen level price testing data frame
lvl_tr_df = ag_df.loc[:'2019-12'][:]

#adf test and print result
lvl_adf = pd.DataFrame.from_dict(adf_test(lvl_tr_df))
lvl_adf.set_index(' ',inplace= True)
print('The adf result for level price is:\n', lvl_adf)
check_adf(lvl_adf)

#generate df containing ln and dln
lag_df=ag_df.loc[:][:]
lag_df[lag_df.columns]=lag_df[lag_df.columns].apply(pd.to_numeric)
for i in lag_df.columns:
    lag_df['ln '+i]=np.log(lag_df.loc[:,i])

#gen data frame for ln drop 2000-01 cos Nan
ln_tr_df = lag_df.loc[:, lag_df.columns.str.startswith('ln')]
ln_tr_df = ln_tr_df.loc[:'2019-12'][:]
ln_tr_df = ln_tr_df.drop(index=['2000-01'])

#adf test and print result
ln_adf = pd.DataFrame.from_dict(adf_test(ln_tr_df))
ln_adf.set_index(' ',inplace= True)
print('The adf result for level price is:\n', ln_adf)
check_adf(ln_adf)

#generate dln
lag1_df=ag_df.loc[:][:]
lag1_df[lag1_df.columns]=lag1_df[lag1_df.columns].apply(pd.to_numeric)
for i in lag1_df.columns:
    lag1_df['ln '+i]=np.log(lag1_df.loc[:,i])
    lag1_df['ln '+i+'1']=lag1_df['ln '+i].shift(1)
    lag1_df['dln '+i]=lag1_df['ln '+i]-lag1_df['ln '+i+'1']

#gen data frame for ln drop 2000-01 cos Nan
mod_df = lag1_df.loc[:, lag1_df.columns.str.startswith('dln')]
mod_df = mod_df.drop(index=['2000-01'])

#split into testing (bef 2020) and training (after 2020)
tr_df = mod_df.loc['2000-02':'2019-12'][:]
ts_df = mod_df.loc['2020-01':'2022-01'][:]


#adf test and print result
dln_adf = pd.DataFrame.from_dict(adf_test(tr_df))
dln_adf.set_index(' ',inplace= True)
print('The adf result for level price is:\n', dln_adf)
check_adf(dln_adf)

#---------------------------------------------determine the lags in VECM-----------------------------------
# c is constant; ct constant and linear; ctt constant,linear, and quadratic; n no trend
order1 = VAR(tr_df).select_order(10,trend='c')
print(order1.summary())

order2 = VAR(tr_df).select_order(10,trend='ct')
print(order2.summary())

order3 = VAR(tr_df).select_order(10,trend='ctt')
print(order3.summary())

order4 = VAR(tr_df).select_order(10,trend='n')
print(order4.summary())

#We choose 1 and 2 lags


#---------------------------------------------Johansen Test-----------------------------------
# 1/2 lags
# det_oreder is the option for trend: -1 - no deterministic terms; 0 - constant term; 1 - linear trend

rank1 = vecm.select_coint_rank(tr_df, det_order=-1, k_ar_diff=2, signif=0.01)
print(rank1.summary())

rank2 = vecm.select_coint_rank(tr_df, det_order=0, k_ar_diff=2, signif=0.01)
print(rank2.summary())

rank3 = vecm.select_coint_rank(tr_df, det_order=1, k_ar_diff=2, signif=0.01)
print(rank3.summary())

rank4 = vecm.select_coint_rank(tr_df, det_order=-1, k_ar_diff=1, signif=0.01)
print(rank4.summary())

rank5 = vecm.select_coint_rank(tr_df, det_order=0, k_ar_diff=1, signif=0.01)
print(rank5.summary())

rank6 = vecm.select_coint_rank(tr_df, det_order=1, k_ar_diff=1, signif=0.01)
print(rank6.summary())




#-------------------------------------------------VECM model-----------------------------------------
#run across 1/2 lags
#trend:n - none; co - constant outside of cointegration; ci - constant inside of coitegration;
#lo - linear outside; li - linear inside; co/ci can be combine with ci/co
#18 models in total


vecm_res = {}
for i in [1,2]:
    for x in ['n','co','ci','lo','li']:
        vecm_res['vecm'+','+str(i)+','+x]=vecm.VECM(tr_df, k_ar_diff=i, deterministic= x, coint_rank=10).fit()
for j in [1,2]:
    for x in ['co','ci']:
        for y in ['lo','li']:
            vecm_res['vecm'+','+str(j)+','+x+y]=vecm.VECM(tr_df, k_ar_diff=j, deterministic= x+y, coint_rank=10).fit()

#the results are stored inside a dictionary vecm_res

ts_columns = ts_df.columns
vecm_df = ts_df.loc[:][:]
for i in vecm_res:
    df1 = pd.DataFrame(vecm_res[i].predict(steps=25),index=ts_df.index, columns=[s +','+ i for s in ts_columns])
    vecm_df =  pd.concat([vecm_df,df1],axis=1)
print('The vecm forecasted value: \n', vecm_df)


#VECM rmse and selection

vecm_rmse_dict =  {}
for i in ts_columns:
    vecm_rmse_dict[i] =[]
    for j in vecm_res:
        vecm_rmse_dict[i].append(np.sqrt(mean_squared_error(vecm_df[i], vecm_df[i+','+j])))
vecm_rmse = pd.DataFrame.from_dict(vecm_rmse_dict)
vecm_rmse.index = [i for i in vecm_res]
#between c*l* the difference is very small

vecm_sel = {}
for i in vecm_rmse:
    vecm_sel[i]=[vecm_rmse[[i]].idxmin()[0],vecm_rmse[[i]].min()[0]]
print('vecm selection result:/n',vecm_sel)

#select 1 lag and lo


#----------------------------------------------------Grangercausalitytests----------------------------------------------------

#Granger causality test 
#run only on 1 lag
## H0 is no G-causality H1 is yes
## Report LR

vecm_granger_dict = {}
list1 = ['dln Steak, sirloin','dln Steak, round','dln Round roast', 'dln Chuck roast',
                 'dln All Uncooked Other Beef (Excluding Veal)', 'dln All uncooked ground beef', 'dln All Uncooked Beef Roasts',
                 'dln All Uncooked Beef Steaks','dln Boxed beef cutout choice','dln Boxed beef cutout select']
for i in list1:
    vecm_granger_dict[i] = []
    for j in list1:
        if i != j:
            vecm_granger_dict[i].append(grangercausalitytests(
                tr_df[[i,j]], [1
                               ])[1][0]['lrtest'][1]
        )          
        else:
            vecm_granger_dict[i].append(None)
            
vecm_granger = pd.DataFrame.from_dict(vecm_granger_dict)
vecm_granger.index=list1
print (vecm_granger)



#---------------------------------------------------print out the regression report for the best fit model-----------------------------------
print('the summary for the best-fit vecm model is: /n',vecm_res['vecm,1,lo'].summary())





#---------------------------------------------------IRF plotting based on vecm,1,lo---------------------------------
font = {'weight' : 'bold',
        'size'   : 10}
matplotlib.rc('font', **font)
irf_fig1 = vecm_res['vecm,1,lo'].irf(15).plot(impulse='dln Boxed beef cutout choice')
irf_fig1.set_size_inches(15,30)


irf_fig2 = vecm_res['vecm,1,lo'].irf(15).plot(impulse='dln Boxed beef cutout select')
irf_fig2.set_size_inches(15,30)





#---------------------------------------------------VAR as benchmark models--------------------------------------------------

#We run 8 models based on 1. AIC/BIC 2. 4 different trends: constant, linear, quadratic, none


#Store as dictionary to compensate
#hqic alsways chooses 1 and aic/fpe always 2, bic 1 when n and 0 o.w.

var_res = {}
for j in ['aic','hqic']:
    for x in ['n','c','ct','ctt']:
        if j == 'aic':
            var_res['var'+',2,'+x]=VAR(tr_df).fit(maxlags=10, ic =j, trend=x)
        if j == 'hqic':
            var_res['var'+',1,'+x]=VAR(tr_df).fit(maxlags=10, ic =j, trend=x)
#call from dictionary for result

#calucluate predicted value
var_df = ts_df[:][:]
ts_columns = ts_df.columns
for i in var_res:
    df2 = pd.DataFrame(var_res[i].forecast(tr_df.values[-2:],steps=25),index=ts_df.index, columns=[s +','+ i for s in ts_columns])
    var_df =  pd.concat([var_df,df2],axis=1)
print('The forecasted value of VAR:',var_df)

var_rmse_dict =  {}
for i in ts_columns:
    var_rmse_dict[i] =[]
    for j in var_res:
        var_rmse_dict[i].append(np.sqrt(mean_squared_error(var_df[i], var_df[i+','+j])))
var_rmse = pd.DataFrame.from_dict(var_rmse_dict)
var_rmse.index = [i for i in var_res]


var_sel = {}
for i in var_rmse:
    var_sel[i]=[var_rmse[[i]].idxmin()[0],var_rmse[[i]].min()[0]]
print('The VAR selection results:', var_sel)


# ## evaluate vs VAR
print('Result comparing VECM and VAR. It shows true if VECM outperfomrs. /n', (var_rmse.loc['var,2,c'][:]>= vecm_rmse.loc['vecm,1,lo']))


#-------------------------------------------------------------------Evaluate against VECM W/O specific data about cut/grade-----------------------------------
#seperate sets
tr1_df = tr_df[['dln All Uncooked Other Beef (Excluding Veal)',
               'dln All uncooked ground beef',
               'dln All Uncooked Beef Roasts',
               'dln All Uncooked Beef Steaks','dln Boxed beef cutout choice','dln Boxed beef cutout select']]
ts1_df = ts_df[['dln All Uncooked Other Beef (Excluding Veal)',
               'dln All uncooked ground beef',
               'dln All Uncooked Beef Roasts',
               'dln All Uncooked Beef Steaks','dln Boxed beef cutout choice','dln Boxed beef cutout select']]
tr1_df= tr1_df.astype('float32')
ts1_df=ts1_df.astype('float32')


#the process is the same as VECM above
vecm_res1 = {}
for i in [1,2]:
    for x in ['n','co','ci','lo','li']:
        vecm_res1['vecm'+','+str(i)+','+x]=vecm.VECM(tr1_df, k_ar_diff=i, deterministic= x, coint_rank=10).fit()
for j in [1,2]:
    for x in ['co','ci']:
        for y in ['lo','li']:
            vecm_res1['vecm'+','+str(j)+','+x+y]=vecm.VECM(tr1_df, k_ar_diff=j, deterministic= x+y, coint_rank=10).fit()

ts1_columns = ts1_df.columns
vecm1_df = ts1_df
for i in vecm_res1:
    df3 = pd.DataFrame(vecm_res1[i].predict(steps=25),index=ts1_df.index, columns=[s +','+ i for s in ts1_columns])
    vecm1_df =  pd.concat([vecm1_df,df3],axis=1)

vecm_rmse_dict1 =  {}
for i in ts1_columns:
    vecm_rmse_dict1[i] =[]
    for j in vecm_res1:
        vecm_rmse_dict1[i].append(np.sqrt(mean_squared_error(vecm1_df[i], vecm1_df[i+','+j])))
vecm_rmse1 = pd.DataFrame.from_dict(vecm_rmse_dict1)
vecm_rmse1.index = [i for i in vecm_res1]
vecm_rmse1 #between c*l* the difference is very small

vecm_sel1 = {}
for i in vecm_rmse1:
    vecm_sel1[i]=[vecm_rmse1[[i]].idxmin()[0],vecm_rmse1[[i]].min()[0]]
print('The selection is /n', vecm_sel1)


#choose 1, lo

print('If True, then vecm with specific data outperforms the one w/o')
print((vecm_rmse1.loc['vecm,1,lo'][ts1_columns] >= vecm_rmse.loc['vecm,1,lo'][ts1_columns]))
