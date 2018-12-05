#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[ ]:





# In[2]:


import numpy as np


# In[3]:


import seaborn as sns


# In[4]:


import matplotlib.pyplot as plt


def plot_cont_vars(dataframe):

    plt.figure(figsize=(20,40))


    for i in range(len(floats_list)):
        #print (floats_list[i])
        var = str(floats_list[i])
        std_text = std[var].values
        kur_text = kur[var].values
        skew_text = skew[var].values
        #print(std_text)
        plt.subplot(10,2,i+1)
        bp = sns.distplot(dataframe[var], bins=10)
        bp.set_title('Std:{}, Kur:{}, Skew{}'.format(std_text, kur_text, skew_text))


# In[5]:


complaints_df = pd.read_csv("C:\\Hackathon_Data\\Complaints data.csv",encoding = "ISO-8859-1")
len(complaints_df.index)


# In[6]:


claims_df = pd.read_csv("C:\\Hackathon_Data\\R_CLAIM_EXPORT.txt", encoding = "ISO-8859-1", error_bad_lines=False )
policy_with_claim = claims_df['policy_bk']
policy_with_claim = list(policy_with_claim.values)


# In[7]:


complaints_df.columns


# In[8]:


complaints_df.head()


# In[9]:


pd.to_datetime(complaints_df["Registered Date"]).max()


# In[10]:


pd.to_datetime(complaints_df["Registered Date"]).min()


# In[11]:


policy_df = pd.read_csv("C:\\Hackathon_Data\\R_POLICY_EXPORT.csv",encoding = "ISO-8859-1", error_bad_lines=False )
len(policy_df.index)


# In[12]:


policy_df.columns


# In[13]:


policy_df["Has_Claim"] = policy_df["policy_bk"].isin(policy_with_claim).astype(int)


# In[14]:


target_variable = "Has_Claim"


# In[15]:


policy_df["Has_Claim"].nunique()


# In[16]:


policy_df.groupby(by=["Has_Claim"])["policy_bk"].nunique()


# In[17]:


policy_number_of_rows = len(policy_df.index)
policy_number_of_rows


# In[18]:


policy_df_first_clean = policy_df.copy()


# In[19]:


policy_df_first_clean = policy_df_first_clean.dropna(subset=['policy_bk','issue_date','job_postcode','cover_type',
                                                             'architect_name',
                                                            'current_ind',
                                                            'certificate_status_cd',
                                                            'limit_of_insurance',
                                                            'existing_hbcf_claim'])


# In[20]:


policy_df_first_clean = policy_df_first_clean.set_index('policy_bk')


# In[21]:


for col in policy_df_first_clean.columns:
    na_count = policy_df_first_clean[col].isna().sum()
    if na_count > 0:
        print(col, na_count, " so will be deleted")
        policy_df_first_clean.drop([col], axis=1, inplace=True)
    else:
        print(col, na_count, "so will be kept")
        
    #claims_df.isna().sum()


# In[ ]:





# In[22]:


policy_date_fields = list(policy_df_first_clean.filter(regex=(".*date.*")).columns)


# In[23]:


for col in policy_date_fields:
    policy_df_first_clean[col] = pd.to_datetime(policy_df_first_clean[col])
    


# In[48]:


policy_data_types = pd.DataFrame(policy_df_first_clean.dtypes, columns=['data_type'])
data_unique_values = pd.DataFrame(policy_df_first_clean.nunique() , columns=['unique_values'])
data_description = pd.concat([data_unique_values, policy_data_types], axis=1)
data_description


# In[ ]:





# In[ ]:





# In[29]:


policy_df_first_clean["job_postcode"] =  policy_df_first_clean["job_postcode"].astype(str)
policy_df_first_clean["certificate_status_cd"] =  policy_df_first_clean["certificate_status_cd"].astype(str)
policy_df_first_clean["insurance_agent"] =  policy_df_first_clean["insurance_agent"].astype(str)
policy_df_first_clean["owner_postcode"] =  policy_df_first_clean["owner_postcode"].astype(str)
policy_df_first_clean["Has_Claim"] =  policy_df_first_clean["Has_Claim"].astype(str)


# In[30]:


policy_df_first_clean.drop(["policy_sk"], axis=1, inplace=True)


# In[31]:


policy_df_first_clean.drop(["job_number"], axis=1, inplace=True)


# In[32]:


policy_df_first_clean.drop(["architect_phonenum"], axis=1, inplace=True)


# In[33]:


policy_df_first_clean.drop(["owner_name"], axis=1, inplace=True)


# In[34]:


policy_df_first_clean.drop(["owner_street_address"], axis=1, inplace=True)


# In[35]:


policy_df_first_clean.drop(["owner_state_cd"], axis=1, inplace=True)


# In[36]:


policy_df_first_clean.drop(["owner_phone"], axis=1, inplace=True)


# In[37]:


policy_df_first_clean.drop(["owner_mobile_ph"], axis=1, inplace=True)


# In[38]:


policy_df_first_clean.drop(["owner_abn"], axis=1, inplace=True)


# In[39]:


policy_df_first_clean.drop(["owner_email"], axis=1, inplace=True)


# In[ ]:





# In[40]:


policy_df_first_clean.drop(["details_of_related_party"], axis=1, inplace=True)


# In[41]:


policy_df_first_clean.drop(["architect_name"], axis=1, inplace=True)


# In[42]:


policy_df_first_clean.drop(["job_state_cd"], axis=1, inplace=True)


# In[43]:


#policy_df_first_clean.drop(["owner_email"], axis=1, inplace=True)


# In[44]:


policy_df_first_clean.describe(include = 'all')


# In[67]:


policy_df_second_clean = policy_df_first_clean.copy()
policy_data_types_2 = pd.DataFrame(policy_df_second_clean.dtypes, columns=['data_type'])
data_unique_values_2 = pd.DataFrame(policy_df_second_clean.nunique() , columns=['unique_values'])
data_description_2 = pd.concat([data_unique_values_2, policy_data_types_2], axis=1)
data_description_2 = data_description_2[data_description_2['data_type']=="object"]


# In[68]:


data_description_2 = data_description_2[data_description_2['unique_values'] > 1]


# In[70]:


cat_vars_to_plot = list(data_description_2[data_description_2['unique_values'] < 100]['unique_values'].index)
cat_vars_to_plot


# In[58]:


data_types = pd.DataFrame(policy_df_first_clean.dtypes, columns=['data_type'])
#print(data_types)
floats = data_types[(data_types['data_type']=='float64')]
floats_list = list(floats.index)
print(floats_list)

ints = data_types[(data_types['data_type']=='int64')]
ints_list = list(ints.index)
print(ints_list)

floats_list = ints_list + floats_list 

categoricals = data_types[data_types['data_type']=='object']
categoricals_list = list(categoricals.index)
print(categoricals_list)


# In[59]:



floats_list.remove('adt_batch_id')


# In[60]:


kur = pd.DataFrame(policy_df_first_clean[floats_list].kurtosis(axis=0), columns=['Kurtosis'])
kur = kur.T
kur

skew = pd.DataFrame(policy_df_first_clean[floats_list].skew(axis=0), columns=['Skew'])
skew = skew.T
skew

std = pd.DataFrame(policy_df_first_clean[floats_list].std(axis=0), columns=['Std'])
std = std.T
std

frames = [std,skew,kur]

all_stats = pd.concat(frames)
all_stats


# In[61]:


def log_transform(dataframe): 
    dataframe_transform = dataframe.copy()

    for i in range(len(floats_list)):
        var = str(floats_list[i])
        std_text = std[var].values
        kur_text = kur[var].values
        skew_text = skew[var].values
        skew_true = skew_text > 2 or skew_text < -2
        kur_true = kur_text > 2 or kur_text < -2

        if kur_true == True or skew_true == True:
            transform = True
        else:
            transform = False

        print(var,'is', transform,'and has kur:',kur_text,'and skew:',skew_text)



        if transform == True:    
            dataframe_transform[var] = (dataframe_transform[var]+1).apply(np.log)
        else:
            dataframe_transform[var] = dataframe_transform[var]
            
    return dataframe_transform


# In[62]:


policy_df_transform = log_transform(policy_df_first_clean)


# In[ ]:


plot_cont_vars(policy_df_transform)


# In[ ]:


g = sns.PairGrid(policy_df_transform, vars=floats_list,
                 hue='Has_Claim', palette='RdBu_r')
g.map(plt.scatter, alpha=0.8)
g.add_legend();


# In[72]:


def plot_cont_kdes(data_frame,target_list,floats_list,target_var_name):    
    target = target_list
    plt.figure(figsize=(20,40))
    fig_no = 1

    for col in floats_list:
        df = data_frame[[target_var_name,col]]
        #print(df.head())
        plt.subplot(8,2,fig_no)   
        for cols in target:
            #print(target_var_name == cols)
            #print(cols)
            df_sub = df[df[target_var_name] == cols]
            #print(df_sub.head())
            df_sub = df_sub[col]
            df_sub = df_sub.rename(cols)
            #print(df_sub.head())
            sns.kdeplot(df_sub, shade=True, legend=True)
            plt.title(col)

        fig_no = fig_no+1


# In[73]:


plot_cont_kdes(policy_df_transform,['1','0'],floats_list,'Has_Claim')


# In[ ]:


categoricals_list.remove("broker_bk")


# In[74]:


def cat_plots(dataframe, target_var, categoricals_list, target_var_list):    
    dataframe_final = dataframe[categoricals_list]

    for cat in categoricals_list:    
        df = dataframe_final.groupby([target_var, cat])[cat].count()
       
        df_pivot = df.unstack(level=0)
        df_pivot[target_var_list[0]] = df_pivot[target_var_list[0]]/df_pivot[target_var_list[0]].sum()
        df_pivot[target_var_list[1]] = df_pivot[target_var_list[1]]/df_pivot[target_var_list[1]].sum()
        

        pos = list(range(len(df_pivot.index)))
        #print(pos)

        width = 0.4
        
        print(df_pivot.index)

        # Plotting the bars
        fig, ax = plt.subplots(figsize=(20,10))

        # Create a bar with pre_score data,
        # in position pos,
        plt.bar(pos, 
                #using df['pre_score'] data,
                df_pivot[target_var_list[0]], 
                # of width
                width, 
                # with alpha 0.5
                alpha=0.5, 
                # with color
                color='#EE3224', 
                # with label the first value in first_name
                label=df_pivot.index[0]) 

        # Create a bar with mid_score data,
        # in position pos + some width buffer,
        plt.bar([p + width for p in pos], 
                #using df['mid_score'] data,
                df_pivot[target_var_list[1]],
                # of width
                width, 
                # with alpha 0.5
                alpha=0.5, 
                # with color
                color='#F78F1E', 
                # with label the second value in first_name
                label=df_pivot.index[1]) 


        # Set the y axis label
        ax.set_ylabel('pct')

        # Set the chart's title
        ax.set_title(cat)

        # Set the position of the x ticks
        ax.set_xticks([p + (0.5*width) for p in pos])

        # Set the labels for the x ticks
        ax.set_xticklabels(df_pivot.index.values , rotation=45)

        # Setting the x-axis and y-axis limits
        #plt.xlim(min(pos)-width, max(pos)+width*4)
        #plt.ylim([0, max(df['pre_score'] + df['mid_score'] + df['post_score'])] )

        # Adding the legend and showing the plot
        plt.legend(target_var_list, loc='upper left')
        plt.savefig("cat_"+cat+".png")
        plt.grid()
        plt.show()
        plt.clf()


# In[75]:


cat_plots(policy_df_transform, 'Has_Claim', cat_vars_to_plot, ['1','0'])


# In[ ]:


categoricals_list


# In[ ]:





# In[ ]:


policy_df_transform[categoricals_list]


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


claims_df = pd.read_csv("C:\\Hackathon_Data\\R_CLAIM_EXPORT.txt", encoding = "ISO-8859-1", error_bad_lines=False )
policy_with_claim = claims_df['policy_bk']
policy_with_claim = list(policy_with_claim.values)


# In[ ]:





# In[ ]:


claims_df["date_originally_created"] = pd.to_datetime(claims_df["date_originally_created"])


# In[ ]:


claims_df.tail()


# In[ ]:


claims_df['certificate_bk'].nunique()


# In[ ]:


claim_builders = claims_df.groupby(by=["certificate_bk"])["cims_claim_number_sk"].nunique()
claim_builders


# In[ ]:


number_of_rows = len(claims_df.index)
number_of_rows


# In[ ]:


claims_df_first_clean = claims_df.copy()


# In[ ]:


for col in claims_df_first_clean.columns:
    na_count = claims_df_first_clean[col].isna().sum()
    if na_count > 0:
        print(col, na_count, " so will be deleted")
        claims_df_first_clean.drop([col], axis=1, inplace=True)
    else:
        print(col, na_count, "so will be kept")
        
    #claims_df.isna().sum()


# In[ ]:


claims_df_first_clean.columns


# In[ ]:


claim_unique_vals = claims_df_first_clean.nunique().sort_values(axis=0, ascending=False)
claim_unique_vals[claim_unique_vals < 10]
final_cat_list = claim_unique_vals.index
final_cat_list


# In[ ]:


for col in claims_df_first_clean:
    if claim_unique_vals[col] < 10:
        grouped = claims_df_first_clean.groupby(by=[col])["cims_claim_number_sk"].count()
        print(grouped)
    else:
        pass


# In[ ]:


grouped = claims_df.groupby(by=["claim_status_code"])["cims_claim_number_sk"].count()
grouped


# In[ ]:


claims_df_second_clean = claims_df_first_clean.copy()
len(claims_df_second_clean)


# In[ ]:


claims_df_a = claims_df_second_clean[claims_df_second_clean['claim_status_code'] == 'A']
claims_df_f = claims_df_second_clean[claims_df_second_clean['claim_status_code'] == 'F']
a_and_f = [claims_df_a, claims_df_f]
claims_df_second_clean = pd.concat(a_and_f)


# In[ ]:


len(claims_df_second_clean.index)


# In[ ]:


data_types = pd.DataFrame(claims_df_second_clean.dtypes, columns=['data_type'])
print(data_types)
floats = data_types[(data_types['data_type']=='float64')]
floats_list = list(floats.index)
print(floats_list)

ints = data_types[(data_types['data_type']=='int64')]
ints_list = list(ints.index)
print(ints_list)

floats_list = ints_list + floats_list 

categoricals = data_types[data_types['data_type']=='object']
categoricals_list = list(categoricals.index)
print(categoricals_list)


# In[ ]:


kur = pd.DataFrame(claims_df_second_clean[floats_list].kurtosis(axis=0), columns=['Kurtosis'])
kur = kur.T
kur

skew = pd.DataFrame(claims_df_second_clean[floats_list].skew(axis=0), columns=['Skew'])
skew = skew.T
skew

std = pd.DataFrame(claims_df_second_clean[floats_list].std(axis=0), columns=['Std'])
std = std.T
std

frames = [std,skew,kur]

all_stats = pd.concat(frames)
all_stats


# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(20,40))


for i in range(len(floats_list)):
    #print (floats_list[i])
    var = str(floats_list[i])
    std_text = std[var].values
    kur_text = kur[var].values
    skew_text = skew[var].values
    #print(std_text)
    plt.subplot(10,2,i+1)
    bp = sns.distplot(claims_df_second_clean[var], bins=10)
    bp.set_title('Std:{}, Kur:{}, Skew{}'.format(std_text, kur_text, skew_text))
    


# In[ ]:


claims_df_transform = claims_df_second_clean.copy()

for i in range(len(floats_list)):
    var = str(floats_list[i])
    std_text = std[var].values
    kur_text = kur[var].values
    skew_text = skew[var].values
    skew_true = skew_text > 2 or skew_text < -2
    kur_true = kur_text > 2 or kur_text < -2
    
    if kur_true == True or skew_true == True:
        transform = True
    else:
        transform = False
    
    print(var,'is', transform,'and has kur:',kur_text,'and skew:',skew_text)
        
            
        
    if transform == True:    
        claims_df_transform[var] = (claims_df_transform[var]+1).apply(np.log)
    else:
        claims_df_transform[var] = claims_df_transform[var]
    


# In[ ]:


kur_trans = pd.DataFrame(claims_df_transform[floats_list].kurtosis(axis=0), columns=['Kurtosis'])
kur_trans = kur_trans.T
skew_trans = pd.DataFrame(claims_df_transform[floats_list].skew(axis=0), columns=['Skew'])
skew_trans = skew_trans.T
std_trans = pd.DataFrame(claims_df_transform[floats_list].std(axis=0), columns=['Std'])
std_trans = std_trans.T
std_trans


# In[ ]:


plt.figure(figsize=(20,40))

from math import sqrt

for i in range(len(floats_list)):
    #print (floats_list[i])
    var = str(floats_list[i])
    std_text = std_trans[var].values
    kur_text = kur_trans[var].values
    skew_text = skew_trans[var].values
    #print(std_text)
    plt.subplot(10,2,i+1)
    
    nbins = int(round(sqrt(claims_df_transform[var].max())+10,0))
    #print (nbins)
    
    bp = sns.distplot(claims_df_transform[var], bins=nbins)
    bp.set_title('{} -- Std:{}, Kur:{}, Skew{}'.format(var, std_text, kur_text, skew_text))


# In[ ]:


target = ['F','A']
plt.figure(figsize=(20,40))
fig_no = 1

for col in floats_list:
    df = claims_df_transform[['claim_status_code',col]]
    plt.subplot(8,2,fig_no)   
    for cols in target:
        #print(cols)
        df_sub = df[df['claim_status_code'] == cols]
        #print(df_sub.head())
        df_sub = df_sub[col]
        df_sub = df_sub.rename(cols)
        #print(df_sub.head())
        sns.kdeplot(df_sub, shade=True, legend=True)
        plt.title(col)
        
    fig_no = fig_no+1


# In[ ]:


g = sns.PairGrid(claims_df_transform, vars=floats_list,
                 hue='claim_status_code', palette='RdBu_r')
g.map(plt.scatter, alpha=0.8)
g.add_legend();


# In[ ]:


categoricals_list_for_plots = categoricals_list.copy()

categoricals_list_for_plots.remove('claim_status_code')

categoricals_list_for_plots


# In[ ]:


df = claims_df_transform.groupby(['claim_status_code','claim_number'])['claim_number'].count()

print(df.index, type(df))


# In[ ]:


claims_df_final = claims_df_second_clean[final_cat_list]

for cat in final_cat_list:    
    df = claims_df_final.groupby(['claim_status_code', cat])[cat].count()
    df_pivot = df.unstack(level=0)
    df_pivot['F'] = df_pivot['F']/df_pivot['F'].sum()
    df_pivot['A'] = df_pivot['A']/df_pivot['A'].sum()
    print (df_pivot)
    
    pos = list(range(len(df_pivot.index)))
    #print(pos)

    width = 0.4

    # Plotting the bars
    fig, ax = plt.subplots(figsize=(20,10))

    # Create a bar with pre_score data,
    # in position pos,
    '''plt.bar(pos, 
            #using df['pre_score'] data,
            df_pivot['F'], 
            # of width
            width, 
            # with alpha 0.5
            alpha=0.5, 
            # with color
            color='#EE3224', 
            # with label the first value in first_name
            label=df_pivot.index[0]) 

    # Create a bar with mid_score data,
    # in position pos + some width buffer,
    plt.bar([p + width for p in pos], 
            #using df['mid_score'] data,
            df_pivot['A'],
            # of width
            width, 
            # with alpha 0.5
            alpha=0.5, 
            # with color
            color='#F78F1E', 
            # with label the second value in first_name
            label=df_pivot.index[1]) 


    # Set the y axis label
    ax.set_ylabel('pct')

    # Set the chart's title
    ax.set_title(cat)

    # Set the position of the x ticks
    ax.set_xticks([p + (0.5*width) for p in pos])

    # Set the labels for the x ticks
    ax.set_xticklabels(df_pivot.index.values , rotation=45)

    # Setting the x-axis and y-axis limits
    #plt.xlim(min(pos)-width, max(pos)+width*4)
    #plt.ylim([0, max(df['pre_score'] + df['mid_score'] + df['post_score'])] )

    # Adding the legend and showing the plot
    plt.legend(['F','A'], loc='upper left')
    plt.grid()
    plt.show()
    plt.clf()'''


# In[ ]:


pos = list(range(len(df_pivot.index)))
print(pos)

width = 0.4

# Plotting the bars
fig, ax = plt.subplots(figsize=(10,5))

# Create a bar with pre_score data,
# in position pos,
plt.bar(pos, 
        #using df['pre_score'] data,
        df_pivot['Bad'], 
        # of width
        width, 
        # with alpha 0.5
        alpha=0.5, 
        # with color
        color='#EE3224', 
        # with label the first value in first_name
        label=df_pivot.index[0]) 

# Create a bar with mid_score data,
# in position pos + some width buffer,
plt.bar([p + width for p in pos], 
        #using df['mid_score'] data,
        df_pivot['Good'],
        # of width
        width, 
        # with alpha 0.5
        alpha=0.5, 
        # with color
        color='#F78F1E', 
        # with label the second value in first_name
        label=df_pivot.index[1]) 
 

# Set the y axis label
ax.set_ylabel('pct')

# Set the chart's title
ax.set_title('Home Onwnership Split')

# Set the position of the x ticks
ax.set_xticks([p + (0.5*width) for p in pos])

# Set the labels for the x ticks
ax.set_xticklabels(df_pivot.index.values)

# Setting the x-axis and y-axis limits
#plt.xlim(min(pos)-width, max(pos)+width*4)
#plt.ylim([0, max(df['pre_score'] + df['mid_score'] + df['post_score'])] )

# Adding the legend and showing the plot
plt.legend(['Bad','Good'], loc='upper left')
plt.grid()
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


np.sort(policy_df.columns.values)

cert_df = pd.read_csv("C:\\Hackathon_Data\\R_BUILDER_EXPORT.txt", encoding = "ISO-8859-1", error_bad_lines=False )
# In[ ]:


builder_df = pd.read_csv("C:\\Hackathon_Data\\R_BUILDER_EXPORT.txt", encoding = "ISO-8859-1", error_bad_lines=False )


# In[ ]:


builder_df


# In[ ]:




