#!/usr/bin/env python
# coding: utf-8

# # Project: Investigate a Dataset (Replace this with something more specific!)
# 
# ## Table of Contents
# <ul>
# <li><a href="#intro">Introduction</a></li>
# <li><a href="#wrangling">Data Wrangling</a></li>
# <li><a href="#eda">Exploratory Data Analysis</a></li>
# <li><a href="#conclusions">Conclusions</a></li>
# </ul>

# <a id='intro'></a>
# ## Introduction
# 
# This dataset collects information from 100k medical appointments in Brazil and is focused on the question of whether or not patients show up for their appointment. A number of characteristics about the patient are included in each row.
# 
# The questions that will be answered will be two typical sterotypes - 1. Does older people tned to book and plan eariler than younger people? And 2. Are people who frequent doctors more healthy?

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# <a id='wrangling'></a>
# ## Data Wrangling
# 
# 
# 
# ### General Properties

# In[2]:


df = pd.read_csv('noshowappointments-may-2016.csv')


# In[3]:


df.head()


# #### Data Columns Explained
# AppointmentID - Identification of each appointment 
# 
# Gender = Male or Female . Female is the greater proportion, woman takes way more care of they health in comparison to man. 
# 
# ScheduledDay = The day someone called or registered the appointment, this is before appointment of course. 
# 
# AppointmentDay = The day of the actuall appointment, when they have to visit the doctor. 
# 
# Age = How old is the patient. 
# 
# Neighbourhood = Where the appointment takes place. 
# 
# Scholarship = Ture of False . Observation, this is a broad topic, consider reading this article https://en.wikipedia.org/wiki/Bolsa_Fam%C3%ADlia 
# 
# Hipertension = True or False 
# 
# Diabetes = True or False 
# 
# Alcoholism = True or False 
# 
# Handcap = True or False SMS_received = 1 or more messages sent to the patient. 
# 
# No-show = True or False.

# In[4]:


df.shape


# In[5]:


df.describe()


# In[6]:


df.info()


# ### Data Cleaning (Replace this with more specific notes!)

# In[7]:


df.rename(columns={'No-show': 'no_show'}, inplace= True) 
df.rename(columns=lambda x: x.strip().lower().replace(" ",""), inplace=True)


# Rename No-show to no_show and renaming every column to delete space and all lower case

# In[9]:


def convert_to_datetime():
    df['scheduledday'] = pd.to_datetime(df['scheduledday'])
    df['appointmentday'] = pd.to_datetime(df['appointmentday'])
    return df.head(1)
convert_to_datetime()


# function for converting date time from str to datetime

# In[10]:


df['scheduleddate'] = [d.date() for d in df['scheduledday']]
df['scheduledtime'] = [d.time() for d in df['scheduledday']]
df['appointmentdate'] = [d.date() for d in df['appointmentday']]
df['appointmenttime'] = [d.time() for d in df['appointmentday']]
df.drop(['scheduledday','appointmentday'], axis=1, inplace=True)


# Change the ScheduledDay and AppointmentDay from str to date time format then split ScheduledDay and AppointmentDay into sperate date and time colums ScheduledDate, ScheduledTime, AppointmentDate, AppointmentTime.

# In[11]:


df['no_show']= df['no_show'].map({'Yes':1, 'No':0}).astype(int)


# Replace yes and no with 1 and two for no_show

# In[12]:


df.head(5)


# In[13]:


df.hist(figsize=(20,10));


# Here we can start to see some trends.

# <a id='eda'></a>
# ## Exploratory Data Analysis
# 
# 
# 
# ### Does older people schedual their appoint more in advanced?

# In[14]:


df['waitdays'] = df.appointmentdate - df.scheduleddate
df.waitdays = df.waitdays.dt.days
df.plot(x='waitdays', y='age', kind='scatter')
plt.title('Age vs Days of Wait Time')
plt.xlabel('Days')
plt.ylabel('Age');


# Here we have a scatter plot showing the amount of time people schedual their appointment in advance compare to their age. Based on this information, older people does not schedual their time more in advanced compared to younger people.

# ### How healthy are the people going to the appointments?

# In[16]:


hipertension_yes = df.query('hipertension == 1')['age'].count()
hipertension_no = df.query('hipertension == 0')['age'].count()
diabetes_yes = df.query('diabetes == 1')['age'].count()
diabetes_no = df.query('diabetes == 0')['age'].count()
alcoholism_yes = df.query('alcoholism == 1')['age'].count()
alcoholism_no = df.query('alcoholism == 0')['age'].count()
locations = [1, 2, 3, 4, 5, 6]
heights = [hipertension_yes, hipertension_no, diabetes_yes, diabetes_no, alcoholism_yes, alcoholism_no]
labels = ['Hipt Y', 'Hipt N', 'Diab Y', 'Diab N', 'Alcoh Y', 'Alcoh N']
plt.bar(locations, heights, tick_label=labels)
plt.title('People with Pre-Condition')
plt.ylabel('Count of paitents')
plt.xlabel('Health Condition')
plt.rcParams["figure.figsize"] = [10,5]


# By counting people with different conditions, this chart is able to be drawn. It appeares that majority amount of people does not have any serious pre-conditions.

# <a id='conclusions'></a>
# ## Conclusions
# 
# After looking at paitnet data provided to us from Brazil's medical appointments it has clearly answered both of our answers. First is that older people do no have tendency to book appointments any eariler than younger people.
# Second people with no serious health problems tends to go to the doctors more than those that have serious health problems.
# 
# ### Limitations
# This data doesn't provide enough information for accurate prediction of if paitents will show up. For example the distance between the paitnet and the clinc and the reason paitent booked the apointment all are significant to predicting if paitent will show up for appointment or not.

# In[ ]:




