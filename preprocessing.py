import pandas as pd
import numpy as np
import lightgbm
from sklearn.model_selection import train_test_split

def binary_map(feature):
    return feature.map({'Yes':1, 'No':0})
def split_multicolumn(col_series):
    result_df = col_series.to_frame()
    options = []
    # Iterate over the column
    for idx, value  in col_series[col_series.notnull()].iteritems():
        # Break each value into list of options
        for option in value.split(';'):
            # Add the option as a column to result
            if not option in result_df.columns:
                options.append(option)
                result_df[option] = False
            # Mark the value in the option column as Tru   
            result_df.at[idx, option] = True
            
    return result_df[options]

def get_dummies_for_object (df):
    for i in df.select_dtypes(include=['object']).columns:
        if df[f'{i}'].nunique()<50:
            dumm=pd.get_dummies(df[i], prefix=i,dummy_na=False)
            df = pd.concat([df, dumm], axis=1)
            col=[col for col in df.columns if col is not i]
            df=df.loc[:,col]
        else : pass
    return df     
def fill_accordingto_specific_corelation(df,feature_fill, feature_from,feature_from2,feature_from3):
    index_nan = list(df[feature_fill][df[feature_fill].isnull()].index)
    for i in index_nan:
        feature_pred = df[feature_fill][
            ((df[feature_from] == df.iloc[i][feature_from]) & (df[feature_from2] == df.iloc[i][feature_from2]) & (df[feature_from3] == df.iloc[i][feature_from3])) 
            ].median()
        feature_med = df[feature_fill].median()
        if not np.isnan(feature_pred):
            df[feature_fill].iloc[i] = feature_pred
        else:
            df[feature_fill].iloc[i] = feature_med    

def preprocess(df, option):
    cols = ['Country_extremely_high_90k_198k',
 'Country_very_high_65k_90k',
 'YearsCodingProf_0_2_years',
 'Country_high_45k_65k',
 'YearsCodingProf_3_5_years',
 'AssessBenefits2_Bad',
 'DevType_Student',
 'Employment_self_employed',
 'CompanySize_10000_or_more_employees',
 'YearsCodingProf_6_8_years',
 'Age_18_24_years_old',
 'CareerSatisfaction_Extremelysatisfied',
 'Employment_full_time',
 'LanguageWorkedWith_PHP',
 'Age_35_44_years_old',
 'CareerSatisfaction_Moderatelysatisfied',
 'CommunicationTools_Confluence',
 'CommunicationTools_Slack',
 'DevType_Engineering_manager',
 'EducationTypes_Contributed_to_open_source_software',
 'FormalEducation_Master',
 'Country_moderate_25k_45k',
 'Dependents_No',
 'PlatformWorkedWith_Windows_Desktop_or_Server',
 'LanguageWorkedWith_Cplusplus',
 'EducationTypes_Participated_in_a_hackathon',
 'ErgonomicDevices_Standing_desk',
 'LastNewJob',
 'PlatformWorkedWith_AWS',
 'PlatformWorkedWith_Mac_OS',
 'CommunicationTools_Jira',
 'IDE_Notepadplusplus',
 'Age_25_34_years_old',
 'Methodology_Agile',
 'HypotheticalTools2_interested',
 'CompanySize_Fewer_than_10_employees',
 'AdsPriorities6_Bad',
 'JobEmailPriorities6_Good',
 'AdsActions_Paid_to_access_a_website_advertisement_free',
 'Country_somewhat_low_10k_25k',
 'Exercise_I_dont_typically_exercise',
 'Age_45_54_years_old',
 'DevType_Database_administrator',
 'CompanySize_10_to_19_employees',
 'AdBlockerReasons_I_wanted_to_support_the_website_I_was_visiting_by_viewing_their_ads',
 'CompanySize_1000_to_4999_employees',
 'JobContactPriorities5_Bad',
 'ErgonomicDevices_Ergonomic_keyboard_or_mouse',
 'PlatformDesireNextYear_Android',
 'WakeTime_Between8_01_9_00AM',
 'HoursOutside_1_2_hours',
 'DevType_Designer',
 'AssessBenefits8_Bad',
 'DevType_System_administrator',
 'SelfTaughtTypes_The_official_documentation_and_or_standards_for_the_technology',
 'LanguageDesireNextYear_JavaScript',
 'CommunicationTools_Trello',
 'JobContactPriorities3_Good',
 'Exercise_3_4_times_per_week',
 'HypotheticalTools3_interested',
 'IDE_IntelliJ',
 'Methodology_Extreme_programming__XP_',
 'LanguageWorkedWith_TypeScript',
 'AdsPriorities7_Bad',
 'HoursOutside_Less_than_30_minutes',
 'AssessJob7_Bad',
 'HackathonReasons_To_improve_my_ability_to_work_on_a_team_with_other_programmers',
 'LanguageWorkedWith_HTML',
 'HopeFiveYears_Anotherrole',
 'YearsCodingProf_9_11_years',
 'HopeFiveYears_Havingown_company',
 'SelfTaughtTypes_A_book_or_e_book_from_O’Reilly_Apress_or_a_similar_publisher',
 'LanguageDesireNextYear_HTML',
 'IDE_PHPStorm',
 'DatabaseWorkedWith_SQL_Server',
 'LanguageWorkedWith_Go',
 'LanguageDesireNextYear_SQL',
 'IDE_Android_Studio',
 'FormalEducation_Studywithoutdegree',
 'PlatformDesireNextYear_AWS',
 'DevType_Desktop_or_enterprise_applications_developer',
 'JobEmailPriorities6_Bad',
 'FrameworkDesireNextYear_Angular',
 'DatabaseDesireNextYear_MongoDB',
 'CommunicationTools_Google_Hangouts_Chat',
 'FrameworkWorkedWith_React',
 'DevType_QA_or_test_developer',
 'AssessJob8_Good',
 'PlatformWorkedWith_Google_Cloud_Platform_App_Engine',
 'CommunicationTools_Facebook',
 'JobSatisfaction_Moderatelysatisfied',
 'Gender_Male',
 'DatabaseWorkedWith_PostgreSQL',
 'YearsCodingProf_30_or_more_years',
 'DatabaseDesireNextYear_MySQL',
 'EthicsChoice_Dependsonwhatitis',
 'AssessBenefits2_Good',
 'IDE_Sublime_Text',
 'PlatformDesireNextYear_Raspberry_Pi',
 'HopeFiveYears_Engineeringmanager',
 'Employment_part_time',
 'NumberMonitors_1',
 'JobSatisfaction_Extremelysatisfied',
 'AssessJob1_Good',
 'LanguageDesireNextYear_PHP',
 'AssessBenefits8_Good',
 'DevType_DevOps_specialist',
 'IDE_Eclipse',
 'EducationTypes_Participated_in_online_coding_competitions__e.g._HackerRank_CodeChef_TopCoder_',
 'PlatformDesireNextYear_Serverless',
 'Methodology_Formal_standard_such_as_ISO_9001_or_IEEE_12207__aka_“waterfall”_methodologies_',
 'CommunicationTools_HipChat',
 'JobContactPriorities1_Good',
 'PlatformDesireNextYear_Linux',
 'PlatformWorkedWith_Raspberry_Pi',
 'StackOverflowVisit_Afewtimespermonthorweekly',
 'CareerSatisfaction_Moderatelydissatisfied',
 'PlatformWorkedWith_Azure',
 'DatabaseWorkedWith_MariaDB',
 'DatabaseWorkedWith_Memcached',
 'JobContactPriorities2_Good',
 'CommunicationTools_Other_chat_system__IRC_proprietary_software__',
 'AssessJob5_Bad',
 'AssessBenefits11_Good',
 'FormalEducation_Associatedegree',
 'CompanySize_500_to_999_employees',
 'DevType_Mobile_developer',
 'HoursOutside_30_59_minutes',
 'LanguageWorkedWith_CSS',
 'DatabaseWorkedWith_Elasticsearch',
 'JobSatisfaction_Moderatelydissatisfied',
 'DevType_Product_manager',
 'HackathonReasons_Because_I_find_it_enjoyable']
      

    if (option == "Online"):

        binary_list=[ 'CompanySize_Small_5_60','Dependents_No']
        df[binary_list] = df[binary_list].apply(binary_map)
        
        df = pd.get_dummies(df).reindex(columns=cols, fill_value=0)
        
        return df
    
    elif (option == "Batch"):
        '''try:
            print(x)
        except:
            print("An exception occurred")'''

        num_col=['Salary','ConvertedSalary','CompanySize','YearsCoding','YearsCodingProf','NumberMonitors','HoursComputer','HoursOutside','Exercise','Age']
        categoric_cols=['AssessJob1', 'AssessJob2', 'AssessJob3', 'AssessJob4','AssessJob5', 'AssessJob6', 'AssessJob7', 'AssessJob8', 'AssessJob9',
        'AssessJob10', 'AssessBenefits1', 'AssessBenefits2', 'AssessBenefits3','AssessBenefits4', 'AssessBenefits5', 'AssessBenefits6','AssessBenefits7', 'AssessBenefits8', 'AssessBenefits9',
        'AssessBenefits10', 'AssessBenefits11', 'JobContactPriorities1','JobContactPriorities2', 'JobContactPriorities3','JobContactPriorities4', 'JobContactPriorities5', 'JobEmailPriorities1',
        'JobEmailPriorities2', 'JobEmailPriorities3', 'JobEmailPriorities4','JobEmailPriorities5', 'JobEmailPriorities6', 'JobEmailPriorities7','AdsPriorities1', 'AdsPriorities2', 'AdsPriorities3',
        'AdsPriorities4', 'AdsPriorities5', 'AdsPriorities6', 'AdsPriorities7','Hobby', 'OpenSource', 'Country', 'Student', 'Employment','FormalEducation', 'UndergradMajor', 'DevType','JobSatisfaction','CareerSatisfaction', 'HopeFiveYears', 'JobSearchStatus', 'LastNewJob',
        'UpdateCV', 'Currency', 'SalaryType', 'CurrencySymbol','CommunicationTools', 'TimeFullyProductive', 'EducationTypes','SelfTaughtTypes', 'TimeAfterBootcamp', 'HackathonReasons',
        'AgreeDisagree1', 'AgreeDisagree2', 'AgreeDisagree3','LanguageWorkedWith', 'LanguageDesireNextYear', 'DatabaseWorkedWith','DatabaseDesireNextYear', 'PlatformWorkedWith','PlatformDesireNextYear', 'FrameworkWorkedWith',
        'FrameworkDesireNextYear', 'IDE', 'OperatingSystem', 'NumberMonitors','Methodology', 'VersionControl', 'CheckInCode', 'AdBlocker','AdBlockerDisable', 'AdBlockerReasons', 'AdsAgreeDisagree1',
        'AdsAgreeDisagree2', 'AdsAgreeDisagree3', 'AdsActions', 'AIDangerous','AIInteresting', 'AIResponsible', 'AIFuture', 'EthicsChoice','EthicsReport', 'EthicsResponsible', 'EthicalImplications',
        'StackOverflowRecommend', 'StackOverflowVisit','StackOverflowHasAccount', 'StackOverflowParticipate','StackOverflowJobs', 'StackOverflowDevStory','StackOverflowJobsRecommend', 'StackOverflowConsiderMember',
        'HypotheticalTools1', 'HypotheticalTools2', 'HypotheticalTools3','HypotheticalTools4', 'HypotheticalTools5', 'WakeTime', 'HoursComputer','HoursOutside', 'SkipMeals', 'ErgonomicDevices', 'Gender','SexualOrientation', 'EducationParents', 'RaceEthnicity',
        'Dependents', 'MilitaryUS', 'SurveyTooLong', 'SurveyEasy']
        #FormalEducation
        d = {"Bachelor’s degree (BA, BS, B.Eng., etc.)": "Bachelor",'Master’s degree (MA, MS, M.Eng., MBA, etc.)':'Master','Some college/university study without earning a degree':'Study without degree','Secondary school (e.g. American high school, German Realschule or Gymnasium, etc.)':'Secondary school','Other doctoral degree (Ph.D, Ed.D., etc.)':'doctoral degree'
            ,'Professional degree (JD, MD, etc.)':'Professional degree','I never completed any formal education':'Not graduated'}
        for i in list(d):df['FormalEducation']=df['FormalEducation'].map(lambda x: d[i] if (x== i) else x)
        #Employment
        d1 = {'Employed full-time': 'full-time','Employed part-time':'part-time','Not employed, but looking for work':'looking for work','Independent contractor, freelancer, or self-employed':'self-employed','Not employed, and not looking for work':'not want to work'}
        for i in list(d1):df['Employment']=df['Employment'].map(lambda x: d1[i] if (x== i) else x)
        #Student
        d2 = {'Yes, full-time':'Student full-time ','Yes, part-time':'Student part-time'}
        for i in list(d2):df['Student']=df['Student'].map(lambda x: d2[i] if (x== i) else x)  
        #JobSearchStatus
        d={'I’m not actively looking, but I am open to new opportunities':'Not actively',"I am not interested in new job opportunities":"Not-interested","I am actively looking for a job":"Actively"}
        for i in list(d):df['JobSearchStatus']=df['JobSearchStatus'].map(lambda x: d[i] if (x== i) else x)  
        #LastNewJob (Daha once bir iste calisti mi?)
        d={'Less than a year ago':1,'Between 1 and 2 years ago':2,'More than 4 years ago':5,'Between 2 and 4 years ago':3,"I've never had a job":0}
        for i in list(d):df['LastNewJob']=df['LastNewJob'].map(lambda x: d[i] if (x== i) else x)
        #HopeFiveYears
        d={"Working in a different or more specialized technical role than the one I'm in now": 'Another role',
        'Working as a founder or co-founder of my own company' : 'Having own-company',                                
        'Doing the same work' :'same',                                                                  
        'Working as an engineering manager or other functional manager':'Engineering manager',                        
        'Working as a product manager or project manager':'Product Manager',                                       
        'Working in a career completely unrelated to software development': 'Unrelated Career',                       
        'Retirement':'Retirement' }
        for i in list(d):df['HopeFiveYears']=df['HopeFiveYears'].map(lambda x: d[i] if (x== i) else x)
                
        #UndergradMajor
        d={'Computer science, computer engineering, or software engineering':'Computer science',           
        'Another engineering discipline (ex. civil, electrical, mechanical)':'Another engineering',        
        'Information systems, information technology, or system administration':'information technology',     
        'A natural science (ex. biology, chemistry, physics)':'Natural Science',                       
        'Mathematics or statistics':'Math/stats',                                                 
        'Web development or web design':'Web development',                                              
        'A business discipline (ex. accounting, finance, marketing)':'Business Discipline',                 
        'A humanities discipline (ex. literature, history, philosophy)':'Humanities Discipline',              
        'A social science (ex. anthropology, psychology, political science)':'Social Science',         
        'Fine arts or performing arts (ex. graphic design, music, studio art)':'Arts',       
        'I never declared a major':'Never declerad major',                                                    
        'A health science (ex. nursing, pharmacy, radiology)':'Health Science'}
        for i in list(d):df['UndergradMajor']=df['UndergradMajor'].map(lambda x: d[i] if (x== i) else x)
        #Gender
        index_male=df['Gender'][df['Gender']=='Male'].index.tolist()
        index_female=df['Gender'][df['Gender']=='Female'].index.tolist()
        index_NaN=df['Gender'][df['Gender'].isnull()==True].index.tolist()
        index_ = set(df['Gender'].index.tolist()).difference(set(index_male+index_female+index_NaN))
        for i in index_: df['Gender'][i]='LGBTQ'
        #CompanySize
        d={'20 to 99 employees':60,'100 to 499 employees':300,'10,000 or more employees':12500,'10 to 19 employees':15,'1,000 to 4,999 employees':3000,'Fewer than 10 employees':5,'500 to 999 employees':750,'5,000 to 9,999 employees':7500}  
        for i in list(d):df['CompanySize']=df['CompanySize'].map(lambda x: d[i] if (x== i) else x)
        #YearsCoding
        d={'3-5 years':2.5,'6-8 years':7,'9-11 years':10,'0-2 years':1,'12-14 years':13,'15-17 years':16,'18-20 years':19,'30 or more years':35,'21-23 years':22,'24-26 years':25,'27-29 years':28  }       
        for i in list(d):df['YearsCoding']=df['YearsCoding'].map(lambda x: d[i] if (x== i) else x)
        #YearsCodingProf    
        d={'0-2 years':1,'3-5 years':4,'6-8 years':7,'9-11 years':10,'12-14 years': 13,'15-17 years':16,'18-20 years':19,'21-23 years': 22,'30 or more years':35,'24-26 years': 25, '27-29 years':28  }
        for i in list(d):df['YearsCodingProf']=df['YearsCodingProf'].map(lambda x: d[i] if (x== i) else x) 
        #Age
        d={'25 - 34 years old':29.5,'18 - 24 years old':21,'6-8 years':7,'35 - 44 years old':39,'12-14 years': 13,'45 - 54 years old':50,'18-20 years':19,'Under 18 years old': 17,'55 - 64 years old':60,'65 years or older': 66}
        for i in list(d):df['Age']=df['Age'].map(lambda x: d[i] if (x== i) else x) 
        #HoursComputer
        d={'9 - 12 hours':10.5,'5 - 8 hours':6.5,'Over 12 hours':15,'1 - 4 hours':2.5,'Less than 1 hour': 0.1}
        for i in list(d):df['HoursComputer']=df['HoursComputer'].map(lambda x: d[i] if (x== i) else x) 
        #HoursOutside
        d={'1 - 2 hours':1.5,'30 - 59 minutes':0.45,'Less than 30 minutes':0.2,'3 - 4 hours':3.5,'Over 4 hours': 5}
        for i in list(d):df['HoursOutside']=df['HoursOutside'].map(lambda x: d[i] if (x== i) else x) 
        #Exercise
        d={"I don't typically exercise":0,'1 - 2 times per week':1.5,'3 - 4 times per week':3.5,'Daily or almost every day':7}
        for i in list(d):df['Exercise']=df['Exercise'].map(lambda x: d[i] if (x== i) else x) 
        del1=['AdsPriorities1','AssessBenefits9','AssessJob10','AssessJob2','JobEmailPriorities2','Respondent','AssessJob9','JobEmailPriorities1','AdsPriorities2','AdsPriorities5','AssessBenefits4','AssessJob6','AdsPriorities4',
        'JobEmailPriorities7','AssessJob4','AssessJob3','AdsPriorities3']
        df=df.drop(df[del1],axis=1)
        #YearsCodingProf and YearsCoding have more than %80 Correlation,then we will delete one of them
        del2=['YearsCoding']
        df=df.drop(df[del2],axis=1)
        #The features showing below are deleted because we will use ConvertedSalary in place of Salary and the other related columns with it.
        del3=["Salary","Currency","CurrencySymbol","SalaryType"] #Respondent is already removed
        df=df.drop(df[del3],axis=1)
        del4=["AIResponsible","AdsAgreeDisagree2","TimeFullyProductive","AIDangerous","AdsAgreeDisagree3","JobSearchStatus","EthicsReport","HypotheticalTools4","AIFuture","AdsAgreeDisagree1","HypotheticalTools5",'UpdateCV',
        'HypotheticalTools1','OperatingSystem','EducationParents'] #without Gender
        df=df.drop(df[del4],axis=1)
        delll=['SurveyEasy','SurveyTooLong','AdBlockerDisable','StackOverflowDevStory','EthicsResponsible','AgreeDisagree2','StackOverflowHasAccount','SkipMeals','AgreeDisagree3', 'StackOverflowParticipate', 'StackOverflowJobsRecommend', 'AgreeDisagree1' ,'StackOverflowRecommend', 'StackOverflowConsiderMember', 'Hobby' ,'OpenSource']
        df=df.drop(df[delll],axis=1)
        df=df.drop(df[['MilitaryUS']],axis=1)
        #Outliers
        df['ConvertedSalary']=df['ConvertedSalary'].apply(lambda x: 196000 if x>= 196000 else  x)
        del_nan=['TimeAfterBootcamp']
        df=df.drop(df[del_nan],axis=1)
    
        #Filling NaN
        from sklearn.impute import SimpleImputer
        imp_most= SimpleImputer(missing_values=np.nan, strategy='most_frequent')
        #We impute features with less than 3k NaN numbers in aimed data (not in all data),these have less correlation btw
        df['Employment']=imp_most.fit_transform(df['Employment'].values.reshape(-1,1))[:,0]  #154
        df['Student']=imp_most.fit_transform(df['Student'].values.reshape(-1,1))[:,0] #400
        df['CareerSatisfaction']=imp_most.fit_transform(df['CareerSatisfaction'].values.reshape(-1,1))[:,0] #737
        df['StackOverflowVisit']=imp_most.fit_transform(df['StackOverflowVisit'].values.reshape(-1,1))[:,0] #1209
        df['FormalEducation']=imp_most.fit_transform(df['FormalEducation'].values.reshape(-1,1))[:,0] #694
        df['HopeFiveYears']=imp_most.fit_transform(df['HopeFiveYears'].values.reshape(-1,1))[:,0] #859
        df['WakeTime']=imp_most.fit_transform(df['WakeTime'].values.reshape(-1,1))[:,0] #1673
        df['NumberMonitors']=imp_most.fit_transform(df['NumberMonitors'].values.reshape(-1,1))[:,0] #1322 
        df['StackOverflowJobs']=imp_most.fit_transform(df['StackOverflowJobs'].values.reshape(-1,1))[:,0] #1436
        df['CheckInCode']=imp_most.fit_transform(df['CheckInCode'].values.reshape(-1,1))[:,0] #2600
        df['Country']=imp_most.fit_transform(df['Country'].values.reshape(-1,1))[:,0] #0 #salary kisminda nan yok
        #
        with_nan= df[['Dependents','Age']][df['Dependents'].isnull()==True]
        with_nan1=with_nan[(with_nan['Age']==39.0)]
        n_index=with_nan1.index.to_list()
        for i in n_index: df['Dependents'][i]='Yes'    
        with_nan1=with_nan[(with_nan['Age']==66.0)]
        n_index=with_nan1.index.to_list()
        for i in n_index: df['Dependents'][i]='Yes'    
        df[['Dependents']]=imp_most.fit_transform(df[['Dependents']])
        #
        with_nan= df[['EthicalImplications','EthicsChoice']][df['EthicalImplications'].isnull()==True]
        with_nan1=with_nan[(with_nan['EthicsChoice']=='Depends on what it is')]
        n_index=with_nan1.index.to_list()
        for i in n_index: df['EthicalImplications'][i]="Unsure/ I don't know"    
        #
        with_nan= df[['EthicalImplications','JobSatisfaction']][df['EthicalImplications'].isnull()==True]
        with_nan1=with_nan[(with_nan['JobSatisfaction']=="Slightly satisfied")]
        n_index=with_nan1.index.to_list()
        for i in n_index: df['EthicalImplications'][i]="Unsure/I don't know"    
        #
        with_nan= df[['EthicalImplications','HypotheticalTools3']][df['EthicalImplications'].isnull()==True]
        with_nan1=with_nan[(with_nan['HypotheticalTools3']=="Very interested")]
        n_index=with_nan1.index.to_list()
        for i in n_index: df['EthicalImplications'][i]="No"    
        #
        with_nan= df[['EthicalImplications','HypotheticalTools3']][df['EthicalImplications'].isnull()==True]
        with_nan1=with_nan[(with_nan['HypotheticalTools3']=="A little bit interested")]
        n_index=with_nan1.index.to_list()
        for i in n_index: df['EthicalImplications'][i]="Unsure / I don't know"    
        df[['EthicalImplications']]=imp_most.fit_transform(df[['EthicalImplications']])
        #
        with_nan= df[['HypotheticalTools3','UndergradMajor']][df['HypotheticalTools3'].isnull()==True]
        with_nan1=with_nan[(with_nan['UndergradMajor']=="Another engineering")]
        n_index=with_nan1.index.to_list()
        for i in n_index: df['HypotheticalTools3'][i]="A little bit interested"    
        #
        with_nan= df[['HypotheticalTools3','UndergradMajor']][df['HypotheticalTools3'].isnull()==True]
        with_nan1=with_nan[(with_nan['UndergradMajor']=="Humanities Discipline")]
        n_index=with_nan1.index.to_list()
        for i in n_index: df['HypotheticalTools3'][i]="Not at all interested"  
        #
        with_nan= df[['HypotheticalTools3','JobSatisfaction']][df['HypotheticalTools3'].isnull()==True]
        with_nan1=with_nan[(with_nan['JobSatisfaction']=="Slightly satisfied")]
        n_index=with_nan1.index.to_list()
        for i in n_index: df['HypotheticalTools3'][i]="Very interested"    
        df[['HypotheticalTools3']]=imp_most.fit_transform(df[['HypotheticalTools3']])
        #
        with_nan= df[['Gender','CareerSatisfaction']][df['Gender'].isnull()==True]
        with_nan1=with_nan[(with_nan['CareerSatisfaction']=="Slightly satisfied")]
        n_index=with_nan1.index.to_list()
        for i in n_index: df['Gender'][i]="Female"    
        df[['Gender']]=imp_most.fit_transform(df[['Gender']]) 
        # 
        with_nan= df[['Gender','HopeFiveYears']][df['Gender'].isnull()==True]
        with_nan1=with_nan[(with_nan['HopeFiveYears']=="same")]
        n_index=with_nan1.index.to_list()
        for i in n_index: df['Gender'][i]="Female"    
        #
        with_nan= df[['Gender','FormalEducation']][df['Gender'].isnull()==True]
        with_nan1=with_nan[(with_nan['FormalEducation']=="Not graduated")]
        n_index=with_nan1.index.to_list()
        for i in n_index: df['Gender'][i]="LGBTQ"    
        #
        with_nan= df[['HypotheticalTools2','Age']][df['HypotheticalTools2'].isnull()==True]
        with_nan1=with_nan[(with_nan['Age']==39)]
        n_index=with_nan1.index.to_list()
        for i in n_index: df['HypotheticalTools2'][i]="Not at all interested"    
        #
        with_nan= df[['HypotheticalTools2','Age']][df['HypotheticalTools2'].isnull()==True]
        with_nan1=with_nan[(with_nan['Age']==39)]
        n_index=with_nan1.index.to_list()
        for i in n_index: df['HypotheticalTools2'][i]="Not at all interested"    
        #            
        with_nan= df[['HypotheticalTools2','Age']][df['HypotheticalTools2'].isnull()==True]
        with_nan1=with_nan[(with_nan['Age']==39)]
        n_index=with_nan1.index.to_list()
        for i in n_index: df['HypotheticalTools2'][i]="Not at all interested"    
        df[['HypotheticalTools2']]=imp_most.fit_transform(df[['HypotheticalTools2']])
        #
        with_nan= df[['JobSatisfaction','CompanySize']][df['JobSatisfaction'].isnull()==True]
        with_nan1=with_nan[(with_nan['CompanySize']==df['CompanySize'].mean())]
        n_index=with_nan1.index.to_list()
        for i in n_index: df['JobSatisfaction'][i]='Neither satisfied nor dissatisfied'
        df[['JobSatisfaction']]=imp_most.fit_transform(df[['JobSatisfaction']])
        #
        with_nan= df[['JobSatisfaction','CompanySize']][df['JobSatisfaction'].isnull()==True]
        with_nan1=with_nan[(with_nan['CompanySize']==df['CompanySize'].mean())]
        n_index=with_nan1.index.to_list()
        for i in n_index: df['JobSatisfaction'][i]='Neither satisfied nor dissatisfied'
        df[['JobSatisfaction']]=imp_most.fit_transform(df[['JobSatisfaction']])
        #
        with_nan= df[['UndergradMajor','CareerSatisfaction']][df['UndergradMajor'].isnull()==True]
        with_nan1=with_nan[(with_nan['CareerSatisfaction']=='Slightly satisfied')]
        n_index=with_nan1.index.to_list()
        for i in n_index: df['UndergradMajor'][i]='information technology'
        df[['UndergradMajor']]=imp_most.fit_transform(df[['UndergradMajor']])
        #
        with_nan= df[['EthicsChoice','EthicalImplications']][df['EthicsChoice'].isnull()==True]
        with_nan1=with_nan[(with_nan['EthicalImplications']=='Depends on what it is')]
        n_index=with_nan1.index.to_list()
        for i in n_index: df['EthicsChoice'][i]='No'
        df[['EthicsChoice']]=imp_most.fit_transform(df[['EthicsChoice']])
        with_nan= df[['EthicsChoice','AIInteresting']][df['AIInteresting'].isnull()==True]
        with_nan1=with_nan[(with_nan['EthicsChoice']=='No')]
        n_index=with_nan1.index.to_list()
        for i in n_index: df['AIInteresting'][i]='Increasing automation of jobs'
        df[['AIInteresting']]=imp_most.fit_transform(df[['AIInteresting']])
        #
        with_nan= df[['StackOverflowVisit','StackOverflowJobs']][df['StackOverflowVisit'].isnull()==True]
        with_nan1=with_nan[(with_nan['StackOverflowJobs']=="No, I didn't know that Stack Overflow had a jobs board")]
        n_index=with_nan1.index.to_list()
        for i in n_index: df['StackOverflowVisit'][i]='Less than once per month or monthly'

        df[['StackOverflowVisit']]=imp_most.fit_transform(df[['StackOverflowVisit']])
        #
        fillna_missing=[ 'AssessJob1','AssessJob5', 'AssessJob7', 'AssessJob8', 'AssessBenefits1','AssessBenefits2', 'AssessBenefits3', 'AssessBenefits5','AssessBenefits6', 
                'AssessBenefits7','AssessBenefits8','AssessBenefits10', 'AssessBenefits11', 'JobContactPriorities1','JobContactPriorities2', 'JobContactPriorities3','JobContactPriorities4', 'JobContactPriorities5',
                'JobEmailPriorities3','JobEmailPriorities4', 'JobEmailPriorities5', 'JobEmailPriorities6', 'AdsPriorities6', 'AdsPriorities7']
        for i in fillna_missing:
            df[[i]]=df[[i]].fillna('missing')
        #
        from sklearn.impute import SimpleImputer
        imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')

        #CompanySize has correlation lower than %10,we will fill it with mean and it has 6742 nans in half data
        df[['CompanySize']]=imp_mean.fit_transform(df[['CompanySize']])
        #Exercise has correlation lower than %10,we will fill it with mean and it has 1697 nans in half data
        df[['Exercise']]=imp_mean.fit_transform(df[['Exercise']])

        #Other numeric columns will be filled with median of the most correlated 3 columns
        fill_accordingto_specific_corelation(df,'Age', 'CompanySize','YearsCodingProf','LastNewJob')  
        #HoursComputer --1718
        fill_accordingto_specific_corelation(df,'HoursComputer', 'Exercise','HoursOutside','CompanySize')    
        #HoursOutside --1772
        fill_accordingto_specific_corelation(df,'HoursOutside', 'Exercise','LastNewJob','YearsCodingProf') 
        #LastNewJob -- 92
        fill_accordingto_specific_corelation(df,'LastNewJob', 'Age','YearsCodingProf','HoursOutside')    
        #YearsCodingProf --737
        fill_accordingto_specific_corelation(df,'YearsCodingProf', 'Age','LastNewJob','ConvertedSalary')    
       
        #FEATURE ENGINEERING
        def convert_features(feature):
            d = {1.0:'Good',2.0:'Good',3.0:'Good',4.0:'Good',5.0:'Good',6.0:'Bad',7.0:'Bad',8.0:'Bad',9.0:'Bad',10.0:'Bad'}
            for i in list(d):df[feature]=df[feature].map(lambda x: d[i] if (x==i) else x)
        for i in ['AssessJob1','AssessJob5','AssessJob7','AssessJob8']:
            convert_features(i)        
        def convert2_features(feature):
            d = {1.0:'Good',2.0:'Good',3.0:'Good',4.0:'Good',5.0:'Good',6.0:'Bad',7.0:'Bad',8.0:'Bad',9.0:'Bad',10.0:'Bad',11.0:'Bad'}
            for i in list(d):df[feature]=df[feature].map(lambda x: d[i] if (x==i) else x)
        for i in ['AssessBenefits1','AssessBenefits2', 'AssessBenefits3', 'AssessBenefits5','AssessBenefits6', 'AssessBenefits7', 'AssessBenefits8','AssessBenefits10', 'AssessBenefits11']:
            convert2_features(i)       
        def convert3_features(feature):
            d = {1.0:'Good',2.0:'Good',3.0:'Good',4.0:'Bad',5.0:'Bad'}
            for i in list(d):df[feature]=df[feature].map(lambda x: d[i] if (x==i) else x)
        for i in ['JobContactPriorities1','JobContactPriorities2', 'JobContactPriorities3','JobContactPriorities4', 'JobContactPriorities5']:
            convert3_features(i)
        def convert4_features(feature):
            d = {1.0:'Good',2.0:'Good',3.0:'Good',4.0:'Good',5.0:'Bad',6.0:'Bad',7.0:'Bad'}
            for i in list(d):df[feature]=df[feature].map(lambda x: d[i] if (x==i) else x)
        for i in ['JobEmailPriorities3','JobEmailPriorities4', 'JobEmailPriorities5', 'JobEmailPriorities6','AdsPriorities6','AdsPriorities7']:
            convert4_features(i)
        def convert5_features(feature):
            d = {'Not at all interested':'not_interested','Somewhat interested':'interested','A little bit interested':'not_interested','Very interested':'interested','Extremely interested':'interested'}
            for i in list(d):df[feature]=df[feature].map(lambda x: d[i] if (x==i) else x)
        for i in ['HypotheticalTools2','HypotheticalTools3']:
            convert5_features(i)
        #CompanySize
        d={60:'20_to_99 employees',300:'100_to_499 employees',12500:'10000_or_more_employees',15:'10_to_19_employees',3000:'1000_to_4999_employees',5:'Fewer_than_10_employees',750:'500_to_999_employees',7500:'5000_to_9999_employees'}  
        for i in list(d):df['CompanySize']=df['CompanySize'].map(lambda x: d[i] if (x== i) else x)
        #YearsCoding
        #d={2.5:'3_5_years',7:'6_8_years',10:'9_11_years',1:'0_2_years',13:'12_14 years',16:'15_17 years',19:'18_20 years',35:'30_or_more_years',22:'21_23_years',25:'24_26_years',28:'27_29_years'}       
        #for i in list(d):df['YearsCoding']=df['YearsCoding'].map(lambda x: d[i] if (x== i) else x)
        #YearsCodingProf    
        d={1:'0_2_years',4:'3_5_years',7:'6_8_years',10:'9_11_years',13:'12_14_years',16:'15_17_years',19:'18_20_years',22:'21_23_years',35:'30_or_more_years',25:'24_26_years',28: '27_29_years'}
        for i in list(d):df['YearsCodingProf']=df['YearsCodingProf'].map(lambda x: d[i] if (x== i) else x) 
        #Age
        d={29.5:'25_34_years_old',21:'18_24_years_old',7:'6_8_years',39:'35_44_years_old',13:'12_14_years',50:'45_54_years_old',19:'18_20_years',17:'Under_18_years_old',60:'55_64_years_old',66:'65_years_or_older'}
        for i in list(d):df['Age']=df['Age'].map(lambda x: d[i] if (x== i) else x) 
        #HoursComputer
        d={10.5:'9_12_hours',6.5:'5_8_hours',15:'Over_12_hours',2.5:'1_4 hours',0.1:'Less_than_1_hour'}
        for i in list(d):df['HoursComputer']=df['HoursComputer'].map(lambda x: d[i] if (x== i) else x) 
        #HoursOutside
        d={1.5:'1_2_hours',0.45:'30_59_minutes',0.2:'Less_than_30_minutes',3.5:'3_4_hours',5:'Over_4_hours'}
        for i in list(d):df['HoursOutside']=df['HoursOutside'].map(lambda x: d[i] if (x== i) else x) 
        #Exercise
        d={0:"I_dont_typically_exercise",1.5:'1_2_times_per_week',3.5:'3_4_times_per_week',7:'Daily_or_almost_every_day'}
        for i in list(d):df['Exercise']=df['Exercise'].map(lambda x: d[i] if (x== i) else x) 

        dff=df.dropna(subset=['ConvertedSalary'])
        temp=dff.groupby(['Country'],as_index=False).ConvertedSalary.mean().sort_values('ConvertedSalary',ascending=False)
        #df=dff.copy()
        #Convert Country
        #extremely high
        extremely_high_90k_198k=[]
        temp=dff.groupby(['Country'],as_index=False).ConvertedSalary.mean().sort_values('ConvertedSalary',ascending=False)
        res=temp[temp['ConvertedSalary']>90000]
        for i in res['Country']:extremely_high_90k_198k.append(i)
        #very high
        very_high_65k_90k=[]
        res=temp[(temp['ConvertedSalary']>65000) & (temp['ConvertedSalary']<=90000)]
        for i in res['Country']:very_high_65k_90k.append(i)  
        #high
        high_45k_65k=[]
        res=temp[(temp['ConvertedSalary']>45000) & (temp['ConvertedSalary']<=65000)]
        for i in res['Country']:high_45k_65k.append(i)   
        #moderate
        moderate_25k_45k=[]
        res=temp[(temp['ConvertedSalary']>25000) & (temp['ConvertedSalary']<=45000)]
        for i in res['Country']:moderate_25k_45k.append(i)      
        #somewhat low
        somewhat_low_10k_25k=[]
        res=temp[(temp['ConvertedSalary']>10000) & (temp['ConvertedSalary']<=25000)]
        for i in res['Country']:somewhat_low_10k_25k.append(i)  
        #low
        low_less_10k=[]
        res=temp[temp['ConvertedSalary']<=10000]
        for i in res['Country']:low_less_10k.append(i)

        for i in extremely_high_90k_198k: df['Country']=df['Country'].map(lambda x: 'extremely_high_90k_198k' if (x==i) else x)
        for i in very_high_65k_90k: df['Country']=df['Country'].map(lambda x: 'very_high_65k_90k' if (x==i) else x)
        for i in high_45k_65k: df['Country']=df['Country'].map(lambda x: 'high_45k_65k' if (x==i) else x)
        for i in moderate_25k_45k: df['Country']=df['Country'].map(lambda x: 'moderate_25k_45k' if (x==i) else x)
        for i in somewhat_low_10k_25k: df['Country']=df['Country'].map(lambda x: 'somewhat_low_10k_25k' if (x==i) else x)
        for i in low_less_10k: df['Country']=df['Country'].map(lambda x: 'low_less_10k' if (x==i) else x)
        val=['Angola',
            'Antigua and Barbuda', 'Haiti', 'Gabon', 'Burkina Faso', 'North Korea',
            'Guinea', 'Cape Verde', 'Liberia', 'Mauritania',
            "Democratic People's Republic of Korea",'Central African Republic',
            'Palau', 'Micronesia, Federated States of...', 'Niger', 'Djibouti',
            'Mali', 'Brunei Darussalam', 'Grenada', 'Nauru', 'Solomon Islands',
            'Timor-Leste', 'Belize', 'Burundi', 'San Marino', 'Guinea-Bissau']
        for i in val: df['Country']=df['Country'].map(lambda x: 'others' if (x==i) else x)   
        df['LastNewJob']=df['LastNewJob'].astype(int)    
        #df=df.dropna(subset=['ConvertedSalary'])
        splitted_cols=['DevType','CommunicationTools','EducationTypes','SelfTaughtTypes','LanguageWorkedWith','LanguageDesireNextYear',
                'DatabaseWorkedWith','DatabaseDesireNextYear','PlatformWorkedWith','PlatformDesireNextYear','FrameworkWorkedWith',
                'FrameworkDesireNextYear','IDE','Methodology','VersionControl','SexualOrientation','RaceEthnicity','AdBlocker','HackathonReasons','AdBlockerReasons','AdsActions']
        def split_multicolumn_dummy(col_series,prefix):
            result_df = col_series.to_frame()
            options = []
            # Iterate over the column
            for idx, value  in col_series[col_series.notnull()].iteritems():
                # Break each value into list of options
                for option in value.split(';'):
                    # Add the option as a column to result
                    if not option in result_df.columns:
                        options.append(option)
                        result_df[option] = False
                    # Mark the value in the option column as Tru   
                    result_df.at[idx, option] = True
            return result_df[options].add_prefix(prefix+'_')
        df_DevType=split_multicolumn_dummy(df['DevType'],'DevType')
        df_CommunicationTools=split_multicolumn_dummy(df['CommunicationTools'],'CommunicationTools')
        df_EducationTypes=split_multicolumn_dummy(df['EducationTypes'],'EducationTypes')
        df_SelfTaughtTypes=split_multicolumn_dummy(df['SelfTaughtTypes'],'SelfTaughtTypes')
        df_HackathonReasons=split_multicolumn_dummy(df['HackathonReasons'],'HackathonReasons')
        df_LanguageWorkedWith=split_multicolumn_dummy(df['LanguageWorkedWith'],'LanguageWorkedWith')
        df_LanguageDesireNextYear=split_multicolumn_dummy(df['LanguageDesireNextYear'],'LanguageDesireNextYear')
        df_DatabaseWorkedWith=split_multicolumn_dummy(df['DatabaseWorkedWith'],'DatabaseWorkedWith')
        df_DatabaseDesireNextYear=split_multicolumn_dummy(df['DatabaseDesireNextYear'],'DatabaseDesireNextYear')
        df_PlatformDesireNextYear=split_multicolumn_dummy(df['PlatformDesireNextYear'],'PlatformDesireNextYear')
        df_PlatformWorkedWith=split_multicolumn_dummy(df['PlatformWorkedWith'],'PlatformWorkedWith')
        df_FrameworkWorkedWith=split_multicolumn_dummy(df['FrameworkWorkedWith'],'FrameworkWorkedWith')
        df_FrameworkDesireNextYear=split_multicolumn_dummy(df['FrameworkDesireNextYear'],'FrameworkDesireNextYear')
        df_IDE=split_multicolumn_dummy(df['IDE'],'IDE')
        df_Methodology=split_multicolumn_dummy(df['Methodology'],'Methodology')
        df_VersionControl=split_multicolumn_dummy(df['VersionControl'],'VersionControl')
        df_AdBlockers=split_multicolumn_dummy(df['AdBlocker'],'AdBlocker')
        df_AdBlockerReasons=split_multicolumn_dummy(df['AdBlockerReasons'],'AdBlockerReasons')
        df_RaceEthnicity=split_multicolumn_dummy(df['RaceEthnicity'],'RaceEthnicity')
        df_SexualOrientation=split_multicolumn_dummy(df['SexualOrientation'],'SexualOrientation')
        df_AdsActions=split_multicolumn_dummy(df['AdsActions'],'AdsActions')
        df_ErgonomicDevices=split_multicolumn_dummy(df['ErgonomicDevices'],'ErgonomicDevices')
        df_splitted_dummied=pd.concat([df_DevType,df_CommunicationTools,df_EducationTypes,df_SelfTaughtTypes,df_HackathonReasons,df_LanguageWorkedWith,df_LanguageDesireNextYear,df_DatabaseWorkedWith,
                 df_DatabaseDesireNextYear,df_PlatformWorkedWith,df_PlatformDesireNextYear,df_FrameworkWorkedWith,df_FrameworkDesireNextYear,df_IDE,df_Methodology,df_VersionControl,df_AdBlockers,
                 df_AdBlockerReasons,df_RaceEthnicity,df_SexualOrientation,df_AdsActions,df_ErgonomicDevices], axis=1, join='inner')
        df_splitted_dummied.columns = df_splitted_dummied.columns.str.replace(' ', '_')
        df_splitted_dummied=df_splitted_dummied.replace(True,1)
        df_splitted_dummied=df_splitted_dummied.replace(False,0)
        splitted_cols=['DevType','CommunicationTools','EducationTypes','SelfTaughtTypes','LanguageWorkedWith','LanguageDesireNextYear',
              'DatabaseWorkedWith','DatabaseDesireNextYear','PlatformWorkedWith','PlatformDesireNextYear','FrameworkWorkedWith',
               'FrameworkDesireNextYear','IDE','Methodology','VersionControl','SexualOrientation','RaceEthnicity','AdBlocker','HackathonReasons','AdBlockerReasons','AdsActions']
        get_dummies_cols=set(df.columns).difference(set(splitted_cols))
        get_dummies_cols=list(get_dummies_cols.difference(set(['ConvertedSalary' ])))
        df_getdum=get_dummies_for_object(df.loc[:,get_dummies_cols])
        df_num=df.loc[:,[ 'ConvertedSalary' ]]
        df=pd.concat([df_num,df_splitted_dummied,df_getdum], axis=1, join='inner')
        d=df.columns[df.columns.str.contains('missing')==True]
        df=df.drop(columns=d)
        df.columns=df.columns.str.replace(' ','')
        df.columns=df.columns.str.replace('/','_')
        df.columns=df.columns.str.replace('+','plus')
        df.columns=df.columns.str.replace(",",'')
        df.columns=df.columns.str.replace("-",'_')
        df.columns=df.columns.str.replace(":",'_')
        df.columns=df.columns.str.replace("'",'_')
        df.columns=df.columns.str.replace('"','_')
        df.columns=df.columns.str.replace('(','_')
        df.columns=df.columns.str.replace(')','_')
        df.columns=df.columns.str.replace("etc.",'')
        df=df.drop(columns='ConvertedSalary')
        ################################################## 
        df = pd.get_dummies(df).reindex(columns=cols, fill_value=0)
        print(df.columns)
        return df  
    else:
        print("Incorrect operational options")
        
    return df

        
