import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
 
from preprocessing import preprocess
#load the model from disk
import joblib
filename = 'model_lgbm.sav'
model = joblib.load(filename)

def main():

    st.set_page_config(page_title='StackOverflowDeveloperSurvey')
    st.title('StackOverflowDeveloperSurvey2018')
    st.markdown("""
                :dart:  This Streamlit app is made to predict Salary of the developers who participated in the 2018 StackOverflow survey.
    The application is functional for both online prediction and batch data prediction. \n
    """)
    st.markdown("<h3></h3>", unsafe_allow_html=True)
    add_selectbox = st.sidebar.selectbox(
    "How would you like to predict?", ("Online", "Batch"))
    st.sidebar.info('This app is created to predict Salary')
    if add_selectbox == "Online":
        st.info("")
        #Based on our optimal features selection
        st.subheader("***")

        #AGE
        st.subheader("Enter a number.")  
        age = st.number_input('1-What is your age?', min_value=18,value=18, max_value=75 , key = "ager")
        age_18_24_val=0
        age_25_34_val=0
        age_35_44_val=0
        age_45_66_val=0
        if 18<age<=24:age_18_24_val=1
        if 25<age<=34:age_25_34_val=1
        if 35<age<=44:age_35_44_val=1
        if 45<age<=66:age_45_66_val=1
        

        #Employment
        st.subheader("Select an option.")
        emp_slf_val=0
        emp_fll_val=0
        emp_prt_val=0
        emp =['Full Time','Part Time','Self Employed','Retired','Not Wanttowork','Looking Forwork']
        empw = st.selectbox('2-Which of the following best describes your current employment status?',   (emp), key = "emp")
        if empw =='Self Employed':emp_slf_val=1
        if empw =='Full Time':emp_fll_val=1
        if empw =='Part Time':emp_prt_val=1
        else :emp_prt_val=1


        #Dependents
        st.subheader("Select an option.") 
        Dependents_No = st.selectbox('3-Do you have any children or other dependents that you care for? If you prefer not to answer, you may leave this question blank.', ('Yes','No'), key = "d")
        
        #YearsCodingProf
        yc_0_2_val=0
        yc_3_5_val=0
        yc_6_8_val=0
        yc_9_11_val=0
        yc_11_30_val=0

        st.subheader("Select an option.") 
        YearsCodingProf = st.number_input('4-For how many years have you coded professionally (as a part of your work)?', min_value=0, max_value=30, value=10, key = "yy")

        if 0<YearsCodingProf<=2:age_18_24_val=1
        elif 3<YearsCodingProf<=5:age_25_34_val=1
        elif 6<YearsCodingProf<=8:age_35_44_val=1
        elif 9<YearsCodingProf<=11:age_45_66_val=1
        elif 11<YearsCodingProf<=30:age_45_66_val=1
        else:pass

        #PlatformWorkedWith
        st.subheader("Select an option.")
        plt_aws_val=0
        plt_windw_val=0
        plt_mac_val=0
        plt_ras_val=0
        plt_azr_val=0
        plt_ggl_cld_val=0
        plt_and_val=0
        plt =['Linux','Arduino','Windows Desktop or Server','Amazon Echo','AWS','iOS','Mac OS','Serverless','Android','Firebase',
        'Azure','Drupal','WordPress','Heroku','Raspberry','IBM Cloud or Watson','Mainframe','Apple Watch or Apple TV',
        'ESP8266','Google Cloud Platform App Engine','SharePoint','Predix','Salesforce','Windows Phone','Google Home','Gaming Console']
        pltw = st.multiselect('5-Which of the following platforms have you done extensive development work for over the past year?', (plt), key = "plt")
        for i in pltw:
            if pltw =='Mac OS':plt_mac_val=1
            if pltw =='Azure': plt_azr_val=1
            if pltw =='Raspberry':plt_ras_val=1
            if pltw =='Google Cloud Platform App Engine':plt_ggl_cld_val=1
            if pltw =='AWS':plt_aws_val=1
            if pltw =='Windows Desktop or Server':plt_windw_val=1
            if pltw =='Android':plt_and_val=1
            else :plt_and_val=1

        #CareerSatisfaction
        st.subheader("Select an option.")
        cr_exs_val=0
        cr_ms_val=0
        cr_md_val=0
        crry=['Extremely satisfied','Moderately satisfied','Slightly satisfied','Neither satisfied nor dissatisfied','Slightly dissatisfied','Moderately dissatisfied','Extremely dissatisfied']
        crryw= st.selectbox('6-Overall, how satisfied are you with your career thus far?',  (crry), key = "crry")
        
        if crryw=='Extremely satisfied':cr_exs_val=1
        if crryw=='Moderately satisfied':cr_ms_val=1
        if crryw=='Moderately dissatisfied':cr_md_val=1
        else :cr_md_val=1

        #DevType
        st.subheader("Select an option.") 
        dv_std_val=0
        dv_em_val=0
        dv_da_val=0
        dv_d_val=0
        dv_ent_val=0
        dv_sys_val=0
        dv_qa_val=0
        dv_dev_val=0
        dv_prd_val=0
        devs=['Database administrator','DevOps specialist','Full stack developer','System administrator','Data or business_analyst','Desktop or enterprise applications developer',
        'Game or graphics developer','QA or test developer','Student','Back end developer','Front end developer','C suite executive CEO/CTO',
            'Engineering manager','Mobile developer','Designer','Marketing or sales professional','Embedded applications or devices developer','Educator or academic researcher',
            'Data scientist or machine learning specialist','Product manager']
        devsw= st.multiselect('7-Which of the following describe you? Please select all that apply.',  (devs), key = "devs")
        for i in devsw:
            if i=='Student':dv_std_val=1
            if i=='Engineering manager':dv_em_val=1
            if i=='Database administrator':dv_da_val=1
            if i=='Designer':dv_d_val=1
            if i=='Desktop or enterprise applications developer':dv_ent_val=1
            if i=='System administrator':dv_sys_val=1
            if i=='QA or test developer':dv_qa_val=1
            if i=='DevOps specialist':dv_dev_val=1
            if i=='Product manager':dv_prd_val=1
            else :dv_prd_val=1
       #LanguageWithWork
        st.subheader("Select an option.") 
        worked_php=0
        worked_cplus=0
        worked_html=0
        worked_go=0
        worked_css=0
        worked_type=0
        languages=['JavaScript','Python','Bash','C','Cplusplus','Java','Matlab','R','SQL','TypeScript','HTML','CSS','Assembly','CoffeeScript','Erlang','Go','Lua','Ruby','PHP','VB.NET','C#',
        'Swift','Kotlin','Objective','Rust','Groovy','Scala','F#','Haskell','Perl','Visual','VBA','Ocaml','Delphi','Julia','Hack','Clojure','Cobol']
        languageWorkedWith= st.multiselect('8-Which of the following programming, scripting, and markup languages have you done extensive development work in over the past year, and which do you want to work in over the next year?',  (languages), key = "te")
        for i in languageWorkedWith:
            if i=='PHP':worked_php=1
            if i=='C++':worked_cplus=1
            if i=='HTML':worked_html=1
            if i=='Go':worked_go=1
            if i=='Css':worked_css=1
            if i=='TypeScript':worked_type=1
            else :worked_type=1

        #CompanySize
        st.subheader("Select an option.")
        CompanySize_Small_5_60 = st.selectbox('9-Approximately how many people are employed by the company or organization you work for?', ('No','Yes'),key = "cc")
    
        #Country
        st.subheader("Select an option.")
        ccs=['Andorra','Liechtenstein','Venezuela, Bolivarian Republic of...','Ireland','Botswana','Saint Lucia','Togo','Luxembourg','Iceland','United States','Norway','Switzerland','New Zealand','South Korea','United Kingdom','Australia','Sierra Leone','United Arab Emirates','Tunisia','Canada','Denmark','Israel','Italy','Mongolia','Other Country (Not Listed Above)','Uruguay','Zimbabwe','Germany','Qatar','Singapore','France','Bahamas','Spain','Netherlands','Malta','Japan','Finland','Sweden','Austria','Jamaica','Saudi Arabia','Belgium','South Africa','Bosnia and Herzegovina','Hong Kong (S.A.R.)','Cuba','Afghanistan','Libyan Arab Jamahiriya','Brazil','Oman','Chile','Slovenia','Cyprus','Republic of Moldova','Romania','Greece','Estonia','Malaysia','Portugal','Colombia','Bulgaria','Taiwan','Czech Republic','Marshall Islands','Mauritius','Serbia','Costa Rica','Hungary','Lithuania','Kuwait','Ecuador','Turkey','Lesotho','Armenia','Thailand','Montenegro','Poland','Argentina','China','Democratic Republic of the Congo','Latvia','Panama','The former Yugoslav Republic of Macedonia','Republic of Korea','Dominican Republic','Slovakia','Croatia','Paraguay','Kenya','Nicaragua','Jordan','Lebanon','Bahrain','Mexico','Tajikistan','Belarus','Trinidad and Tobago','India','Georgia','Ukraine','Bangladesh','United Republic of Tanzania','Russian Federation','Cameroon','Benin','Algeria','Nigeria','Madagascar','Philippines','Albania','Guatemala','El Salvador','Cambodia','Morocco','Peru','Honduras','Senegal','Viet Nam','Suriname','Barbados','Mozambique','Malawi','Uganda','Azerbaijan','Pakistan','Rwanda','Kazakhstan','Myanmar','Indonesia','Somalia','Iran, Islamic Republic of...','Namibia','Uzbekistan','Ghana','Sri Lanka','Nepal','Maldives','Bolivia','Turkmenistan','Congo, Republic of the...','Egypt','Fiji','Syrian Arab Republic','Yemen','Kyrgyzstan','Sudan','Ethiopia','Iraq','Guyana','Bhutan','Gambia',"Côte d'Ivoire",'Eritrea','Swaziland','Zambia','Dominica','Monaco']
        ccs=sorted(ccs)
        country= st.selectbox('10-In which country do you currently reside?', (ccs),key = "co")

        extremely_high_90k_198k= ['Andorra','Liechtenstein','Venezuela, Bolivarian Republic of...','Ireland','Botswana','Saint Lucia','Togo','Luxembourg','Iceland','United States','Norway','Switzerland','New Zealand','South Korea','United Kingdom','Australia','Sierra Leone','United Arab Emirates','Tunisia','Canada','Denmark','Israel','Italy','Mongolia','Other Country (Not Listed Above)','Uruguay','Zimbabwe','Germany','Qatar']
        very_high_65k_90k =['Singapore','France','Bahamas','Spain','Netherlands','Malta','Japan','Finland','Sweden','Austria','Jamaica','Saudi Arabia','Belgium','South Africa']
        high_45k_65k=['Bosnia and Herzegovina','Hong Kong (S.A.R.)','Cuba','Afghanistan','Libyan Arab Jamahiriya','Brazil','Oman','Chile','Slovenia','Cyprus','Republic of Moldova','Romania','Greece','Estonia','Malaysia','Portugal','Colombia','Bulgaria']
        moderate_25k_45k=['Taiwan','Czech Republic','Marshall Islands','Mauritius','Serbia','Costa Rica','Hungary','Lithuania','Kuwait','Ecuador','Turkey','Lesotho','Armenia','Thailand','Montenegro','Poland','Argentina','China','Democratic Republic of the Congo','Latvia','Panama','The former Yugoslav Republic of Macedonia','Republic of Korea','Dominican Republic','Slovakia','Croatia','Paraguay','Kenya','Nicaragua','Jordan','Lebanon','Bahrain','Mexico','Tajikistan','Belarus','Trinidad and Tobago','India','Georgia','Ukraine','Bangladesh','United Republic of Tanzania','Russian Federation','Cameroon']
        somewhat_low_10k_25k=['Benin','Algeria','Nigeria','Madagascar','Philippines','Albania','Guatemala','El Salvador','Cambodia','Morocco','Peru','Honduras','Senegal','Viet Nam','Suriname','Barbados','Mozambique','Malawi','Uganda','Azerbaijan','Pakistan','Rwanda','Kazakhstan','Myanmar','Indonesia','Somalia','Iran, Islamic Republic of...','Namibia','Uzbekistan','Ghana','Sri Lanka','Nepal','Maldives','Bolivia','Turkmenistan','Congo, Republic of the...','Egypt','Fiji','Syrian Arab Republic']
        low_less_10k =['Yemen','Kyrgyzstan','Sudan','Ethiopia','Iraq','Guyana','Bhutan','Gambia',"Côte d'Ivoire",'Eritrea','Swaziland','Zambia','Dominica','Monaco']

        extremely_val=0
        very_high_val=0
        high_val=0
        moderate_val=0
        somewhat_val=0
        low_less_val=0

        for i in extremely_high_90k_198k:
            if i==country:extremely_val=1
        for i in very_high_65k_90k:       
            if i==country:very_high_val=1
        for i in high_45k_65k:       
            if i==country:high_val=1 
        for i in moderate_25k_45k:       
            if i==country:moderate_val=1
        for i in somewhat_low_10k_25k:       
            if i==country:somewhat_val=1
        for i in low_less_10k:       
            if i==country:low_less_val=1
        
        #IDE
        st.subheader("Select an option.")
        ide_ntpdpls_val=0
        ide_eclp_val=0
        ide_phpstr_val=0
        ide_subtxt_val=0
        ide_intl_val=0
        ide_vscde_val=0
        ide_andr_val=0
        idem=['Python Jupyter','Sublime Text','Vim','Notepadplusplus','Visual Studio','Visual Studio Code','Intellij',
        'PyCharm','Atom','Android Studio','Xcode','PHPStorm','Eclipse','NetBeans','Emacs',
        'RStudio','Coda','RubyMine','Zend','Komodo','TextMate','Light Table']

        idew = st.multiselect('11-Which development environment(s) do you use regularly?  Please check all that apply.' ,idem, key = "ide")
        for i in idew:
            if idew =='Notepadplusplus':ide_ntpdpls_val=1
            if idew =='Eclipse':ide_eclp_val=1
            if idew =='PHPStorm':ide_phpstr_val=1
            if idew =='Sublime Text':ide_subtxt_val=1
            if idew =='Intellij':ide_intl_val=1
            if idew =='Visual Studio Code':ide_vscde_val=1
            if idew =='Android Studio':ide_andr_val=1
            else :ide_andr_val=1



        #CommunicationTools
        st.subheader("Select an option.")
        ct_sl_val=0
        ct_co_val=0
        ct_ji_val=0
        ct_tr_val=0
        ct_gh_val=0
        ct_ot_val=0
        ctool=['Slack','Jira','Google Hangouts Chat','Facebook','Trello','Confluence','HipChat','Stack Overflow Enterprise','Office Productivity Suite Microsoft Office Google Suite','Other Wiki Tool Github Google Sites Proprietary Software','Other Chat System IRC Proprietary Software']
        ctoolw=st.multiselect('12-Which of the following tools do you use to communicate, coordinate, or share knowledge with your coworkers? Please select all that apply.',  (ctool), key = "ctool")
        for i in ctoolw:
            if i=='Slack':ct_sl_val=1
            if i=='Confluence':ct_co_val=1
            if i=='Jira':ct_ji_val=1
            if i=='Trello':ct_tr_val=1
            if i=='Google Hangouts Chat':ct_gh_val=1
            if i=='Other Wiki Tool Github Google Sites Proprietary Software':ct_ot_val=1
            else :ct_ot_val=1

        #ASSESSBENEFITS8
        #st.subheader("Select an option.")
        #ab8_go_val=0
        #ab8=['Good','Bad']
        #ab8w=st.selectbox("Would you mind an computer/office equipment allowance?",  (ab8), key = "ab8")
        #if ab8w=='Good':ab8_go_val=1
        
        #PlatformDesireNextYear
        st.subheader("Select an option.")
        pltdsr_and_val=0
        pltdsr_rasp_val=0
        pltdsr_her_val=0
        pltdsr_srvr_val=0
        pltdsr_aws_val=0
        pltdsr=['Linux','Arduino','Windows Desktop or Server','AWS','Mac OS','Serverless''Android','Google Cloud Platform App Engine','iOS','Firebase',
        'Google Home','Raspberry','Azure','Heroku','Salesforce','SharePoint','WordPress','ESP8266','Amazon Echo','Apple Watch or Apple TV',
        'Gaming Console','IBM Cloud or Watson','Mainframe','Windows Phone','Drupal','Predix']
        pltdsrw = st.multiselect('13-Which of the following platforms have you done extensive development work for over the past year?', (pltdsr), key = "pltdsr")
        for i in pltdsrw:
            if pltdsrw =='Android':pltdsr_and_val=1
            if pltdsrw =='Raspberry':pltdsr_rasp_val=1
            if pltdsrw =='Heroku': pltdsr_her_val=1
            if pltdsrw =='Serverless':pltdsr_srvr_val=1
            if pltdsrw =='AWS':pltdsr_aws_val=1
            else :pltdsr_aws_val=1

        #LanguageDesireNextYear
        st.subheader("Select an option.")
        lng_jvascp_val=0
        lng_sql_val=0
        lng_html_val=0
        lng_css_val=0
        lng_php_val=0
        lng=['Python','Go','Assembly','C','Cplusplus','Matlab''SQL','Bash Shell','C#','Java',
       'JavaScript','TypeScript','HTML','CSS','Erlang','Rust','F#','Haskell','Ocaml','Swift',
       'Kotlin','PHP','Scala','Ruby','CoffeeScript','Perl','Delphi Object Pascal','Groovy','Hack','VB NET',
       'R','Objective C','VBA','Lua','Julia','Clojure','Visual Basic','Cobol']
        lngw = st.multiselect('14-Which of the following programming, scripting, and markup languages have you done extensive development work in over the past year, and which do you want to work in over the next year? ', (lng), key = "lng")
        for i in lngw:
            if lngw =='JavaScript':lng_jvascp_val=1
            if lngw =='SQL':lng_sql_val=1
            if lngw =='HTML':lng_html_val=1
            if lngw =='CSS':lng_css_val=1
            if lngw =='PHP':lng_php_val=1
            else :lng_php_val=1

        #DatabaseDesireNextYear
        st.subheader("Select an option.")
        db_mngdb_val=0
        db_sqlsrv_val=0
        db_mysql_val=0
        db=['PostgreSQL Oracle','IBM Db','Redis','Amazon DynamoDB','Apache Hive','Amazon RDS Aurora''Neo4j',
            'SQL Server','Elasticsearch','SQLite','Google BigQuery','MySQL','MongoDB','MariaDB''Memcached',
            'Microsoft AzureTables CosmosDB SQL','Google Cloud Storage','Cassandra','Apache HBase','Amazon Redshift'] 
        dbw = st.multiselect('15-Which of the following database environments have you done extensive development work in over the past year, and which do you want to work in over the next year?', (db), key = "db")
            
        for i in dbw:
            if dbw =='MongoDB':db_mngdb_val=1
            if dbw =='SQL Server':db_sqlsrv_val=1
            if dbw =='MySQL':db_mysql_val=1
            else :db_mysql_val=1
            

        #HopeFiveYears
        st.subheader("Select an option")
        hop_engman_val=0
        hop_hvng_val=0
        hop_anthr_val=0
        hop=['Anotherrole','Engineeringmanager','Havingown Company','ProductManager','Retirement','UnrelatedCareer''Same']
        hopw = st.selectbox('16-Which of the following best describes what you hope to be doing in five years?', (hop), key = "hop")
        if hopw =='Engineeringmanager':hop_engman_val=1
        if hopw =='Havingown Company':hop_hvng_val=1
        if hopw =='Anotherrole':hop_anthr_val=1
        else :hop_anthr_val=1




        #DATAFRAME
        # 'AssessBenefits8_Good':ab8_go_val,            
        data = {'Dependents_No':Dependents_No,'YearsCodingProf':YearsCodingProf,'CompanySize_Small_5_60':CompanySize_Small_5_60,'Country_extremely_high_90k_198k':extremely_val,'Country_very_high_65k_90k':very_high_val,
               'Country_high_45k_65k':high_val,'Country_moderate_25k_45k':moderate_val,'Country_somewhat_low_10k_25k':somewhat_val,'Country_low_less_10k':low_less_val,
               'Age_18_24_years_old':age_18_24_val,'Age_35_44_years_old':age_35_44_val,'Age_25_34_years_old':age_25_34_val,'Age_45_54_years_old':age_45_66_val,
               'YearsCodingProf_0_2_years':yc_0_2_val,'YearsCodingProf_3_5_years':yc_3_5_val,'YearsCodingProf_6_8_years':yc_6_8_val,'YearsCodingProf_9_11_years':yc_9_11_val,'YearsCodingProf_30_or_more_years':yc_11_30_val,
               'LanguageWorkedWith_PHP':worked_php,'LanguageWorkedWith_Cplusplus':worked_cplus,'LanguageWorkedWith_HTML':worked_html,'LanguageWorkedWith_Go':worked_go,'LanguageWorkedWith_CSS':worked_css,'LanguageWorkedWith_TypeScript':worked_type,
                'DevType_Student':dv_std_val,'DevType_Engineering_manager':dv_em_val,'DevType_Database_administrator':dv_da_val,'DevType_Designer':dv_d_val,'DevType_Desktop_or_enterprise_applications_developer':dv_ent_val,'DevType_System_administrator':dv_sys_val,'DevType_QA_or_test_developer':dv_qa_val,'DevType_DevOps_specialist':dv_dev_val,'DevType_Product_manager':dv_prd_val,
                'CareerSatisfaction_Extremelysatisfied':cr_exs_val,'CareerSatisfaction_Moderatelysatisfied':cr_ms_val,'CareerSatisfaction_Moderatelydissatisfied':cr_md_val,
                'CommunicationTools_Slack':ct_sl_val,'CommunicationTools_Confluence':ct_co_val,'CommunicationTools_Jira':ct_ji_val,'CommunicationTools_Trello':ct_tr_val,'CommunicationTools_Google_Hangouts_Chat':ct_gh_val,'CommunicationTools_Other_wiki_tool__Github_Google_Sites_proprietary_software__':ct_ot_val,
                'Employment_self_employed':emp_slf_val,'Employment_full_time':emp_fll_val,'Employment_part_time':emp_prt_val
                 ,'PlatformWorkedWith_AWS':plt_aws_val,'PlatformWorkedWith_Windows_Desktop_or_Server':plt_windw_val,'PlatformWorkedWith_Mac_OS':plt_mac_val,
                'PlatformWorkedWith_Raspberry_Pi':plt_ras_val,'PlatformWorkedWith_Azure':plt_azr_val,'PlatformWorkedWith_Google_Cloud_Platform_App_Engine':plt_ggl_cld_val,'PlatformWorkedWith_Android':plt_and_val,
                'PlatformDesireNextYear_Android':pltdsr_and_val,'PlatformDesireNextYear_Raspberry_Pi':pltdsr_rasp_val,
                'PlatformDesireNextYear_Heroku':pltdsr_her_val,
                'PlatformDesireNextYear_Serverless':pltdsr_srvr_val,'PlatformDesireNextYear_AWS':pltdsr_aws_val,
                'LanguageDesireNextYear_JavaScript':lng_jvascp_val,'LanguageDesireNextYear_SQL':lng_sql_val,
                'LanguageDesireNextYear_HTML':lng_html_val,
                'LanguageDesireNextYear_CSS':lng_css_val,'LanguageDesireNextYear_PHP':lng_php_val,
                'HopeFiveYears_Engineeringmanager':hop_engman_val,'HopeFiveYears_Havingown_company':hop_hvng_val,
                'HopeFiveYears_Anotherrole':hop_anthr_val,'IDE_Notepadplusplus':ide_ntpdpls_val,'IDE_Eclipse':ide_eclp_val,
                'IDE_PHPStorm':ide_phpstr_val,'IDE_Sublime_Text':ide_subtxt_val,'IDE_IntelliJ':ide_intl_val,
                'IDE_Visual_Studio_Code':ide_vscde_val,'IDE_Android_Studio':ide_andr_val,'DatabaseDesireNextYear_MongoDB':db_mngdb_val,'DatabaseDesireNextYear_SQL_Server':db_sqlsrv_val,
                'DatabaseDesireNextYear_MySQL':db_mysql_val

                }
 
        features_df = pd.DataFrame.from_dict([data])
        st.markdown("<h3></h3>", unsafe_allow_html=True)
        st.write('Overview of input is shown below')
        st.markdown("<h3></h3>", unsafe_allow_html=True)
        st.dataframe(features_df)

        #Preprocess inputs
        preprocess_df = preprocess(features_df, 'Online')
        print(preprocess_df.columns)
        prediction = model.predict(preprocess_df)

        if st.button('Predict'):
            
            st.warning(prediction)
            
    else:
        st.subheader("Dataset upload")
        uploaded_file = st.file_uploader("Choose a file")
        if uploaded_file is not None:
            data = pd.read_csv(uploaded_file,encoding= 'utf-8')
            #Get overview of data
            st.write(data.head())
            st.markdown("<h3></h3>", unsafe_allow_html=True)
            #Preprocess inputs
            preprocess_df = preprocess(data, "Batch")
            if st.button('Predict'):
                #Get batch prediction
                prediction = model.predict(preprocess_df)
                prediction_df = pd.DataFrame(prediction, columns=["Predictions"])
                prediction_df = prediction_df.replace({1:'Yes, the passenger survive.', 0:'No, the passenger died'})

                st.markdown("<h3></h3>", unsafe_allow_html=True)
                st.subheader('Prediction')
                st.write(prediction_df)    
        

if __name__ == '__main__':
        main()
