import sys, os, getpass
from matplotlib import pyplot as plt
plt.style.use('ggplot')
if getpass.getuser() == 'LIB3':
    sys.path.append(r'C:\Program Files\IBM\ILOG\CPLEX_Studio1210\cplex\python\3.7\x64_win64')

import pandas
import warnings
import plotly.graph_objects as go
import plotly.express as px  
from plotly.subplots import make_subplots
import numpy as np
from plotly.subplots import make_subplots
from docplex.mp.model import Model
from docplex.util.environment import get_environment
import timeit
import winsound
import time

frequency = 2500  # Set Frequency To 2500 Hertz
duration = 1000  # Set Duration To 1000 ms == 1 second
acre_to_cfs=1.9835
cms_to_cfs=35.31
required_hours_for_min_storage=48
min_storage=0/acre_to_cfs #336 case study
max_storage=570*3/acre_to_cfs #570 case study
min_flow=50
hourly_ramp_factor=.5
hourly_spilage_factor=1
max_flow= 728 #728 case stdy
max_power=18.3    #18.3 case study 
conversion_factor=max_power/max_flow

warnings.simplefilter(action='ignore', category=FutureWarning)
pandas.options.mode.chained_assignment = None  # default='warn'

def read_flow_year_month(flow_year_inp,flow_month_inp, data_frame):
    crit1 = data_frame['Date'].map(lambda x : x.year == int(flow_year_inp))
    crit2 = data_frame['Date'].map(lambda x : x.month == int(flow_month_inp))
    selected_df=data_frame[crit1 & crit2]
    return selected_df
def read_forecast_data (price_year_inp,flow_month_inp, df3):
    crit1 = df3['Date'].map(lambda x : x.year == int(price_year_inp))
    crit2 = df3['Date'].map(lambda x : x.month == int(flow_month_inp))
    forecast=df3[crit1 & crit2 & (df3['lead_time_hours']>=1) 
                          & (df3['lead_time_hours']<=24)]
    return forecast
def prepare_input_data_optimization ( ):
    df3=pandas.read_excel('./LMP_RTM_ANTLER_6_N001.xlsx',
                          sheet_name=price_year_inp)
    selected_hourly_price=read_flow_year_month(price_year_inp,flow_month_inp, df3) #real time price data
    data_frame_avg_hourly=selected_hourly_price.mean(axis=0)
    df3_lmp=pandas.read_excel('./LMP_DAM_ANTLER_6_N001.xlsx',\
                          sheet_name=price_year_inp)
    #print(df3_lmp)
    selected_hourly_price_lmp=read_flow_year_month(price_year_inp,flow_month_inp, df3_lmp)
    
    #read forecast data 
    df3_forecast=pandas.read_csv('./HydroForecast_trinity-center-v6.csv')
    df3_forecast['Date'] = pandas.to_datetime(df3_forecast['Date'])
    forecast=read_forecast_data (price_year_inp,flow_month_inp, df3_forecast)

    df3_forecast2=pandas.read_csv('./Persistence.csv')
    df3_forecast2['Date'] = pandas.to_datetime(df3_forecast2['Date'])
    forecast2=read_forecast_data (price_year_inp,flow_month_inp, df3_forecast2)
    
    df3_obs=pandas.read_csv('./Observations.csv')
    df3_obs['Date'] = pandas.to_datetime(df3_obs['Date'])
    observation=read_flow_year_month(price_year_inp,flow_month_inp, df3_obs)
   
    observation['discharge'].reset_index(drop=True,inplace=True)
    forecast['discharge_q0.5'].reset_index(drop=True,inplace=True)
    forecast2['discharge'].reset_index(drop=True,inplace=True)
    selected_hourly_price_lmp.reset_index(drop=True,inplace=True)
    selected_hourly_price.reset_index(drop=True,inplace=True)
    real_time_price=selected_hourly_price.set_index('Date').stack().reset_index\
    (level=1, drop=True).to_frame('Merged').reset_index()
    day_ahead_price=selected_hourly_price_lmp.set_index('Date').stack().reset_index\
    (level=1, drop=True).to_frame('Merged').reset_index()
    model_input=pandas.DataFrame(observation['discharge']*cms_to_cfs)
    model_input['Upstream_q0.5']=forecast['discharge_q0.5']*cms_to_cfs
    model_input['Persistence']=forecast2['discharge']*cms_to_cfs
    model_input['Real_time_price']=real_time_price['Merged']
    model_input['day_ahead_price']=day_ahead_price['Merged']
    #data_list_input=model_input.values.tolist()
    #print(data_list_input)

    return model_input

def create_model_day_ahead (forecast_flow,price_day_ahead):
    #time_range,data_list_input,read_flow_constraints=prepar_input_data_optimization ( )
    solution_table_milp=pandas.DataFrame()
    inceased_storage_factor=1
    Big_m=1000000
    time_range=range(0, len(forecast_flow))
    #from docplex.mp.model import Model
    start = timeit.default_timer()
    tm = Model(name='MILP_Hydropower_flexibility_valuation_tool')
    tm.parameters.mip.tolerances.mipgap = 0.01
    tm.parameters.timelimit = 1200  
    if forecast_flow.min()>min_flow:
        flow_min=min_flow
    else:
        flow_min=forecast_flow.min()
              
    Q = {(i):tm.continuous_var(name='Q_{0}'.format(i)) for i in time_range} #realeased water via turbine
    R = {(i):tm.continuous_var(name='R_{0}'.format(i)) for i in time_range} #reservoir storage 
    P = {(i):tm.continuous_var(name='P_{0}'.format(i)) for i in time_range} #spilled amount of water 

    total_revenue=tm.sum(
        model_input['day_ahead_price'].loc[i]
        * Q[i]*conversion_factor # A simple linear approximation
        for i in time_range
    )
    tm.maximize(total_revenue)
    #writing constraints
    for i in time_range:
        if i==0:
            tm.add_constraint(Q[i]<=forecast_flow.loc[i])
        tm.add_constraint(Q[i]<=max_flow)
        #tm.add_constraint(R[i]==0) #reservoir flow storage 
        if i>=1:
            tm.add_constraint(R[i]-R[i-1]==forecast_flow.loc[i]-Q[i]-P[i]) #reservoir flow storage 
            #tm.add_constraint(R[i]-R[i-1]==forecast_flow.loc[i]-Q[i]) #reservoir flow storage 
            tm.add_constraint(R[i]<=max_storage) #maximum storage 
            if i>required_hours_for_min_storage:
                tm.add_constraint(R[i]>=min_storage)#minimum storage at reservoir 
            tm.add_constraint(Q[i]-Q[i-1]<=hourly_ramp_factor*Q[i]) #ramp up constraints 
            tm.add_constraint(Q[i-1]-Q[i]<=hourly_ramp_factor*Q[i-1]) #ramp down constraints 
            tm.add_constraint(P[i]-P[i-1]<= hourly_spilage_factor*P[i]) #spillage up constraint 
            tm.add_constraint(P[i-1]-P[i]<= hourly_spilage_factor*P[i-1]) #spillage down constraint
            tm.add_constraint(Q[i]>=flow_min)
        if i==0:
            tm.add_constraint(R[i]==0)
    
    #tm.export_as_lp(basename="Hydropower_%s", path="C:/Users/RONIMS/Documents/C++ project/Hydropower_flexibility_evalutation")
    
    # tm.print_information()
    print('End of creating model')
    #tm.export_as_lp(basename="Hydropower_%s", path="C:/Users/RONIMS/Documents/C++ project/Hydropower_flexibility_evalutation")
    tms = tm.solve(log_output=False)
    # tms.display()
    revenue=tms.objective_value
    for i in time_range:
        power = Q[i].solution_value*conversion_factor
        solution_table_milp=solution_table_milp.append(pandas.DataFrame({"Optimal flow":Q[i].solution_value,\
                                                                         "Storage":R[i].solution_value,
                                                     "By pass":P[i].solution_value, 
                                                       "Power dispatch":power},index=[0]),ignore_index=True)
   
    assert tms is not None, "model can't solve"
    print('End of solving DA model, revenue: {:>.2f}'.format(revenue))
    stop = timeit.default_timer()
    print('Time: ', stop - start)
    return solution_table_milp,revenue

def create_model_real_time(forecast_flow,observed_flow,price_real_time,price_day_ahead):
    
    day_ahead_mdl_output,day_ahead_rev =create_model_day_ahead (forecast_flow,price_day_ahead)
    solution_table_milp=pandas.DataFrame()
    #hourly_ramp_factor=.1
  
    inceased_storage_factor=1
    Big_m=1000000
    time_range=range(0, len(observed_flow))
    if observed_flow.min()>min_flow:
        flow_min=min_flow
    else:
        flow_min=int(observed_flow.min())

    #from docplex.mp.model import Model
    start = timeit.default_timer()
    tm = Model(name='MILP_Hydropower_flexibility_valuation_tool')
    tm.parameters.mip.tolerances.mipgap = 0.01
    tm.parameters.timelimit = 1200  
    Q = {(i):tm.continuous_var(name='Q_{0}'.format(i)) for i in time_range} #realeased water via turbine
    R = {(i):tm.continuous_var(name='R_{0}'.format(i)) for i in time_range} #reservoir storage 
    P = {(i):tm.continuous_var(name='P_{0}'.format(i)) for i in time_range} #spilled amount of water
    A = {(i):tm.continuous_var(name='A_{0}'.format(i)) for i in time_range} #Absolute value
    
    total_revenue=tm.sum(0-A[i] for i in time_range)
    tm.maximize(total_revenue)
    #writing constraints
    for i in time_range:
        if i==0:
            tm.add_constraint(Q[i]<=observed_flow.loc[i])
        tm.add_constraint(Q[i]<=max_flow)
        #tm.add_constraint(R[i]==0) 
        if i>=1:
            tm.add_constraint(R[i]-R[i-1]==observed_flow.loc[i]-Q[i]-P[i]) #reservoir flow storage 
            #tm.add_constraint(R[i]-R[i-1]==observed_flow.loc[i]-Q[i]) #reservoir flow storage 
            tm.add_constraint(R[i]<=max_storage) #maximum storage 
            if i>required_hours_for_min_storage:
                tm.add_constraint(R[i]>=min_storage)#minimum storage at reservoir  
            tm.add_constraint(Q[i]-Q[i-1]<=hourly_ramp_factor*Q[i]) #ramp up constraint 
            tm.add_constraint(Q[i-1]-Q[i]<=hourly_ramp_factor*Q[i-1]) #ramp down constraint 
            tm.add_constraint(P[i]-P[i-1]<= hourly_spilage_factor*P[i]) #spillage up constraint 
            tm.add_constraint(P[i-1]-P[i]<= hourly_spilage_factor*P[i-1]) #spillage down constraint
            tm.add_constraint(Q[i]>=flow_min)
        if i==0:
            tm.add_constraint(R[i]==0)
            
        tm.add_constraint(
            A[i]>=price_real_time.loc[i]*(
                Q[i]*conversion_factor - day_ahead_mdl_output.iloc[i][3]
            )
        )
        tm.add_constraint(
            A[i]>=price_real_time.loc[i]*(
                day_ahead_mdl_output.iloc[i][3] - Q[i]*conversion_factor
            )
        )

    
    #tm.export_as_lp(basename="Hydropower_%s", path="C:/Users/RONIMS/Documents/C++ project/Hydropower_flexibility_evalutation")
    
    # tm.print_information()
    print('End of creating model')
    # tm.export_as_lp(basename="Hydropower_%s", path="C:/Users/RONIMS/Documents/C++ project/Hydropower_flexibility_evalutation")
    tms = tm.solve(log_output=False)
    assert tms is not None, "model can't solve"
    revenue=tms.objective_value
    total_revenue=revenue+day_ahead_rev
    # tms.display()
    for i in time_range:
        power = Q[i].solution_value*conversion_factor
        solution_table_milp=solution_table_milp.append(pandas.DataFrame({"Optimal flow":Q[i].solution_value,\
                                                                         "Storage":R[i].solution_value,
                                                     "By pass":P[i].solution_value, 
                                                       "Power dispatch":power},index=[0]),ignore_index=True)
    solution_table_milp.loc[:, 'DA_LMP'] = price_day_ahead.values
    solution_table_milp.loc[:, 'RT_LMP'] = price_real_time.values
    solution_table_milp.loc[:, 'FLOW_OBS'] = observed_flow.values
    solution_table_milp.loc[:, 'FLOW_FCS'] = forecast_flow.values
    solution_table_milp.loc[:, 'DA_FLOW_STORE'] = day_ahead_mdl_output.loc[:, 'Storage'].values
    solution_table_milp.loc[:, 'RT_FLOW_STORE'] = solution_table_milp.loc[:, 'Storage'].values
    solution_table_milp.loc[:, 'DA_FLOW_BYPASS'] = day_ahead_mdl_output.loc[:, 'By pass'].values
    solution_table_milp.loc[:, 'RT_FLOW_BYPASS'] = solution_table_milp.loc[:, 'By pass'].values
    solution_table_milp.loc[:, 'DA_FLOW_DISPATCH'] = day_ahead_mdl_output.loc[:, 'Optimal flow'].values
    solution_table_milp.loc[:, 'RT_FLOW_DISPATCH'] = solution_table_milp.loc[:, 'Optimal flow'].values
    solution_table_milp.loc[:, 'DA_POWER'] = day_ahead_mdl_output.loc[:, 'Power dispatch'].values
    solution_table_milp.loc[:, 'RT_POWER'] = solution_table_milp.loc[:, 'Power dispatch'].values
 
    print('End of solving RT model, total revenue: {:>.2f}'.format(total_revenue))
    stop = timeit.default_timer()
    print('Time: ', stop - start)
    return solution_table_milp,total_revenue, flow_min
if __name__=='__main__':
    ls_month = range(1, 13)
    dict_revenue_p50 = dict()
    dict_time_p50 = dict()
    dict_df_rt_p50 = dict()
    min_flow_list=[]
    for m in ls_month:
        print('Month: ' + str(m))
        flow_year_inp, flow_month_inp, price_year_inp = '2020', str(m), '2020'
        model_input = prepare_input_data_optimization()

        # First, use upstream p50 forecasts
        forecast_flow   = model_input['Upstream_q0.5']
        #forecast_flow = model_input['Persistence']
        #forecast_flow = model_input['discharge']
        price_day_ahead = model_input['day_ahead_price']
        observed_flow   = model_input['discharge']
        price_real_time = model_input['Real_time_price']
        t0 = time.time()
   
        try:
            solution_table_milp, revenue, flow_min = create_model_real_time(forecast_flow, observed_flow, price_real_time, price_day_ahead)
            dict_revenue_p50[m] = revenue
            min_flow_list.append(flow_min)
            dict_df_rt_p50[m] = solution_table_milp
        except:
            dict_revenue_p50[m] = None
            dict_df_rt_p50[m] = None
            print('Errors in model solving')

        dict_time_p50[m] = time.time() - t0

        #print minflow
        ls_rev_p50 = list()
    for m in ls_month:
        ls_rev_p50.append(dict_revenue_p50[m])
    with open('C:/Users/RONIMS/source/repos/Hydropower Flexibility Valuation Tool/min_flow.txt', 'w') as f:
        f.write("%s\n" % min_flow_list)
        f.write("%s\n" % ls_rev_p50)
    df_rev = pandas.DataFrame(
        {
            #'Constant outflow': ls_rev_obs,
            'Revenue ': ls_rev_p50,
            #'No storage between days': ls_rev_per,
            #'CNRFC': ls_rev_sfl,
        },
        index=['Jan', 'Feb','Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'],
        #index=['Jan', 'Feb','Mar', 'Apr', 'May', 'Jun'],
        #index=ls_month,
    )
    (df_rev/1E3).plot.bar()
    plt.ylabel('Total revenue(Thousand $)',fontsize=20)
    plt.xlabel('Month',fontsize=20)
    plt.legend(prop={"size":20})
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.show() 
