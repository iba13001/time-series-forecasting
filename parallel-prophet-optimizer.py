# BEFORE RUNNING, dfv (DATAFRAME THAT HAS ALL TIMESERIES) AND ts_ids (ALL UNIQUE TIME-SERIES IDS) NEED TO BE DEFINED FIRST

def optimize_prophet(ts_id, custom_holiday = True):
  # PICK THE TIME SERIES TO OPTIMIZE
    dfvv = dfv[dfv.ts_id == ts_id]
    max_date_v = dfvv.ds.max()
  
  # CROSS VALIDATE EVERY MONTH FOR THE LAST 3 MONTHS
    cutoffs = pd.to_datetime([max_date_v - datetime.timedelta(days = 97),
                          max_date_v - datetime.timedelta(days = 67),
                          max_date_v - datetime.timedelta(days = 37)])
    mapes = []
  
  # DO NOT OPTIMIZE IF THE TIME-SERIES HAS LESS THAN TWO YEAR DATA AND USE DEFAULT VALUES
    if (dfvv.shape[0] < 2*52):
        print('less than 2-year training data for: ' + ts_id)
        opt_flex = pd.DataFrame()
        opt_flex['ts_id'] = [ts_id]
        opt_flex['flex'] = [0.05]
        return(opt_flex)
    #changepoint_ratio = 1 - 5/dfvv.shape[0]
  # ALLOW THE DETECTION OF CHANGE POINT ON ALL THE TRAIN DATA
    changepoint_ratio = 1
  # START OPTIMIZING
    for flex in flexs:
        #df.rename(columns = {'greg_amc_week_end_dt' : 'ds', s_metric : 'y'}, inplace = True)
        if custom_holiday:
            m = Prophet(interval_width = 0.8, weekly_seasonality = False, daily_seasonality = False,
                        holidays = holidays,
                        changepoint_prior_scale=flex,
                       changepoint_range = changepoint_ratio)
        else:
            m = Prophet(interval_width = 0.8, weekly_seasonality = False, daily_seasonality = False,
                        changepoint_prior_scale=flex,
                       changepoint_range = changepoint_ratio)
            
        m.add_country_holidays(country_name='US')
      # CHANGE THE ORDER OF FOURIER SERIES IF NEEDED
        #m.add_seasonality(name='monthly', period=30.5, fourier_order=10)
      # ADD ANY EXOG REGRESSORS IF NEEDED
        #m.add_regressor('cpi_all_items',standardize = True)
  
      # FIT
        m.fit(dfvv)
                
        # CROSS-VALIDATE
        df_cv = cross_validation(m, cutoffs=cutoffs, horizon = horizon, parallel = 'threads')
        mapes.append(performance_metrics(df_cv,rolling_window = 1).rmse[0])
        
        
    opt_flex = pd.DataFrame()
    opt_flex['ts_id'] = [ts_id]
  
  # IF THE VALUE IS > 0.5, IT'S VERY LIKELY AN OVERFITTED TREND. USE A MEDIAN VALUE 0.26
    if flexs[np.argmin(mapes)] > 0.5:
        opt_flex['flex'] = 0.26
    else:
        opt_flex['flex'] = flexs[np.argmin(mapes)]
    
    #opt_flex['flex'] = flexs[np.argmin(mapes)]
    print('FINISHED: ' + ts_id)
    
    return(opt_flex)

###########################
## RUN FUNC IN PARALLEL
###########################

start_time = time.time()
p = Pool(cpu_count())
results_holidays = p.map(partial(optimize_prophet, custom_holiday = True), ts_ids)
results_no_holidays = p.map(partial(optimize_prophet, custom_holiday = True), ts_ids)
p.close()
p.join()
print("--- %s seconds ---" % (time.time() - start_time))
