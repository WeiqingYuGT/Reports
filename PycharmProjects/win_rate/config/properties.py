sample_rate = 0.01

feature_list = ['r_timestamp','sl_adjusted_confidence','uid_type','device_type', 'os',
            'device_make','bundle','banner_size','sp_user_gender',
            'pub_bid_rates','isp','adomain','device_model']

target_set = ['winbid','ad_impression','xad_revenue','pub_revenue']

feature_list_bk = ['tsrc_id','sp_iab_category','user_iab_category','fp_sic','adv_bid_rates','category',
                   'city','zip', 'int_banner','adgroup_id','state','sp_user_age','pub_type','carrier']

feature_to_use = ['sl_adjusted_confidence', 'uid_type', 'device_type', 'os',
                    'device_make', 'bundle', 'banner_size', 'sp_user_gender',
                    'pub_bid_rates', 'isp', 'adomain', 'device_model','hour','weekday']

neptune_vendor_list = [0,19,96]

daily_path = 's3a://xad-science/rti/daily_etl/'

lookback_window = 15

n_split = 1

retension_days = 40

dump_loc = './data/'

bid_price_lower_bound = 0.00028

chain_model_location = './data/models/wr_model_object.data'

deliver_model_location = './data/models/delivery_model_object.data'

upload_location = 's3://xad-science/rti/models/'

n_estimators = 100
max_depth = 12
n_jobs = 8
min_impurity_decrease = 0.0001
max_features = 0.5

bid_price_quantiles = None