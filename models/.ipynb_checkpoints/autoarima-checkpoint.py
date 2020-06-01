import pmdarima as pm
import pandas as pd
import datetime

cod_ccaa_cell = {'AN': 'Andalucía',
                 'AR': 'Aragón',
                 'AS': 'Principado de Asturias',
                 'IB': 'Islas Baleares',
                 'CN': 'Canarias',
                 'CB': 'Cantabria',
                 'CM': 'Castilla-La Mancha',
                 'CL': 'Castilla y León',
                 'CT': 'Cataluña',
                 'CE': 'Ceuta',
                 'VC': 'Comunidad Valenciana',
                 'EX': 'Extremadura',
                 'GA': 'Galicia',
                 'MD': 'Comunidad de Madrid',
                 # 'ME': 'Melilla',
                 'MC': 'Región de Murcia',
                 'NC': 'Comunidad Foral de Navarra',
                 'PV': 'País Vasco',
                 'RI': 'La Rioja'}

# results_url = "https://covid19.isciii.es/resources/serie_historica_acumulados.csv"
results_url = "../data/serie_historica_acumulados.csv"

end_date_training = datetime.date(2020, 4, 15)
end_date_testing = datetime.date(2020, 4, 30)

df = pd.read_csv(results_url, engine='python')
df = df.fillna(0)

df = df[df['CCAA'].isin(cod_ccaa_cell.keys())]

df['FECHA'] = pd.to_datetime(df['FECHA'], format="%d/%m/%Y")

df = df.set_index('FECHA')

df['CASOS'] = df['CASOS'] + df['PCR+']
df = df.drop(columns=['PCR+', 'TestAc+'])

n_periods = 7
for date_predict in [end_date_training + datetime.timedelta(days=x) for x in range(2, 17)]:
    df_result = pd.DataFrame(columns=["CCAA", "FECHA", 'Recuperados', 'CASOS', 'Fallecidos', 'UCI', 'Hospitalizados'])

    train_df = df[:date_predict]
    for ccaa_key in cod_ccaa_cell.keys():
        ccaa_df_train = train_df[train_df['CCAA'] == ccaa_key]
        for feature_daily in ['CASOS', 'Recuperados', 'Fallecidos']:
            ccaa_df_train[feature_daily] = ccaa_df_train[feature_daily].diff().fillna(0)

        df_ca_result = pd.DataFrame(
            columns=["CCAA", "FECHA", 'Recuperados', 'CASOS', 'Fallecidos', 'UCI', 'Hospitalizados'])

        for feature in ['Recuperados', 'CASOS', 'Fallecidos', 'UCI', 'Hospitalizados']:

            model = pm.auto_arima(ccaa_df_train[feature],
                                  start_p=1, start_q=1,
                                  test='adf',  # use adftest to find optimal 'd'
                                  max_p=3, max_q=3,  # maximum p and q
                                  m=1,  # frequency of series
                                  d=None,  # let model determine 'd'
                                  seasonal=False,  # No Seasonality
                                  start_P=0,
                                  D=0,
                                  trace=True,
                                  error_action='ignore',
                                  suppress_warnings=True,
                                  stepwise=True)

            # Forecast
            fc, confint = model.predict(n_periods=n_periods, return_conf_int=True)
            index_of_fc = [date_predict + datetime.timedelta(days=x) for x in range(n_periods)]

            # make series for plotting purpose
            fc_series = pd.Series(fc, index=index_of_fc)
            df_ca_result[feature] = fc_series

        df_ca_result['CCAA'] = ccaa_key
        df_ca_result = df_ca_result.reset_index()
        df_ca_result['FECHA'] = df_ca_result['index']
        df_ca_result = df_ca_result.drop(columns=['index'])

        df_result = df_result.append(df_ca_result)

    df_result = df_result.reset_index(drop=True)
    df_result.to_csv("../output/ESL_MTM{0}_{1}_{2}.csv".format(date_predict.day, date_predict.month, date_predict.year),
                     index=False)
