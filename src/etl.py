import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn import tree
from sklearn.model_selection import cross_val_score
from sklearn import svm
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score

def latency_data_prep(file18, file19, file20, file21, file22):
  latency_10 = pd.read_csv(file18)
  latency_50 = pd.read_csv(file19)
  latency_100 = pd.read_csv(file20)
  latency_150 = pd.read_csv(file21)
  latency_200 = pd.read_csv(file22)
  
  def pckt_count(a):
    a = a.split(';')
    a = [int(i) for i in a[:-1]]
    return len(a)

  latency_10['pck_ct'] = latency_10['packet_times'].apply(pckt_count)
  latency_50['pck_ct'] = latency_50['packet_times'].apply(pckt_count)
  latency_100['pck_ct'] = latency_100['packet_times'].apply(pckt_count)
  latency_150['pck_ct'] = latency_150['packet_times'].apply(pckt_count)
  latency_200['pck_ct'] = latency_200['packet_times'].apply(pckt_count)

  latency_10['packet_size_total'] = latency_10['1->2Pkts'] + latency_10['2->1Pkts']
  latency_50['packet_size_total'] = latency_50['1->2Pkts'] + latency_50['2->1Pkts']
  latency_100['packet_size_total'] = latency_100['1->2Pkts'] + latency_100['2->1Pkts']
  latency_150['packet_size_total'] = latency_150['1->2Pkts'] + latency_150['2->1Pkts']
  latency_200['packet_size_total'] = latency_200['1->2Pkts'] + latency_200['2->1Pkts']

  latency_10['total_bytes'] = latency_10['1->2Bytes'] + latency_10['2->1Bytes']
  latency_50['total_bytes'] = latency_50['1->2Bytes'] + latency_50['2->1Bytes']
  latency_100['total_bytes'] = latency_100['1->2Bytes'] + latency_100['2->1Bytes']
  latency_150['total_bytes'] = latency_150['1->2Bytes'] + latency_150['2->1Bytes']
  latency_200['total_bytes'] = latency_200['1->2Bytes'] + latency_200['2->1Bytes']

  latency_10['latency'] = [10] * len(latency_10)
  latency_50['latency'] = [50] * len(latency_50)
  latency_100['latency'] = [100] * len(latency_100)
  latency_150['latency'] = [150] * len(latency_150)
  latency_200['latency'] = [200] * len(latency_200)

  #number of changes in direction in "packet_dirs"

  def agg_10sec(df):
      new_df = pd.DataFrame()
      min_time = df["Time"][0]
      latency = df["latency"][0]
      while min_time < df["Time"][len(df)-1]:
          temp_df = df[(df["Time"] >= min_time) & (df["Time"] < min_time+10)]
          row = temp_df[["1->2Bytes", "2->1Bytes", "1->2Pkts", "2->1Pkts", "packet_size_total", "pck_ct", "total_bytes"]].sum().to_frame().T
          row["packet_sizes_var"] = temp_df["packet_size_total"].var()
          row["Time"] = min_time
        row["latency"] = latency
        new_df = new_df.append(row)
        min_time += 10
    return new_df.reset_index(drop=True)

  agg_10 = agg_10sec(latency_10)
  agg_50 = agg_10sec(latency_50)
  agg_100 = agg_10sec(latency_100)
  agg_150 = agg_10sec(latency_150)
  agg_200 = agg_10sec(latency_200)
  
  full_df = pd.concat([agg_10, agg_50, agg_100, agg_150, agg_200])
  
  return full_df

def loss_data_prep(file1, file2, file3, file4, file5, file6, file7, file8, file9, file10, file11, file12, file13, file14, file15, file16, file17):
  run_200_300 = pd.read_csv(file1)
  run_200_1000 = pd.read_csv(file2)
  run_200_25000 = pd.read_csv(file3)
  run_200_5000 = pd.read_csv(file4)
  run_200_500 = pd.read_csv(file5)
  run_200_750 = pd.read_csv(file6)
  run_200_200 = pd.read_csv(file7)
  run_200_400 = pd.read_csv(file8)
  run_200_600 = pd.read_csv(file9)
  run_200_800 = pd.read_csv(file10)
  run_200_1000 = pd.read_csv(file11)
  run_200_1200 = pd.read_csv(file12)
  run_200_800 = pd.read_csv(file13)
  run_200_2000 = pd.read_csv(file14)
  run_200_1400 = pd.read_csv(file15)
  run_200_1800 = pd.read_csv(file16)
  run_200_2500 = pd.read_csv(file17)
  
  def pckt_count(a):
    a = a.split(';')
    a = [int(i) for i in a[:-1]]
    return len(a)
  
  run_200_300['pck_ct'] = run_200_300['packet_times'].apply(pckt_count)
  run_200_1000['pck_ct'] = run_200_1000['packet_times'].apply(pckt_count)
  run_200_25000['pck_ct'] = run_200_25000['packet_times'].apply(pckt_count)
  run_200_5000['pck_ct'] = run_200_5000['packet_times'].apply(pckt_count)
  run_200_500['pck_ct'] = run_200_500['packet_times'].apply(pckt_count)
  run_200_750['pck_ct'] = run_200_750['packet_times'].apply(pckt_count)
  run_200_200['pck_ct'] = run_200_200['packet_times'].apply(pckt_count)
  run_200_400['pck_ct'] = run_200_400['packet_times'].apply(pckt_count)
  run_200_600['pck_ct'] = run_200_600['packet_times'].apply(pckt_count)
  run_200_800['pck_ct'] = run_200_800['packet_times'].apply(pckt_count)
  run_200_1000['pck_ct'] =  run_200_1000['packet_times'].apply(pckt_count)
  run_200_1200['pck_ct'] =  run_200_1200['packet_times'].apply(pckt_count)
  run_200_800['pck_ct'] = run_200_800['packet_times'].apply(pckt_count)
  run_200_2000['pck_ct'] =  run_200_2000['packet_times'].apply(pckt_count)
  run_200_1400['pck_ct'] =  run_200_1400['packet_times'].apply(pckt_count)
  run_200_1800['pck_ct'] = run_200_1800['packet_times'].apply(pckt_count)
  run_200_2500['pck_ct'] =  run_200_2500['packet_times'].apply(pckt_count)
  
  run_200_5000['packet_size_total'] = run_200_5000['1->2Pkts'] + run_200_5000['2->1Pkts']
  run_200_300['packet_size_total'] = run_200_300['1->2Pkts'] + run_200_300['2->1Pkts']
  run_200_1000['packet_size_total'] = run_200_1000['1->2Pkts'] + run_200_1000['2->1Pkts']
  run_200_25000['packet_size_total'] = run_200_25000['1->2Pkts'] + run_200_25000['2->1Pkts']
  run_200_500['packet_size_total'] =run_200_500['1->2Pkts'] + run_200_25000['2->1Pkts']
  run_200_750['packet_size_total'] =run_200_750['1->2Pkts'] + run_200_25000['2->1Pkts']
  run_200_200['packet_size_total'] =run_200_200['1->2Pkts'] + run_200_25000['2->1Pkts']
  run_200_400['packet_size_total'] =run_200_400['1->2Pkts'] + run_200_25000['2->1Pkts']
  run_200_600['packet_size_total'] =run_200_600['1->2Pkts'] + run_200_25000['2->1Pkts']
  run_200_800['packet_size_total'] =run_200_800['1->2Pkts'] + run_200_25000['2->1Pkts']
  run_200_1000['packet_size_total'] = run_200_1000['1->2Pkts'] + run_200_25000['2->1Pkts']
  run_200_1200['packet_size_total'] = run_200_1200['1->2Pkts'] + run_200_25000['2->1Pkts']
  run_200_800['packet_size_total'] = run_200_800['1->2Pkts'] + run_200_25000['2->1Pkts']
  run_200_2000['packet_size_total'] = run_200_2000['1->2Pkts'] + run_200_25000['2->1Pkts']
  run_200_1400['packet_size_total'] = run_200_1400['1->2Pkts'] + run_200_25000['2->1Pkts']
  run_200_1800['packet_size_total'] = run_200_1800['1->2Pkts'] + run_200_25000['2->1Pkts']
  run_200_2500['packet_size_total'] = run_200_2500['1->2Pkts'] + run_200_25000['2->1Pkts']
  
  ratio_200_5000  = [0.04 for i in range(310)] 
  ratio_200_300 = [0.67 for i in range(324)]
  ratio_200_1000 = [0.2 for i in range(318)]
  ratio_200_25000 = [0.008 for i in range(len(run_200_25000))]
  ratio_200_500 = [0.4 for i in range(len(run_200_500))] 
  ratio_200_750= [0.266 for i in range(len(run_200_750))]
  ratio_200_200= [1 for i in range(len(run_200_200))]
  ratio_200_400= [0.5 for i in range(len(run_200_400))]
  ratio_200_600= [0.333 for i in range(len(run_200_600))]
  ratio_200_800= [0.25 for i in range(len(run_200_800))]
  ratio_200_1000= [0.2 for i in range(len(run_200_1000))]
  ratio_200_1200= [0.166 for i in range(len(run_200_1200))]
  ratio_200_800= [0.25 for i in range(len(run_200_800))]
  ratio_200_2000= [0.1 for i in range(len(run_200_2000))]
  ratio_200_1400= [0.142 for i in range(len(run_200_1400))]
  ratio_200_1800= [0.111 for i in range(len(run_200_1800))]
  ratio_200_2500= [0.08 for i in range(len(run_200_2500))]
  
  run_200_5000['ratio'] = ratio_200_5000
  run_200_300['ratio'] = ratio_200_300
  run_200_1000['ratio'] = ratio_200_1000
  run_200_25000['ratio'] = ratio_200_25000
  run_200_500['ratio'] = ratio_200_500
  run_200_750['ratio'] = ratio_200_750
  run_200_200['ratio'] = ratio_200_200
  run_200_400['ratio'] = ratio_200_400
  run_200_600['ratio'] = ratio_200_600
  run_200_800['ratio'] = ratio_200_800
  run_200_1000['ratio'] = ratio_200_1000
  run_200_1200['ratio'] = ratio_200_1200
  run_200_800['ratio'] = ratio_200_800
  run_200_2000['ratio'] = ratio_200_2000
  run_200_1400['ratio'] = ratio_200_1400
  run_200_1800['ratio'] = ratio_200_1800
  run_200_2500['ratio'] = ratio_200_2500
  
  def avg_time_delt(a):
    return pd.Series(a.split(';')[:-1]).astype(int).diff().mean()
  
  def agg_10sec(df):
    new_df = pd.DataFrame()
    min_time = df["Time"][25]
    ratio = df["ratio"][25]
    while min_time < df["Time"][len(df)-1]:
        temp_df = df[(df["Time"] >= min_time) & (df["Time"] < min_time+10)]
        row = temp_df[["1->2Bytes", "2->1Bytes", "1->2Pkts", "2->1Pkts", "packet_size_total", "pck_ct"]].sum().to_frame().T
        row["packet_sizes_var"] = temp_df["packet_size_total"].var()
        row['avg_time_delta'] = temp_df['packet_times'].apply(avg_time_delt).mean()
        row["Time"] = min_time
        row["ratio"] = ratio
        new_df = new_df.append(row)
        min_time += 10
    return new_df.reset_index(drop=True)
  
  agg_200_300 = agg_10sec(run_200_300)
  agg_200_1000 = agg_10sec(run_200_1000)
  agg_200_5000 = agg_10sec(run_200_5000)
  agg_200_25000 = agg_10sec(run_200_25000)
  agg_200_500 = agg_10sec(run_200_500)
  agg_200_750 = agg_10sec(run_200_750)
  agg_200_200 = agg_10sec(run_200_200)
  agg_200_400 = agg_10sec(run_200_400)
  agg_200_600 = agg_10sec(run_200_600)
  agg_200_800 = agg_10sec(run_200_800)
  agg_200_1000 =  agg_10sec(run_200_1000)
  agg_200_2000 =  agg_10sec(run_200_2000)
  agg_200_1400 =  agg_10sec(run_200_1400)
  agg_200_1800 =  agg_10sec(run_200_1800)
  agg_200_2500 =  agg_10sec(run_200_2500)
  
  full_df = pd.concat([agg_200_300, agg_200_1000, agg_200_5000, agg_200_25000, agg_200_500, agg_200_750, agg_200_200, agg_200_600, agg_200_1000, agg_200_2000])
  
  #start of linear regression
  
  ###ENTIRE DATASET
  full_df = pd.concat([agg_200_300, agg_200_1000, agg_200_5000, agg_200_25000, agg_200_500, agg_200_750, agg_200_200, agg_200_600, agg_200_1000, agg_200_2000])

  # Load the diabetes dataset
  features = ['packet_size_total', 'pck_ct', 'packet_sizes_var','avg_time_delta', 'Time']
  df_X = full_df[features]
  df_y = full_df['ratio']

  # Use only one feature
  #diabetes_X = diabetes_X[:, np.newaxis, 2]

  # Split the data into training/testing sets
  X_train, X_rem, y_train, y_rem = train_test_split(df_X, df_y, train_size=0.8, random_state=42)

  # new data 

  full_df = pd.concat([agg_200_400, agg_200_800, agg_200_1400, agg_200_1800])

  # Load the diabetes dataset
  features = ['packet_size_total', 'pck_ct', 'packet_sizes_var','avg_time_delta', 'Time']
  df_X = full_df[features]
  df_y = full_df['ratio']

  filler, X_rem, filler, y_rem = train_test_split(df_X, df_y, train_size=0.8, random_state=42)

  X_valid, X_test, y_valid, y_test = train_test_split(X_rem, y_rem, test_size=0.5)

  # Create linear regression object
  regr = linear_model.LinearRegression()

  # Train the model using the training sets
  regr.fit(X_train, y_train)

  # Make predictions using the testing set
  y_pred = regr.predict(X_test)

  # The coefficients
  #print("Coefficients: \n", regr.coef_)
  # The mean squared error
  #print("Mean squared error: %.2f" % mean_squared_error(y_test, y_pred))
  # The coefficient of determination: 1 is perfect prediction
  #print("Coefficient of determination: %.2f" % r2_score(y_test, y_pred))
  
  return r2_score(y_test, y_pred) #r2_score(y_test.reset_index(drop=True), y_pred), y_test.reset_index(drop=True), y_pred

def latency_linear_reg(df):
  # Load the dataset 'packet_sizes_var', "Time"
  features = ['packet_size_total', 'pck_ct', "total_bytes", 'packet_sizes_var']
  df_X = df[features]
  df_y = df['latency']

  # Split the data into training/testing sets
  X_train, X_rem, y_train, y_rem = train_test_split(df_X, df_y, train_size=0.8, random_state=42)

  X_valid, X_test, y_valid, y_test = train_test_split(X_rem, y_rem, test_size=0.5)

  # Create linear regression object
  regr = linear_model.LinearRegression()

  # Train the model using the training sets
  regr.fit(X_train, y_train)

  # Make predictions using the testing set
  y_pred = regr.predict(X_test)

  # The coefficients
  #print("Coefficients: \n", regr.coef_)

  # The mean squared error
  #print("Mean squared error: %.2f" % mean_squared_error(y_test, y_pred))

  # The coefficient of determination: 1 is perfect prediction
  #print("Coefficient of determination: %.2f" % r2_score(y_test, y_pred))
  return r2_score(y_test.reset_index(drop=True), y_pred)#, y_test.reset_index(drop=True), y_pred

def decision_tree(df):
  # Load the dataset 'packet_sizes_var', "Time"
  features = ['packet_size_total', 'pck_ct', "total_bytes", 'packet_sizes_var']
  df_X = df[features]
  df_y = df['latency']

  # Split the data into training/testing sets
  X_train, X_rem, y_train, y_rem = train_test_split(df_X, df_y, train_size=0.8, random_state=42)

  X_valid, X_test, y_valid, y_test = train_test_split(X_rem, y_rem, test_size=0.5)

  # Create linear regression object
  clf = tree.DecisionTreeClassifier()

  # Train the model using the training sets
  clf = clf.fit(X_train, y_train)

  # Make predictions using the testing set
  y_pred = clf.predict(X_test)

  # The coefficients
  #print("Coefficients: \n", regr.coef_)

  # The mean squared error
  #print("Mean squared error: %.2f" % mean_squared_error(y_test, y_pred))

  # The coefficient of determination: 1 is perfect prediction
  #print("Coefficient of determination: %.2f" % r2_score(y_test, y_pred))

  return clf.score(X_test, y_test.reset_index(drop=True))#, y_test.reset_index(drop=True), y_pred
  
def svm(df):
  # Load the dataset 'packet_sizes_var', "Time"
  features = ['packet_size_total', 'pck_ct', "total_bytes", 'packet_sizes_var', 'Time']
  df_X = df[features]
  df_y = df['latency']

  # Split the data into training/testing sets
  X_train, X_rem, y_train, y_rem = train_test_split(df_X, df_y, train_size=0.8, random_state=42)

  X_valid, X_test, y_valid, y_test = train_test_split(X_rem, y_rem, test_size=0.5)

  # Create linear regression object
  clf = make_pipeline(StandardScaler(), SVC(gamma='auto'))

  # Train the model using the training sets
  clf = clf.fit(X_train, y_train)

  # Make predictions using the testing set
  y_pred = clf.predict(X_test)

  # The coefficients
  #print("Coefficients: \n", regr.coef_)

  # The mean squared error
  #print("Mean squared error: %.2f" % mean_squared_error(y_test, y_pred))

  # The coefficient of determination: 1 is perfect prediction
  #print("Coefficient of determination: %.2f" % r2_score(y_test, y_pred))

  return clf.score(X_test, y_test.reset_index(drop=True))#, y_test.reset_index(drop=True), y_pred
