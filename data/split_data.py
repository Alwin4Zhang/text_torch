from sklearn.model_selection import train_test_split
import csv
import pandas as pd
from os.path import dirname,join as join_path

X,y = [],[]
path_root = dirname(__file__)
data_path = join_path(path_root,'ch_auto.csv')
df = pd.read_csv(data_path)
list_label = df.columns.values
list_data = df.values.tolist()
for row in list_data:
    X.append(row[-1])
    y.append(row[1])

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=0)

x_test,x_val,y_test,y_val = train_test_split(X_test,y_test,test_size=0.5,random_state=0)

train_dataset =[('label','text')] + list(zip(y_train,X_train))
test_dataset =[('label','text')] + list(zip(y_test,x_test))
val_dataset =[('label','text')] + list(zip(y_val,x_val))

with open(join_path(path_root,'train.tsv'),'w') as csvfile:
    writer = csv.writer(csvfile)    
    writer.writerows(train_dataset)
    csvfile.close()

with open(join_path(path_root,'test.tsv'),'w') as csvfile:
    writer = csv.writer(csvfile)    
    writer.writerows(test_dataset)
    csvfile.close()

with open(join_path(path_root,'dev.tsv'),'w') as csvfile:
    writer = csv.writer(csvfile)    
    writer.writerows(val_dataset)
    csvfile.close()