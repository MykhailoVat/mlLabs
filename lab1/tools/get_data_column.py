import csv

def get_data_column(i_column):
     with open("../data_zdo/zdo_data.csv", 'r', newline='') as data:
         reader = csv.reader(data)
         next(reader)

         values = []
         for row in reader:
             value = float(row[i_column])
             values.append(value)

         return values


