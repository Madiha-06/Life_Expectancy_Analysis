import pandas as pd

class DataCollector:
      def __init__(self,link):
            self.link = link

      def data_collect(self):
         pd.set_option('display.max_columns', None)
         dataset=pd.read_csv(self.link)
         return dataset
