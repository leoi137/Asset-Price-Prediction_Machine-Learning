import pandas as pd


class CSVData():
    def __init__(self, csv_dir, symbols, start, end, kind):
        """
        Parameters:

        csv_dir: Location of the csv file, its directory

        symbols: Symbols used in a list format

        start: Start date in string format

        end: End date in string format
        
        kind: The type of asset you are using (currently only works
         with forex downloaded from QuantConnect due to it's format)
        """

        self.csv_dir = csv_dir
        self.symbols = symbols
        self.start = start
        self.end = end
        self.kind = kind
        
        self.data = {}
        
    def more_data(self, more_syms, more_csv_dir, frame):
        for s in more_syms:
            self.data['{}{}'.format(s, frame)] = pd.read_csv(more_csv_dir + s + '.csv', header = None)
    
    def get_data(self):
        if self.kind == 'Forex':
            
            for s in self.symbols:
                self.data[s] = pd.read_csv(self.csv_dir + s + '.csv', header = None)
                
            for s in self.symbols:
                self.data[s]['date'] = self.data[s][0]
                self.data[s].drop([0, 5, 10], axis = 1, inplace = True)
                self.data[s].set_index(pd.to_datetime(self.data[s]['date']), inplace = True)
                self.data[s].drop('date', axis = 1, inplace = True)
                self.data[s].columns = ['bidopen', 'bidhigh', 'bidlow', 'bidclose',
                                        'askopen', 'askhigh','asklow', 'askclose']
                self.data[s] = self.data[s][self.start:self.end]
                
            for s in self.symbols:
                self.data[s]['open'] = (self.data[s]['bidopen'] + self.data[s]['askopen']) / 2
                self.data[s]['high'] = (self.data[s]['bidhigh'] + self.data[s]['askhigh']) / 2
                self.data[s]['low'] = (self.data[s]['bidlow'] + self.data[s]['asklow']) / 2
                self.data[s]['close'] = (self.data[s]['bidclose'] + self.data[s]['askclose']) / 2
            
        return self.data
