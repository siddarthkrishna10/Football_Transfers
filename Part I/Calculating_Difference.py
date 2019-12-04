import pandas as pd
import numpy as np

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

#Reading the dataset into an object
a = pd.read_csv('https://github.com/siddarthkrishna10/Football_Transfers/blob/master/Part%20I/Transfers.csv')

#Cleaning the Dataset of NaN values
a1 = a.dropna()

#Grouping the dataset by each Season
a2 = a1.groupby('Season')

#Find the difference for each player across 14 seasons
a3 = a2.get_group('2004-2005')
a3['Difference'] = a3['Transfer_fee'] - a3['Market_value']
#Finding the mean of this difference for the season
mean3 = round(np.mean(a3.Difference))

a4 = a2.get_group('2005-2006')
a4['Difference'] = a4['Transfer_fee'] - a4['Market_value']
mean4 = round(np.mean(a4.Difference))

a5 = a2.get_group('2006-2007')
a5['Difference'] = a5['Transfer_fee'] - a5['Market_value']
mean5 = round(np.mean(a5.Difference))

a6 = a2.get_group('2007-2008')
a6['Difference'] = a6['Transfer_fee'] - a6['Market_value']
mean6 = round(np.mean(a6.Difference))

a7 = a2.get_group('2008-2009')
a7['Difference'] = a7['Transfer_fee'] - a7['Market_value']
mean7 = round(np.mean(a7.Difference))

a8 = a2.get_group('2009-2010')
a8['Difference'] = a8['Transfer_fee'] - a8['Market_value']
mean8 = round(np.mean(a8.Difference))

a9 = a2.get_group('2010-2011')
a9['Difference'] = a9['Transfer_fee'] - a9['Market_value']
mean9 = round(np.mean(a9.Difference))

a10 = a2.get_group('2011-2012')
a10['Difference'] = a10['Transfer_fee'] - a10['Market_value']
mean10 = round(np.mean(a10.Difference))

a11 = a2.get_group('2012-2013')
a11['Difference'] = a11['Transfer_fee'] - a11['Market_value']
mean11 = round(np.mean(a11.Difference))

a12 = a2.get_group('2013-2014')
a12['Difference'] = a12['Transfer_fee'] - a12['Market_value']
mean12 = round(np.mean(a12.Difference))

a13 = a2.get_group('2014-2015')
a13['Difference'] = a13['Transfer_fee'] - a13['Market_value']
mean13 = round(np.mean(a13.Difference))

a14 = a2.get_group('2015-2016')
a14['Difference'] = a14['Transfer_fee'] - a14['Market_value']
mean14 = round(np.mean(a14.Difference))

a15 = a2.get_group('2016-2017')
a15['Difference'] = a15['Transfer_fee'] - a15['Market_value']
mean15 = round(np.mean(a15.Difference))

a16 = a2.get_group('2017-2018')
a16['Difference'] = a16['Transfer_fee'] - a16['Market_value']
mean16 = round(np.mean(a16.Difference))

#Writing all the 14 mean values into a CSV file
mean = pd.DataFrame({'Season': ['2004/2005', '2005/2006', '2006/2007', '2007/2008', '2008/2009', '2009/2010',
                                '2010/2011', '2011/2012', '2012/2013', '2013/2014', '2014/2015', '2015/2016',
                                '2016/2017', '2017/2018'], 'Average Difference': ['%f' % mean3, '%f' % mean4,
                                                                                  '%f' % mean5, '%f' % mean6,
                                                                                  '%f' % mean7, '%f' % mean8,
                                                                                  '%f' % mean9, '%f' % mean10,
                                                                                  '%f' % mean11, '%f' % mean12,
                                                                                  '%f' % mean13, '%f' % mean14,
                                                                                  '%f' % mean15, '%f' % mean16]})
mean.to_csv('Mean_Table.csv')
