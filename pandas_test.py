import pandas as pd
import numpy as np
#add comment
data = {
    'name': ['Xavier', 'Ann', 'Jana', 'Yi', 'Robin', 'Amal', 'Nori'],
    'city': ['Mexico City', 'Toronto', 'Prague', 'Shanghai',
             'Manchester', 'Cairo', 'Osaka'],
    'age': [41, 28, 33, 34, 38, 31, 37],
    'py-score': [88.0, 79.0, 81.0, 80.0, 68.0, 61.0, 84.0]
}
row_labels = [10, 11, 12, 13, 14, 15, 16]

df = pd.DataFrame(data=data, index=row_labels)

#начало, конец фрейма
df.head(n=2)
df.tail(n=4)

#столбец с индексами
cities = df['city']
#numpy.ndarray
print(cities.values)

#Создание
d = {'x': [1, 2, 3], 'y': np.array([2, 4, 8]), 'z': 100}
df1 = pd.DataFrame(d)

#изменение порядка столбцов, индексации
df1 = pd.DataFrame(d, index=[100, 200, 300], columns=['z', 'y', 'x'])

#Создание из списка
l = [{'x': 1, 'y': 2, 'z': 100},
     {'x': 2, 'y': 4, 'z': 100},
     {'x': 3, 'y': 8, 'z': 100}]
pd.DataFrame(l)

#Создание из numpy array
#copy = False (по умолчанию) - данные в датафрейме меняются при изменении массива
arr = np.array([[1, 2, 100],
                [2, 4, 100],
                [3, 8, 100]])
df_ = pd.DataFrame(arr, columns=['x', 'y', 'z'], copy = True) 
arr[0, 0] = 1000

#Открытие, сохранение csv
df.to_csv('D:\proj\data.csv')
df = pd.read_csv('D:\proj\data.csv', index_col=0)

#метки строк столбцов
df.index
df.columns
#Данные без меток
df.values
df.to_numpy() #можно использовать dtype, copy

#Типы данных
df.dtypes
df_ = df.astype(dtype={'age': np.int32, 'py-score': np.float32})

#размерности
df.ndim #измерения
df.shape #размер NxM - кортеж
df.size #общее число элементов

#полуение данных
df.loc[101] #строка с меткой -тут при срезе включены оба конца
df.iloc[0] #строка по номеру -тут при срезе правый не включен

df.iloc[1:6:2, 0] #через каждую вторую строку

df['city'][101]

#изменение через loc
df.loc[:, 'py-score']
df.loc[:13, 'py-score'] = [40, 50, 60, 70]
df.loc[14:, 'py-score'] = 0

#изменение через iloc
df.iloc[:, -1] = np.array([88.0, 79.0, 81.0, 80.0, 68.0, 61.0, 84.0])
df['py-score']



df1 = pd.DataFrame({'name': ['Jeff', 'Esha', 'Jia'], 
                    'city' : ['New York', 'Mexico', 'Atlanta'],
                   'age': [30, 56, 8]})
df1
#изменение значений в столбце по условию
df1.loc[(df1['city'] == 'New York') | (df1['city'] == 'Atlanta'), 'age'] += 1
df1
#cut для диапазонов (10 включен в первый диапазон 0-10)
    age_bins = [0, 10, 18, 30, 50, 65, 80, 120]
    age_labels = ['0-10', '11-18', '19-30', '31-50', '51-65', '66-80', '81-120']
    
    df1['age_bracket'] = pd.cut(df1['age'], bins=age_bins, labels=age_labels)
type(df1['city'][0])


####Вставка, удаление строк
john = pd.Series(data=['John', 'Boston', 34, 79], index=df.columns, name=17)

#вставка в конец
df = df.append(john)
df
#удаление
df_new = df.drop(labels=[17])
df_new
df.drop(labels=[17])
df.drop(labels=[17], inplace = True) # inplace = True - изменяет исходный датафрейм, False(по умолч) - не изменяет
df

####Вставка, удаление столбцов

df['js-score'] = np.array([71.0, 95.0, 88.0, 79.0, 91.0, 91.0, 80.0])
df['total-score'] = 0.0
df
#Вставка в указанное место
df.insert(loc=4, column='django-score', value=np.array([86.0, 81.0, 78.0, 88.0, 74.0, 70.0, 81.0]))
df
#Удаление столбца
del df['total-score']
df
#удаляет и возвращает столбец
a = df.pop('total-score')
print(a)
#тоже удаление
df.drop(labels='age', axis=1) #не изменяет исходный, а inplace = True - изменяет исходный
df


#операции
(df['py-score'] * 0.96445532).round(2)
df['total'] = 0.4 * df['py-score'] + 0.3 * df['django-score'] + 0.3 * df['js-score']
df


#среднее numpy с весами (лин комбинац), передается часть df
del df['total']
df['total'] = np.average(df.iloc[:, 2:5], axis=1, weights=[0.4, 0.3, 0.3])
df

#СОртировка
df.sort_values(by='js-score', ascending=False) #inplace=True - изменяет исходный df
df.sort_values(by=['total', 'py-score'], ascending=[False, False])


#Фильтрация
filter_ = df['django-score'] >= 80
filter_
#df[filter_] выдаёт Pandas DataFrame со строками df, где в filter_ стоит True
df[filter_]

df[(df['django-score'] >= 80) ^ (df['js-score'] >= 80)] #^ XOR исключающее или(True когда только одно True)

df['django-score'].where(df['django-score'] >= 80, 1)


#статистика для числовых полей
df.describe()

#NAN  float('nan'), math.nan или numpy.nan
df_ = pd.DataFrame({'x': [1, 2, np.nan, 4]})
df_
df_.mean() #Пропускает Nan
df_.mean(skipna=False)

df_.fillna(0) #Замена на значение
df_.fillna(method='ffill') #замена на значение выше
df_.fillna(method='bfill') #замена на значение ниже

df_.interpolate()
# inplace тоже можно

#удалить строки с NAN
df_.dropna() #можно inplae = True




