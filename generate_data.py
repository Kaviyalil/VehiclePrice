import pandas as pd, numpy as np
np.random.seed(42)

makes = {
    'Toyota':    {'models':['Camry','Corolla','RAV4','Highlander','Prius','Sienna','4Runner','Tacoma','Sequoia','Venza','Land Cruiser'],'base':18000,'engines':[1.8,2.0,2.5,3.5,4.0,5.7]},
    'Honda':     {'models':['Civic','Accord','CR-V','Pilot','Odyssey','Passport','HR-V','Ridgeline'],'base':16000,'engines':[1.5,1.8,2.4,3.5]},
    'Ford':      {'models':['Mustang','F-150','Explorer','Bronco','Expedition','Edge','Fusion','Escape'],'base':20000,'engines':[1.5,2.0,2.3,3.5,5.0]},
    'BMW':       {'models':['3 Series','5 Series','X5','X3','X1','7 Series','i4','X7'],'base':40000,'engines':[2.0,3.0,4.4]},
    'Mercedes':  {'models':['C-Class','E-Class','ML-Class','GLC','GLE','S-Class'],'base':45000,'engines':[2.0,3.0,4.7]},
    'Audi':      {'models':['A4','A6','Q5','Q3','A3','A5','e-tron'],'base':38000,'engines':[1.8,2.0,3.0]},
    'Chevrolet': {'models':['Malibu','Equinox','Tahoe','Traverse','Blazer','Silverado'],'base':22000,'engines':[1.5,2.5,3.6,5.3]},
    'Hyundai':   {'models':['Elantra','Sonata','Tucson','Santa Fe','Ioniq 5','Palisade'],'base':17000,'engines':[2.0,2.4,3.8]},
    'Nissan':    {'models':['Altima','Sentra','Rogue','Pathfinder','Murano','Maxima','Frontier'],'base':16000,'engines':[1.8,2.5,3.5,3.8]},
    'Kia':       {'models':['Optima','Sorento','Sportage','Telluride','EV6','Stinger'],'base':18000,'engines':[2.0,2.4,3.8]},
    'Mazda':     {'models':['Mazda6','CX-5','CX-9','Mazda3','MX-5','CX-3'],'base':17000,'engines':[2.0,2.5]},
    'Volkswagen':{'models':['Passat','Tiguan','Atlas','Jetta','Golf'],'base':19000,'engines':[1.4,1.8,2.0,3.6]},
    'Subaru':    {'models':['Outback','Forester','Ascent','Impreza','Legacy'],'base':18000,'engines':[2.0,2.4,2.5]},
    'Tesla':     {'models':['Model 3','Model Y'],'base':45000,'engines':[0.0]},
    'Tata':      {'models':['Nexon','Harrier','Safari','Tiago','Altroz','Punch','Tigor','Curvv'],'base':8000,'engines':[1.2,1.5,2.0]},
    'Renault':   {'models':['Kwid','Duster','Triber','Kiger','Lodgy','Captur'],'base':6000,'engines':[1.0,1.5]},
}

rows = []
for make, info in makes.items():
    for model in info['models']:
        for _ in range(80):
            year         = int(np.random.randint(2013, 2025))
            engine       = float(np.random.choice(info['engines']))
            mileage      = int(np.random.randint(3000, 130000))
            transmission = np.random.choice(['Automatic','Manual'])
            fuel         = np.random.choice(['Petrol','Diesel','Electric','Hybrid']) if engine > 0 else 'Electric'
            owner        = np.random.choice(['First','Second','Third'])
            age          = 2024 - year
            base         = info['base']
            price        = base - age*(base*0.07) - mileage*0.04 + engine*1500
            if transmission == 'Automatic': price += 1500
            if fuel == 'Diesel':            price += 1000
            if fuel == 'Electric':          price += 5000
            if fuel == 'Hybrid':            price += 3000
            if owner == 'Second':           price *= 0.85
            if owner == 'Third':            price *= 0.72
            price = max(2000, price + np.random.normal(0, base*0.05))
            rows.append({'Year':year,'Make':make,'Model':model,'Engine_Size':engine,
                         'Mileage':mileage,'Transmission':transmission,
                         'Fuel_Type':fuel,'Owner_Type':owner,'Price':round(price,2)})

df = pd.DataFrame(rows)
df.to_csv('dataset.csv', index=False)
print('Total rows:', len(df))
print(df['Price'].describe())
